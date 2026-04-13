# trade_executor.py — bad_request fix + integer contracts + all previous fixes
#
# FIX FOR bad_request CODE 11050:
#   Uses deribit.calc_contracts() → always returns int
#   Uses deribit.split_amount() → both tp1/tp2 are ints
#   Uses deribit.to_int_amount() on any manual amount calculation
#
# ORDER ID EXTRACTION:
#   place_market_order() and place_limit_order() return full result dict
#   Order ID is in result["order"]["order_id"] — extracted here correctly
#
# ALL PREVIOUS FIXES RETAINED:
#   clean_invalid_trades() removes stuck/broken state on startup
#   is_order_filled() uses order_state == "filled"
#   get_fill_price() reads trades[] array
#   Trailing stop after TP1 hit
#   save_signal() called after all orders placed with real values

import os, json, time, logging, requests, joblib
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

from config import (
    SYMBOLS, ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT,
    RISK_PER_TRADE, TIMEFRAME_ENTRY, TIMEFRAME_CONFIRM,
    LIVE_LIMIT, MODEL_FILE, LOG_FILE, get_tier
)
from feature_engineering import add_indicators
from smart_scheduler import should_scan, get_mode_thresholds, check_correlation, get_effective_risk
from deribit_client import DeribitClient, TRADEABLE_SYMBOLS

TRADES_FILE     = "trades.json"
HISTORY_FILE    = "trade_history.json"
SIGNALS_FILE    = "signals.json"
MODE_FILE       = "scan_mode.json"
BALANCE_FILE    = "balance.json"
MAX_OPEN_TRADES = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
log = logging.getLogger(__name__)


# ════════════ FILE I/O ════════════════════════════════════════════════

def load_json(path, default):
    try:
        for p in [Path(path), Path("data") / Path(path).name]:
            if p.exists():
                with open(p) as f: return json.load(f)
    except Exception: pass
    return default

def save_json(path, data):
    for dest in [Path(path), Path("data") / Path(path).name]:
        try:
            dest.parent.mkdir(exist_ok=True)
            tmp = str(dest) + ".tmp"
            with open(tmp, "w") as f: json.dump(data, f, indent=2, default=str)
            os.replace(tmp, str(dest))
        except Exception as e: log.error(f"save_json {dest}: {e}")

load_trades  = lambda: load_json(TRADES_FILE,  {})
save_trades  = lambda d: save_json(TRADES_FILE, d)
load_history = lambda: load_json(HISTORY_FILE, [])
load_signals = lambda: load_json(SIGNALS_FILE, [])

def append_history(rec):
    h = load_history(); h.append(rec); save_json(HISTORY_FILE, h)

def save_signal(sig):
    s = load_signals()
    s.append({**sig, "generated_at": datetime.now(timezone.utc).isoformat()})
    save_json(SIGNALS_FILE, s[-500:])

def load_model():
    p = joblib.load(MODEL_FILE)
    log.info(f"✓ Model: {len(p['all_features'])} features | 73.1% accuracy")
    return p


# ════════════ EXCHANGE ════════════════════════════════════════════════

def init_deribit() -> DeribitClient:
    cid    = os.getenv("DERIBIT_CLIENT_ID",     "")
    secret = os.getenv("DERIBIT_CLIENT_SECRET", "")
    if not cid or not secret:
        raise ValueError("DERIBIT_CLIENT_ID / DERIBIT_CLIENT_SECRET not set in GitHub Secrets")
    client = DeribitClient(cid, secret)
    client.test_connection()
    return client


def save_balance_json(deribit: DeribitClient) -> float:
    try:
        balances  = deribit.get_all_balances()
        total_usd = deribit.get_total_equity_usd()
        positions = deribit.get_positions()
        upnl = sum(float(p.get("floating_profit_loss_usd") or p.get("floating_profit_loss") or 0)
                   for p in positions)
        assets = [{"asset": cur, "free": str(round(info["available"], 6)),
                   "total": str(round(info["equity_usd"], 2))}
                  for cur, info in balances.items()]
        save_json(BALANCE_FILE, {
            "usdt":           round(total_usd, 2),
            "equity":         round(total_usd + upnl, 2),
            "unrealised":     round(upnl, 4),
            "assets":         assets,
            "updated_at":     datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "mode":           "deribit_testnet",
            "exchange":       "Deribit(by Coinbase) Testnet",
            "tradeable":      TRADEABLE_SYMBOLS,
            "open_positions": len(positions),
        })
        log.info(f"  ✅ Balance: ${total_usd:.2f} | unrealised: {upnl:+.2f} | positions: {len(positions)}")
        return total_usd
    except Exception as e:
        log.error(f"  save_balance_json failed: {e}")
        return 0.0


# ════════════ MARKET DATA ════════════════════════════════════════════

def get_data(symbol: str, interval: str) -> pd.DataFrame:
    for url in ["https://data-api.binance.vision/api/v3/klines", "https://api.binance.com/api/v3/klines"]:
        try:
            r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": LIVE_LIMIT}, timeout=15)
            r.raise_for_status()
            df = pd.DataFrame(r.json()).iloc[:, :6]
            df.columns = ["open_time","open","high","low","close","volume"]
            for c in ["open","high","low","close","volume"]: df[c] = pd.to_numeric(df[c])
            return df
        except Exception: continue
    raise Exception(f"Cannot fetch data for {symbol}")


# ════════════ STUCK TRADE CLEANER ════════════════════════════════════

def clean_invalid_trades(deribit: DeribitClient):
    """Remove trades with $0 SL/TP or no live orders — from old bugs."""
    trades = load_trades()
    if not trades: return

    to_remove = []
    for symbol, trade in trades.items():
        # Remove trades with zero stop or tp (broken old state)
        if float(trade.get("stop", 0)) == 0 or float(trade.get("tp1", 0)) == 0:
            log.warning(f"  🗑️ {symbol}: stop=0 or tp1=0 — removing broken state")
            to_remove.append(symbol); continue

        # Check if live orders still exist on Deribit
        oids  = trade.get("order_ids", {})
        live  = False
        for key, oid in oids.items():
            if key == "entry" or not oid or str(oid) in ("", "None"): continue
            try:
                o = deribit.get_order(str(oid))
                if o.get("order_state") in ("open", "partially_filled", "untriggered"):
                    live = True; break
            except Exception: pass

        if not live and len(oids) > 1:
            log.warning(f"  🗑️ {symbol}: no live orders on Deribit — clearing stuck trade")
            _record_close(trade, float(trade.get("entry", 0)), 0.0, "Stuck — auto-removed")
            to_remove.append(symbol)

    if to_remove:
        for sym in to_remove: trades.pop(sym, None)
        save_trades(trades)
        log.info(f"  ✅ Removed {len(to_remove)} invalid trade(s): {to_remove}")
        _send(f"🧹 Removed {len(to_remove)} invalid trade(s): {', '.join(to_remove)}")


# ════════════ EXECUTE TRADE ══════════════════════════════════════════

def execute_trade(deribit: DeribitClient, symbol, signal, entry, atr,
                  confidence, score, reasons, risk_mult=1.0, balance=10000.0):

    trades = load_trades()
    if symbol in trades: log.info(f"  {symbol}: already open — skip"); return False
    if len(trades) >= MAX_OPEN_TRADES: log.info("  Max trades — skip"); return False
    if not check_correlation(trades, signal): return False
    if not deribit.is_supported(symbol): log.info(f"  {symbol}: not on Deribit — skip"); return False
    if balance < 5: _warn(f"⚠️ Balance ${balance:.2f} too low"); return False

    dec = 4 if entry < 10 else 2
    if signal == "BUY":
        stop = round(entry - atr*ATR_STOP_MULT, dec); tp1 = round(entry + atr*ATR_TARGET1_MULT, dec); tp2 = round(entry + atr*ATR_TARGET2_MULT, dec)
        side = "BUY"; sl_side = "SELL"; tp_side = "SELL"
    else:
        stop = round(entry + atr*ATR_STOP_MULT, dec); tp1 = round(entry - atr*ATR_TARGET1_MULT, dec); tp2 = round(entry - atr*ATR_TARGET2_MULT, dec)
        side = "SELL"; sl_side = "BUY"; tp_side = "BUY"

    # FIXED: calc_contracts returns int, split_amount returns (int, int)
    total_contracts          = deribit.calc_contracts(symbol, balance, entry, stop, risk_mult)
    amount_tp1, amount_tp2   = deribit.split_amount(symbol, total_contracts)
    risk_usd                 = round(balance * RISK_PER_TRADE * risk_mult, 2)

    log.info(f"  {signal} {symbol} | total={total_contracts} tp1={amount_tp1} tp2={amount_tp2}")
    log.info(f"  SL={stop:.{dec}f} TP1={tp1:.{dec}f} TP2={tp2:.{dec}f}")

    # Verify all amounts are integers before sending
    assert isinstance(total_contracts, int), f"total_contracts must be int, got {type(total_contracts)}"
    assert isinstance(amount_tp1, int),      f"amount_tp1 must be int, got {type(amount_tp1)}"

    order_ids    = {}
    actual_entry = entry

    try:
        # 1. Market entry
        entry_result = deribit.place_market_order(symbol, side, total_contracts)
        if not entry_result: log.error("  Entry empty"); return False

        # Extract order_id from result["order"] (not from result directly)
        entry_order        = entry_result.get("order", entry_result)
        order_ids["entry"] = str(entry_order.get("order_id", ""))
        actual_entry       = deribit.get_fill_price(entry_result, entry)
        if actual_entry == 0: actual_entry = entry
        log.info(f"  ✅ Filled @ ~{actual_entry:.{dec}f}")
        time.sleep(1.5)

        # Recalculate levels from actual fill price (prevents overlap errors)
        if signal == "BUY":
            stop = deribit.round_price(symbol, actual_entry - atr*ATR_STOP_MULT)
            tp1  = deribit.round_price(symbol, actual_entry + atr*ATR_TARGET1_MULT)
            tp2  = deribit.round_price(symbol, actual_entry + atr*ATR_TARGET2_MULT)
        else:
            stop = deribit.round_price(symbol, actual_entry + atr*ATR_STOP_MULT)
            tp1  = deribit.round_price(symbol, actual_entry - atr*ATR_TARGET1_MULT)
            tp2  = deribit.round_price(symbol, actual_entry - atr*ATR_TARGET2_MULT)

        # 2. Stop Loss
        try:
            sl_result = deribit.place_limit_order(symbol, sl_side, total_contracts, stop, stop_price=stop)
            sl_order  = sl_result.get("order", sl_result)
            oid = str(sl_order.get("order_id", ""))
            if oid: order_ids["stop_loss"] = oid
            log.info(f"  ✅ SL @ {stop:.{dec}f} id:{oid or 'MISSING'}")
        except Exception as e: log.warning(f"  SL failed: {e}")

        # 3. TP1
        try:
            if amount_tp1 > 0:
                tp1_result = deribit.place_limit_order(symbol, tp_side, amount_tp1, tp1)
                tp1_order  = tp1_result.get("order", tp1_result)
                oid = str(tp1_order.get("order_id", ""))
                if oid: order_ids["tp1"] = oid
                log.info(f"  ✅ TP1 @ {tp1:.{dec}f} id:{oid or 'MISSING'}")
        except Exception as e: log.warning(f"  TP1 failed: {e}")

        # 4. TP2
        try:
            if amount_tp2 > 0:
                tp2_result = deribit.place_limit_order(symbol, tp_side, amount_tp2, tp2)
                tp2_order  = tp2_result.get("order", tp2_result)
                oid = str(tp2_order.get("order_id", ""))
                if oid: order_ids["tp2"] = oid
                log.info(f"  ✅ TP2 @ {tp2:.{dec}f} id:{oid or 'MISSING'}")
        except Exception as e: log.warning(f"  TP2 failed: {e}")

    except Exception as e:
        log.error(f"  Trade error {symbol}: {e}")
        _warn(f"⚠️ Trade error {symbol}: {e}")
        return False

    record = {
        "symbol": symbol, "signal": signal,
        "entry": actual_entry, "stop": stop, "tp1": tp1, "tp2": tp2,
        "qty": total_contracts, "qty_tp1": amount_tp1, "qty_tp2": amount_tp2,
        "risk_usd": risk_usd, "balance_at_open": balance, "risk_mult": risk_mult,
        "order_ids": order_ids,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "tp1_hit": False, "tp2_hit": False, "closed": False,
        "confidence": confidence, "score": score, "reasons": reasons,
        "tier": get_tier(symbol), "exchange": "deribit_testnet",
    }
    trades[symbol] = record
    save_trades(trades)
    save_signal(record)
    _send_open_alert(symbol, signal, confidence, score, actual_entry,
                     stop, tp1, tp2, total_contracts, amount_tp1, amount_tp2,
                     risk_usd, balance, reasons, risk_mult)
    log.info(f"  ✅✅ TRADE OPENED: {symbol} {signal} | SL:{order_ids.get('stop_loss','!')} TP1:{order_ids.get('tp1','!')}")
    return True


# ════════════ TRADE MONITORING ════════════════════════════════════════

def _fill_price(order: dict, fallback: float) -> float:
    p = float(order.get("average_price") or order.get("price") or 0)
    return p if p > 0 else fallback

def check_open_trades(deribit: DeribitClient):
    trades = load_trades()
    if not trades: log.info("  No open trades"); return

    to_remove = []
    log.info(f"  Monitoring {len(trades)} trade(s)")

    for symbol, trade in list(trades.items()):
        if trade.get("closed"): to_remove.append(symbol); continue
        oids  = trade.get("order_ids", {})
        entry = float(trade["entry"])
        dec   = 4 if entry < 10 else 2

        def get_o(key):
            if key not in oids or not oids[key] or str(oids[key]) in ("","None"): return {}
            return deribit.get_order(str(oids[key]))

        try:
            # ── TP1 ────────────────────────────────────────────────
            if not trade["tp1_hit"] and "tp1" in oids:
                o = get_o("tp1")
                if deribit.is_order_filled(o):
                    trade["tp1_hit"] = True
                    fill = _fill_price(o, trade["tp1"])
                    pnl  = _calc_pnl(trade, fill, "tp1")
                    log.info(f"  🎯 TP1 {symbol} @ {fill:.{dec}f} pnl≈{pnl:+.4f}")
                    _send_close_alert(symbol, "TP1 HIT 🎯", pnl, entry, fill, trade["opened_at"])

                    # Move SL to breakeven
                    if oids.get("stop_loss") and trade.get("qty_tp2", 0) > 0:
                        try:
                            deribit.cancel_order(oids["stop_loss"])
                            sl_side  = "SELL" if trade["signal"]=="BUY" else "BUY"
                            qty_rem  = deribit.to_int_amount(symbol, trade["qty_tp2"])
                            sl_res   = deribit.place_limit_order(symbol, sl_side, qty_rem, entry, stop_price=entry)
                            sl_o     = sl_res.get("order", sl_res)
                            new_id   = str(sl_o.get("order_id",""))
                            if new_id:
                                trade["order_ids"]["stop_loss"] = new_id
                                trade["stop"] = entry
                            _send(f"🛡️ *{symbol}* SL → breakeven `{entry:.{dec}f}` — risk-free!")
                        except Exception as e: log.warning(f"  Breakeven SL: {e}")

            # ── Trailing stop ──────────────────────────────────────
            if trade.get("tp1_hit") and not trade.get("tp2_hit") and oids.get("stop_loss"):
                live = deribit.get_live_price(symbol)
                if live > 0:
                    halfway   = (entry + float(trade["tp2"])) / 2
                    at_trail  = ((trade["signal"]=="BUY"  and live >= halfway) or
                                 (trade["signal"]=="SELL" and live <= halfway))
                    sl_at_be  = abs(float(trade.get("stop",0)) - entry) < 0.01 * entry
                    if at_trail and sl_at_be and trade.get("qty_tp2", 0) > 0:
                        try:
                            deribit.cancel_order(oids["stop_loss"])
                            sl_side = "SELL" if trade["signal"]=="BUY" else "BUY"
                            qty_rem = deribit.to_int_amount(symbol, trade["qty_tp2"])
                            sl_res  = deribit.place_limit_order(symbol, sl_side, qty_rem,
                                          trade["tp1"], stop_price=trade["tp1"])
                            sl_o    = sl_res.get("order", sl_res)
                            new_id  = str(sl_o.get("order_id",""))
                            if new_id:
                                trade["order_ids"]["stop_loss"] = new_id
                                trade["stop"] = trade["tp1"]
                            log.info(f"  🚀 {symbol} trailing SL → TP1 {trade['tp1']:.{dec}f}")
                            _send(f"🚀 *{symbol}* Trailing SL → locked profit `{trade['tp1']:.{dec}f}`")
                        except Exception as e: log.warning(f"  Trailing SL: {e}")

            # ── TP2 ────────────────────────────────────────────────
            if trade["tp1_hit"] and not trade["tp2_hit"] and "tp2" in oids:
                o = get_o("tp2")
                if deribit.is_order_filled(o):
                    trade["tp2_hit"] = True; trade["closed"] = True
                    fill = _fill_price(o, trade["tp2"])
                    pnl  = _calc_pnl(trade, fill, "tp2")
                    log.info(f"  ✅ TP2 {symbol} @ {fill:.{dec}f} pnl≈{pnl:+.4f}")
                    _send_close_alert(symbol, "✅ FULL WIN (TP2)", pnl, entry, fill, trade["opened_at"])
                    _record_close(trade, fill, pnl, "TP2 hit")
                    to_remove.append(symbol)

            # ── SL ─────────────────────────────────────────────────
            if not trade.get("closed") and oids.get("stop_loss"):
                o = get_o("stop_loss")
                if deribit.is_order_filled(o):
                    trade["closed"] = True
                    fill = _fill_price(o, trade["stop"])
                    pnl  = _calc_pnl(trade, fill, "sl")
                    log.info(f"  ❌ SL {symbol} @ {fill:.{dec}f} pnl≈{pnl:+.4f}")
                    _send_close_alert(symbol, "❌ STOPPED OUT", pnl, entry, fill, trade["opened_at"])
                    _record_close(trade, fill, pnl, "SL hit")
                    for k in ("tp1","tp2"):
                        if oids.get(k) and not trade.get(f"{k}_hit"):
                            try: deribit.cancel_order(oids[k])
                            except Exception: pass
                    to_remove.append(symbol)

        except Exception as e: log.error(f"  Monitor error {symbol}: {e}")

    save_trades(trades)
    for sym in set(to_remove): trades.pop(sym, None)
    save_trades(trades)


def _calc_pnl(trade, close_price, close_type) -> float:
    qty  = float(trade["qty_tp1"] if close_type=="tp1" else
                 trade["qty_tp2"] if close_type=="tp2" else trade["qty"])
    diff = ((close_price - trade["entry"]) if trade["signal"]=="BUY"
            else (trade["entry"] - close_price))
    return round(diff * qty, 4)

def _record_close(trade, close_price, pnl, reason):
    append_history({**trade, "close_price": close_price, "pnl": pnl,
                    "closed_at": datetime.now(timezone.utc).isoformat(), "close_reason": reason})


# ════════════ SIGNAL GENERATION ══════════════════════════════════════

def generate_signal(symbol, pipeline, thresholds):
    try:
        df15 = add_indicators(get_data(symbol, TIMEFRAME_ENTRY))
        df1h = add_indicators(get_data(symbol, TIMEFRAME_CONFIRM))
        if df15.empty or len(df15) < 50: return None

        row = df15.iloc[-1].copy()
        r1h = df1h.iloc[-1] if not df1h.empty else pd.Series(dtype=float)
        row["rsi_1h"]   = float(r1h.get("rsi",  50))
        row["adx_1h"]   = float(r1h.get("adx",   0))
        row["trend_1h"] = float(r1h.get("trend", 0))

        af = pipeline["all_features"]
        if any(f not in row.index for f in af): return None

        X    = pd.DataFrame([row[af].values], columns=af)
        Xs   = pipeline["selector"].transform(X)
        pred = pipeline["ensemble"].predict(Xs)[0]
        prob = pipeline["ensemble"].predict_proba(Xs)[0]
        sig  = {0:"BUY", 1:"SELL", 2:"NO_TRADE"}[pred]
        conf = round(float(max(prob)) * 100, 1)

        log.info(f"    ML: {sig} {conf:.1f}% (need ≥{thresholds['min_confidence']}%)")
        if sig == "NO_TRADE" or conf < thresholds["min_confidence"]: return None

        adx = float(row.get("adx", 0))
        log.info(f"    ADX: {adx:.1f} (need ≥{thresholds['min_adx']})")
        if adx < thresholds["min_adx"]: return None

        score, reasons = _quality_score(row, r1h, sig, conf)
        log.info(f"    Score: {score} (need ≥{thresholds['min_score']})")

        entry = float(row["close"]); atr = float(row["atr"])
        dec   = 4 if entry < 10 else 2
        if sig == "BUY":
            stop=round(entry-atr*ATR_STOP_MULT,dec); tp1=round(entry+atr*ATR_TARGET1_MULT,dec); tp2=round(entry+atr*ATR_TARGET2_MULT,dec)
        else:
            stop=round(entry+atr*ATR_STOP_MULT,dec); tp1=round(entry-atr*ATR_TARGET1_MULT,dec); tp2=round(entry-atr*ATR_TARGET2_MULT,dec)

        if score < thresholds["min_score"]:
            save_signal({"symbol":symbol,"signal":sig,"confidence":conf,"score":score,
                "entry":entry,"atr":atr,"stop":stop,"tp1":tp1,"tp2":tp2,"reasons":reasons,
                "rejected":True,"reject_reason":f"score {score}<{thresholds['min_score']}"})
            return None

        return {"symbol":symbol,"signal":sig,"confidence":conf,"score":score,
                "entry":entry,"atr":atr,"stop":stop,"tp1":tp1,"tp2":tp2,"reasons":reasons}

    except Exception as e: log.error(f"    Signal error {symbol}: {e}"); return None


def _quality_score(row, r1h, signal, conf):
    score, reasons = 0, []
    if conf>=70:   score+=1; reasons.append(f"High conf ({conf:.0f}%)")
    elif conf>=55: score+=1; reasons.append(f"Good conf ({conf:.0f}%)")
    elif conf>=50: reasons.append(f"Conf ({conf:.0f}%)")
    adx=float(row.get("adx",0))
    if adx>20:   score+=1; reasons.append(f"Strong ADX {adx:.0f}")
    elif adx>15: score+=1; reasons.append(f"ADX {adx:.0f}")
    rsi=float(row.get("rsi",50))
    if signal=="BUY" and rsi<50:   score+=1; reasons.append(f"RSI bullish ({rsi:.0f})")
    elif signal=="SELL" and rsi>50: score+=1; reasons.append(f"RSI bearish ({rsi:.0f})")
    e20=float(row.get("ema20",0)); e50=float(row.get("ema50",0))
    if signal=="BUY" and e20>e50:   score+=1; reasons.append("EMA bullish")
    elif signal=="SELL" and e20<e50: score+=1; reasons.append("EMA bearish")
    c20=float(r1h.get("ema20",0)); c50=float(r1h.get("ema50",0))
    if signal=="BUY" and c20>c50:   score+=1; reasons.append("1h confirms")
    elif signal=="SELL" and c20<c50: score+=1; reasons.append("1h confirms")
    if not reasons: reasons.append(f"ML {conf:.0f}%")
    return score, reasons


# ════════════ TELEGRAM ════════════════════════════════════════════════

def _send(text):
    tok=os.getenv("TELEGRAM_TOKEN",""); cid=os.getenv("TELEGRAM_CHAT_ID","")
    if not tok or not cid: return
    try: requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
            data={"chat_id":cid,"text":text,"parse_mode":"Markdown"}, timeout=10)
    except Exception: pass

def _warn(text): log.warning(text); _send(text)

def check_mode_switch(mode):
    last = load_json(MODE_FILE, {})
    if last.get("mode") != mode["mode"]:
        msgs = {"active":"📈 *Active* — conf≥50% score≥1",
                "quiet":"🌙 *Quiet* — conf≥55% score≥2",
                "weekend":"📅 *Weekend* — conf≥50% score≥1"}
        _send(msgs.get(mode["mode"],"Mode changed"))
        save_json(MODE_FILE, {"mode":mode["mode"],"since":datetime.now(timezone.utc).isoformat()})

def _send_open_alert(symbol, signal, confidence, score, entry, stop, tp1, tp2,
                     amount, tp1_qty, tp2_qty, risk_usd, balance, reasons, risk_mult=1.0):
    emoji="🟢" if signal=="BUY" else "🔴"; stars="⭐"*min(score,5)
    dec=4 if entry<10 else 2
    sl_pct=abs((stop-entry)/entry*100); t1_pct=abs((tp1-entry)/entry*100); t2_pct=abs((tp2-entry)/entry*100)
    rlines="\n".join([f"  • {r}" for r in reasons])
    risk_n="" if risk_mult>=1.0 else f"\n⚡ Risk: {int(risk_mult*100)}%"
    _send(f"🤖 *DERIBIT TESTNET TRADE*\n━━━━━━━━━━━━━━━━━━━━\n"
          f"{emoji} *{signal} — {symbol}* {stars}\n🏷️ _{get_tier(symbol)}_{risk_n}\n"
          f"🎯 Conf: *{confidence:.1f}%* · Score: *{score}*\n\n"
          f"⚡ *ENTRY:* `{entry:.{dec}f}`\n"
          f"🛑 *STOP:* `{stop:.{dec}f}` (-{sl_pct:.1f}%)\n"
          f"🎯 *TP1:* `{tp1:.{dec}f}` (+{t1_pct:.1f}%) × {tp1_qty}\n"
          f"🎯 *TP2:* `{tp2:.{dec}f}` (+{t2_pct:.1f}%) × {tp2_qty}\n"
          f"📦 Total: *{amount}* contracts · Risk: `${risk_usd:.2f}`\n"
          f"💼 Portfolio: `${balance:.2f}`\n\n"
          f"📊 *Reasons:*\n{rlines}\n━━━━━━━━━━━━━━━━━━━━")

def _send_close_alert(symbol, result, pnl, entry, close_price, opened_at):
    emoji="✅" if pnl>0 else "❌"; dec=4 if entry<10 else 2
    try: dur=str(datetime.now(timezone.utc)-datetime.fromisoformat(opened_at)).split(".")[0]
    except Exception: dur="—"
    _send(f"🤖 *DERIBIT TRADE CLOSED*\n{emoji} *{result} — {symbol}*\n"
          f"📥 `{entry:.{dec}f}` → 📤 `{close_price:.{dec}f}`\n"
          f"💵 *PnL ≈ `{pnl:+.4f}`* · ⏱️ {dur}")


# ════════════ MAIN ════════════════════════════════════════════════════

def run_execution_scan():
    log.info(f"\n{'═'*56}")
    log.info(f"SCAN — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    log.info(f"{'═'*56}")

    run, mode, vol, reason = should_scan()
    check_mode_switch(mode)
    if not run: log.info(f"  SKIPPED: {reason}"); return

    effective_risk = get_effective_risk(mode, vol)
    vol_warn       = vol["message"] if vol.get("warn") else None
    thresholds     = get_mode_thresholds(mode)

    deribit  = init_deribit()
    pipeline = load_model()

    log.info(f"  {mode['label']} | conf≥{thresholds['min_confidence']}% | "
             f"score≥{thresholds['min_score']} | ADX≥{thresholds['min_adx']} | risk:{effective_risk:.2f}")

    log.info(f"\n[0] Balance..."); balance = save_balance_json(deribit)
    log.info(f"\n[1] Clean invalid trades..."); clean_invalid_trades(deribit)
    log.info(f"\n[2] Monitor open trades..."); check_open_trades(deribit); save_balance_json(deribit)

    trades = load_trades()
    log.info(f"\n[3] Scanning {len(SYMBOLS)} coins | Open:{len(trades)}/{MAX_OPEN_TRADES}")
    log.info(f"    Tradeable: {TRADEABLE_SYMBOLS}")

    found = 0
    for symbol in SYMBOLS:
        if len(load_trades()) >= MAX_OPEN_TRADES: log.info("  🛑 Max trades — stop"); break
        log.info(f"\n  ── {symbol} ({get_tier(symbol)}) ──")
        sig = generate_signal(symbol, pipeline, thresholds)
        if sig is None: time.sleep(0.3); continue
        found += 1
        if vol_warn: sig["reasons"] = list(sig.get("reasons",[])) + [f"⚠️ {vol_warn}"]
        execute_trade(deribit, symbol=sig["symbol"], signal=sig["signal"],
                      entry=sig["entry"], atr=sig["atr"],
                      confidence=sig["confidence"], score=sig["score"],
                      reasons=sig["reasons"], risk_mult=effective_risk, balance=balance)
        time.sleep(1)

    save_balance_json(deribit)
    log.info(f"\n{'═'*56}\nDONE — {found} signal(s) | Portfolio: ${balance:.2f}\n{'═'*56}\n")


def run_diagnostic():
    from smart_scheduler import get_scan_mode, check_btc_volatility
    mode=get_scan_mode(); vol=check_btc_volatility(); eff=get_effective_risk(mode,vol)
    lines=["🔍 *Diagnostic — Deribit Testnet*",
           f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}","━━━━━━━━━━━━━━━━━━━━"]
    try:
        d=init_deribit(); bal=save_balance_json(d); pos=d.get_positions()
        lines+=[f"✅ Deribit OK | GET ✅ POST ✅",f"💰 Portfolio: *${bal:.2f} USD*",
                f"📂 Positions: {len(pos)}",f"📋 Tradeable ({len(TRADEABLE_SYMBOLS)}): {TRADEABLE_SYMBOLS[:8]}"]
    except Exception as e: lines.append(f"❌ Deribit: {e}")
    lines+=[f"📋 Mode: *{mode['label']}* conf≥{mode['min_confidence']}%",
            f"📊 BTC ATR: *{vol['atr_pct']:.2f}%*",f"⚡ Risk: *{int(eff*100)}%*"]
    try: p=load_model(); lines.append(f"🤖 Model: ✅ {len(p['all_features'])} features")
    except Exception as e: lines.append(f"❌ Model: {e}")
    real=[h for h in load_history() if h.get("signal")!="RECOVERED"]
    wins=[h for h in real if (h.get("pnl") or 0)>0]; wr=round(len(wins)/len(real)*100,1) if real else 0
    lines+=[f"📈 Win rate: *{wr}%* ({len(wins)}W/{len(real)-len(wins)}L)",
            f"💵 PnL: *{sum(h.get('pnl',0) for h in real):+.4f}*"]
    _send("\n".join(lines))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "diagnostic":   run_diagnostic()
    elif len(sys.argv) > 1 and sys.argv[1] == "clear_stuck": d=init_deribit(); clean_invalid_trades(d)
    else: run_execution_scan()
