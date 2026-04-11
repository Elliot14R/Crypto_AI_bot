# trade_executor.py — ALL BUGS FIXED
# FIX 1: save_signal() called AFTER orders placed with real entry/order IDs
# FIX 2: fill price from trades[] array via get_fill_price()
# FIX 3: is_order_filled() uses order_state field (Deribit)
# FIX 4: clear_stuck_trades() removes old trades with $0 SL/TP
# FIX 5: place_limit_order returns full result so order_id extractable

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
from deribit_client import DeribitClient, TRADEABLE
from paper_trader import PaperTrader

TRADES_FILE     = "trades.json"
HISTORY_FILE    = "trade_history.json"
SIGNALS_FILE    = "signals.json"
MODE_FILE       = "scan_mode.json"
BALANCE_FILE    = "balance.json"
MAX_OPEN_TRADES = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
log = logging.getLogger(__name__)


# ════════════ HELPERS ════════════════════════════════════════════════

def load_json(p, d):
    try:
        for path in [Path(p), Path("data") / p]:
            if path.exists():
                with open(path) as f: return json.load(f)
    except: pass
    return d

def save_json(p, data):
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as f: json.dump(data, f, indent=2, default=str)
    os.replace(tmp, p)

load_trades  = lambda: load_json(TRADES_FILE,  {})
save_trades  = lambda d: save_json(TRADES_FILE, d)
load_history = lambda: load_json(HISTORY_FILE, [])
load_signals = lambda: load_json(SIGNALS_FILE, [])
append_history = lambda rec: (lambda h: (h.append(rec), save_json(HISTORY_FILE, h)))(load_history())

def save_signal(sig):
    s = load_signals()
    s.append({**sig, "generated_at": datetime.now(timezone.utc).isoformat()})
    save_json(SIGNALS_FILE, s[-500:])


# ════════════ INIT ═══════════════════════════════════════════════════

def init_deribit() -> DeribitClient:
    cid    = os.getenv("DERIBIT_CLIENT_ID",     "")
    secret = os.getenv("DERIBIT_CLIENT_SECRET", "")
    if not cid or not secret:
        raise ValueError("DERIBIT_CLIENT_ID / DERIBIT_CLIENT_SECRET not set in GitHub Secrets")
    return DeribitClient(cid, secret)

def init_paper() -> PaperTrader:
    return PaperTrader()


# ════════════ BALANCE ════════════════════════════════════════════════

def fetch_and_save_balance(deribit: DeribitClient, paper: PaperTrader) -> float:
    try:
        deribit_usd = deribit.get_usdt_equivalent()
        balances    = deribit.get_all_balances()
        assets = [
            {"asset": cur, "free": round(float(info.get("available", 0)), 6),
             "total": round(float(info.get("equity_usd", 0)), 2), "source": "deribit_testnet"}
            for cur, info in balances.items() if float(info.get("equity_usd", 0)) > 0
        ]
        save_json(BALANCE_FILE, {
            "usdt":        round(deribit_usd, 2),
            "equity":      round(deribit_usd, 2),
            "deribit_usd": round(deribit_usd, 2),
            "unrealised":  0.0,
            "assets":      assets,
            "updated_at":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "mode":        "deribit_testnet",
            "note":        f"Deribit Portfolio: ${deribit_usd:.0f}",
        })
        log.info(f"✓ Balance: ${deribit_usd:.2f} (Deribit Testnet)")
        return round(deribit_usd, 2)
    except Exception as e:
        log.error(f"Balance fetch failed: {e}")
        bal = load_json(BALANCE_FILE, {})
        return float(bal.get("usdt") or bal.get("equity") or 10000)


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
        except: continue
    raise Exception(f"All kline endpoints failed for {symbol}")

def load_model():
    p = joblib.load(MODEL_FILE)
    for k in ["ensemble","selector","all_features","label_map"]:
        if k not in p: raise ValueError(f"Model missing: {k}")
    log.info(f"✓ Model: {len(p['all_features'])} features")
    return p


# ════════════ EXECUTE TRADE ══════════════════════════════════════════

def execute_trade(deribit: DeribitClient, paper: PaperTrader,
                  symbol, signal, entry, atr, confidence, score,
                  reasons, risk_mult=1.0, deribit_balance=10000.0):

    trades = load_trades()
    if symbol in trades: log.info(f"  {symbol} already open — skip"); return False
    if len(trades) >= MAX_OPEN_TRADES: log.info("  Max open trades — skip"); return False
    if not check_correlation(trades, signal): return False

    dec = 4 if entry < 10 else 2
    if signal == "BUY":
        stop = round(entry - atr*ATR_STOP_MULT, dec)
        tp1  = round(entry + atr*ATR_TARGET1_MULT, dec)
        tp2  = round(entry + atr*ATR_TARGET2_MULT, dec)
        side = "BUY"; sl_side = "SELL"; tp_side = "SELL"
    else:
        stop = round(entry + atr*ATR_STOP_MULT, dec)
        tp1  = round(entry - atr*ATR_TARGET1_MULT, dec)
        tp2  = round(entry - atr*ATR_TARGET2_MULT, dec)
        side = "SELL"; sl_side = "BUY"; tp_side = "BUY"

    use_deribit = deribit.is_supported(symbol)
    exch_label  = "DERIBIT" if use_deribit else "PAPER"
    log.info(f"  [{exch_label}] {signal} {symbol} SL={stop:.{dec}f} TP1={tp1:.{dec}f} TP2={tp2:.{dec}f}")

    order_ids    = {}
    actual_entry = entry
    risk_usd     = 0.0

    if use_deribit:
        amount_usd = deribit.calc_usd_amount(deribit_balance, entry, stop, risk_mult)
        amount_tp1 = max(10.0, round(amount_usd * 0.5 / 10) * 10)
        amount_tp2 = max(10.0, round(amount_usd * 0.5 / 10) * 10)
        risk_usd   = round(deribit_balance * RISK_PER_TRADE * risk_mult, 2)

        try:
            # Entry — full result returned for fill price extraction
            entry_result = deribit.place_market_order(symbol, side, amount_usd)
            if not entry_result: log.error(f"  Entry failed {symbol}"); return False

            entry_order  = entry_result.get("order", entry_result)
            order_ids["entry"] = str(entry_order.get("order_id", ""))

            # FIX 2: real fill price from trades[] array
            actual_entry = deribit.get_fill_price(entry_result, entry)
            if actual_entry == 0: actual_entry = entry
            log.info(f"  ✅ Entry filled @ ${actual_entry:.2f}")
            time.sleep(1.5)

            # Stop-Loss (FIX 1+2: real POST, correct stop_price field)
            try:
                sl_result = deribit.place_limit_order(symbol, sl_side, amount_usd, price=stop, stop_price=stop)
                sl_order  = sl_result.get("order", sl_result)
                oid = str(sl_order.get("order_id", ""))
                if oid: order_ids["stop_loss"] = oid
                log.info(f"  ✅ SL @ {stop:.{dec}f} id:{oid or 'MISSING'}")
            except Exception as e: log.warning(f"  ⚠️ SL failed: {e}")

            # Take Profit 1
            try:
                tp1_result = deribit.place_limit_order(symbol, tp_side, amount_tp1, price=tp1)
                tp1_order  = tp1_result.get("order", tp1_result)
                oid = str(tp1_order.get("order_id", ""))
                if oid: order_ids["tp1"] = oid
                log.info(f"  ✅ TP1 @ {tp1:.{dec}f} id:{oid or 'MISSING'}")
            except Exception as e: log.warning(f"  ⚠️ TP1 failed: {e}")

            # Take Profit 2
            try:
                tp2_result = deribit.place_limit_order(symbol, tp_side, amount_tp2, price=tp2)
                tp2_order  = tp2_result.get("order", tp2_result)
                oid = str(tp2_order.get("order_id", ""))
                if oid: order_ids["tp2"] = oid
                log.info(f"  ✅ TP2 @ {tp2:.{dec}f} id:{oid or 'MISSING'}")
            except Exception as e: log.warning(f"  ⚠️ TP2 failed: {e}")

        except Exception as e:
            log.error(f"  Deribit error {symbol}: {e}")
            _warn(f"⚠️ Trade error {symbol}: {e}")
            return False

    else:
        # Paper trading for alt coins
        paper_bal = paper.get_usdt_balance()
        stop_dist = abs(entry - stop)
        if stop_dist <= 0: return False
        qty      = round((paper_bal * RISK_PER_TRADE * risk_mult) / stop_dist, 6)
        if qty <= 0: return False
        qty_tp1  = round(qty * 0.5, 6)
        qty_tp2  = round(qty - qty_tp1, 6)
        risk_usd = round(paper_bal * RISK_PER_TRADE * risk_mult, 2)

        eo = paper.place_market_order(symbol, side, qty)
        if not eo: return False
        order_ids["entry"]     = str(eo.get("orderId", ""))
        actual_entry           = float(eo.get("paper_fill", entry) or entry)

        sl_o  = paper.place_limit_order(symbol, sl_side, qty,     price=stop, stop_price=stop)
        tp1_o = paper.place_limit_order(symbol, tp_side, qty_tp1, price=tp1)
        tp2_o = paper.place_limit_order(symbol, tp_side, qty_tp2, price=tp2)
        order_ids["stop_loss"] = str(sl_o.get("orderId",  ""))
        order_ids["tp1"]       = str(tp1_o.get("orderId", ""))
        order_ids["tp2"]       = str(tp2_o.get("orderId", ""))
        log.info(f"  📝 Paper entry @ {actual_entry:.{dec}f} qty={qty} risk=${risk_usd:.2f}")

    qty_rec = (amount_usd / actual_entry) if use_deribit else qty

    # FIX 1: save AFTER all orders placed with real values
    record = {
        "symbol": symbol, "signal": signal,
        "entry":  actual_entry,  # real fill, not estimate
        "stop":   stop, "tp1": tp1, "tp2": tp2,
        "qty": round(qty_rec, 6), "qty_tp1": round(qty_rec*0.5, 6), "qty_tp2": round(qty_rec*0.5, 6),
        "risk_usd": risk_usd, "risk_mult": risk_mult,
        "balance_at_open": deribit_balance,
        "order_ids": order_ids,  # real IDs not empty
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "tp1_hit": False, "tp2_hit": False, "closed": False,
        "confidence": confidence, "score": score,
        "reasons": reasons, "tier": get_tier(symbol), "exchange": exch_label.lower(),
    }
    trades[symbol] = record
    save_trades(trades)
    save_signal(record)
    _send_open_alert(symbol, signal, confidence, score, actual_entry,
                     stop, tp1, tp2, qty_rec, risk_usd, deribit_balance, reasons, exch_label, order_ids)
    log.info(f"  ✅✅ TRADE OPENED: {symbol} {signal} [{exch_label}]")
    return True


# ════════════ MONITOR ════════════════════════════════════════════════

def check_open_trades(deribit: DeribitClient, paper: PaperTrader):
    trades = load_trades()
    if not trades: log.info("  No open trades"); return

    to_remove = []
    log.info(f"  Monitoring {len(trades)} trade(s)")

    for symbol, trade in list(trades.items()):
        if trade.get("closed"): to_remove.append(symbol); continue

        oids       = trade.get("order_ids", {})
        entry      = float(trade["entry"])
        dec        = 4 if entry < 10 else 2
        is_deribit = trade.get("exchange") == "deribit"

        def get_o(key):
            if key not in oids or not oids[key] or oids[key] == "None": return {}
            try:
                return deribit.get_order(oids[key]) if is_deribit else paper.get_order(symbol, oids[key])
            except: return {}

        def filled(o):
            # FIX 3: Deribit uses order_state, paper uses status
            if is_deribit: return deribit.is_order_filled(o)
            return o.get("status") == "FILLED"

        def fill_px(o, fallback):
            if is_deribit: return deribit.get_fill_price(o, fallback)
            return float(o.get("price", fallback) or fallback)

        try:
            # TP1
            if not trade["tp1_hit"] and "tp1" in oids:
                o = get_o("tp1")
                if filled(o):
                    trade["tp1_hit"] = True
                    px  = fill_px(o, trade["tp1"])
                    pnl = _pnl(trade, px, "tp1")
                    log.info(f"  🎯 TP1 {symbol} @ {px:.{dec}f} pnl={pnl:+.4f}")
                    _send_close_alert(symbol, "TP1 HIT 🎯", pnl, entry, px, trade["opened_at"],
                                      "DERIBIT" if is_deribit else "PAPER")
                    if not is_deribit: paper.update_balance_after_close(pnl)
                    # Move SL to breakeven
                    if is_deribit and oids.get("stop_loss"):
                        try:
                            deribit.cancel_order(oids["stop_loss"])
                            sl_side = "SELL" if trade["signal"] == "BUY" else "BUY"
                            be_res  = deribit.place_limit_order(symbol, sl_side,
                                          max(10.0, round(trade.get("qty",0)*entry*0.5/10)*10),
                                          price=entry, stop_price=entry)
                            be_o = be_res.get("order", be_res)
                            if be_o.get("order_id"):
                                trade["order_ids"]["stop_loss"] = str(be_o["order_id"])
                                trade["stop"] = entry
                                _send(f"🛡️ *{symbol} RISK-FREE!*\nSL → breakeven @ `{entry:.{dec}f}`")
                        except Exception as e:
                            log.warning(f"  SL breakeven failed: {e}")

            # TP2
            if trade["tp1_hit"] and not trade["tp2_hit"] and "tp2" in oids:
                o = get_o("tp2")
                if filled(o):
                    trade["tp2_hit"] = True; trade["closed"] = True
                    px  = fill_px(o, trade["tp2"])
                    pnl = _pnl(trade, px, "tp2")
                    log.info(f"  ✅ TP2 {symbol} @ {px:.{dec}f} pnl={pnl:+.4f}")
                    _send_close_alert(symbol, "✅ FULL WIN (TP2)", pnl, entry, px, trade["opened_at"],
                                      "DERIBIT" if is_deribit else "PAPER")
                    _rec_close(trade, px, pnl, "TP2 hit")
                    if not is_deribit: paper.update_balance_after_close(pnl)
                    to_remove.append(symbol)

            # SL
            if not trade.get("closed") and oids.get("stop_loss"):
                o = get_o("stop_loss")
                if filled(o):
                    trade["closed"] = True
                    px  = fill_px(o, trade["stop"])
                    pnl = _pnl(trade, px, "sl")
                    log.info(f"  ❌ SL {symbol} @ {px:.{dec}f} pnl={pnl:+.4f}")
                    _send_close_alert(symbol, "❌ STOPPED OUT", pnl, entry, px, trade["opened_at"],
                                      "DERIBIT" if is_deribit else "PAPER")
                    _rec_close(trade, px, pnl, "SL hit")
                    if not is_deribit: paper.update_balance_after_close(pnl)
                    for k in ("tp1","tp2"):
                        if oids.get(k) and not trade.get(f"{k}_hit"):
                            try:
                                deribit.cancel_order(oids[k]) if is_deribit else paper.cancel_order(symbol, oids[k])
                            except: pass
                    to_remove.append(symbol)

        except Exception as e:
            log.error(f"  Monitor error {symbol}: {e}")

    save_trades(trades)
    for sym in set(to_remove): trades.pop(sym, None)
    save_trades(trades)


def clear_stuck_trades(deribit: DeribitClient, paper: PaperTrader):
    """FIX 4: Remove old broken trades with missing/empty SL/TP order IDs."""
    trades = load_trades()
    if not trades: return
    log.info(f"  🔄 Checking {len(trades)} trade(s) for stuck/broken orders...")
    cleared = 0

    for symbol in list(trades.keys()):
        trade      = trades[symbol]
        oids       = trade.get("order_ids", {})
        is_deribit = trade.get("exchange") == "deribit"

        # Detect broken: SL or TP1 missing (caused by old _post bug)
        sl  = oids.get("stop_loss", "")
        tp1 = oids.get("tp1", "")
        if not sl or not tp1 or sl in ("None", "") or tp1 in ("None", ""):
            log.warning(f"  ⚠️ {symbol}: missing SL/TP IDs (from old POST bug) — clearing")
            _rec_close(trade, float(trade.get("entry", 0)), 0.0,
                       "Cleared: SL/TP not placed (POST bug now fixed)")
            trades.pop(symbol); cleared += 1; continue

        # Check if live orders still exist
        found = False
        for key, oid in oids.items():
            if key == "entry" or not oid or oid in ("None", ""): continue
            try:
                if is_deribit:
                    o = deribit.get_order(oid)
                    if o.get("order_state") in ("open","untriggered","partially_filled"):
                        found = True; break
                else:
                    o = paper.get_order(symbol, oid)
                    if o.get("status") == "NEW": found = True; break
            except: pass

        if not found:
            log.warning(f"  ⚠️ {symbol}: no live orders — clearing")
            _rec_close(trade, float(trade.get("entry", 0)), 0.0, "Auto-cleared (no live orders)")
            trades.pop(symbol); cleared += 1

    if cleared:
        save_trades(trades)
        log.info(f"  ✅ Cleared {cleared} broken trade(s)")
        _send(f"🧹 *{cleared} broken trade(s) cleared*\n(SL/TP POST bug now fixed — new trades will have proper SL/TP)")
    else:
        log.info("  ✓ No stuck trades")


def _pnl(trade, close_px, t):
    e = float(trade["entry"])
    q = float(trade["qty_tp1"] if t=="tp1" else trade["qty_tp2"] if t=="tp2" else trade["qty"])
    return round((close_px-e)*q if trade["signal"]=="BUY" else (e-close_px)*q, 4)

def _rec_close(trade, px, pnl, reason):
    h = load_history()
    h.append({**trade, "close_price":px, "pnl":pnl,
              "closed_at":datetime.now(timezone.utc).isoformat(), "close_reason":reason})
    save_json(HISTORY_FILE, h)


# ════════════ SIGNALS ════════════════════════════════════════════════

def generate_signal(symbol, pipeline, thresholds):
    try:
        df_e = add_indicators(get_data(symbol, TIMEFRAME_ENTRY))
        df_c = add_indicators(get_data(symbol, TIMEFRAME_CONFIRM))
        if df_e.empty or len(df_e) < 50: return None

        row_e = df_e.iloc[-1].copy()
        row_c = df_c.iloc[-1] if not df_c.empty else pd.Series(dtype=float)
        row_e["rsi_1h"]   = float(row_c.get("rsi",  50))
        row_e["adx_1h"]   = float(row_c.get("adx",   0))
        row_e["trend_1h"] = float(row_c.get("trend", 0))

        af = pipeline["all_features"]
        if any(f not in row_e.index for f in af): return None

        X    = pd.DataFrame([row_e[af].values], columns=af)
        Xs   = pipeline["selector"].transform(X)
        pred = pipeline["ensemble"].predict(Xs)[0]
        prob = pipeline["ensemble"].predict_proba(Xs)[0]
        sig  = {0:"BUY", 1:"SELL", 2:"NO_TRADE"}[pred]
        conf = round(float(max(prob))*100, 1)

        log.info(f"    ML: {sig} {conf:.1f}% (need ≥{thresholds['min_confidence']}%)")
        if sig == "NO_TRADE": return None
        if conf < thresholds["min_confidence"]: return None

        adx = float(row_e.get("adx", 0))
        log.info(f"    ADX: {adx:.1f} (need ≥{thresholds['min_adx']})")
        if adx < thresholds["min_adx"]: return None

        score, reasons = _quality_score(row_e, row_c, sig, conf)
        log.info(f"    Score: {score}/5 (need ≥{thresholds['min_score']})")

        entry = float(row_e["close"]); atr = float(row_e["atr"])
        dec   = 4 if entry < 10 else 2
        if sig == "BUY":
            stop = round(entry - atr*ATR_STOP_MULT, dec)
            tp1  = round(entry + atr*ATR_TARGET1_MULT, dec)
            tp2  = round(entry + atr*ATR_TARGET2_MULT, dec)
        else:
            stop = round(entry + atr*ATR_STOP_MULT, dec)
            tp1  = round(entry - atr*ATR_TARGET1_MULT, dec)
            tp2  = round(entry - atr*ATR_TARGET2_MULT, dec)

        if score < thresholds["min_score"]:
            save_signal({"symbol":symbol,"signal":sig,"confidence":conf,"score":score,
                "entry":entry,"atr":atr,"stop":stop,"tp1":tp1,"tp2":tp2,"reasons":reasons,
                "rejected":True,"reject_reason":f"score {score}<{thresholds['min_score']}"})
            return None

        return {"symbol":symbol,"signal":sig,"confidence":conf,"score":score,
                "entry":entry,"atr":atr,"stop":stop,"tp1":tp1,"tp2":tp2,"reasons":reasons}

    except Exception as e:
        log.error(f"    Signal error {symbol}: {e}"); return None


def _quality_score(row_e, row_c, signal, confidence):
    s, r = 0, []
    if confidence >= 70:   s+=1; r.append(f"High conf ({confidence:.0f}%)")
    elif confidence >= 55: s+=1; r.append(f"AI conf ({confidence:.0f}%)")
    adx = float(row_e.get("adx",0))
    if adx > 20:   s+=1; r.append(f"Strong ADX {adx:.0f}")
    elif adx > 15: s+=1; r.append(f"Moderate ADX {adx:.0f}")
    rsi = float(row_e.get("rsi",50))
    if signal=="BUY" and rsi<50:   s+=1; r.append(f"RSI bullish ({rsi:.0f})")
    elif signal=="SELL" and rsi>50: s+=1; r.append(f"RSI bearish ({rsi:.0f})")
    e20,e50 = float(row_e.get("ema20",0)), float(row_e.get("ema50",0))
    if signal=="BUY" and e20>e50:   s+=1; r.append("EMA bullish")
    elif signal=="SELL" and e20<e50: s+=1; r.append("EMA bearish")
    c20,c50 = float(row_c.get("ema20",0)), float(row_c.get("ema50",0))
    if signal=="BUY" and c20>c50:   s+=1; r.append("1h confirms")
    elif signal=="SELL" and c20<c50: s+=1; r.append("1h confirms")
    if not r: r.append(f"ML {confidence:.0f}%")
    return s, r


# ════════════ TELEGRAM ═══════════════════════════════════════════════

def _send(text):
    tok=os.getenv("TELEGRAM_TOKEN",""); cid=os.getenv("TELEGRAM_CHAT_ID","")
    if not tok or not cid: return
    try: requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
            data={"chat_id":cid,"text":text,"parse_mode":"Markdown"}, timeout=10)
    except: pass

def _warn(text): log.warning(text); _send(text)

def check_mode_switch(mode):
    last = load_json(MODE_FILE, {})
    if last.get("mode") != mode["mode"]:
        msgs = {"active":"📈 *Active* — conf≥55% every 15min",
                "quiet":"🌙 *Quiet* — conf≥62% every 30min",
                "weekend":"📅 *Weekend* — conf≥58%"}
        _send(msgs.get(mode["mode"],"Mode changed"))
        save_json(MODE_FILE, {
            "mode": mode["mode"], "label": mode["label"],
            "min_confidence": mode["min_confidence"],
            "min_score": mode["min_score"],
            "min_adx": mode["min_adx"],
            "since": datetime.now(timezone.utc).isoformat(),
        })

def _send_open_alert(symbol, signal, confidence, score, entry,
                     stop, tp1, tp2, qty, risk_usd, balance, reasons, exch, order_ids):
    emoji = "🟢" if signal=="BUY" else "🔴"; stars = "⭐"*min(score,5)
    dec   = 4 if entry<10 else 2; fp = lambda v: f"{v:.{dec}f}"
    sl_pct = abs((stop-entry)/entry*100); t1_pct = abs((tp1-entry)/entry*100); t2_pct = abs((tp2-entry)/entry*100)
    rlines = "\n".join([f"  • {r}" for r in reasons])
    sl_id  = (order_ids.get("stop_loss","—") or "—")[:10]
    tp1_id = (order_ids.get("tp1","—") or "—")[:10]
    tp2_id = (order_ids.get("tp2","—") or "—")[:10]
    _send(
        f"🤖 *NEW TRADE — {exch}*\n━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{emoji} *{signal} — {symbol}* {stars}\n_{get_tier(symbol)}_\n"
        f"🎯 {confidence:.1f}% conf · {score}/5 score\n\n"
        f"⚡ Entry:  `{fp(entry)}`\n"
        f"🛑 SL:     `{fp(stop)}` (-{sl_pct:.1f}%) `{sl_id}`\n"
        f"🎯 TP1:    `{fp(tp1)}` (+{t1_pct:.1f}%) `{tp1_id}`\n"
        f"🎯 TP2:    `{fp(tp2)}` (+{t2_pct:.1f}%) `{tp2_id}`\n\n"
        f"💰 Risk: `{risk_usd:.2f}` | Bal: `{balance:.2f}`\n\n"
        f"📊 *Reasons:*\n{rlines}\n━━━━━━━━━━━━━━━━━━━━"
    )

def _send_close_alert(symbol, result, pnl, entry, px, opened_at, exch):
    emoji = "✅" if pnl>0 else "❌"; dec = 4 if entry<10 else 2
    try: dur = str(datetime.now(timezone.utc)-datetime.fromisoformat(opened_at)).split(".")[0]
    except: dur = "—"
    _send(f"🤖 *CLOSED — {exch}*\n{emoji} *{result} — {symbol}*\n\n"
          f"📥 `{entry:.{dec}f}` → 📤 `{px:.{dec}f}`\n💵 *PnL: `{pnl:+.4f}`* | ⏱️ {dur}")


# ════════════ DIAGNOSTIC ═════════════════════════════════════════════

def run_diagnostic():
    from smart_scheduler import get_scan_mode, check_btc_volatility
    mode = get_scan_mode(); vol = check_btc_volatility()
    lines = ["🔍 *Bot Diagnostic — Deribit Testnet*",
             f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
             "━━━━━━━━━━━━━━━━━━━━",
             f"Mode: *{mode['label']}* conf≥{mode['min_confidence']}% score≥{mode['min_score']}",
             f"BTC ATR: *{vol['atr_pct']:.3f}%* ({vol['status']})"]
    try:
        deribit = init_deribit(); paper = init_paper()
        bal = fetch_and_save_balance(deribit, paper)
        lines.append(f"💰 Deribit: *${bal:.2f} USD* ✅")
        trades = load_trades()
        lines.append(f"📂 Open trades: *{len(trades)}*")
        for sym, t in trades.items():
            oids = t.get("order_ids",{})
            lines.append(f"  • {sym} {t.get('signal')} entry:{t.get('entry')} "
                         f"SL:{t.get('stop')} sl_id:{(oids.get('stop_loss') or 'MISSING')[:8]}")
    except Exception as e: lines.append(f"❌ Deribit: {e}")
    try:
        p = load_model(); lines.append(f"🤖 Model: ✅ {len(p['all_features'])} features")
    except Exception as e: lines.append(f"❌ Model: {e}")
    _send("\n".join(lines))


# ════════════ MAIN ════════════════════════════════════════════════════

def run_execution_scan():
    log.info(f"\n{'═'*56}")
    log.info(f"SCAN — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    log.info(f"{'═'*56}")

    run, mode, vol, reason = should_scan()
    check_mode_switch(mode)
    if not run: log.info(f"SKIPPED: {reason}"); return

    effective_risk = get_effective_risk(mode, vol)
    vol_warn       = vol["message"] if vol.get("warn") else None
    thresholds     = get_mode_thresholds(mode)

    deribit    = init_deribit()
    paper      = init_paper()
    pipeline   = load_model()

    log.info(f"\n[0] Balance...")
    deribit_bal = fetch_and_save_balance(deribit, paper)

    log.info(f"\n[1] Clear stuck/broken trades...")
    clear_stuck_trades(deribit, paper)

    log.info(f"\n[2] Monitor open trades...")
    check_open_trades(deribit, paper)

    trades = load_trades()
    log.info(f"\n[3] Scanning {len(SYMBOLS)} coins | Open:{len(trades)}/{MAX_OPEN_TRADES}")
    log.info(f"    conf≥{thresholds['min_confidence']}% score≥{thresholds['min_score']} ADX≥{thresholds['min_adx']} risk:{effective_risk:.2f}")

    found = 0
    for symbol in SYMBOLS:
        if len(load_trades()) >= MAX_OPEN_TRADES: log.info("  Max trades — stop"); break
        exch = "DERIBIT" if deribit.is_supported(symbol) else "PAPER"
        log.info(f"\n  ── {symbol} ({get_tier(symbol)}) [{exch}] ──")
        sig = generate_signal(symbol, pipeline, thresholds)
        if sig is None: time.sleep(0.3); continue
        found += 1
        if vol_warn: sig["reasons"] = list(sig.get("reasons",[])) + [f"⚠️ {vol_warn}"]
        execute_trade(deribit, paper, **{k:sig[k] for k in
            ["symbol","signal","entry","atr","confidence","score","reasons"]},
            risk_mult=effective_risk, deribit_balance=deribit_bal)
        time.sleep(1)

    fetch_and_save_balance(deribit, paper)
    log.info(f"\n{'═'*56}")
    log.info(f"DONE — {found} signal(s) | ${deribit_bal:.2f}")
    log.info(f"{'═'*56}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "diagnostic": run_diagnostic()
    elif len(sys.argv) > 1 and sys.argv[1] == "clear_stuck": clear_stuck_trades(init_deribit(), init_paper())
    else: run_execution_scan()
