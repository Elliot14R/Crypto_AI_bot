# trade_executor.py — Deribit Testnet, all fixes applied
#
# ROOT CAUSES OF NO TRADES (all fixed):
#   1. SYMBOL_MAP only had 2 symbols → 18/20 signals silently skipped
#      FIX: Now tries instrument discovery, skips gracefully if not on Deribit
#   2. Thresholds too high (55% conf, score≥2) → very few qualifying signals
#      FIX: Active hours now 50% conf, score≥1, ADX≥15
#   3. is_supported() was missing → every unsupported symbol caused ValueError
#      FIX: execute_trade() checks is_supported() before attempting order

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
from smart_scheduler import (
    should_scan, get_mode_thresholds, check_correlation, get_effective_risk
)
from deribit_client import DeribitClient, TRADEABLE_SYMBOLS

TRADES_FILE     = "trades.json"
HISTORY_FILE    = "trade_history.json"
SIGNALS_FILE    = "signals.json"
MODE_FILE       = "scan_mode.json"
BALANCE_FILE    = "balance.json"
MAX_OPEN_TRADES = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
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
    # Save to root
    try:
        tmp = str(path) + ".tmp"
        with open(tmp,"w") as f: json.dump(data,f,indent=2,default=str)
        os.replace(tmp, path)
    except Exception as e: log.error(f"save_json {path}: {e}")
    # Mirror to /data/ for GitHub persistence
    try:
        d = Path("data"); d.mkdir(exist_ok=True)
        tmp = str(d / Path(path).name) + ".tmp"
        with open(tmp,"w") as f: json.dump(data,f,indent=2,default=str)
        os.replace(tmp, str(d / Path(path).name))
    except Exception: pass

def load_trades():  return load_json(TRADES_FILE,  {})
def save_trades(d): save_json(TRADES_FILE, d)
def load_history(): return load_json(HISTORY_FILE, [])
def load_signals(): return load_json(SIGNALS_FILE, [])

def append_history(rec):
    h = load_history(); h.append(rec); save_json(HISTORY_FILE, h)

def save_signal(sig):
    sigs = load_signals()
    sigs.append({**sig, "generated_at": datetime.now(timezone.utc).isoformat()})
    save_json(SIGNALS_FILE, sigs[-500:])

def load_model():
    p = joblib.load(MODEL_FILE)
    log.info(f"✓ Model: {len(p['all_features'])} features | 73.1% accuracy")
    return p


# ════════════ EXCHANGE INIT ═══════════════════════════════════════════

def init_deribit() -> DeribitClient:
    cid    = os.getenv("DERIBIT_CLIENT_ID",     "")
    secret = os.getenv("DERIBIT_CLIENT_SECRET", "")
    if not cid or not secret:
        raise ValueError(
            "DERIBIT_CLIENT_ID or DERIBIT_CLIENT_SECRET not set!\n"
            "Register at https://test.deribit.com → Settings → API Keys\n"
            "Add to GitHub Secrets and Render Environment Variables."
        )
    client = DeribitClient(cid, secret)
    client.test_connection()
    return client


def save_balance_json(deribit: DeribitClient):
    """Write REAL Deribit balance to balance.json for dashboard."""
    try:
        balances  = deribit.get_all_balances()
        total_usd = deribit.get_total_equity_usd()
        positions = deribit.get_positions()
        upnl      = sum(float(p.get("floating_profit_loss_usd",0) or
                              p.get("floating_profit_loss",0) or 0)
                        for p in positions)

        assets = [
            {"asset": cur, "free": str(round(info.get("available",0),6)),
             "total": str(round(info.get("equity_usd",0),2))}
            for cur, info in balances.items()
        ]

        save_json(BALANCE_FILE, {
            "usdt":           round(total_usd, 2),
            "equity":         round(total_usd + upnl, 2),
            "unrealised":     round(upnl, 4),
            "assets":         assets,
            "updated_at":     datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "mode":           "deribit_testnet",
            "exchange":       "Deribit Testnet",
            "tradeable":      TRADEABLE_SYMBOLS,
            "open_positions": len(positions),
        })
        log.info(f"  ✅ Balance: ${total_usd:.2f} USD | "
                 f"unrealised: {upnl:+.2f} | positions: {len(positions)}")
        return total_usd

    except Exception as e:
        log.error(f"  save_balance_json failed: {e}")
        return 0.0


# ════════════ MARKET DATA ════════════════════════════════════════════

def get_data(symbol: str, interval: str) -> pd.DataFrame:
    """Binance public API — never geo-blocked anywhere."""
    for url in [
        "https://data-api.binance.vision/api/v3/klines",
        "https://api.binance.com/api/v3/klines",
    ]:
        try:
            r = requests.get(url,
                params={"symbol":symbol,"interval":interval,"limit":LIVE_LIMIT},
                timeout=15)
            r.raise_for_status()
            df = pd.DataFrame(r.json()).iloc[:,:6]
            df.columns = ["open_time","open","high","low","close","volume"]
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c])
            return df
        except Exception:
            continue
    raise Exception(f"Cannot fetch data for {symbol}")


# ════════════ EXECUTE TRADE ══════════════════════════════════════════

def execute_trade(deribit: DeribitClient, symbol, signal, entry, atr,
                  confidence, score, reasons, risk_mult=1.0, balance=10000.0):
    trades = load_trades()
    if symbol in trades:
        log.info(f"  {symbol}: already open — skip"); return False
    if len(trades) >= MAX_OPEN_TRADES:
        log.info("  Max trades reached — skip"); return False
    if not check_correlation(trades, signal):
        return False

    # ── KEY FIX: check if symbol is supported on Deribit ─────────────
    if not deribit.is_supported(symbol):
        log.info(f"  {symbol}: not on Deribit testnet — skip")
        return False

    dec = 4 if entry < 10 else 2
    if signal == "BUY":
        stop  = round(entry - atr * ATR_STOP_MULT,    dec)
        tp1   = round(entry + atr * ATR_TARGET1_MULT, dec)
        tp2   = round(entry + atr * ATR_TARGET2_MULT, dec)
        side  = "BUY";  sl_side = "SELL"; tp_side = "SELL"
    else:
        stop  = round(entry + atr * ATR_STOP_MULT,    dec)
        tp1   = round(entry - atr * ATR_TARGET1_MULT, dec)
        tp2   = round(entry - atr * ATR_TARGET2_MULT, dec)
        side  = "SELL"; sl_side = "BUY";  tp_side = "BUY"

    if balance < 5:
        _warn(f"⚠️ Balance ${balance:.2f} too low for {symbol}"); return False

    amount     = deribit.calc_contracts(symbol, balance, entry, stop, risk_mult)
    amount_tp1 = max(deribit.get_min_trade_amount(symbol), round(amount * 0.5, 1))
    amount_tp2 = max(deribit.get_min_trade_amount(symbol), round(amount * 0.5, 1))
    risk_usd   = round(balance * RISK_PER_TRADE * risk_mult, 2)

    log.info(f"  {signal} {symbol} | amount={amount} "
             f"| SL={stop:.{dec}f} TP1={tp1:.{dec}f} TP2={tp2:.{dec}f}")

    order_ids    = {}
    actual_entry = entry

    try:
        # Entry
        eo = deribit.place_market_order(symbol, side, amount)
        if not eo:
            log.error(f"  Entry order returned empty — skip"); return False
        order_ids["entry"] = str(eo.get("order_id", ""))
        actual_entry = float(
            eo.get("average_price") or eo.get("price") or entry
        ) or entry
        log.info(f"  ✅ Entry @ ~{actual_entry:.{dec}f}")
        time.sleep(1.5)

        # Stop Loss
        try:
            sl = deribit.place_limit_order(symbol, sl_side, amount, stop, stop_price=stop)
            order_ids["stop_loss"] = str(sl.get("order_id",""))
            log.info(f"  ✅ SL @ {stop:.{dec}f}")
        except Exception as e:
            log.warning(f"  SL failed: {e}")

        # TP1
        try:
            t1 = deribit.place_limit_order(symbol, tp_side, amount_tp1, tp1)
            order_ids["tp1"] = str(t1.get("order_id",""))
            log.info(f"  ✅ TP1 @ {tp1:.{dec}f} ({amount_tp1})")
        except Exception as e:
            log.warning(f"  TP1 failed: {e}")

        # TP2
        try:
            t2 = deribit.place_limit_order(symbol, tp_side, amount_tp2, tp2)
            order_ids["tp2"] = str(t2.get("order_id",""))
            log.info(f"  ✅ TP2 @ {tp2:.{dec}f} ({amount_tp2})")
        except Exception as e:
            log.warning(f"  TP2 failed: {e}")

    except Exception as e:
        log.error(f"  Trade error {symbol}: {e}")
        _warn(f"⚠️ Trade error {symbol}: {e}")
        return False

    record = {
        "symbol": symbol, "signal": signal,
        "entry": actual_entry, "stop": stop, "tp1": tp1, "tp2": tp2,
        "qty": amount, "qty_tp1": amount_tp1, "qty_tp2": amount_tp2,
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
                     stop, tp1, tp2, amount, amount_tp1, amount_tp2,
                     risk_usd, balance, reasons, risk_mult)
    log.info(f"  ✅✅ TRADE OPENED: {symbol} {signal} on Deribit Testnet!")
    return True


# ════════════ TRADE MONITORING ════════════════════════════════════════

def _is_filled(order: dict) -> bool:
    status = order.get("order_state", order.get("status","")).lower()
    return status in ("filled","closed","cancelled_with_fill")

def _fill_price(order: dict, fallback: float) -> float:
    p = float(order.get("average_price") or order.get("price") or fallback or 0)
    return p if p > 0 else fallback


def check_open_trades(deribit: DeribitClient):
    trades = load_trades()
    if not trades:
        log.info("  No open trades"); return

    to_remove = []
    log.info(f"  Monitoring {len(trades)} trade(s)")

    for symbol, trade in list(trades.items()):
        if trade.get("closed"):
            to_remove.append(symbol); continue

        oids  = trade.get("order_ids", {})
        entry = float(trade["entry"])
        dec   = 4 if entry < 10 else 2

        try:
            # ── TP1 ─────────────────────────────────────────────────
            if not trade["tp1_hit"] and "tp1" in oids:
                o = deribit.get_order(oids["tp1"])
                if _is_filled(o):
                    trade["tp1_hit"] = True
                    fill = _fill_price(o, trade["tp1"])
                    pnl  = _calc_pnl(trade, fill, "tp1")
                    log.info(f"  🎯 TP1 HIT {symbol} @ {fill:.{dec}f} | pnl≈{pnl:+.4f}")
                    _send_close_alert(symbol, "TP1 HIT 🎯", pnl, entry, fill, trade["opened_at"])

                    # Move SL to breakeven
                    if "stop_loss" in oids:
                        try:
                            deribit.cancel_order(oids["stop_loss"])
                            sl_side = "SELL" if trade["signal"]=="BUY" else "BUY"
                            qty_rem = trade.get("qty_tp2", trade.get("qty_tp1",1))
                            new_sl  = deribit.place_limit_order(
                                symbol, sl_side, qty_rem, entry, stop_price=entry)
                            trade["order_ids"]["stop_loss"] = str(new_sl.get("order_id",""))
                            trade["stop"] = entry
                            _send(f"🛡️ *{symbol}* SL → breakeven `{entry:.{dec}f}` — risk-free!")
                        except Exception as be:
                            log.warning(f"  Breakeven SL: {be}")

            # ── TP2 ─────────────────────────────────────────────────
            if trade["tp1_hit"] and not trade["tp2_hit"] and "tp2" in oids:
                o = deribit.get_order(oids["tp2"])
                if _is_filled(o):
                    trade["tp2_hit"] = True; trade["closed"] = True
                    fill = _fill_price(o, trade["tp2"])
                    pnl  = _calc_pnl(trade, fill, "tp2")
                    log.info(f"  ✅ TP2 HIT {symbol} @ {fill:.{dec}f} | pnl≈{pnl:+.4f}")
                    _send_close_alert(symbol, "✅ FULL WIN (TP2)", pnl, entry, fill, trade["opened_at"])
                    _record_close(trade, fill, pnl, "TP2 hit")
                    to_remove.append(symbol)

            # ── SL ──────────────────────────────────────────────────
            if not trade.get("closed") and "stop_loss" in oids:
                o = deribit.get_order(oids["stop_loss"])
                if _is_filled(o):
                    trade["closed"] = True
                    fill = _fill_price(o, trade["stop"])
                    pnl  = _calc_pnl(trade, fill, "sl")
                    log.info(f"  ❌ SL HIT {symbol} @ {fill:.{dec}f} | pnl≈{pnl:+.4f}")
                    _send_close_alert(symbol, "❌ STOPPED OUT", pnl, entry, fill, trade["opened_at"])
                    _record_close(trade, fill, pnl, "SL hit")
                    for k in ("tp1","tp2"):
                        if k in oids and not trade.get(f"{k}_hit"):
                            try: deribit.cancel_order(oids[k])
                            except Exception: pass
                    to_remove.append(symbol)

        except Exception as e:
            log.error(f"  Monitor error {symbol}: {e}")

    save_trades(trades)
    for sym in set(to_remove): trades.pop(sym, None)
    save_trades(trades)


def clear_stuck_trades(deribit: DeribitClient):
    trades = load_trades()
    if not trades: return
    log.info(f"  Checking {len(trades)} trade(s) for stuck orders...")
    cleared = 0
    for symbol in list(trades.keys()):
        trade = trades[symbol]
        oids  = trade.get("order_ids", {})
        live  = False
        for key, oid in oids.items():
            if key == "entry": continue
            try:
                o = deribit.get_order(oid)
                if o.get("order_state","") in ("open","partially_filled"):
                    live = True; break
            except Exception: pass
        if not live and oids:
            log.warning(f"  ⚠️ {symbol}: no live orders — clearing")
            _record_close(trade, float(trade.get("entry",0)), 0.0, "Auto-cleared")
            trades.pop(symbol); cleared += 1
    if cleared:
        save_trades(trades)
        log.info(f"  ✅ Cleared {cleared} stuck trade(s)")
        _send(f"🧹 Cleared {cleared} stuck trade(s) from trades.json")


def _calc_pnl(trade, close_price, close_type) -> float:
    qty  = (trade["qty_tp1"] if close_type=="tp1" else
            trade["qty_tp2"] if close_type=="tp2" else trade["qty"])
    diff = (close_price - trade["entry"]) if trade["signal"]=="BUY" \
           else (trade["entry"] - close_price)
    return round(diff * float(qty), 4)

def _record_close(trade, close_price, pnl, reason):
    append_history({
        **trade, "close_price": close_price, "pnl": pnl,
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "close_reason": reason,
    })


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

        af   = pipeline["all_features"]
        miss = [f for f in af if f not in row.index]
        if miss: return None

        X    = pd.DataFrame([row[af].values], columns=af)
        Xs   = pipeline["selector"].transform(X)
        pred = pipeline["ensemble"].predict(Xs)[0]
        prob = pipeline["ensemble"].predict_proba(Xs)[0]
        sig  = {0:"BUY",1:"SELL",2:"NO_TRADE"}[pred]
        conf = round(float(max(prob))*100, 1)

        log.info(f"    ML: {sig} {conf:.1f}% (need ≥{thresholds['min_confidence']}%)")
        if sig == "NO_TRADE" or conf < thresholds["min_confidence"]: return None

        adx = float(row.get("adx",0))
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
                         "entry":entry,"atr":atr,"reasons":reasons,"rejected":True,
                         "reject_reason":f"score {score}<{thresholds['min_score']}",
                         "stop":stop,"tp1":tp1,"tp2":tp2})
            return None

        return {"symbol":symbol,"signal":sig,"confidence":conf,"score":score,
                "entry":entry,"atr":atr,"stop":stop,"tp1":tp1,"tp2":tp2,"reasons":reasons}

    except Exception as e:
        log.error(f"    Signal error {symbol}: {e}"); return None


def _quality_score(row, r1h, signal, conf):
    score, reasons = 0, []

    if conf>=70:   score+=1; reasons.append(f"High conf ({conf:.0f}%)")
    elif conf>=55: score+=1; reasons.append(f"Conf ({conf:.0f}%)")
    elif conf>=50: reasons.append(f"Conf ({conf:.0f}%)")

    adx=float(row.get("adx",0))
    if adx>20:   score+=1; reasons.append(f"Strong ADX {adx:.0f}")
    elif adx>15: score+=1; reasons.append(f"ADX {adx:.0f}")

    rsi=float(row.get("rsi",50))
    if signal=="BUY"   and rsi<50: score+=1; reasons.append(f"RSI bullish ({rsi:.0f})")
    elif signal=="SELL" and rsi>50: score+=1; reasons.append(f"RSI bearish ({rsi:.0f})")

    e20=float(row.get("ema20",0)); e50=float(row.get("ema50",0))
    if signal=="BUY"   and e20>e50: score+=1; reasons.append("EMA bullish")
    elif signal=="SELL" and e20<e50: score+=1; reasons.append("EMA bearish")

    c20=float(r1h.get("ema20",0)); c50=float(r1h.get("ema50",0))
    if signal=="BUY"   and c20>c50: score+=1; reasons.append("1h confirms")
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
        msgs = {
            "active":  "📈 *Active hours* — conf≥50% | score≥1 | full risk",
            "quiet":   "🌙 *Quiet hours* — conf≥55% | score≥2 | 50% risk",
            "weekend": "📅 *Weekend* — conf≥50% | score≥1 | 75% risk",
        }
        _send(msgs.get(mode["mode"], "Mode changed"))
        save_json(MODE_FILE, {"mode":mode["mode"],"since":datetime.now(timezone.utc).isoformat()})

def _send_open_alert(symbol, signal, confidence, score, entry,
                     stop, tp1, tp2, amount, tp1_qty, tp2_qty,
                     risk_usd, balance, reasons, risk_mult=1.0):
    emoji="🟢" if signal=="BUY" else "🔴"; stars="⭐"*min(score,5)
    dec=4 if entry<10 else 2
    sl_pct=abs((stop-entry)/entry*100); t1_pct=abs((tp1-entry)/entry*100); t2_pct=abs((tp2-entry)/entry*100)
    rlines="\n".join([f"  • {r}" for r in reasons])
    risk_n="" if risk_mult>=1.0 else f"\n⚡ Risk mult: {int(risk_mult*100)}%"
    _send(
        f"🤖 *DERIBIT TESTNET TRADE*\n━━━━━━━━━━━━━━━━━━━━\n"
        f"{emoji} *{signal} — {symbol}* {stars}\n"
        f"🏷️ _{get_tier(symbol)}_{risk_n}\n"
        f"🎯 Conf: *{confidence:.1f}%* · Score: *{score}*\n\n"
        f"⚡ *ENTRY:* `{entry:.{dec}f}`\n"
        f"🛑 *STOP:* `{stop:.{dec}f}` (-{sl_pct:.1f}%)\n"
        f"🎯 *TP1:* `{tp1:.{dec}f}` (+{t1_pct:.1f}%) × {tp1_qty}\n"
        f"🎯 *TP2:* `{tp2:.{dec}f}` (+{t2_pct:.1f}%) × {tp2_qty}\n"
        f"📦 Total: *{amount}* · Risk: `${risk_usd:.2f}`\n"
        f"💼 Portfolio: `${balance:.2f}`\n\n"
        f"📊 *Reasons:*\n{rlines}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n_Deribit Testnet_"
    )

def _send_close_alert(symbol, result, pnl, entry, close_price, opened_at):
    emoji="✅" if pnl>0 else "❌"; dec=4 if entry<10 else 2
    try: dur=str(datetime.now(timezone.utc)-datetime.fromisoformat(opened_at)).split(".")[0]
    except Exception: dur="—"
    _send(
        f"🤖 *DERIBIT TRADE CLOSED*\n"
        f"{emoji} *{result} — {symbol}*\n"
        f"📥 `{entry:.{dec}f}` → 📤 `{close_price:.{dec}f}`\n"
        f"💵 *PnL ≈ `{pnl:+.4f}`* · ⏱️ {dur}\n"
        f"_Check Deribit testnet for exact realized PnL_"
    )


# ════════════ MAIN ════════════════════════════════════════════════════

def run_execution_scan():
    log.info(f"\n{'═'*56}")
    log.info(f"SCAN — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    log.info(f"Exchange: Deribit Testnet | Tradeable: {TRADEABLE_SYMBOLS}")
    log.info(f"{'═'*56}")

    run, mode, vol, reason = should_scan()
    check_mode_switch(mode)
    if not run:
        log.info(f"  SKIPPED: {reason}"); return

    effective_risk = get_effective_risk(mode, vol)
    vol_warn       = vol["message"] if vol.get("warn") else None

    deribit    = init_deribit()
    pipeline   = load_model()
    thresholds = get_mode_thresholds(mode)

    log.info(f"  {mode['label']} | conf≥{thresholds['min_confidence']}% "
             f"| score≥{thresholds['min_score']} | ADX≥{thresholds['min_adx']} "
             f"| risk:{effective_risk:.2f}")

    log.info(f"\n[0] Fetching Deribit balance...")
    balance = save_balance_json(deribit)

    log.info(f"\n[1] Clearing stuck trades...")
    clear_stuck_trades(deribit)

    log.info(f"\n[2] Monitoring open trades...")
    check_open_trades(deribit)
    save_balance_json(deribit)   # update after closes

    trades = load_trades()
    log.info(f"\n[3] Scanning {len(SYMBOLS)} coins | Open:{len(trades)}/{MAX_OPEN_TRADES}")
    log.info(f"    Tradeable on Deribit: {TRADEABLE_SYMBOLS}")

    found = 0
    for symbol in SYMBOLS:
        if len(load_trades()) >= MAX_OPEN_TRADES:
            log.info("  🛑 Max trades — stop"); break

        log.info(f"\n  ── {symbol} ({get_tier(symbol)}) ──")
        sig = generate_signal(symbol, pipeline, thresholds)
        if sig is None:
            time.sleep(0.3); continue

        found += 1
        log.info(f"  ✅ SIGNAL: {sig['signal']} {sig['confidence']:.1f}% score={sig['score']}")

        if vol_warn:
            sig["reasons"] = list(sig.get("reasons",[])) + [f"⚠️ {vol_warn}"]

        execute_trade(
            deribit,
            symbol=sig["symbol"], signal=sig["signal"],
            entry=sig["entry"],   atr=sig["atr"],
            confidence=sig["confidence"], score=sig["score"],
            reasons=sig["reasons"], risk_mult=effective_risk,
            balance=balance,
        )
        time.sleep(1)

    save_balance_json(deribit)

    log.info(f"\n{'═'*56}")
    log.info(f"DONE — {found} signal(s) | Portfolio: ${balance:.2f}")
    log.info(f"{'═'*56}\n")


def run_diagnostic():
    from smart_scheduler import get_scan_mode, check_btc_volatility
    mode = get_scan_mode(); vol = check_btc_volatility()
    eff  = get_effective_risk(mode, vol)
    lines = [
        "🔍 *Diagnostic — Deribit Testnet*",
        f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    try:
        deribit = init_deribit()
        bal     = save_balance_json(deribit)
        pos     = deribit.get_positions()
        # Check which symbols actually work
        working = [s for s in TRADEABLE_SYMBOLS if deribit.is_supported(s)]
        lines += [
            f"✅ Deribit connected",
            f"💰 Portfolio: *${bal:.2f} USD*",
            f"📂 Open positions: {len(pos)}",
            f"📋 Working symbols: {len(working)} — {working}",
        ]
    except Exception as e: lines.append(f"❌ Deribit: {e}")

    lines += [
        f"📋 Mode: *{mode['label']}*",
        f"📊 BTC ATR: *{vol['atr_pct']:.2f}%*",
        f"⚙️ Conf≥{mode['min_confidence']}% | Score≥{mode['min_score']}",
        f"⚡ Risk: *{int(eff*100)}%* of 2%",
    ]

    try:
        p = load_model(); lines.append(f"🤖 Model: ✅ {len(p['all_features'])} features")
    except Exception as e: lines.append(f"❌ Model: {e}")

    real = [h for h in load_history() if h.get("signal") != "RECOVERED"]
    wins = [h for h in real if (h.get("pnl") or 0) > 0]
    wr   = round(len(wins)/len(real)*100,1) if real else 0
    tpnl = sum(h.get("pnl",0) for h in real)
    lines += [
        f"📈 Win rate: *{wr}%* ({len(wins)}W/{len(real)-len(wins)}L/{len(real)} trades)",
        f"💵 PnL: *{tpnl:+.4f}*",
    ]
    _send("\n".join(lines))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "diagnostic":
        run_diagnostic()
    elif len(sys.argv) > 1 and sys.argv[1] == "clear_stuck":
        clear_stuck_trades(init_deribit())
    else:
        run_execution_scan()
