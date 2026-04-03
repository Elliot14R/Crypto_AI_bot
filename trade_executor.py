# trade_executor.py — Paper Trading Version
# Uses real Binance PUBLIC market data for signals (never geo-blocked)
# Simulates trade execution locally — no API keys needed for execution
# Balance, orders, PnL all tracked in JSON files

import os, json, time, logging, requests, joblib
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from persistence import save_json, load_json

load_dotenv(dotenv_path=".env", override=True)

from config import (
    SYMBOLS, ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT,
    RISK_PER_TRADE, TIMEFRAME_ENTRY, TIMEFRAME_CONFIRM,
    LIVE_LIMIT, MODEL_FILE, LOG_FILE, get_tier, MAX_SAME_DIRECTION
)
from feature_engineering import add_indicators
from smart_scheduler import (
    should_scan, get_mode_thresholds,
    check_correlation, get_effective_risk
)
from paper_trader import PaperTrader

TRADES_FILE     = "trades.json"
HISTORY_FILE    = "trade_history.json"
SIGNALS_FILE    = "signals.json"
MODE_FILE       = "scan_mode.json"
BALANCE_FILE    = "balance.json"
MAX_OPEN_TRADES = 3
TP1_CLOSE_PCT   = 0.5
TP2_CLOSE_PCT   = 0.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# FILE I/O
# ══════════════════════════════════════════════════════════

# def load_json(path, default):
#     try:
#         if Path(path).exists():
#             with open(path) as f:
#                 return json.load(f)
#     except Exception: pass
#     return default


# def save_json(path, data):
#     try:
#         tmp = str(path) + ".tmp"
#         with open(tmp, "w") as f:
#             json.dump(data, f, indent=2, default=str)
#         os.replace(tmp, path)
#     except Exception as e:
#         log.error(f"save_json {path}: {e}")


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
    pipeline = joblib.load(MODEL_FILE)
    required = ["ensemble", "selector", "all_features", "best_features", "label_map"]
    missing  = [k for k in required if k not in pipeline]
    if missing:
        raise ValueError(f"Model missing keys: {missing}")
    log.info(f"✓ Model: {len(pipeline['all_features'])} features")
    return pipeline


# ══════════════════════════════════════════════════════════
# MARKET DATA — public API, never geo-blocked
# ══════════════════════════════════════════════════════════

def get_data(symbol: str, interval: str) -> pd.DataFrame:
    url    = "https://data-api.binance.vision/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": LIVE_LIMIT}
    resp   = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()).iloc[:, :6]
    df.columns = ["open_time","open","high","low","close","volume"]
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c])
    return df


def calc_pos_size(balance: float, entry: float, stop: float, risk_mult: float = 1.0):
    risk_usd = balance * RISK_PER_TRADE * risk_mult
    dist     = abs(entry - stop)
    if dist <= 0: return 0.0, 0.0
    qty = risk_usd / dist
    max_usd = balance * 0.20
    if qty * entry > max_usd:
        qty = max_usd / entry
    return round(qty, 6), round(risk_usd, 2)


# ══════════════════════════════════════════════════════════
# TRADE EXECUTION — paper simulation
# ══════════════════════════════════════════════════════════

def execute_trade(ex: PaperTrader, symbol, signal, entry, atr,
                  confidence, score, reasons, risk_mult=1.0):
    trades = load_trades()
    if symbol in trades:
        log.info(f"  {symbol}: already open — skip")
        return False
    if len(trades) >= MAX_OPEN_TRADES:
        log.info(f"  Max trades ({MAX_OPEN_TRADES}) reached — skip")
        return False
    if not check_correlation(trades, signal):
        return False

    dec = 4 if entry < 10 else 2
    if signal == "BUY":
        stop    = round(entry - atr * ATR_STOP_MULT,    dec)
        tp1     = round(entry + atr * ATR_TARGET1_MULT, dec)
        tp2     = round(entry + atr * ATR_TARGET2_MULT, dec)
        side    = "BUY";  sl_side = "SELL"; tp_side = "SELL"
    else:
        stop    = round(entry + atr * ATR_STOP_MULT,    dec)
        tp1     = round(entry - atr * ATR_TARGET1_MULT, dec)
        tp2     = round(entry - atr * ATR_TARGET2_MULT, dec)
        side    = "SELL"; sl_side = "BUY";  tp_side = "BUY"

    balance = ex.get_usdt_balance()
    if balance < 10:
        log.warning(f"  Balance too low ({balance:.2f} USDT)")
        return False

    qty, risk_usd = calc_pos_size(balance, entry, stop, risk_mult)
    if qty <= 0:
        log.warning(f"  Zero position size for {symbol}")
        return False

    qty_tp1   = round(qty * TP1_CLOSE_PCT, 6)
    qty_tp2   = round(qty * TP2_CLOSE_PCT, 6)
    order_ids = {}

    log.info(f"  Placing {signal} {symbol} qty={qty} risk={risk_usd:.2f} USDT")

    try:
        eo = ex.place_market_order(symbol, side, qty)
        order_ids["entry"] = str(eo.get("orderId", ""))
        actual_entry = float(eo.get("paper_fill", entry) or
                             eo.get("fills",[{}])[0].get("price", entry) or entry)
        log.info(f"  ✅ Entry @ {actual_entry:.{dec}f}")

        sl = ex.place_limit_order(symbol, sl_side, qty, stop, stop_price=stop)
        order_ids["stop_loss"] = str(sl.get("orderId",""))
        log.info(f"  ✅ SL @ {stop}")

        t1 = ex.place_limit_order(symbol, tp_side, qty_tp1, tp1)
        order_ids["tp1"] = str(t1.get("orderId",""))
        log.info(f"  ✅ TP1 @ {tp1}")

        t2 = ex.place_limit_order(symbol, tp_side, qty_tp2, tp2)
        order_ids["tp2"] = str(t2.get("orderId",""))
        log.info(f"  ✅ TP2 @ {tp2}")

    except Exception as e:
        log.error(f"  Order error {symbol}: {e}")
        return False

    record = {
        "symbol": symbol, "signal": signal,
        "entry": actual_entry, "stop": stop, "tp1": tp1, "tp2": tp2,
        "qty": qty, "qty_tp1": qty_tp1, "qty_tp2": qty_tp2,
        "risk_usd": risk_usd, "balance_at_open": balance,
        "risk_mult": risk_mult, "order_ids": order_ids,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "tp1_hit": False, "tp2_hit": False, "closed": False,
        "confidence": confidence, "score": score, "reasons": reasons,
        "tier": get_tier(symbol), "mode": "paper_trading",
    }
    trades[symbol] = record
    save_trades(trades)
    save_signal(record)
    _send_open_alert(symbol, signal, confidence, score, actual_entry,
                     stop, tp1, tp2, qty, risk_usd, balance, reasons, risk_mult)
    log.info(f"  ✅✅ PAPER TRADE OPENED: {symbol} {signal}")
    return True


# ══════════════════════════════════════════════════════════
# MONITORING
# ══════════════════════════════════════════════════════════

def check_open_trades(ex: PaperTrader):
    trades = load_trades()
    if not trades:
        log.info("  No open trades")
        return

    to_remove = []
    log.info(f"  Monitoring {len(trades)} trade(s)")

    for symbol, trade in list(trades.items()):
        if trade.get("closed"):
            to_remove.append(symbol)
            continue
        try:
            oids  = trade.get("order_ids", {})
            entry = float(trade["entry"])

            # Check TP1
            if not trade["tp1_hit"] and "tp1" in oids:
                try:
                    o = ex.get_order(symbol, oids["tp1"])
                    if o.get("status") == "FILLED":
                        trade["tp1_hit"] = True
                        fill = float(o.get("price") or trade["tp1"])
                        pnl  = _calc_pnl(trade, fill, "tp1")
                        log.info(f"  🎯 TP1 HIT {symbol} pnl={pnl:+.4f}")
                        _send_close_alert(symbol, "TP1 HIT 🎯", pnl, entry, fill, trade["opened_at"])
                        # Move SL to breakeven
                        try:
                            sl_side = "SELL" if trade["signal"]=="BUY" else "BUY"
                            new_sl  = ex.place_limit_order(symbol, sl_side, trade["qty_tp2"], entry, stop_price=entry)
                            trade["order_ids"]["stop_loss"] = str(new_sl.get("orderId",""))
                            trade["stop"] = entry
                            _send(f"🛡️ *{symbol}* SL moved to entry — risk-free now!")
                        except Exception as be:
                            log.warning(f"  Breakeven SL failed: {be}")
                except Exception as e:
                    log.warning(f"  TP1 check {symbol}: {e}")

            # Check TP2
            if trade["tp1_hit"] and not trade["tp2_hit"] and "tp2" in oids:
                try:
                    o = ex.get_order(symbol, oids["tp2"])
                    if o.get("status") == "FILLED":
                        trade["tp2_hit"] = True; trade["closed"] = True
                        fill = float(o.get("price") or trade["tp2"])
                        pnl  = _calc_pnl(trade, fill, "tp2")
                        log.info(f"  ✅ TP2 HIT {symbol} pnl={pnl:+.4f}")
                        _send_close_alert(symbol, "✅ FULL WIN (TP2)", pnl, entry, fill, trade["opened_at"])
                        _record_close(trade, fill, pnl, "TP2 hit")
                        ex.update_balance_after_close(pnl)
                        to_remove.append(symbol)
                except Exception as e:
                    log.warning(f"  TP2 check {symbol}: {e}")

            # Check SL
            if not trade.get("closed") and "stop_loss" in oids:
                try:
                    o = ex.get_order(symbol, oids["stop_loss"])
                    if o.get("status") == "FILLED":
                        trade["closed"] = True
                        fill = float(o.get("price") or trade["stop"])
                        pnl  = _calc_pnl(trade, fill, "sl")
                        log.info(f"  ❌ SL HIT {symbol} pnl={pnl:+.4f}")
                        _send_close_alert(symbol, "❌ STOPPED OUT", pnl, entry, fill, trade["opened_at"])
                        _record_close(trade, fill, pnl, "SL hit")
                        ex.update_balance_after_close(pnl)
                        to_remove.append(symbol)
                except Exception as e:
                    log.warning(f"  SL check {symbol}: {e}")

        except Exception as e:
            log.error(f"  Monitor error {symbol}: {e}")

    save_trades(trades)
    for sym in set(to_remove):
        trades.pop(sym, None)
    save_trades(trades)


def _calc_pnl(trade, close_price, close_type):
    qty = (trade["qty_tp1"] if close_type=="tp1" else
           trade["qty_tp2"] if close_type=="tp2" else trade["qty"])
    if trade["signal"] == "BUY":
        return round((close_price - trade["entry"]) * qty, 4)
    return round((trade["entry"] - close_price) * qty, 4)


def _record_close(trade, close_price, pnl, reason):
    append_history({
        **trade, "close_price": close_price, "pnl": pnl,
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "close_reason": reason,
    })


# ══════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ══════════════════════════════════════════════════════════

def generate_signal(symbol, pipeline, thresholds):
    try:
        df15 = add_indicators(get_data(symbol, TIMEFRAME_ENTRY))
        df1h = add_indicators(get_data(symbol, TIMEFRAME_CONFIRM))
        if df15.empty or len(df15) < 50: return None

        row = df15.iloc[-1].copy()
        r1h = df1h.iloc[-1] if not df1h.empty else pd.Series(dtype=float)
        row["rsi_1h"]   = float(r1h.get("rsi",   50))
        row["adx_1h"]   = float(r1h.get("adx",    0))
        row["trend_1h"] = float(r1h.get("trend",  0))

        all_feat = pipeline["all_features"]
        missing  = [f for f in all_feat if f not in row.index]
        if missing:
            log.warning(f"    Missing features: {missing[:5]}")
            return None

        X_raw  = pd.DataFrame([row[all_feat].values], columns=all_feat)
        X_sel  = pipeline["selector"].transform(X_raw)
        pred   = pipeline["ensemble"].predict(X_sel)[0]
        prob   = pipeline["ensemble"].predict_proba(X_sel)[0]
        signal = {0:"BUY", 1:"SELL", 2:"NO_TRADE"}[pred]
        conf   = round(float(max(prob)) * 100, 1)

        log.info(f"    ML: {signal} {conf:.1f}% (need ≥{thresholds['min_confidence']}%)")
        if signal == "NO_TRADE" or conf < thresholds["min_confidence"]: return None

        adx = float(row.get("adx", 0))
        log.info(f"    ADX: {adx:.1f} (need ≥{thresholds['min_adx']})")
        if adx < thresholds["min_adx"]: return None

        score, reasons = _quality_score(row, r1h, signal, conf)
        log.info(f"    Score: {score}/6 (need ≥{thresholds['min_score']})")

        entry = float(row["close"])
        atr   = float(row["atr"])
        dec   = 4 if entry < 10 else 2

        if signal == "BUY":
            stop = round(entry - atr*ATR_STOP_MULT, dec)
            tp1  = round(entry + atr*ATR_TARGET1_MULT, dec)
            tp2  = round(entry + atr*ATR_TARGET2_MULT, dec)
        else:
            stop = round(entry + atr*ATR_STOP_MULT, dec)
            tp1  = round(entry - atr*ATR_TARGET1_MULT, dec)
            tp2  = round(entry - atr*ATR_TARGET2_MULT, dec)

        if score < thresholds["min_score"]:
            save_signal({"symbol":symbol,"signal":signal,"confidence":conf,"score":score,
                         "entry":entry,"atr":atr,"reasons":reasons,"rejected":True,
                         "reject_reason":f"score {score}<{thresholds['min_score']}",
                         "stop":stop,"tp1":tp1,"tp2":tp2})
            return None

        return {"symbol":symbol,"signal":signal,"confidence":conf,"score":score,
                "entry":entry,"atr":atr,"stop":stop,"tp1":tp1,"tp2":tp2,"reasons":reasons}

    except Exception as e:
        log.error(f"    Signal error {symbol}: {e}")
        return None


def _quality_score(row, r1h, signal, confidence):
    score, reasons = 0, []
    if confidence >= 75:   score+=1; reasons.append(f"High AI conf ({confidence:.0f}%)")
    elif confidence >= 60: score+=1; reasons.append(f"Good AI conf ({confidence:.0f}%)")
    elif confidence >= 55: reasons.append(f"AI conf ({confidence:.0f}%)")

    adx = float(row.get("adx", 0))
    if adx > 20:   score+=1; reasons.append(f"Strong trend ADX {adx:.0f}")
    elif adx > 15: score+=1; reasons.append(f"Moderate ADX {adx:.0f}")

    rsi = float(row.get("rsi", 50))
    if signal=="BUY"   and rsi<50:  score+=1; reasons.append(f"RSI bullish ({rsi:.0f})")
    elif signal=="SELL" and rsi>50: score+=1; reasons.append(f"RSI bearish ({rsi:.0f})")

    e20=float(row.get("ema20",0)); e50=float(row.get("ema50",0))
    if signal=="BUY"   and e20>e50: score+=1; reasons.append("EMA20>EMA50 uptrend")
    elif signal=="SELL" and e20<e50: score+=1; reasons.append("EMA20<EMA50 downtrend")

    c20=float(r1h.get("ema20",0)); c50=float(r1h.get("ema50",0))
    if signal=="BUY"   and c20>c50: score+=1; reasons.append("1h uptrend confirmed")
    elif signal=="SELL" and c20<c50: score+=1; reasons.append("1h downtrend confirmed")

    return score, reasons


# ══════════════════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════════════════

def _send(text):
    token   = os.getenv("TELEGRAM_TOKEN",   "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id: return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id":chat_id,"text":text,"parse_mode":"Markdown"}, timeout=10)
    except Exception: pass

def _warn(text): log.warning(text); _send(text)

def check_mode_switch(mode):
    last = load_json(MODE_FILE, {})
    if last.get("mode") != mode["mode"]:
        msgs = {
            "active":  "📈 *Active hours* — conf≥65% | score≥3 | full risk",
            "quiet":   "🌙 *Quiet hours* — conf≥72% | score≥4 | 50% risk",
            "weekend": "📅 *Weekend* — conf≥65% | score≥3 | 75% risk",
        }
        _send(msgs.get(mode["mode"], "Mode changed"))
        save_json(MODE_FILE, {"mode":mode["mode"],"since":datetime.now(timezone.utc).isoformat()})

def _send_open_alert(symbol, signal, confidence, score, entry,
                     stop, tp1, tp2, qty, risk_usd, balance, reasons, risk_mult=1.0):
    emoji  = "🟢" if signal=="BUY" else "🔴"
    stars  = "⭐"*min(score,5)
    dec    = 4 if entry<10 else 2
    sl_pct = abs((stop-entry)/entry*100)
    t1_pct = abs((tp1-entry)/entry*100)
    risk_n = "" if risk_mult>=1.0 else f"\n⚡ Risk: {int(risk_mult*100)}%"
    rlines = "\n".join([f"  • {r}" for r in reasons])
    _send(
        f"🤖 *PAPER TRADE OPENED*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{emoji} *{signal} — {symbol}* {stars}\n"
        f"🏷️ {get_tier(symbol)}{risk_n}\n"
        f"🎯 Conf: *{confidence:.1f}%* · Score: *{score}/6*\n\n"
        f"⚡ *ENTRY:* `{entry:.{dec}f}`\n"
        f"🛑 *STOP:* `{stop:.{dec}f}` (-{sl_pct:.1f}%)\n"
        f"🎯 *TP1:* `{tp1:.{dec}f}` (+{t1_pct:.1f}%)\n"
        f"💰 Pos: `{qty*entry:.2f} USDT` · Risk: `{risk_usd:.2f} USDT`\n\n"
        f"📊 *Reasons:*\n{rlines}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_Paper Trading (Real market data)_"
    )

def _send_close_alert(symbol, result, pnl, entry, close_price, opened_at):
    emoji = "✅" if pnl>0 else "❌"
    dec   = 4 if entry<10 else 2
    try:
        dur = str(datetime.now(timezone.utc)-datetime.fromisoformat(opened_at)).split(".")[0]
    except Exception: dur = "—"
    _send(
        f"🤖 *PAPER TRADE CLOSED*\n"
        f"{emoji} *{result} — {symbol}*\n"
        f"📥 `{entry:.{dec}f}` → 📤 `{close_price:.{dec}f}`\n"
        f"💵 *PnL: `{pnl:+.4f} USDT`* · ⏱️ {dur}\n"
        f"_Paper Trading_"
    )


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def run_execution_scan():
    log.info(f"\n{'═'*56}")
    log.info(f"SCAN START — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    log.info(f"Method: Paper Trading (real market data, simulated fills)")
    log.info(f"{'═'*56}")

    run, mode, vol, reason = should_scan()
    check_mode_switch(mode)
    if not run:
        log.info(f"  Scan SKIPPED: {reason}")
        return

    effective_risk = get_effective_risk(mode, vol)
    vol_warning    = vol["message"] if vol.get("warn") else None

    log.info(f"  Mode:{mode['label']} conf≥{mode['min_confidence']}% "
             f"score≥{mode['min_score']} ADX≥{mode['min_adx']} risk:{effective_risk:.2f}")

    # Initialize paper trader (no API keys needed)
    ex = PaperTrader()
    ex.test_connection()

    pipeline   = load_model()
    thresholds = get_mode_thresholds(mode)

    # Step 1: Save balance snapshot (uses live prices for unrealised PnL)
    log.info(f"\n[0] Saving balance snapshot...")
    ex.save_balance_snapshot()

    # Step 2: Check open trades (SL/TP triggered by live prices)
    log.info(f"\n[1] Checking open trades...")
    check_open_trades(ex)

    # Step 3: Save updated balance after any closures
    ex.save_balance_snapshot()

    # Step 4: Scan for new signals
    trades = load_trades()
    log.info(f"\n[2] Scanning {len(SYMBOLS)} coins | Open:{len(trades)}/{MAX_OPEN_TRADES}")
    log.info(f"    conf≥{thresholds['min_confidence']}% | score≥{thresholds['min_score']} | ADX≥{thresholds['min_adx']} | risk:{effective_risk:.2f}")

    signals_found = 0
    for symbol in SYMBOLS:
        if len(load_trades()) >= MAX_OPEN_TRADES:
            log.info("  🛑 Max trades reached — scan complete")
            break

        log.info(f"\n  ── {symbol} ({get_tier(symbol)}) ──")
        sig = generate_signal(symbol, pipeline, thresholds)
        if sig is None:
            time.sleep(0.3)
            continue

        signals_found += 1
        log.info(f"  ✅ SIGNAL: {sig['signal']} {sig['confidence']:.1f}% score={sig['score']}")
        log.info(f"  Levels: entry={sig['entry']} SL={sig['stop']} TP1={sig['tp1']} TP2={sig['tp2']}")

        if vol_warning:
            sig["reasons"] = list(sig.get("reasons",[])) + [f"⚠️ {vol_warning}"]

        execute_trade(ex, sig["symbol"], sig["signal"], sig["entry"],
                      sig["atr"], sig["confidence"], sig["score"],
                      sig["reasons"], effective_risk)
        time.sleep(0.5)

    # Step 5: Final balance save
    ex.save_balance_snapshot()

    log.info(f"\n{'═'*56}")
    log.info(f"SCAN DONE — {signals_found} signal(s) found")
    log.info(f"{'═'*56}\n")


def run_diagnostic():
    from smart_scheduler import get_scan_mode, check_btc_volatility
    lines = ["🔍 *Bot Diagnostic — Paper Trading Mode*",
             f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
             "━━━━━━━━━━━━━━━━━━━━"]
    try:
        mode = get_scan_mode(); vol = check_btc_volatility()
        eff  = get_effective_risk(mode, vol)
        lines += [f"📋 Mode: *{mode['label']}*",
                  f"📊 BTC ATR: *{vol['atr_pct']:.2f}%*",
                  f"⚙️ Conf≥{mode['min_confidence']}% | Score≥{mode['min_score']}/6",
                  f"⚡ Risk: *{int(eff*100)}%*"]
    except Exception as e: lines.append(f"❌ Scheduler: {e}")

    ex  = PaperTrader()
    bal = ex.get_usdt_balance()
    lines += [f"💰 Paper balance: *{bal:.2f} USDT*",
              f"📂 Open trades: *{len(load_trades())}*",
              f"✅ No exchange connection needed — geo-block bypassed!"]

    try:
        p = load_model(); lines.append(f"🤖 Model: ✅ {len(p['all_features'])} features")
    except Exception as e: lines.append(f"❌ Model: {e}")

    real = [h for h in load_history() if h.get("signal") != "RECOVERED"]
    wins = [h for h in real if (h.get("pnl") or 0) > 0]
    wr   = round(len(wins)/len(real)*100,1) if real else 0
    tpnl = sum(h.get("pnl",0) for h in real)
    lines += [f"📈 Win rate: *{wr}%* ({len(wins)}W/{len(real)-len(wins)}L)",
              f"💵 Total PnL: *{tpnl:+.4f} USDT*"]
    _send("\n".join(lines))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "diagnostic":
        run_diagnostic()
    else:
        run_execution_scan()
