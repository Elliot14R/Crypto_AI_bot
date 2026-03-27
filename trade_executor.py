# trade_executor.py  (smart scheduler edition)
# Integrates active/quiet/weekend modes + BTC volatility check
# Sends Telegram alerts when mode switches

import os, json, time, logging, requests, joblib, pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

try:
    import ccxt
except ImportError:
    raise ImportError("Run: pip install ccxt")

from config import (
    SYMBOLS, FEATURES, ATR_STOP_MULT, ATR_TARGET1_MULT,
    ATR_TARGET2_MULT, RISK_PER_TRADE, TIMEFRAME_ENTRY,
    TIMEFRAME_CONFIRM, TIMEFRAME_TREND, LIVE_LIMIT,
    MODEL_FILE, LOG_FILE
)
from feature_engineering import add_indicators
from smart_scheduler import should_scan, get_mode_thresholds, check_btc_volatility

TRADES_FILE     = "trades.json"
HISTORY_FILE    = "trade_history.json"
SIGNALS_FILE    = "signals.json"         # ← new: for dashboard signals page
MODE_FILE       = "scan_mode.json"       # ← new: persist last mode for Telegram switch alert
MAX_OPEN_TRADES = 3
TP1_CLOSE_PCT   = 0.5
TP2_CLOSE_PCT   = 0.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
log = logging.getLogger(__name__)


# ════════════ HELPERS ════════════════════════════════════

def load_json(path, default):
    try:
        if Path(path).exists():
            with open(path) as f: return json.load(f)
    except Exception: pass
    return default

def save_json(path, data):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f: json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)

def load_trades():  return load_json(TRADES_FILE, {})
def save_trades(d): save_json(TRADES_FILE, d)
def load_history(): return load_json(HISTORY_FILE, [])
def load_signals(): return load_json(SIGNALS_FILE, [])

def append_history(rec):
    h = load_history(); h.append(rec); save_json(HISTORY_FILE, h)

def save_signal(sig):
    """Save every qualifying signal for the dashboard signals page."""
    sigs = load_signals()
    sigs.append({**sig, "generated_at": datetime.now(timezone.utc).isoformat()})
    sigs = sigs[-200:]   # keep last 200
    save_json(SIGNALS_FILE, sigs)


# ════════════ EXCHANGE ════════════════════════════════════

def init_exchange():
    key    = os.getenv("BINANCE_API_KEY","")
    secret = os.getenv("BINANCE_SECRET","")
    if not key or not secret:
        raise ValueError("Missing BINANCE_API_KEY or BINANCE_SECRET in .env")
    ex = ccxt.binance({"apiKey":key,"secret":secret,"options":{"defaultType":"spot"},"enableRateLimit":True})
    ex.set_sandbox_mode(True)
    log.info("Exchange: Binance TESTNET")
    return ex

def load_model():
    p = joblib.load(MODEL_FILE)
    log.info(f"Model loaded")
    return p

def get_balance_usdt(ex):
    try:
        b = ex.fetch_balance()
        return float(b["USDT"]["free"])
    except Exception as e:
        log.error(f"Balance: {e}"); return 0.0


# ════════════ MARKET DATA ══════════════════════════════════

def get_data(symbol, interval):
    url    = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": LIVE_LIMIT}
    resp   = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()).iloc[:, :6]
    df.columns = ["open_time","open","high","low","close","volume"]
    for c in ["open","high","low","close","volume"]: df[c] = pd.to_numeric(df[c])
    return df

def calc_pos_size(balance, entry, stop):
    risk  = balance * RISK_PER_TRADE
    dist  = abs(entry - stop)
    return round(risk / dist, 6) if dist > 0 else 0.0


# ════════════ EXECUTION ════════════════════════════════════

def execute_trade(ex, symbol, signal, entry, atr, confidence, score, reasons):
    trades = load_trades()
    if symbol in trades:
        log.info(f"  {symbol}: already open"); return False
    if len(trades) >= MAX_OPEN_TRADES:
        log.info(f"  Max trades reached"); return False

    dec = 4 if entry < 10 else 2
    if signal == "BUY":
        stop = round(entry - atr * ATR_STOP_MULT,    dec)
        tp1  = round(entry + atr * ATR_TARGET1_MULT, dec)
        tp2  = round(entry + atr * ATR_TARGET2_MULT, dec)
        side = "buy"
    else:
        stop = round(entry + atr * ATR_STOP_MULT,    dec)
        tp1  = round(entry - atr * ATR_TARGET1_MULT, dec)
        tp2  = round(entry - atr * ATR_TARGET2_MULT, dec)
        side = "sell"

    balance  = get_balance_usdt(ex)
    if balance < 10:
        _warn(f"⚠️ Low balance {balance:.2f} USDT — skip {symbol}"); return False

    qty      = calc_pos_size(balance, entry, stop)
    risk_usd = round(balance * RISK_PER_TRADE, 2)
    qty_tp1  = round(qty * TP1_CLOSE_PCT, 6)
    qty_tp2  = round(qty * TP2_CLOSE_PCT, 6)
    if qty <= 0: return False

    log.info(f"  {signal} {symbol} qty={qty} entry~{entry} SL={stop} TP1={tp1} TP2={tp2}")
    order_ids = {}

    try:
        eo = ex.create_order(symbol, "market", side, qty)
        order_ids["entry"] = eo["id"]
        actual_entry = float(eo.get("average", entry) or entry)
        time.sleep(1)

        sl_side = "sell" if signal == "BUY" else "buy"
        tp_side = sl_side

        for attempt in ["stop_loss_limit","limit"]:
            try:
                o = ex.create_order(symbol, attempt, sl_side, qty, stop,
                    params={"stopPrice":stop,"timeInForce":"GTC"})
                order_ids["stop_loss"] = o["id"]; break
            except Exception: pass

        for attempt in ["take_profit_limit","limit"]:
            try:
                o = ex.create_order(symbol, attempt, tp_side, qty_tp1, tp1,
                    params={"stopPrice":tp1,"timeInForce":"GTC"})
                order_ids["tp1"] = o["id"]; break
            except Exception: pass

        for attempt in ["take_profit_limit","limit"]:
            try:
                o = ex.create_order(symbol, attempt, tp_side, qty_tp2, tp2,
                    params={"stopPrice":tp2,"timeInForce":"GTC"})
                order_ids["tp2"] = o["id"]; break
            except Exception: pass

    except ccxt.InsufficientFunds: _warn(f"⚠️ Insufficient funds {symbol}"); return False
    except ccxt.NetworkError as e: _warn(f"⚠️ Network {symbol}: {e}"); return False
    except ccxt.ExchangeError as e: _warn(f"⚠️ Exchange {symbol}: {e}"); return False
    except Exception as e: _warn(f"⚠️ Error {symbol}: {e}"); return False

    record = {
        "symbol":symbol,"signal":signal,"entry":actual_entry,
        "stop":stop,"tp1":tp1,"tp2":tp2,"qty":qty,
        "qty_tp1":qty_tp1,"qty_tp2":qty_tp2,"risk_usd":risk_usd,
        "balance_at_open":balance,"order_ids":order_ids,
        "opened_at":datetime.now(timezone.utc).isoformat(),
        "tp1_hit":False,"tp2_hit":False,"closed":False,
        "confidence":confidence,"score":score,"reasons":reasons,
    }
    trades[symbol] = record
    save_trades(trades)
    save_signal({**record, "entry":actual_entry})
    _send_open_alert(symbol,signal,confidence,score,actual_entry,stop,tp1,tp2,qty,risk_usd,balance,reasons)
    log.info(f"  ✅ Trade opened: {symbol}")
    return True


# ════════════ MONITORING ═══════════════════════════════════

def check_open_trades(ex):
    trades = load_trades()
    if not trades: return
    to_close = []

    for symbol, trade in trades.items():
        if trade.get("closed"): to_close.append(symbol); continue
        try:
            oids, entry = trade.get("order_ids",{}), trade["entry"]

            if not trade["tp1_hit"] and "tp1" in oids:
                try:
                    o = ex.fetch_order(oids["tp1"], symbol)
                    if o["status"] == "closed":
                        trade["tp1_hit"] = True
                        pnl = _pnl(trade, o["average"], "tp1")
                        log.info(f"  TP1 HIT {symbol} {pnl:+.4f}")
                        _send_close_alert(symbol,"TP1 HIT 🎯",pnl,entry,o["average"],trade["opened_at"])
                except Exception as e: log.warning(f"  TP1 {symbol}: {e}")

            if trade["tp1_hit"] and not trade["tp2_hit"] and "tp2" in oids:
                try:
                    o = ex.fetch_order(oids["tp2"], symbol)
                    if o["status"] == "closed":
                        trade["tp2_hit"] = True; trade["closed"] = True
                        pnl = _pnl(trade, o["average"], "tp2")
                        log.info(f"  TP2 HIT {symbol} {pnl:+.4f}")
                        _send_close_alert(symbol,"✅ FULL WIN (TP2)",pnl,entry,o["average"],trade["opened_at"])
                        _record_close(trade, o["average"], pnl, "TP2 hit")
                        to_close.append(symbol)
                except Exception as e: log.warning(f"  TP2 {symbol}: {e}")

            if not trade.get("closed") and "stop_loss" in oids:
                try:
                    o = ex.fetch_order(oids["stop_loss"], symbol)
                    if o["status"] == "closed":
                        trade["closed"] = True
                        pnl = _pnl(trade, o["average"], "sl")
                        log.info(f"  SL HIT {symbol} {pnl:+.4f}")
                        _send_close_alert(symbol,"❌ STOPPED OUT",pnl,entry,o["average"],trade["opened_at"])
                        _record_close(trade, o["average"], pnl, "SL hit")
                        _cancel_remaining(ex, symbol, oids, trade)
                        to_close.append(symbol)
                except Exception as e: log.warning(f"  SL {symbol}: {e}")

        except Exception as e: log.error(f"  Monitor {symbol}: {e}")

    save_trades(trades)
    for sym in set(to_close):
        if trades.get(sym,{}).get("closed"): trades.pop(sym,None)
    save_trades(trades)


def _pnl(trade, close_price, t):
    entry = trade["entry"]
    qty   = trade["qty_tp1"] if t=="tp1" else trade["qty_tp2"] if t=="tp2" else trade["qty"]
    return round((close_price-entry)*qty if trade["signal"]=="BUY" else (entry-close_price)*qty, 4)

def _cancel_remaining(ex, symbol, oids, trade):
    for key in ("tp1","tp2"):
        if key in oids and not trade.get(f"{key}_hit"):
            try: ex.cancel_order(oids[key], symbol)
            except Exception: pass

def _record_close(trade, close_price, pnl, reason):
    append_history({**trade,"close_price":close_price,"pnl":pnl,
                    "closed_at":datetime.now(timezone.utc).isoformat(),"close_reason":reason})


# ════════════ SIGNAL GENERATION ════════════════════════════

def generate_signal(symbol, pipeline, thresholds):
    try:
        df_entry   = add_indicators(get_data(symbol, TIMEFRAME_ENTRY))
        df_confirm = add_indicators(get_data(symbol, TIMEFRAME_CONFIRM))
        df_trend   = add_indicators(get_data(symbol, TIMEFRAME_TREND))
        if df_entry.empty or len(df_entry) < 50: return None

        row_entry   = df_entry.iloc[-1]
        row_confirm = df_confirm.iloc[-1] if not df_confirm.empty else pd.Series(dtype=float)

        all_feat = pipeline["all_features"]
        selector = pipeline["selector"]
        ensemble = pipeline["ensemble"]
        missing  = [f for f in all_feat if f not in df_entry.columns]
        if missing: return None

        X_raw = pd.DataFrame([row_entry[all_feat].values], columns=all_feat)
        X_sel = selector.transform(X_raw)
        pred  = ensemble.predict(X_sel)[0]
        prob  = ensemble.predict_proba(X_sel)[0]
        label = {0:"BUY",1:"SELL",2:"NO_TRADE"}
        signal     = label[pred]
        confidence = round(float(max(prob))*100, 1)

        # Use mode-aware thresholds
        if signal == "NO_TRADE" or confidence < thresholds["min_confidence"]: return None
        adx_val = float(row_entry.get("adx", 0))
        if adx_val < thresholds["min_adx"]: return None

        score, reasons = _quality_score(row_entry, row_confirm, signal, confidence)
        if score < thresholds["min_score"]: return None

        return {
            "symbol":symbol,"signal":signal,"confidence":confidence,
            "score":score,"entry":float(row_entry["close"]),
            "atr":float(row_entry["atr"]),"reasons":reasons,
            "stop": round(float(row_entry["close"]) - float(row_entry["atr"]) * ATR_STOP_MULT, 4),
            "tp1":  round(float(row_entry["close"]) + float(row_entry["atr"]) * ATR_TARGET1_MULT, 4),
            "tp2":  round(float(row_entry["close"]) + float(row_entry["atr"]) * ATR_TARGET2_MULT, 4),
        }
    except Exception as e:
        log.error(f"  Signal {symbol}: {e}"); return None


def _quality_score(row_entry, row_confirm, signal, confidence):
    score, reasons = 0, []
    if confidence >= 75:
        score+=1; reasons.append(f"High AI confidence ({confidence:.0f}%)")
    elif confidence >= 65:
        reasons.append(f"AI confidence ({confidence:.0f}%)")
    adx = row_entry.get("adx",0)
    if adx > 25:   score+=1; reasons.append(f"Strong trend ADX {adx:.0f}")
    elif adx > 20: score+=1; reasons.append(f"Moderate trend ADX {adx:.0f}")
    rsi = row_entry.get("rsi",50)
    if signal=="BUY"  and rsi<40: score+=1; reasons.append(f"RSI oversold {rsi:.0f}")
    if signal=="SELL" and rsi>60: score+=1; reasons.append(f"RSI overbought {rsi:.0f}")
    e20,e50,e200=row_entry.get("ema20",0),row_entry.get("ema50",0),row_entry.get("ema200",0)
    if signal=="BUY"  and e20>e50>e200: score+=1; reasons.append("EMA uptrend")
    if signal=="SELL" and e20<e50<e200: score+=1; reasons.append("EMA downtrend")
    if signal=="BUY"  and row_confirm.get("ema20",0)>row_confirm.get("ema50",0):
        score+=1; reasons.append("1h EMA confirms uptrend")
    if signal=="SELL" and row_confirm.get("ema20",0)<row_confirm.get("ema50",0):
        score+=1; reasons.append("1h EMA confirms downtrend")
    return score, reasons


# ════════════ TELEGRAM ════════════════════════════════════

def _send(text):
    token   = os.getenv("TELEGRAM_TOKEN","")
    chat_id = os.getenv("TELEGRAM_CHAT_ID","")
    if not token or not chat_id: return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id":chat_id,"text":text,"parse_mode":"Markdown"}, timeout=10)
    except Exception as e: log.warning(f"Telegram: {e}")

def _warn(text):
    log.warning(text); _send(text)

def send_mode_switch_alert(mode: dict):
    """Send Telegram when switching between active/quiet/weekend."""
    if mode["mode"] == "active":
        _send("📈 *Active trading hours started*\nScanning every 15 min · Min conf 65% · Full signals")
    elif mode["mode"] == "quiet":
        _send("🌙 *Quiet hours started*\nScanning every 30 min · Min conf 72% · ADX 30+ only")
    elif mode["mode"] == "weekend":
        _send("📅 *Weekend mode*\nConf raised to 70% · Lower volume · Trade carefully")

def check_mode_switch(mode: dict):
    """Detect if mode has changed since last run and alert if so."""
    last = load_json(MODE_FILE, {})
    if last.get("mode") != mode["mode"]:
        log.info(f"  Mode switch: {last.get('mode','?')} → {mode['mode']}")
        send_mode_switch_alert(mode)
        save_json(MODE_FILE, {"mode": mode["mode"], "since": datetime.now(timezone.utc).isoformat()})

def _send_open_alert(symbol,signal,confidence,score,entry,stop,tp1,tp2,qty,risk_usd,balance,reasons):
    emoji = "🟢" if signal=="BUY" else "🔴"
    stars = "⭐"*min(score,5)
    dec   = 4 if entry<10 else 2
    fp    = lambda v: f"{v:,.{dec}f}"
    sl_pct= abs((stop-entry)/entry*100)
    t1_pct= abs((tp1 -entry)/entry*100)
    t2_pct= abs((tp2 -entry)/entry*100)
    rlines= "\n".join([f"  - {r}" for r in reasons])
    _send(
        f"🤖 *LIVE TEST TRADE OPENED*\n━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{emoji} *{signal}  —  {symbol}* {stars}\n"
        f"🎯 Confidence: *{confidence:.1f}%*\n\n"
        f"⚡ *ENTRY:*     `{fp(entry)}`\n"
        f"🛑 *STOP LOSS:* `{fp(stop)}`  (-{sl_pct:.1f}%)\n"
        f"🎯 *TARGET 1:*  `{fp(tp1)}`  (+{t1_pct:.1f}%)\n"
        f"🎯 *TARGET 2:*  `{fp(tp2)}`  (+{t2_pct:.1f}%)\n\n"
        f"💰 *Position:*  `{round(qty*entry,2):.2f} USDT`\n"
        f"⚠️  *Risk:*      `{risk_usd:.2f} USDT` (1%)\n"
        f"💼 *Balance:*   `{balance:.2f} USDT`\n\n"
        f"📊 *Why:*\n{rlines}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n_Binance Testnet — paper trading only_"
    )

def _send_close_alert(symbol,result,pnl,entry,close_price,opened_at):
    emoji = "✅" if pnl>0 else "❌"
    dec   = 4 if entry<10 else 2
    try:
        dur = str(datetime.now(timezone.utc)-datetime.fromisoformat(opened_at)).split(".")[0]
    except Exception: dur="—"
    _send(
        f"🤖 *TRADE CLOSED*\n━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{emoji} *{result}  —  {symbol}*\n\n"
        f"📥 Entry: `{entry:.{dec}f}`\n"
        f"📤 Close: `{close_price:.{dec}f}`\n"
        f"💵 *PnL: `{pnl:+.4f} USDT`*\n"
        f"⏱️ Duration: {dur}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n_Binance Testnet_"
    )


# ════════════ MAIN ═════════════════════════════════════════

def run_execution_scan():
    log.info(f"\n{'═'*52}")
    log.info(f"Scan — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    log.info(f"{'═'*52}")

    # Smart scheduler decision
    run, mode, vol, reason = should_scan()
    log.info(f"  {mode['emoji']} {mode['label']} — {reason}")

    # Alert if mode switched
    check_mode_switch(mode)

    if not run:
        log.info(f"  Scan SKIPPED: {reason}")
        _warn(f"⏭️ Scan skipped — {reason}") if vol.get("skip") else None
        return

    # Volatility warning in every signal if high
    vol_warning = vol["message"] if vol.get("warn") else None

    exchange   = init_exchange()
    pipeline   = load_model()
    thresholds = get_mode_thresholds(mode)

    log.info(f"  Thresholds: conf≥{thresholds['min_confidence']}% "
             f"score≥{thresholds['min_score']} ADX≥{thresholds['min_adx']}")

    log.info("\n[1/2] Checking open trades...")
    check_open_trades(exchange)

    trades = load_trades()
    log.info(f"\n[2/2] Scanning {len(SYMBOLS)} symbols | Open: {len(trades)}/{MAX_OPEN_TRADES}")

    for symbol in SYMBOLS:
        log.info(f"  {symbol}...")
        sig = generate_signal(symbol, pipeline, thresholds)
        if not sig:
            log.info(f"    No signal"); time.sleep(0.5); continue

        log.info(f"    SIGNAL: {sig['signal']} {sig['confidence']}% score={sig['score']}")

        # Append vol warning to reasons if needed
        if vol_warning:
            sig["reasons"] = list(sig.get("reasons",[])) + [vol_warning]

        execute_trade(
            exchange, sig["symbol"], sig["signal"], sig["entry"],
            sig["atr"], sig["confidence"], sig["score"], sig["reasons"]
        )
        time.sleep(1)

    log.info("Scan complete.")


if __name__ == "__main__":
    run_execution_scan()
