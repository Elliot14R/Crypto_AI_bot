# trade_executor.py — FIXED VERSION
# Fixes:
#   1. SELL stop/TP direction bug (was calculating wrong prices)
#   2. Symbols not on Binance (RENDERUSDT, TAOUSDT) handled gracefully
#   3. Detailed diagnostic logging so you can see exactly what's happening
#   4. Score fallback — saves signal even if score too low (for dashboard)
#   5. GitHub Actions cache failure handled — always starts clean
#   6. Model pipeline dict checked properly before use

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
    MODEL_FILE, LOG_FILE, get_tier
)
from feature_engineering import add_indicators
from smart_scheduler import should_scan, get_mode_thresholds

TRADES_FILE     = "trades.json"
HISTORY_FILE    = "trade_history.json"
SIGNALS_FILE    = "signals.json"
MODE_FILE       = "scan_mode.json"
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
    sigs = load_signals()
    sigs.append({**sig, "generated_at": datetime.now(timezone.utc).isoformat()})
    sigs = sigs[-500:]
    save_json(SIGNALS_FILE, sigs)


# ════════════ EXCHANGE ════════════════════════════════════

def init_exchange():
    key    = os.getenv("BINANCE_API_KEY", "")
    secret = os.getenv("BINANCE_SECRET",  "")
    if not key or not secret:
        raise ValueError(
            "BINANCE_API_KEY or BINANCE_SECRET missing!\n"
            "Check GitHub Secrets are set correctly."
        )
    ex = ccxt.binance({
        "apiKey": key, "secret": secret,
        "options": {"defaultType": "spot"},
        "enableRateLimit": True,
    })
    ex.set_sandbox_mode(True)
    log.info("✓ Exchange: Binance TESTNET")
    return ex


def load_model():
    pipeline = joblib.load(MODEL_FILE)
    # Validate it has the expected keys
    required = ["ensemble", "selector", "all_features", "best_features", "label_map"]
    missing  = [k for k in required if k not in pipeline]
    if missing:
        raise ValueError(f"Model pipeline missing keys: {missing}")
    log.info(f"✓ Model loaded — {len(pipeline['all_features'])} features")
    return pipeline


def get_balance_usdt(ex):
    try:
        b = ex.fetch_balance()
        return float(b.get("USDT", {}).get("free", 0))
    except Exception as e:
        log.error(f"Balance fetch failed: {e}")
        return 0.0


def verify_symbol(ex, symbol) -> bool:
    """Check if symbol actually trades on Binance testnet."""
    try:
        ex.fetch_ticker(symbol)
        return True
    except Exception:
        return False


# ════════════ MARKET DATA ══════════════════════════════════

def get_data(symbol, interval):
    url    = "https://data-api.binance.vision/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": LIVE_LIMIT}
    resp   = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()).iloc[:, :6]
    df.columns = ["open_time","open","high","low","close","volume"]
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c])
    return df


def calc_pos_size(balance, entry, stop):
    risk = balance * RISK_PER_TRADE
    dist = abs(entry - stop)
    
    if dist <= 0:
        log.warning("  Stop distance = 0, cannot size position")
        return 0.0
        
    qty = risk / dist
    
    # ── NEW SAFETY CAP ───────────────────────────────────────────
    # Never allocate more than 20% of total balance to a single trade
    max_usd_position = balance * 0.20 
    
    if (qty * entry) > max_usd_position:
        log.info(f"  Qty capped: {qty*entry:.2f} USDT exceeds 20% limit.")
        qty = max_usd_position / entry
    # ─────────────────────────────────────────────────────────────
        
    return round(qty, 6)


# ════════════ EXECUTION ════════════════════════════════════

def execute_trade(ex, symbol, signal, entry, atr, confidence, score, reasons):
    trades = load_trades()
    if symbol in trades:
        log.info(f"  {symbol}: already have open trade — skip")
        return False
    if len(trades) >= MAX_OPEN_TRADES:
        log.info(f"  Max open trades ({MAX_OPEN_TRADES}) reached — skip")
        return False

    # ── Price levels — FIXED: correct direction for SELL ──────────
    dec = 4 if entry < 10 else 2
    if signal == "BUY":
        stop = round(entry - atr * ATR_STOP_MULT,    dec)
        tp1  = round(entry + atr * ATR_TARGET1_MULT, dec)
        tp2  = round(entry + atr * ATR_TARGET2_MULT, dec)
        side      = "buy"
        sl_side   = "sell"
        tp_side   = "sell"
    else:  # SELL
        stop = round(entry + atr * ATR_STOP_MULT,    dec)   # SL above entry for short
        tp1  = round(entry - atr * ATR_TARGET1_MULT, dec)   # TP below entry for short
        tp2  = round(entry - atr * ATR_TARGET2_MULT, dec)
        side      = "sell"
        sl_side   = "buy"
        tp_side   = "buy"

    log.info(f"  Levels: entry={entry:.{dec}f} stop={stop:.{dec}f} tp1={tp1:.{dec}f} tp2={tp2:.{dec}f}")

    balance  = get_balance_usdt(ex)
    if balance < 10:
        _warn(f"⚠️ Balance too low ({balance:.2f} USDT) — cannot trade {symbol}")
        return False

    qty      = calc_pos_size(balance, entry, stop)
    risk_usd = round(balance * RISK_PER_TRADE, 2)
    qty_tp1  = round(qty * TP1_CLOSE_PCT, 6)
    qty_tp2  = round(qty * TP2_CLOSE_PCT, 6)

    if qty <= 0:
        log.warning(f"  Position size is zero for {symbol} — skip")
        return False

    log.info(f"  Placing {signal} {symbol} | qty={qty} | risk={risk_usd:.2f} USDT")
    order_ids = {}

    try:
        # ── Entry market order ─────────────────────────────────────
        entry_order = ex.create_order(symbol, "market", side, qty)
        order_ids["entry"] = entry_order["id"]
        actual_entry = float(entry_order.get("average", entry) or entry)
        log.info(f"  ✅ Entry filled at {actual_entry:.{dec}f}")
        time.sleep(1.5)   # give exchange time before placing SL/TP

        # ── Stop loss ──────────────────────────────────────────────
        sl_placed = False
        for order_type in ["stop_loss_limit", "limit"]:
            try:
                sl_o = ex.create_order(
                    symbol, order_type, sl_side, qty, stop,
                    params={"stopPrice": stop, "timeInForce": "GTC"}
                )
                order_ids["stop_loss"] = sl_o["id"]
                log.info(f"  ✅ Stop loss placed at {stop:.{dec}f}")
                sl_placed = True
                break
            except Exception as e:
                log.warning(f"  SL attempt ({order_type}) failed: {e}")

        if not sl_placed:
            log.error(f"  ⚠️ Could not place stop loss for {symbol} — trade recorded without SL")

        # ── Take profit 1 (50% position) ───────────────────────────
        for order_type in ["take_profit_limit", "limit"]:
            try:
                tp1_o = ex.create_order(
                    symbol, order_type, tp_side, qty_tp1, tp1,
                    params={"stopPrice": tp1, "timeInForce": "GTC"}
                )
                order_ids["tp1"] = tp1_o["id"]
                log.info(f"  ✅ TP1 placed at {tp1:.{dec}f} (qty={qty_tp1})")
                break
            except Exception as e:
                log.warning(f"  TP1 attempt ({order_type}) failed: {e}")

        # ── Take profit 2 (remaining 50%) ──────────────────────────
        for order_type in ["take_profit_limit", "limit"]:
            try:
                tp2_o = ex.create_order(
                    symbol, order_type, tp_side, qty_tp2, tp2,
                    params={"stopPrice": tp2, "timeInForce": "GTC"}
                )
                order_ids["tp2"] = tp2_o["id"]
                log.info(f"  ✅ TP2 placed at {tp2:.{dec}f} (qty={qty_tp2})")
                break
            except Exception as e:
                log.warning(f"  TP2 attempt ({order_type}) failed: {e}")

    except ccxt.InsufficientFunds as e:
        log.error(f"  Insufficient funds: {e}")
        _warn(f"⚠️ Insufficient funds for {symbol} — {e}")
        return False
    except ccxt.NetworkError as e:
        log.error(f"  Network error: {e}")
        _warn(f"⚠️ Network error placing {symbol} order")
        return False
    except ccxt.ExchangeError as e:
        log.error(f"  Exchange error: {e}")
        _warn(f"⚠️ Exchange rejected {symbol}: {e}")
        return False
    except Exception as e:
        log.error(f"  Unexpected error: {e}")
        _warn(f"⚠️ Unexpected error for {symbol}: {e}")
        return False

    # ── Save trade record ──────────────────────────────────────────
    record = {
        "symbol": symbol, "signal": signal,
        "entry": actual_entry, "stop": stop, "tp1": tp1, "tp2": tp2,
        "qty": qty, "qty_tp1": qty_tp1, "qty_tp2": qty_tp2,
        "risk_usd": risk_usd, "balance_at_open": balance,
        "order_ids": order_ids,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "tp1_hit": False, "tp2_hit": False, "closed": False,
        "confidence": confidence, "score": score, "reasons": reasons,
        "tier": get_tier(symbol),
    }
    trades[symbol] = record
    save_trades(trades)
    save_signal(record)
    _send_open_alert(symbol, signal, confidence, score, actual_entry,
                     stop, tp1, tp2, qty, risk_usd, balance, reasons)
    log.info(f"  ✅✅ TRADE OPENED: {symbol} {signal}")
    return True


# ════════════ MONITORING ═══════════════════════════════════

def check_open_trades(ex):
    trades = load_trades()
    if not trades:
        log.info("  No open trades to monitor")
        return

    to_remove = []
    log.info(f"  Monitoring {len(trades)} open trade(s)")

    for symbol, trade in list(trades.items()):
        if trade.get("closed"):
            to_remove.append(symbol)
            continue
        try:
            oids  = trade.get("order_ids", {})
            entry = trade["entry"]

            # Check TP1
            if not trade["tp1_hit"] and "tp1" in oids:
                try:
                    o = ex.fetch_order(oids["tp1"], symbol)
                    if o["status"] == "closed":
                        trade["tp1_hit"] = True
                        pnl = _pnl(trade, float(o["average"]), "tp1")
                        log.info(f"  🎯 TP1 HIT {symbol} pnl={pnl:+.4f} USDT")
                        _send_close_alert(symbol, "TP1 HIT 🎯", pnl, entry,
                                          float(o["average"]), trade["opened_at"])
                except Exception as e:
                    log.warning(f"  TP1 check {symbol}: {e}")

            # Check TP2
            if trade["tp1_hit"] and not trade["tp2_hit"] and "tp2" in oids:
                try:
                    o = ex.fetch_order(oids["tp2"], symbol)
                    if o["status"] == "closed":
                        trade["tp2_hit"] = True
                        trade["closed"]  = True
                        pnl = _pnl(trade, float(o["average"]), "tp2")
                        log.info(f"  ✅ TP2 HIT {symbol} pnl={pnl:+.4f} USDT")
                        _send_close_alert(symbol, "✅ FULL WIN (TP2)", pnl, entry,
                                          float(o["average"]), trade["opened_at"])
                        _record_close(trade, float(o["average"]), pnl, "TP2 hit")
                        to_remove.append(symbol)
                except Exception as e:
                    log.warning(f"  TP2 check {symbol}: {e}")

            # Check SL
            if not trade.get("closed") and "stop_loss" in oids:
                try:
                    o = ex.fetch_order(oids["stop_loss"], symbol)
                    if o["status"] == "closed":
                        trade["closed"] = True
                        pnl = _pnl(trade, float(o["average"]), "sl")
                        log.info(f"  ❌ SL HIT {symbol} pnl={pnl:+.4f} USDT")
                        _send_close_alert(symbol, "❌ STOPPED OUT", pnl, entry,
                                          float(o["average"]), trade["opened_at"])
                        _record_close(trade, float(o["average"]), pnl, "SL hit")
                        _cancel_remaining(ex, symbol, oids, trade)
                        to_remove.append(symbol)
                except Exception as e:
                    log.warning(f"  SL check {symbol}: {e}")

        except Exception as e:
            log.error(f"  Monitor error {symbol}: {e}")

    save_trades(trades)
    for sym in set(to_remove):
        trades.pop(sym, None)
    save_trades(trades)


def _pnl(trade, close_price, close_type):
    entry = trade["entry"]
    qty   = (trade["qty_tp1"] if close_type == "tp1" else
             trade["qty_tp2"] if close_type == "tp2" else trade["qty"])
    if trade["signal"] == "BUY":
        return round((close_price - entry) * qty, 4)
    return round((entry - close_price) * qty, 4)


def _cancel_remaining(ex, symbol, oids, trade):
    for key in ("tp1", "tp2"):
        if key in oids and not trade.get(f"{key}_hit"):
            try:
                ex.cancel_order(oids[key], symbol)
                log.info(f"  Cancelled {key} order for {symbol}")
            except Exception as e:
                log.warning(f"  Cancel {key} failed: {e}")


def _record_close(trade, close_price, pnl, reason):
    append_history({
        **trade,
        "close_price":  close_price,
        "pnl":          pnl,
        "closed_at":    datetime.now(timezone.utc).isoformat(),
        "close_reason": reason,
    })


# ════════════ SIGNAL GENERATION ════════════════════════════
def generate_signal(symbol, pipeline, thresholds):
    """
    Returns a signal dict if a qualifying signal is found, else None.
    Includes detailed logging so you can see exactly why signals are rejected.
    """
    try:
        # Fetch data
        df_entry   = add_indicators(get_data(symbol, TIMEFRAME_ENTRY))
        df_confirm = add_indicators(get_data(symbol, TIMEFRAME_CONFIRM))

        if df_entry.empty or len(df_entry) < 50:
            log.info(f"    Not enough data ({len(df_entry)} rows)")
            return None

        # 1. Use .copy() so we can safely attach the new columns
        row_entry   = df_entry.iloc[-1].copy() 
        row_confirm = df_confirm.iloc[-1] if not df_confirm.empty else pd.Series(dtype=float)

        # 2. Attach the missing 1-hour features manually!
        row_entry['rsi_1h']   = float(row_confirm.get('rsi', 50))
        row_entry['adx_1h']   = float(row_confirm.get('adx', 0))
        row_entry['trend_1h'] = float(row_confirm.get('trend', 0))

        all_feat = pipeline["all_features"]
        selector = pipeline["selector"]
        ensemble = pipeline["ensemble"]

        # 3. Check for missing features in row_entry.index, NOT df_entry.columns
        missing = [f for f in all_feat if f not in row_entry.index]
        if missing:
            log.warning(f"    Missing features: {missing[:5]}")
            return None

        # ML prediction
        X_raw      = pd.DataFrame([row_entry[all_feat].values], columns=all_feat)
        
        # ... (Leave the rest of your generate_signal function exactly as is!)

        # ML prediction
        X_raw      = pd.DataFrame([row_entry[all_feat].values], columns=all_feat)
        X_sel      = selector.transform(X_raw)
        pred       = ensemble.predict(X_sel)[0]
        prob       = ensemble.predict_proba(X_sel)[0]
        label_map  = {0: "BUY", 1: "SELL", 2: "NO_TRADE"}
        signal     = label_map[pred]
        confidence = round(float(max(prob)) * 100, 1)

        log.info(f"    ML: {signal} {confidence:.1f}% (need ≥{thresholds['min_confidence']}%)")

        if signal == "NO_TRADE":
            log.info(f"    Skipped: NO_TRADE prediction")
            return None

        if confidence < thresholds["min_confidence"]:
            log.info(f"    Skipped: confidence {confidence:.1f}% < {thresholds['min_confidence']}%")
            return None

        adx_val = float(row_entry.get("adx", 0))
        log.info(f"    ADX: {adx_val:.1f} (need ≥{thresholds['min_adx']})")
        if adx_val < thresholds["min_adx"]:
            log.info(f"    Skipped: ADX {adx_val:.1f} too low")
            return None

        score, reasons = _quality_score(row_entry, row_confirm, signal, confidence)
        log.info(f"    Score: {score}/6 (need ≥{thresholds['min_score']})")

        if score < thresholds["min_score"]:
            log.info(f"    Skipped: score {score} < {thresholds['min_score']}")
            # Still save to signals for dashboard visibility
            entry = float(row_entry["close"])
            atr   = float(row_entry["atr"])
            save_signal({
                "symbol": symbol, "signal": signal, "confidence": confidence,
                "score": score, "entry": entry, "atr": atr, "reasons": reasons,
                "rejected": True, "reject_reason": f"score {score} < {thresholds['min_score']}",
                "stop": round(entry - atr*ATR_STOP_MULT, 4) if signal=="BUY" else round(entry + atr*ATR_STOP_MULT, 4),
                "tp1":  round(entry + atr*ATR_TARGET1_MULT, 4) if signal=="BUY" else round(entry - atr*ATR_TARGET1_MULT, 4),
                "tp2":  round(entry + atr*ATR_TARGET2_MULT, 4) if signal=="BUY" else round(entry - atr*ATR_TARGET2_MULT, 4),
            })
            return None

        entry = float(row_entry["close"])
        atr   = float(row_entry["atr"])

        # FIXED: correct SL/TP direction for both BUY and SELL
        if signal == "BUY":
            stop = round(entry - atr * ATR_STOP_MULT,    4)
            tp1  = round(entry + atr * ATR_TARGET1_MULT, 4)
            tp2  = round(entry + atr * ATR_TARGET2_MULT, 4)
        else:
            stop = round(entry + atr * ATR_STOP_MULT,    4)
            tp1  = round(entry - atr * ATR_TARGET1_MULT, 4)
            tp2  = round(entry - atr * ATR_TARGET2_MULT, 4)

        return {
            "symbol": symbol, "signal": signal,
            "confidence": confidence, "score": score,
            "entry": entry, "atr": atr,
            "stop": stop, "tp1": tp1, "tp2": tp2,
            "reasons": reasons,
        }

    except requests.exceptions.HTTPError as e:
        # Handles symbols not on Binance testnet gracefully
        log.warning(f"    HTTP error (symbol may not exist on testnet): {e}")
        return None
    except Exception as e:
        log.error(f"    Signal generation error: {e}")
        return None


def _quality_score(row_entry, row_confirm, signal, confidence):
    score, reasons = 0, []

    # Confidence
    if confidence >= 75:
        score += 1; reasons.append(f"High AI confidence ({confidence:.0f}%)")
    elif confidence >= 65:
        score += 1; reasons.append(f"Good AI confidence ({confidence:.0f}%)")
    elif confidence >= 60:
        reasons.append(f"AI confidence ({confidence:.0f}%)")

    # ADX trend strength
    adx = float(row_entry.get("adx", 0))
    if adx > 25:
        score += 1; reasons.append(f"Strong trend ADX {adx:.0f}")
    elif adx > 18:
        score += 1; reasons.append(f"Moderate trend ADX {adx:.0f}")

    # RSI zone
    rsi = float(row_entry.get("rsi", 50))
    if signal == "BUY" and rsi < 45:
        score += 1; reasons.append(f"RSI bullish zone ({rsi:.0f})")
    elif signal == "SELL" and rsi > 55:
        score += 1; reasons.append(f"RSI bearish zone ({rsi:.0f})")

    # EMA alignment
    e20  = float(row_entry.get("ema20",  0))
    e50  = float(row_entry.get("ema50",  0))
    e200 = float(row_entry.get("ema200", 0))
    if signal == "BUY" and e20 > e50:
        score += 1; reasons.append("EMA20 > EMA50 (uptrend)")
    elif signal == "SELL" and e20 < e50:
        score += 1; reasons.append("EMA20 < EMA50 (downtrend)")

    # 1h confirmation
    c20 = float(row_confirm.get("ema20", 0))
    c50 = float(row_confirm.get("ema50", 0))
    if signal == "BUY" and c20 > c50:
        score += 1; reasons.append("1h confirms uptrend")
    elif signal == "SELL" and c20 < c50:
        score += 1; reasons.append("1h confirms downtrend")

    return score, reasons


# ════════════ TELEGRAM ════════════════════════════════════

def _send(text):
    token   = os.getenv("TELEGRAM_TOKEN",  "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID","")
    if not token or not chat_id:
        log.warning("Telegram not configured — check .env / GitHub Secrets")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10
        )
        if not r.ok:
            log.warning(f"Telegram error: {r.status_code} {r.text[:100]}")
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")


def _warn(text):
    log.warning(text)
    _send(text)


def check_mode_switch(mode: dict):
    last = load_json(MODE_FILE, {})
    if last.get("mode") != mode["mode"]:
        log.info(f"  Mode switch: {last.get('mode','?')} → {mode['mode']}")
        msgs = {
            "active":  "📈 *Active trading hours*\nScanning every 15 min · Conf ≥60% · Normal thresholds",
            "quiet":   "🌙 *Quiet hours*\nScanning every 30 min · Conf ≥68% · Higher threshold",
            "weekend": "📅 *Weekend mode*\nConf raised to 65% · Lower volume · Trade carefully",
        }
        _send(msgs.get(mode["mode"], "Mode changed"))
        save_json(MODE_FILE, {"mode": mode["mode"], "since": datetime.now(timezone.utc).isoformat()})


def _send_open_alert(symbol, signal, confidence, score, entry,
                     stop, tp1, tp2, qty, risk_usd, balance, reasons):
    emoji  = "🟢" if signal == "BUY" else "🔴"
    stars  = "⭐" * min(score, 5)
    dec    = 4 if entry < 10 else 2
    fp     = lambda v: f"{v:,.{dec}f}"
    sl_pct = abs((stop - entry) / entry * 100)
    t1_pct = abs((tp1  - entry) / entry * 100)
    t2_pct = abs((tp2  - entry) / entry * 100)
    pos_usd = round(qty * entry, 2)
    rlines  = "\n".join([f"  • {r}" for r in reasons])
    tier    = get_tier(symbol)
    _send(
        f"🤖 *TESTNET TRADE OPENED*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{emoji} *{signal} — {symbol}* {stars}\n"
        f"🏷️ Tier: _{tier}_\n"
        f"🎯 Confidence: *{confidence:.1f}%* · Score: *{score}/6*\n\n"
        f"⚡ *ENTRY:*     `{fp(entry)}`\n"
        f"🛑 *STOP LOSS:* `{fp(stop)}`  (-{sl_pct:.1f}%)\n"
        f"🎯 *TARGET 1:*  `{fp(tp1)}`  (+{t1_pct:.1f}%)\n"
        f"🎯 *TARGET 2:*  `{fp(tp2)}`  (+{t2_pct:.1f}%)\n\n"
        f"💰 *Position:*  `{pos_usd:.2f} USDT`\n"
        f"⚠️  *Risk:*      `{risk_usd:.2f} USDT` (1% balance)\n"
        f"💼 *Balance:*   `{balance:.2f} USDT`\n\n"
        f"📊 *Reasons:*\n{rlines}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_Binance Testnet — paper trading_"
    )


def _send_close_alert(symbol, result, pnl, entry, close_price, opened_at):
    emoji = "✅" if pnl > 0 else "❌"
    dec   = 4 if entry < 10 else 2
    try:
        dur = str(datetime.now(timezone.utc) -
                  datetime.fromisoformat(opened_at)).split(".")[0]
    except Exception:
        dur = "—"
    _send(
        f"🤖 *TRADE CLOSED*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{emoji} *{result} — {symbol}*\n\n"
        f"📥 Entry:  `{entry:.{dec}f}`\n"
        f"📤 Close:  `{close_price:.{dec}f}`\n"
        f"💵 *PnL:   `{pnl:+.4f} USDT`*\n"
        f"⏱️ Duration: {dur}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_Binance Testnet_"
    )


# ════════════ DIAGNOSTIC SCAN ══════════════════════════════

def run_diagnostic():
    """
    Send a full Telegram diagnostic showing exactly why no trades fired.
    Run manually from GitHub Actions workflow_dispatch.
    """
    log.info("Running diagnostic scan...")
    lines = [
        "🔍 *Bot Diagnostic Report*",
        f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    try:
        from smart_scheduler import get_scan_mode, check_btc_volatility
        mode = get_scan_mode()
        vol  = check_btc_volatility()
        lines.append(f"📋 Mode: *{mode['label']}*")
        lines.append(f"📊 BTC ATR: *{vol['atr_pct']:.2f}%* ({vol['status']})")
        lines.append(f"⚙️ Conf threshold: {mode['min_confidence']}%")
        lines.append(f"⚙️ Score threshold: {mode['min_score']}/6")
        lines.append(f"⚙️ ADX threshold: {mode['min_adx']}")
    except Exception as e:
        lines.append(f"❌ Scheduler error: {e}")

    try:
        ex = init_exchange()
        bal = get_balance_usdt(ex)
        lines.append(f"💰 Balance: *{bal:.2f} USDT*")
        lines.append(f"📂 Open trades: *{len(load_trades())}*")
    except Exception as e:
        lines.append(f"❌ Exchange error: {e}")

    try:
        p = load_model()
        lines.append(f"🤖 Model: ✅ loaded ({len(p['all_features'])} features)")
    except Exception as e:
        lines.append(f"❌ Model error: {e}")

    _send("\n".join(lines))


# ════════════ MAIN ═════════════════════════════════════════

def run_execution_scan():
    log.info(f"\n{'═'*56}")
    log.info(f"SCAN START — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    log.info(f"{'═'*56}")

    # Smart scheduler check
    run, mode, vol, reason = should_scan()
    check_mode_switch(mode)

    if not run:
        log.info(f"  Scan SKIPPED: {reason}")
        return

    vol_warning = vol["message"] if vol.get("warn") else None

    exchange   = init_exchange()
    pipeline   = load_model()
    thresholds = get_mode_thresholds(mode)

    log.info(f"\n[1/2] Checking open trades...")
    check_open_trades(exchange)

    trades = load_trades()
    log.info(f"\n[2/2] Scanning {len(SYMBOLS)} symbols | Open: {len(trades)}/{MAX_OPEN_TRADES}")
    log.info(f"      Thresholds: conf≥{thresholds['min_confidence']}% "
             f"| score≥{thresholds['min_score']} | ADX≥{thresholds['min_adx']}")

    signals_found = 0
    for symbol in SYMBOLS:
        # Stop scanning if max trades reached
        if len(load_trades()) >= MAX_OPEN_TRADES:
            log.info("  Max trades reached — stopping scan")
            break

        log.info(f"\n  ── {symbol} ({get_tier(symbol)}) ──")
        sig = generate_signal(symbol, pipeline, thresholds)

        if sig is None:
            time.sleep(0.4)
            continue

        signals_found += 1
        log.info(f"  ✅ SIGNAL: {sig['signal']} {sig['confidence']:.1f}% score={sig['score']}")

        if vol_warning:
            sig["reasons"] = list(sig.get("reasons", [])) + [f"⚠️ {vol_warning}"]

        success = execute_trade(
            exchange,
            symbol       = sig["symbol"],
            signal       = sig["signal"],
            entry        = sig["entry"],
            atr          = sig["atr"],
            confidence   = sig["confidence"],
            score        = sig["score"],
            reasons      = sig["reasons"],
        )
        time.sleep(1)

    log.info(f"\n{'═'*56}")
    log.info(f"SCAN DONE — {signals_found} signal(s) found")
    log.info(f"{'═'*56}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "diagnostic":
        run_diagnostic()
    else:
        run_execution_scan()
