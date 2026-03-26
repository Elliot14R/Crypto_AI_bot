# trade_executor.py
# Full automated execution engine — Binance Testnet
# Works with GitHub Actions (single-run) or continuous VPS mode
# Loads model pipeline from pro_crypto_ai_model.pkl

import os
import json
import time
import logging
import requests
import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

try:
    import ccxt
except ImportError:
    raise ImportError("Run: pip install ccxt")

from config import (
    SYMBOLS, FEATURES, MIN_CONFIDENCE, MIN_ADX, MIN_SCORE,
    ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT,
    RISK_PER_TRADE, TIMEFRAME_ENTRY, TIMEFRAME_CONFIRM,
    TIMEFRAME_TREND, LIVE_LIMIT, MODEL_FILE, LOG_FILE
)
from feature_engineering import add_indicators

# ── Constants ────────────────────────────────────────────────────
TRADES_FILE    = "trades.json"
MAX_OPEN_TRADES = 3          # never open more than 3 at once
TP1_CLOSE_PCT  = 0.5         # close 50% of position at TP1
TP2_CLOSE_PCT  = 0.5         # close remaining 50% at TP2

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# EXCHANGE SETUP
# ════════════════════════════════════════════════════════════════

def init_exchange():
    """
    Initialize Binance exchange in TESTNET (sandbox) mode.
    Keys must be in .env as BINANCE_API_KEY and BINANCE_SECRET.
    Get testnet keys at: testnet.binance.vision
    """
    api_key = os.getenv("BINANCE_API_KEY", "")
    secret  = os.getenv("BINANCE_SECRET", "")

    if not api_key or not secret:
        raise ValueError(
            "Missing BINANCE_API_KEY or BINANCE_SECRET in .env\n"
            "Get testnet keys at: https://testnet.binance.vision"
        )

    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": secret,
        "options": {"defaultType": "spot"},
        "enableRateLimit": True,
    })
    exchange.set_sandbox_mode(True)   # ← TESTNET — no real money

    log.info("Exchange initialized: Binance TESTNET")
    return exchange


# ════════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════════

def load_model():
    """Load the trained pipeline dict from disk."""
    try:
        pipeline = joblib.load(MODEL_FILE)
        log.info(f"Model loaded from {MODEL_FILE}")
        return pipeline
    except FileNotFoundError:
        log.error(f"Model not found: {MODEL_FILE}")
        raise


# ════════════════════════════════════════════════════════════════
# STATE MANAGEMENT  (trades.json)
# ════════════════════════════════════════════════════════════════

def load_trades() -> dict:
    """Load open trades from JSON file. Returns empty dict if missing."""
    try:
        if Path(TRADES_FILE).exists():
            with open(TRADES_FILE, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        log.warning(f"Could not read {TRADES_FILE}: {e}")
    return {}


def save_trades(trades: dict):
    """Save trades state to JSON file atomically."""
    try:
        tmp = TRADES_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(trades, f, indent=2, default=str)
        os.replace(tmp, TRADES_FILE)
    except IOError as e:
        log.error(f"Could not save {TRADES_FILE}: {e}")


def is_trade_allowed(symbol: str, trades: dict) -> tuple:
    """
    Check if a new trade is allowed.
    Returns (allowed: bool, reason: str)
    """
    if symbol in trades:
        return False, f"Already have open trade for {symbol}"
    if len(trades) >= MAX_OPEN_TRADES:
        return False, f"Max open trades reached ({MAX_OPEN_TRADES})"
    return True, "OK"


# ════════════════════════════════════════════════════════════════
# MARKET DATA
# ════════════════════════════════════════════════════════════════

def get_data(symbol: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV data from Binance public API."""
    url    = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": LIVE_LIMIT}
    resp   = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()).iloc[:, :6]
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c])
    return df


def get_balance_usdt(exchange) -> float:
    """Fetch total USDT balance from exchange."""
    try:
        balance = exchange.fetch_balance()
        return float(balance["USDT"]["free"])
    except Exception as e:
        log.error(f"Could not fetch balance: {e}")
        return 0.0


# ════════════════════════════════════════════════════════════════
# POSITION SIZING
# ════════════════════════════════════════════════════════════════

def calculate_position_size(
    balance_usdt: float,
    entry: float,
    stop: float,
    risk_pct: float = RISK_PER_TRADE
) -> float:
    """
    Risk-based position sizing.
    Formula: risk_amount / stop_distance_per_unit
    Returns quantity in base asset (e.g. BTC amount).
    """
    risk_amount   = balance_usdt * risk_pct        # e.g. 1% of balance
    stop_distance = abs(entry - stop)              # distance in USDT per unit

    if stop_distance <= 0:
        log.warning("Stop distance is zero — skipping position size calc")
        return 0.0

    qty = risk_amount / stop_distance
    return round(qty, 6)


# ════════════════════════════════════════════════════════════════
# TRADE EXECUTION
# ════════════════════════════════════════════════════════════════

def execute_trade(
    exchange,
    symbol:     str,
    signal:     str,       # "BUY" or "SELL"
    entry:      float,
    atr:        float,
    confidence: float,
    score:      int,
    reasons:    list,
) -> bool:
    """
    Execute a trade on Binance Testnet.
    Places market entry + stop loss + two take profit orders.
    Returns True if trade was opened successfully.
    """
    trades  = load_trades()
    allowed, reason = is_trade_allowed(symbol, trades)
    if not allowed:
        log.info(f"  Trade blocked for {symbol}: {reason}")
        return False

    # ── Price levels ────────────────────────────────────────────
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

    # ── Balance and position size ────────────────────────────────
    balance = get_balance_usdt(exchange)
    if balance < 10:
        log.warning(f"  Insufficient balance: {balance:.2f} USDT")
        send_telegram_warning(f"⚠️ Insufficient balance: {balance:.2f} USDT — skipping {symbol}")
        return False

    qty      = calculate_position_size(balance, entry, stop)
    risk_usd = round(balance * RISK_PER_TRADE, 2)
    qty_tp1  = round(qty * TP1_CLOSE_PCT, 6)
    qty_tp2  = round(qty * TP2_CLOSE_PCT, 6)

    if qty <= 0:
        log.warning(f"  Position size zero for {symbol} — skipping")
        return False

    log.info(f"  Executing {signal} {symbol} | qty={qty} | entry≈{entry} | SL={stop} | TP1={tp1} | TP2={tp2}")

    order_ids = {}

    try:
        # 1. Market entry order
        entry_order = exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=qty,
        )
        order_ids["entry"] = entry_order["id"]
        actual_entry = float(entry_order.get("average", entry) or entry)
        log.info(f"  Entry order placed: id={entry_order['id']} price≈{actual_entry}")
        time.sleep(1)   # brief pause before placing SL/TP

        # 2. Stop loss order (opposite side)
        sl_side = "sell" if signal == "BUY" else "buy"
        try:
            sl_order = exchange.create_order(
                symbol=symbol,
                type="stop_loss_limit",
                side=sl_side,
                amount=qty,
                price=stop,
                params={"stopPrice": stop, "timeInForce": "GTC"},
            )
            order_ids["stop_loss"] = sl_order["id"]
            log.info(f"  Stop loss placed at {stop}")
        except Exception as sl_err:
            log.warning(f"  SL order failed (trying limit): {sl_err}")
            # Fallback: plain limit order at stop price
            sl_order = exchange.create_order(
                symbol=symbol,
                type="limit",
                side=sl_side,
                amount=qty,
                price=stop,
                params={"timeInForce": "GTC"},
            )
            order_ids["stop_loss"] = sl_order["id"]

        # 3. Take profit 1 (50% of position)
        tp_side = "sell" if signal == "BUY" else "buy"
        try:
            tp1_order = exchange.create_order(
                symbol=symbol,
                type="take_profit_limit",
                side=tp_side,
                amount=qty_tp1,
                price=tp1,
                params={"stopPrice": tp1, "timeInForce": "GTC"},
            )
            order_ids["tp1"] = tp1_order["id"]
            log.info(f"  TP1 placed at {tp1} for qty={qty_tp1}")
        except Exception as tp1_err:
            log.warning(f"  TP1 order failed (trying limit): {tp1_err}")
            tp1_order = exchange.create_order(
                symbol=symbol,
                type="limit",
                side=tp_side,
                amount=qty_tp1,
                price=tp1,
                params={"timeInForce": "GTC"},
            )
            order_ids["tp1"] = tp1_order["id"]

        # 4. Take profit 2 (remaining 50%)
        try:
            tp2_order = exchange.create_order(
                symbol=symbol,
                type="take_profit_limit",
                side=tp_side,
                amount=qty_tp2,
                price=tp2,
                params={"stopPrice": tp2, "timeInForce": "GTC"},
            )
            order_ids["tp2"] = tp2_order["id"]
            log.info(f"  TP2 placed at {tp2} for qty={qty_tp2}")
        except Exception as tp2_err:
            log.warning(f"  TP2 order failed (trying limit): {tp2_err}")
            tp2_order = exchange.create_order(
                symbol=symbol,
                type="limit",
                side=tp_side,
                amount=qty_tp2,
                price=tp2,
                params={"timeInForce": "GTC"},
            )
            order_ids["tp2"] = tp2_order["id"]

    except ccxt.InsufficientFunds:
        log.error(f"  Insufficient funds for {symbol}")
        send_telegram_warning(f"⚠️ Insufficient funds — could not open {symbol}")
        return False
    except ccxt.NetworkError as e:
        log.error(f"  Network error for {symbol}: {e}")
        send_telegram_warning(f"⚠️ Network error placing {symbol} order: {e}")
        return False
    except ccxt.ExchangeError as e:
        log.error(f"  Exchange error for {symbol}: {e}")
        send_telegram_warning(f"⚠️ Exchange rejected {symbol} order: {e}")
        return False
    except Exception as e:
        log.error(f"  Unexpected error for {symbol}: {e}")
        send_telegram_warning(f"⚠️ Unexpected error for {symbol}: {e}")
        return False

    # ── Save trade to state ──────────────────────────────────────
    trades[symbol] = {
        "symbol":       symbol,
        "signal":       signal,
        "entry":        actual_entry,
        "stop":         stop,
        "tp1":          tp1,
        "tp2":          tp2,
        "qty":          qty,
        "qty_tp1":      qty_tp1,
        "qty_tp2":      qty_tp2,
        "risk_usd":     risk_usd,
        "balance_at_open": balance,
        "order_ids":    order_ids,
        "opened_at":    datetime.utcnow().isoformat(),
        "tp1_hit":      False,
        "tp2_hit":      False,
        "closed":       False,
        "confidence":   confidence,
        "score":        score,
    }
    save_trades(trades)

    # ── Telegram open alert ──────────────────────────────────────
    send_trade_open_alert(
        symbol=symbol, signal=signal, confidence=confidence,
        score=score, entry=actual_entry, stop=stop, tp1=tp1, tp2=tp2,
        qty=qty, risk_usd=risk_usd, balance=balance, reasons=reasons
    )
    return True


# ════════════════════════════════════════════════════════════════
# TRADE MONITORING
# ════════════════════════════════════════════════════════════════

def check_open_trades(exchange):
    """
    Called at the START of every scan.
    Checks if any open orders have been filled (TP or SL hit).
    Updates trades.json and sends Telegram close alerts.
    """
    trades = load_trades()
    if not trades:
        return

    closed_symbols = []

    for symbol, trade in trades.items():
        if trade.get("closed"):
            closed_symbols.append(symbol)
            continue

        try:
            order_ids = trade.get("order_ids", {})
            entry     = trade["entry"]
            opened_at = trade.get("opened_at", "")

            # Check TP1
            if not trade["tp1_hit"] and "tp1" in order_ids:
                try:
                    tp1_order = exchange.fetch_order(order_ids["tp1"], symbol)
                    if tp1_order["status"] == "closed":
                        trade["tp1_hit"] = True
                        pnl = _calc_pnl(trade, tp1_order["average"], "tp1")
                        log.info(f"  TP1 HIT: {symbol} | PnL ≈ {pnl:+.2f} USDT")
                        send_trade_close_alert(symbol, "TP1 HIT", pnl, entry, tp1_order["average"], opened_at)
                except Exception as e:
                    log.warning(f"  Could not check TP1 for {symbol}: {e}")

            # Check TP2
            if trade["tp1_hit"] and not trade["tp2_hit"] and "tp2" in order_ids:
                try:
                    tp2_order = exchange.fetch_order(order_ids["tp2"], symbol)
                    if tp2_order["status"] == "closed":
                        trade["tp2_hit"] = True
                        trade["closed"]  = True
                        pnl = _calc_pnl(trade, tp2_order["average"], "tp2")
                        log.info(f"  TP2 HIT: {symbol} | PnL ≈ {pnl:+.2f} USDT")
                        send_trade_close_alert(symbol, "✅ FULL WIN (TP2)", pnl, entry, tp2_order["average"], opened_at)
                        closed_symbols.append(symbol)
                except Exception as e:
                    log.warning(f"  Could not check TP2 for {symbol}: {e}")

            # Check Stop Loss
            if not trade.get("closed") and "stop_loss" in order_ids:
                try:
                    sl_order = exchange.fetch_order(order_ids["stop_loss"], symbol)
                    if sl_order["status"] == "closed":
                        trade["closed"] = True
                        pnl = _calc_pnl(trade, sl_order["average"], "sl")
                        log.info(f"  SL HIT: {symbol} | PnL ≈ {pnl:+.2f} USDT")
                        send_trade_close_alert(symbol, "❌ STOPPED OUT", pnl, entry, sl_order["average"], opened_at)
                        # Cancel any remaining TP orders
                        _cancel_remaining_orders(exchange, symbol, order_ids, trade)
                        closed_symbols.append(symbol)
                except Exception as e:
                    log.warning(f"  Could not check SL for {symbol}: {e}")

        except Exception as e:
            log.error(f"  Error monitoring {symbol}: {e}")

    # Save updated state
    save_trades(trades)

    # Clean up fully closed trades (keep for 24h record, then remove)
    for sym in closed_symbols:
        if trades.get(sym, {}).get("closed"):
            log.info(f"  Removing closed trade: {sym}")
            trades.pop(sym, None)

    save_trades(trades)


def _calc_pnl(trade: dict, close_price: float, close_type: str) -> float:
    """Calculate approximate PnL in USDT."""
    entry = trade["entry"]
    if close_type in ("tp1",):
        qty = trade["qty_tp1"]
    elif close_type == "tp2":
        qty = trade["qty_tp2"]
    else:  # sl
        qty = trade["qty"]

    if trade["signal"] == "BUY":
        return round((close_price - entry) * qty, 4)
    else:
        return round((entry - close_price) * qty, 4)


def _cancel_remaining_orders(exchange, symbol: str, order_ids: dict, trade: dict):
    """Cancel open TP orders when SL is hit."""
    for key in ("tp1", "tp2"):
        if key in order_ids and not trade.get(f"{key}_hit"):
            try:
                exchange.cancel_order(order_ids[key], symbol)
                log.info(f"  Cancelled {key} order for {symbol}")
            except Exception as e:
                log.warning(f"  Could not cancel {key} for {symbol}: {e}")


# ════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ════════════════════════════════════════════════════════════════

def generate_signal(symbol: str, pipeline: dict) -> dict | None:
    """
    Fetch live data, run feature engineering, get ML prediction.
    Returns signal dict or None if no valid signal.
    """
    try:
        df_entry   = add_indicators(get_data(symbol, TIMEFRAME_ENTRY))
        df_confirm = add_indicators(get_data(symbol, TIMEFRAME_CONFIRM))
        df_trend   = add_indicators(get_data(symbol, TIMEFRAME_TREND))

        if df_entry.empty or len(df_entry) < 50:
            return None

        row_entry   = df_entry.iloc[-1]
        row_confirm = df_confirm.iloc[-1] if not df_confirm.empty else pd.Series(dtype=float)
        row_trend   = df_trend.iloc[-1]   if not df_trend.empty   else pd.Series(dtype=float)

        all_features = pipeline["all_features"]
        selector     = pipeline["selector"]
        ensemble     = pipeline["ensemble"]

        missing = [f for f in all_features if f not in df_entry.columns]
        if missing:
            log.warning(f"  {symbol}: Missing features {missing[:5]}")
            return None

        X_raw      = pd.DataFrame([row_entry[all_features].values], columns=all_features)
        X_sel      = selector.transform(X_raw)
        pred       = ensemble.predict(X_sel)[0]
        prob       = ensemble.predict_proba(X_sel)[0]
        labels     = {0: "BUY", 1: "SELL", 2: "NO_TRADE"}
        signal     = labels[pred]
        confidence = round(float(max(prob)) * 100, 1)

        if signal == "NO_TRADE" or confidence < MIN_CONFIDENCE:
            return None

        adx_val = float(row_entry.get("adx", 0))
        if adx_val < MIN_ADX:
            return None

        # Quality scoring
        score, reasons = _quality_score(row_entry, row_confirm, row_trend, signal, confidence)
        if score < MIN_SCORE:
            return None

        entry = float(row_entry["close"])
        atr   = float(row_entry["atr"])

        return {
            "symbol":     symbol,
            "signal":     signal,
            "confidence": confidence,
            "score":      score,
            "entry":      entry,
            "atr":        atr,
            "reasons":    reasons,
        }

    except Exception as e:
        log.error(f"  Signal generation failed for {symbol}: {e}")
        return None


def _quality_score(row_entry, row_confirm, row_trend, signal, confidence):
    score, reasons = 0, []
    if confidence >= 75:
        score += 1
        reasons.append(f"High AI confidence ({confidence:.0f}%)")
    elif confidence >= 65:
        reasons.append(f"AI confidence ({confidence:.0f}%)")
    adx = row_entry.get("adx", 0)
    if adx > 25:
        score += 1
        reasons.append(f"Strong trend ADX {adx:.0f}")
    elif adx > 20:
        score += 1
        reasons.append(f"Moderate trend ADX {adx:.0f}")
    rsi = row_entry.get("rsi", 50)
    if signal == "BUY" and rsi < 40:
        score += 1
        reasons.append(f"RSI oversold ({rsi:.0f})")
    elif signal == "SELL" and rsi > 60:
        score += 1
        reasons.append(f"RSI overbought ({rsi:.0f})")
    e20, e50, e200 = row_entry.get("ema20", 0), row_entry.get("ema50", 0), row_entry.get("ema200", 0)
    if signal == "BUY" and e20 > e50 > e200:
        score += 1
        reasons.append("EMA uptrend 20>50>200")
    elif signal == "SELL" and e20 < e50 < e200:
        score += 1
        reasons.append("EMA downtrend 20<50<200")
    if signal == "BUY" and row_confirm.get("ema20", 0) > row_confirm.get("ema50", 0):
        score += 1
        reasons.append("1h EMA confirms uptrend")
    elif signal == "SELL" and row_confirm.get("ema20", 0) < row_confirm.get("ema50", 0):
        score += 1
        reasons.append("1h EMA confirms downtrend")
    return score, reasons


# ════════════════════════════════════════════════════════════════
# TELEGRAM ALERTS
# ════════════════════════════════════════════════════════════════

def _send_message(text: str):
    token   = os.getenv("TELEGRAM_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        log.warning("Telegram not configured")
        return
    try:
        url     = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")


def send_trade_open_alert(symbol, signal, confidence, score, entry,
                          stop, tp1, tp2, qty, risk_usd, balance, reasons):
    emoji  = "🟢" if signal == "BUY" else "🔴"
    stars  = "⭐" * min(score, 5)
    dec    = 4 if entry < 10 else 2
    fp     = lambda v: f"{v:,.{dec}f}"
    sl_pct = abs((stop - entry) / entry * 100)
    t1_pct = abs((tp1  - entry) / entry * 100)
    t2_pct = abs((tp2  - entry) / entry * 100)
    pos_usd = round(qty * entry, 2)
    rlines  = "\n".join([f"  - {r}" for r in reasons])
    _send_message(
        f"🤖 *LIVE TEST TRADE OPENED*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{emoji} *{signal}  —  {symbol}* {stars}\n"
        f"🎯 Confidence: *{confidence:.1f}%*\n\n"
        f"⚡ *ENTRY:*      `{fp(entry)}`\n"
        f"🛑 *STOP LOSS:*  `{fp(stop)}`  (-{sl_pct:.1f}%)\n"
        f"🎯 *TARGET 1:*   `{fp(tp1)}`  (+{t1_pct:.1f}%)\n"
        f"🎯 *TARGET 2:*   `{fp(tp2)}`  (+{t2_pct:.1f}%)\n\n"
        f"💰 *Position:*   `{pos_usd:.2f} USDT`\n"
        f"⚠️  *Risk:*       `{risk_usd:.2f} USDT` (1% of balance)\n"
        f"💼 *Balance:*    `{balance:.2f} USDT`\n\n"
        f"📊 *Why:*\n{rlines}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_Binance Testnet — paper trading only_"
    )


def send_trade_close_alert(symbol, result, pnl, entry, close_price, opened_at):
    emoji  = "✅" if pnl > 0 else "❌"
    dec    = 4 if entry < 10 else 2
    try:
        opened = datetime.fromisoformat(opened_at)
        dur    = str(datetime.utcnow() - opened).split(".")[0]
    except Exception:
        dur    = "unknown"
    _send_message(
        f"🤖 *TRADE CLOSED*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{emoji} *{result}  —  {symbol}*\n\n"
        f"📥 Entry:  `{entry:.{dec}f}`\n"
        f"📤 Close:  `{close_price:.{dec}f}`\n"
        f"💵 *PnL:   `{pnl:+.4f} USDT`*\n"
        f"⏱️  Duration: {dur}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_Binance Testnet_"
    )


def send_telegram_warning(text: str):
    _send_message(f"⚠️ *Bot Warning*\n{text}")


# ════════════════════════════════════════════════════════════════
# MAIN SCAN  (called by GitHub Actions / cron)
# ════════════════════════════════════════════════════════════════

def run_execution_scan():
    """
    Single scan — runs once per GitHub Actions trigger.
    1. Check + update open trades
    2. Scan all symbols for new signals
    3. Execute qualifying signals
    """
    log.info(f"\n{'═'*50}")
    log.info(f"Execution scan — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    log.info(f"{'═'*50}")

    # Init exchange and model
    exchange = init_exchange()
    pipeline = load_model()

    # Step 1: Check open trades first
    log.info("\n[1/2] Checking open trades...")
    check_open_trades(exchange)

    # Step 2: Scan for new signals
    trades = load_trades()
    log.info(f"\n[2/2] Scanning {len(SYMBOLS)} symbols | Open trades: {len(trades)}/{MAX_OPEN_TRADES}")

    for symbol in SYMBOLS:
        log.info(f"  Scanning {symbol}...")

        sig = generate_signal(symbol, pipeline)
        if sig is None:
            log.info(f"    No signal")
            time.sleep(0.5)
            continue

        log.info(f"    Signal: {sig['signal']} {sig['confidence']}% score={sig['score']}")

        success = execute_trade(
            exchange=exchange,
            symbol=sig["symbol"],
            signal=sig["signal"],
            entry=sig["entry"],
            atr=sig["atr"],
            confidence=sig["confidence"],
            score=sig["score"],
            reasons=sig["reasons"],
        )

        if success:
            log.info(f"    ✅ Trade opened for {symbol}")
        time.sleep(1)

    log.info("\nScan complete.")


if __name__ == "__main__":
    run_execution_scan()
