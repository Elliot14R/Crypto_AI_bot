# trade_executor.py
import os, json, time, logging, requests, joblib
import pandas as pd
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
    MODEL_FILE, LOG_FILE, get_tier, MAX_SAME_DIRECTION
)
from feature_engineering import add_indicators
from smart_scheduler import (
    should_scan, get_mode_thresholds, check_correlation,
    get_effective_risk
)

# Simple JSON persistence (no external persistence module)
def load_json(path, default):
    try:
        if Path(path).exists():
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return default


def save_json(path, data):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)

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

# ════════════ HELPERS ════════════════════════════════════════

def load_trades():  return load_json(TRADES_FILE,  {})
def save_trades(d): save_json(TRADES_FILE, d)
def load_history(): return load_json(HISTORY_FILE, [])
def load_signals(): return load_json(SIGNALS_FILE, [])

def append_history(rec):
    h = load_history()
    h.append(rec)
    save_json(HISTORY_FILE, h)

def save_signal(sig):
    sigs = load_signals()
    sigs.append({**sig, "generated_at": datetime.now(timezone.utc).isoformat()})
    sigs = sigs[-500:]
    save_json(SIGNALS_FILE, sigs)

def get_real_win_rate() -> dict:
    history = load_history()
    real    = [h for h in history if h.get("signal") != "RECOVERED"]
    wins    = [h for h in real if (h.get("pnl") or 0) > 0]
    losses  = [h for h in real if (h.get("pnl") or 0) <= 0]
    total   = len(real)
    return {
        "total":    total,
        "wins":     len(wins),
        "losses":   len(losses),
        "win_rate": round(len(wins) / total * 100, 1) if total else 0,
        "total_pnl": round(sum(h.get("pnl", 0) for h in real), 4),
        "avg_win":   round(sum(h.get("pnl", 0) for h in wins) / len(wins), 4) if wins else 0,
        "avg_loss":  round(sum(h.get("pnl", 0) for h in losses) / len(losses), 4) if losses else 0,
    }

# ════════════ EXCHANGE ════════════════════════════════════════

def init_exchange():
    key    = os.getenv("BINANCE_API_KEY", "")
    secret = os.getenv("BINANCE_SECRET",  "")
    if not key or not secret:
        raise ValueError("BINANCE_API_KEY or BINANCE_SECRET missing!")
    ex = ccxt.binance({
        "apiKey": key, "secret": secret,
        "options": {"defaultType": "spot"},
        "enableRateLimit": True,
    })
    ex.set_sandbox_mode(True)
    ex.options["warnOnFetchOpenOrdersWithoutSymbol"] = False
    log.info("✓ Exchange: Binance TESTNET")
    return ex

def load_model():
    pipeline = joblib.load(MODEL_FILE)
    return pipeline

def get_balance_usdt(ex):
    """Fetches balance, saves it to GitHub, and catches errors."""
    try:
        b = ex.fetch_balance()
        usdt = float(b.get("USDT", {}).get("free", 0))
        
        assets = []
        for asset, data in b.items():
            if isinstance(data, dict) and float(data.get("free", 0)) > 0:
                if asset not in ("info", "free", "used", "total", "timestamp", "datetime"):
                    assets.append({"asset": asset, "free": float(data["free"])})
        
        save_json(BALANCE_FILE, {
            "usdt": usdt,
            "assets": assets,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "error": None
        })
        return usdt
    except Exception as e:
        log.error(f"Balance fetch failed: {e}")
        # Save the error so we can read it!
        save_json(BALANCE_FILE, {
            "usdt": 0.0,
            "assets": [],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        })
        return 0.0

# ════════════ MARKET DATA ════════════════════════════════════

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

def calc_pos_size(balance, entry, stop, risk_mult=1.0):
    effective_risk = RISK_PER_TRADE * risk_mult
    risk_usd       = balance * effective_risk
    dist           = abs(entry - stop)

    if dist <= 0: return 0.0, 0.0

    qty = risk_usd / dist
    max_usd = balance * 0.20
    if (qty * entry) > max_usd: qty = max_usd / entry

    return round(qty, 6), round(risk_usd, 2)

# ════════════ EXECUTION ══════════════════════════════════════

def execute_trade(ex, symbol, signal, entry, atr, confidence, score, reasons, risk_mult=1.0):
    trades = load_trades()

    if symbol in trades or len(trades) >= MAX_OPEN_TRADES: return False
    if not check_correlation(trades, signal): return False

    dec = 4 if entry < 10 else 2
    if signal == "BUY":
        stop  = round(entry - atr * ATR_STOP_MULT,    dec)
        tp1   = round(entry + atr * ATR_TARGET1_MULT, dec)
        tp2   = round(entry + atr * ATR_TARGET2_MULT, dec)
        side  = "buy";  sl_side = "sell"; tp_side = "sell"
    else:
        stop  = round(entry + atr * ATR_STOP_MULT,    dec)
        tp1   = round(entry - atr * ATR_TARGET1_MULT, dec)
        tp2   = round(entry - atr * ATR_TARGET2_MULT, dec)
        side  = "sell"; sl_side = "buy";  tp_side = "buy"

    balance = get_balance_usdt(ex)
    if balance < 10: return False

    qty, risk_usd = calc_pos_size(balance, entry, stop, risk_mult)
    qty_tp1       = round(qty * TP1_CLOSE_PCT, 6)
    qty_tp2       = round(qty * TP2_CLOSE_PCT, 6)

    if qty <= 0: return False

    log.info(f"  Placing {signal} {symbol} | qty={qty} | risk={risk_usd:.2f} USDT")
    order_ids = {}

    try:
        entry_order  = ex.create_order(symbol, "market", side, qty)
        order_ids["entry"] = entry_order["id"]
        actual_entry = float(entry_order.get("average", entry) or entry)
        time.sleep(1.5)

        sl_placed = False
        for ot in ["stop_loss_limit", "limit"]:
            try:
                sl_o = ex.create_order(symbol, ot, sl_side, qty, stop, params={"stopPrice": stop, "timeInForce": "GTC"})
                order_ids["stop_loss"] = sl_o["id"]
                sl_placed = True
                break
            except Exception: pass

        for ot in ["take_profit_limit", "limit"]:
            try:
                tp1_o = ex.create_order(symbol, ot, tp_side, qty_tp1, tp1, params={"stopPrice": tp1, "timeInForce": "GTC"})
                order_ids["tp1"] = tp1_o["id"]
                break
            except Exception: pass

        for ot in ["take_profit_limit", "limit"]:
            try:
                tp2_o = ex.create_order(symbol, ot, tp_side, qty_tp2, tp2, params={"stopPrice": tp2, "timeInForce": "GTC"})
                order_ids["tp2"] = tp2_o["id"]
                break
            except Exception: pass

    except Exception as e:
        log.error(f"  Trade error: {e}")
        return False

    record = {
        "symbol": symbol, "signal": signal,
        "entry": actual_entry, "stop": stop, "tp1": tp1, "tp2": tp2,
        "qty": qty, "qty_tp1": qty_tp1, "qty_tp2": qty_tp2,
        "risk_usd": risk_usd, "balance_at_open": balance,
        "risk_mult": risk_mult,
        "order_ids": order_ids,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "tp1_hit": False, "tp2_hit": False, "closed": False,
        "confidence": confidence, "score": score, "reasons": reasons,
        "tier": get_tier(symbol),
    }
    trades[symbol] = record
    save_trades(trades)
    save_signal(record)
    _send_open_alert(symbol, signal, confidence, score, actual_entry, stop, tp1, tp2, qty, risk_usd, balance, reasons, risk_mult)
    log.info(f"  ✅✅ TRADE OPENED: {symbol} {signal}")
    return True

# ════════════ MONITORING & RECOVERY ═════════════════════════

def sync_trade_history(ex):
    history = load_history()
    if len(history) > 0: return
    try:
        rebuilt = []
        for sym in SYMBOLS:
            try: orders = ex.fetch_closed_orders(sym, limit=10)
            except Exception: continue
            entries = [o for o in orders if o["type"] == "market" and o["status"] == "closed"]
            exits   = [o for o in orders if o["type"] != "market" and o["status"] == "closed"]
            for entry in entries:
                exit_order = next((x for x in exits if x["timestamp"] > entry["timestamp"]), None)
                if not exit_order: continue
                qty         = float(entry.get("filled") or entry.get("amount") or 0)
                entry_price = float(entry.get("average") or entry.get("price") or 0)
                exit_price  = float(exit_order.get("average") or exit_order.get("price") or 0)
                if qty == 0 or entry_price == 0: continue
                is_buy = entry["side"] == "buy"
                pnl    = (exit_price - entry_price) * qty if is_buy else (entry_price - exit_price) * qty
                rebuilt.append({
                    "symbol": sym, "signal": "BUY" if is_buy else "SELL",
                    "entry": entry_price, "close_price": exit_price, "pnl": round(pnl, 4),
                    "opened_at": datetime.fromtimestamp(entry["timestamp"]/1000, tz=timezone.utc).isoformat(),
                    "closed_at": datetime.fromtimestamp(exit_order["timestamp"]/1000, tz=timezone.utc).isoformat(),
                    "close_reason": "Binance Sync",
                })
        if rebuilt:
            rebuilt.sort(key=lambda x: x["closed_at"])
            save_json(HISTORY_FILE, rebuilt)
    except Exception: pass

def auto_recover_trades(ex):
    trades = load_trades()
    try:
        open_orders    = ex.fetch_open_orders()
        active_symbols = list(set(o["symbol"] for o in open_orders))
        recovered      = 0
        for sym in active_symbols:
            if sym not in trades:
                sym_orders = [o for o in open_orders if o["symbol"] == sym]
                total_qty  = sum(o.get("amount", 0) for o in sym_orders)
                order_ids  = {}
                for o in sym_orders:
                    if o["type"] == "stop_loss_limit" or "stop" in o["type"]: order_ids["stop_loss"] = o["id"]
                    elif "tp1" not in order_ids: order_ids["tp1"] = o["id"]
                    else: order_ids["tp2"] = o["id"]
                trades[sym] = {
                    "symbol": sym, "signal": "RECOVERED",
                    "entry": sym_orders[0].get("average", 0) or sym_orders[0].get("price", 0),
                    "stop": 0, "tp1": 0, "tp2": 0,
                    "qty": total_qty, "qty_tp1": total_qty/2, "qty_tp2": total_qty/2,
                    "risk_usd": 0, "balance_at_open": 0, "risk_mult": 1.0,
                    "order_ids": order_ids,
                    "tp1_hit": False, "tp2_hit": False, "closed": False,
                    "confidence": 100, "score": 6,
                    "reasons": ["🔄 Recovered by Auto-Sync"],
                    "tier": get_tier(sym),
                    "opened_at": sym_orders[0].get("datetime", datetime.now(timezone.utc).isoformat()),
                }
                recovered += 1
        if recovered > 0: save_trades(trades)
    except Exception: pass

def check_open_trades(ex):
    trades    = load_trades()
    if not trades: return
    to_remove = []

    for symbol, trade in list(trades.items()):
        if trade.get("closed"):
            to_remove.append(symbol)
            continue
        try:
            oids  = trade.get("order_ids", {})
            entry = trade["entry"]

            if not trade["tp1_hit"] and "tp1" in oids:
                try:
                    o = ex.fetch_order(oids["tp1"], symbol)
                    if o["status"] == "closed":
                        trade["tp1_hit"] = True
                        pnl = _pnl(trade, float(o["average"]), "tp1")
                        _send_close_alert(symbol, "TP1 HIT 🎯", pnl, entry, float(o["average"]), trade["opened_at"])
                except Exception: pass

            if trade["tp1_hit"] and not trade["tp2_hit"] and "tp2" in oids:
                try:
                    o = ex.fetch_order(oids["tp2"], symbol)
                    if o["status"] == "closed":
                        trade["tp2_hit"] = True
                        trade["closed"]  = True
                        pnl = _pnl(trade, float(o["average"]), "tp2")
                        _send_close_alert(symbol, "✅ FULL WIN (TP2)", pnl, entry, float(o["average"]), trade["opened_at"])
                        _record_close(trade, float(o["average"]), pnl, "TP2 hit")
                        to_remove.append(symbol)
                except Exception: pass

            if not trade.get("closed") and "stop_loss" in oids:
                try:
                    o = ex.fetch_order(oids["stop_loss"], symbol)
                    if o["status"] == "closed":
                        trade["closed"] = True
                        pnl = _pnl(trade, float(o["average"]), "sl")
                        _send_close_alert(symbol, "❌ STOPPED OUT", pnl, entry, float(o["average"]), trade["opened_at"])
                        _record_close(trade, float(o["average"]), pnl, "SL hit")
                        _cancel_remaining(ex, symbol, oids, trade)
                        to_remove.append(symbol)
                except Exception: pass
        except Exception: pass

    save_trades(trades)
    for sym in set(to_remove): trades.pop(sym, None)
    save_trades(trades)

def _pnl(trade, close_price, close_type):
    entry = trade["entry"]
    qty   = (trade["qty_tp1"] if close_type == "tp1" else
             trade["qty_tp2"] if close_type == "tp2" else trade["qty"])
    return round((close_price - entry) * qty if trade["signal"] == "BUY" else (entry - close_price) * qty, 4)

def _cancel_remaining(ex, symbol, oids, trade):
    for key in ("tp1", "tp2"):
        if key in oids and not trade.get(f"{key}_hit"):
            try: ex.cancel_order(oids[key], symbol)
            except Exception: pass

def _record_close(trade, close_price, pnl, reason):
    append_history({
        **trade, "close_price": close_price, "pnl": pnl,
        "closed_at": datetime.now(timezone.utc).isoformat(), "close_reason": reason,
    })

# ════════════ SIGNAL GENERATION ══════════════════════════════

def generate_signal(symbol, pipeline, thresholds):
    try:
        df_entry   = add_indicators(get_data(symbol, TIMEFRAME_ENTRY))
        df_confirm = add_indicators(get_data(symbol, TIMEFRAME_CONFIRM))

        if df_entry.empty or len(df_entry) < 50: return None
        row_entry   = df_entry.iloc[-1].copy()
        row_confirm = df_confirm.iloc[-1] if not df_confirm.empty else pd.Series(dtype=float)

        row_entry["rsi_1h"]   = float(row_confirm.get("rsi", 50))
        row_entry["adx_1h"]   = float(row_confirm.get("adx", 0))
        row_entry["trend_1h"] = float(row_confirm.get("trend", 0))

        all_feat = pipeline["all_features"]
        selector = pipeline["selector"]
        ensemble = pipeline["ensemble"]

        X_raw      = pd.DataFrame([row_entry[all_feat].values], columns=all_feat)
        X_sel      = selector.transform(X_raw)
        pred       = ensemble.predict(X_sel)[0]
        prob       = ensemble.predict_proba(X_sel)[0]
        label_map  = {0: "BUY", 1: "SELL", 2: "NO_TRADE"}
        signal     = label_map[pred]
        confidence = round(float(max(prob)) * 100, 1)

        log.info(f"    ML: {signal} {confidence:.1f}% (need ≥{thresholds['min_confidence']}%)")

        if signal == "NO_TRADE" or confidence < thresholds["min_confidence"]:
            return None

        adx_val = float(row_entry.get("adx", 0))
        if adx_val < thresholds["min_adx"]: return None

        score, reasons = _quality_score(row_entry, row_confirm, signal, confidence)
        entry = float(row_entry["close"])
        atr   = float(row_entry["atr"])
        dec   = 4 if entry < 10 else 2

        if signal == "BUY":
            stop = round(entry - atr * ATR_STOP_MULT, dec)
            tp1  = round(entry + atr * ATR_TARGET1_MULT, dec)
            tp2  = round(entry + atr * ATR_TARGET2_MULT, dec)
        else:
            stop = round(entry + atr * ATR_STOP_MULT, dec)
            tp1  = round(entry - atr * ATR_TARGET1_MULT, dec)
            tp2  = round(entry - atr * ATR_TARGET2_MULT, dec)

        if score < thresholds["min_score"]:
            save_signal({
                "symbol": symbol, "signal": signal, "confidence": confidence,
                "score": score, "entry": entry, "atr": atr, "reasons": reasons,
                "rejected": True, "reject_reason": f"score {score} < {thresholds['min_score']}",
                "stop": stop, "tp1": tp1, "tp2": tp2,
            })
            return None

        return {
            "symbol": symbol, "signal": signal, "confidence": confidence, "score": score,
            "entry": entry, "atr": atr, "stop": stop, "tp1": tp1, "tp2": tp2, "reasons": reasons,
        }

    except Exception as e:
        return None

def _quality_score(row_entry, row_confirm, signal, confidence):
    score, reasons = 0, []
    if confidence >= 75: score += 1; reasons.append(f"High AI confidence ({confidence:.0f}%)")
    elif confidence >= 65: score += 1; reasons.append(f"Good AI confidence ({confidence:.0f}%)")
    elif confidence >= 60: reasons.append(f"AI confidence ({confidence:.0f}%)")

    adx = float(row_entry.get("adx", 0))
    if adx > 25: score += 1; reasons.append(f"Strong trend ADX {adx:.0f}")
    elif adx > 20: score += 1; reasons.append(f"Moderate trend ADX {adx:.0f}")

    rsi = float(row_entry.get("rsi", 50))
    if signal == "BUY" and rsi < 45: score += 1; reasons.append(f"RSI bullish zone ({rsi:.0f})")
    elif signal == "SELL" and rsi > 55: score += 1; reasons.append(f"RSI bearish zone ({rsi:.0f})")

    e20 = float(row_entry.get("ema20", 0)); e50 = float(row_entry.get("ema50", 0))
    if signal == "BUY" and e20 > e50: score += 1; reasons.append("EMA20 > EMA50 uptrend")
    elif signal == "SELL" and e20 < e50: score += 1; reasons.append("EMA20 < EMA50 downtrend")

    c20 = float(row_confirm.get("ema20", 0)); c50 = float(row_confirm.get("ema50", 0))
    if signal == "BUY" and c20 > c50: score += 1; reasons.append("1h confirms uptrend")
    elif signal == "SELL" and c20 < c50: score += 1; reasons.append("1h confirms downtrend")

    return score, reasons

# ════════════ TELEGRAM ═══════════════════════════════════════

def _send(text):
    token   = os.getenv("TELEGRAM_TOKEN",   "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id: return
    try: requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=10)
    except Exception: pass

def _warn(text): log.warning(text); _send(text)

def check_mode_switch(mode: dict):
    last = load_json(MODE_FILE, {})
    if last.get("mode") != mode["mode"]:
        msgs = {
            "active":  "📈 *Active hours* — conf ≥65% | score ≥3 | full risk",
            "quiet":   "🌙 *Quiet hours* — conf ≥72% | score ≥4 | 50% risk",
            "weekend": "📅 *Weekend mode* — conf ≥65% | score ≥3 | 75% risk",
        }
        _send(msgs.get(mode["mode"], "Mode changed"))
        save_json(MODE_FILE, {"mode":  mode["mode"], "since": datetime.now(timezone.utc).isoformat()})

def _send_open_alert(symbol, signal, confidence, score, entry, stop, tp1, tp2, qty, risk_usd, balance, reasons, risk_mult=1.0):
    emoji   = "🟢" if signal == "BUY" else "🔴"
    stars   = "⭐" * min(score, 5)
    dec     = 4 if entry < 10 else 2
    fp      = lambda v: f"{v:,.{dec}f}"
    rlines  = "\n".join([f"  • {r}" for r in reasons])
    risk_note = "" if risk_mult >= 1.0 else f"\n⚡ *Risk reduced to {int(risk_mult*100)}%*"
    _send(
        f"🤖 *TESTNET TRADE OPENED*\n━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{emoji} *{signal} — {symbol}* {stars}\n🏷️ Tier: _{get_tier(symbol)}_{risk_note}\n"
        f"🎯 Confidence: *{confidence:.1f}%* · Score: *{score}/6*\n\n"
        f"⚡ *ENTRY:* `{fp(entry)}`\n🛑 *STOP LOSS:* `{fp(stop)}`\n"
        f"🎯 *TARGET 1:* `{fp(tp1)}`\n🎯 *TARGET 2:* `{fp(tp2)}`\n\n"
        f"💰 *Position:* `{round(qty*entry,2)} USDT`\n⚠️  *Risk:* `{risk_usd:.2f} USDT`\n"
        f"💼 *Balance:* `{balance:.2f} USDT`\n\n📊 *Reasons:*\n{rlines}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n_Binance Testnet_"
    )

def _send_close_alert(symbol, result, pnl, entry, close_price, opened_at):
    emoji = "✅" if pnl > 0 else "❌"
    dec   = 4 if entry < 10 else 2
    try: dur = str(datetime.now(timezone.utc) - datetime.fromisoformat(opened_at)).split(".")[0]
    except Exception: dur = "—"
    _send(f"🤖 *TRADE CLOSED*\n━━━━━━━━━━━━━━━━━━━━\n\n{emoji} *{result} — {symbol}*\n\n"
          f"📥 Entry: `{entry:.{dec}f}`\n📤 Close: `{close_price:.{dec}f}`\n"
          f"💵 *PnL: `{pnl:+.4f} USDT`*\n⏱️ Duration: {dur}\n\n━━━━━━━━━━━━━━━━━━━━\n_Binance Testnet_")

# ════════════ MAIN ═══════════════════════════════════════════

def run_execution_scan():
    log.info(f"\n{'═'*56}\nSCAN START — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n{'═'*56}")

    # 1. Init exchange and FORCE balance check BEFORE anything else
    try:
        exchange = init_exchange()
        get_balance_usdt(exchange)
    except Exception as e:
        log.error(f"Exchange init failed: {e}")
        return

    run, mode, vol, reason = should_scan()
    check_mode_switch(mode)

    if not run:
        log.info(f"  Scan SKIPPED: {reason}")
        return

    effective_risk = get_effective_risk(mode, vol)
    vol_warning    = vol["message"] if vol.get("warn") else None

    log.info(f"\n  Mode: {mode['label']} | conf≥{mode['min_confidence']}% | score≥{mode['min_score']} | risk_mult:{effective_risk:.2f}")

    pipeline   = load_model()
    thresholds = get_mode_thresholds(mode)

    auto_recover_trades(exchange)
    sync_trade_history(exchange)

    log.info(f"\n[1/2] Checking open trades...")
    check_open_trades(exchange)

    trades = load_trades()
    log.info(f"\n[2/2] Scanning {len(SYMBOLS)} symbols | Open: {len(trades)}/{MAX_OPEN_TRADES}")

    signals_found = 0
    for symbol in SYMBOLS:
        if len(load_trades()) >= MAX_OPEN_TRADES: break
        sig = generate_signal(symbol, pipeline, thresholds)
        if sig is None:
            time.sleep(0.4)
            continue
        
        signals_found += 1
        if vol_warning: sig["reasons"] = list(sig.get("reasons", [])) + [f"⚠️ {vol_warning}"]
        execute_trade(exchange, symbol=sig["symbol"], signal=sig["signal"], entry=sig["entry"], atr=sig["atr"], confidence=sig["confidence"], score=sig["score"], reasons=sig["reasons"], risk_mult=effective_risk)
        time.sleep(1)

    log.info(f"\n{'═'*56}\nSCAN DONE — {signals_found} signal(s) found\n{'═'*56}\n")

if __name__ == "__main__":
    run_execution_scan()
