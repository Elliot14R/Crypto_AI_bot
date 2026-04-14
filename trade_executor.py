# trade_executor.py — Detailed Logging + Precision Integrated
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
from deribit_client import DeribitClient
from feature_engineering import add_indicators
from smart_scheduler import should_scan, get_mode_thresholds, check_correlation, get_effective_risk

TRADES_FILE, MAX_OPEN_TRADES = "trades.json", 3
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
log = logging.getLogger(__name__)

# ════════════ FILE I/O ════════════════════════════════════════════════

def load_json(path, default):
    p = Path(path)
    if p.exists():
        try:
            with open(p) as f: return json.load(f)
        except: pass
    return default

def save_json(path, data):
    with open(path, "w") as f: json.dump(data, f, indent=2, default=str)

# ════════════ MAINTENANCE ════════════════════════════════════════════

def clean_invalid_trades(deribit):
    """Position Reconciliation: Sync local memory with actual exchange positions"""
    trades = load_json(TRADES_FILE, {})
    if not trades: return
    live_pos = {}
    for p in deribit.get_positions():
        inst = p['instrument_name']
        base = inst.split('_')[0] if '_' in inst else inst.split('-')[0]
        live_pos[f"{base}USDT"] = float(p['size'])
    to_remove = [s for s in trades if s not in live_pos]
    for s in to_remove: 
        log.warning(f"  🗑️ Clearing ghost trade: {s}")
        trades.pop(s)
    if to_remove: save_json(TRADES_FILE, trades)

# ════════════ CORE EXECUTION ══════════════════════════════════════════

def execute_trade(deribit, symbol, signal, entry, atr, confidence, score, reasons, risk_mult, balance):
    trades = load_json(TRADES_FILE, {})
    if symbol in trades or len(trades) >= MAX_OPEN_TRADES: return False
    
    side, sl_side, tp_side = ("BUY", "SELL", "SELL") if signal == "BUY" else ("SELL", "BUY", "BUY")
    target_q = deribit.calc_contracts(symbol, balance, entry, entry - (atr*ATR_STOP_MULT if signal=="BUY" else -atr*ATR_STOP_MULT), risk_mult)

    try:
        # 1. Market Entry (IOC)
        res = deribit.place_market_order(symbol, side, target_q)
        order = res.get("order", res)
        filled = float(order.get("filled_amount", 0))
        if filled <= 0:
            log.warning(f"  ⚠️ {symbol} IOC skipped: No liquidity.")
            return False

        actual_entry = float(order.get("average_price", entry))
        log.info(f"    ✅ Entry Filled: {symbol} @ {actual_entry}")

        # 2. SL & TP Recalculation
        stop = deribit.round_price(symbol, actual_entry - (atr*ATR_STOP_MULT if signal=="BUY" else -atr*ATR_STOP_MULT))
        tp1 = deribit.round_price(symbol, actual_entry + (atr*ATR_TARGET1_MULT if signal=="BUY" else -atr*ATR_TARGET1_MULT))
        
        # 3. Protective Orders
        sl_res = deribit.place_limit_order(symbol, sl_side, filled, stop, stop_price=stop)
        q1, q2 = deribit.split_amount(symbol, filled)
        tp1_res = deribit.place_limit_order(symbol, tp_side, q1, tp1)

        trades[symbol] = {
            "symbol": symbol, "signal": signal, "entry": actual_entry, 
            "stop": stop, "tp1": tp1, "qty": filled, "qty_tp1": q1, "qty_tp2": q2, 
            "order_ids": {"entry": str(order.get("order_id")), "stop_loss": str(sl_res.get("order", sl_res).get("order_id")), "tp1": str(tp1_res.get("order", tp1_res).get("order_id"))}, 
            "tp1_hit": False, "confidence": confidence, "score": score
        }
        save_json(TRADES_FILE, trades)
        return True
    except Exception as e: 
        log.error(f"    ❌ Execution failed: {e}")
        return False

# ════════════ SIGNAL LOGIC ═══════════════════════════════════════════

def _quality_score(row, r1h, signal, conf):
    score, reasons = 0, []
    if conf >= 70: score += 1; reasons.append(f"High conf ({conf:.0f}%)")
    elif conf >= 55: score += 1; reasons.append(f"Good conf ({conf:.0f}%)")
    adx = float(row.get("adx", 0))
    if adx > 20: score += 1; reasons.append(f"Strong ADX {adx:.0f}")
    rsi = float(row.get("rsi", 50))
    if signal == "BUY" and rsi < 50: score += 1; reasons.append("RSI bullish")
    elif signal == "SELL" and rsi > 50: score += 1; reasons.append("RSI bearish")
    e20, e50 = float(row.get("ema20", 0)), float(row.get("ema50", 0))
    if signal == "BUY" and e20 > e50: score += 1; reasons.append("EMA bullish")
    elif signal == "SELL" and e20 < e50: score += 1; reasons.append("EMA bearish")
    return score, reasons

def get_data(symbol: str, interval: str) -> pd.DataFrame:
    url = "https://data-api.binance.vision/api/v3/klines"
    try:
        r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": 100}, timeout=10)
        df = pd.DataFrame(r.json()).iloc[:, :6]
        df.columns = ["open_time","open","high","low","close","volume"]
        for c in df.columns: df[c] = pd.to_numeric(df[c])
        return df
    except: return pd.DataFrame()

def generate_signal(symbol, pipeline, thresholds):
    try:
        df15 = add_indicators(get_data(symbol, TIMEFRAME_ENTRY))
        df1h = add_indicators(get_data(symbol, TIMEFRAME_CONFIRM))
        if df15.empty or len(df15) < 50: return None

        row = df15.iloc[-1].copy()
        r1h = df1h.iloc[-1] if not df1h.empty else pd.Series(dtype=float)
        row["rsi_1h"], row["adx_1h"], row["trend_1h"] = float(r1h.get("rsi", 50)), float(r1h.get("adx", 0)), float(r1h.get("trend", 0))

        af = pipeline["all_features"]
        X = pd.DataFrame([row[af].values], columns=af)
        Xs = pipeline["selector"].transform(X)
        pred = pipeline["ensemble"].predict(Xs)[0]
        prob = pipeline["ensemble"].predict_proba(Xs)[0]
        sig = {0: "BUY", 1: "SELL", 2: "NO_TRADE"}[pred]
        conf = round(float(max(prob)) * 100, 1)

        # 🟢 DETAILED LOGGING RESTORED
        log.info(f"      ML: {sig} {conf}% (need ≥{thresholds['min_confidence']}%)")
        
        if sig == "NO_TRADE" or conf < thresholds["min_confidence"]: return None

        adx = float(row.get("adx", 0))
        log.info(f"      ADX: {adx:.1f} (need ≥{thresholds['min_adx']})")
        if adx < thresholds["min_adx"]: return None

        score, reasons = _quality_score(row, r1h, sig, conf)
        log.info(f"      Score: {score} (need ≥{thresholds['min_score']})")
        if score < thresholds["min_score"]: return None

        return {"symbol": symbol, "signal": sig, "confidence": conf, "score": score, "entry": float(row["close"]), "atr": float(row["atr"]), "reasons": reasons}
    except: return None

# ════════════ MAIN SCAN LOOP ═════════════════════════════════════════

def run_execution_scan():
    log.info(f"\n{'═'*56}\nSCAN START — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n{'═'*56}")
    run, mode, vol, reason = should_scan()
    if not run:
        log.info(f"  SKIPPED: {reason}"); return

    deribit = DeribitClient(os.getenv("DERIBIT_CLIENT_ID"), os.getenv("DERIBIT_CLIENT_SECRET"))
    deribit.test_connection()
    pipeline = joblib.load(MODEL_FILE)
    thresholds = get_mode_thresholds(mode)
    eff_risk = get_effective_risk(mode, vol)
    balance = deribit.get_total_equity_usd()

    clean_invalid_trades(deribit)
    log.info(f"Scanning {len(SYMBOLS)} coins | Open: {len(load_json(TRADES_FILE, {}))}/3 | Mode: {mode['label']}")

    found = 0
    for symbol in SYMBOLS:
        if len(load_json(TRADES_FILE, {})) >= MAX_OPEN_TRADES:
            log.info("  🛑 Max trades reached. Stopping scan."); break

        # 🟢 COIN HEADER LOGGING
        log.info(f"\n  ── {symbol} ({get_tier(symbol)}) ──")
        sig = generate_signal(symbol, pipeline, thresholds)
        if sig:
            found += 1
            log.info(f"    🚀 SIGNAL FOUND: {sig['signal']} | Conf: {sig['confidence']}%")
            execute_trade(deribit, sig["symbol"], sig["signal"], sig["entry"], sig["atr"], sig["confidence"], sig["score"], sig["reasons"], eff_risk, balance)
            time.sleep(2)
        else:
            time.sleep(0.1)

    log.info(f"\n{'═'*56}\nSCAN COMPLETE\n{'═'*56}")

if __name__ == "__main__":
    run_execution_scan()
