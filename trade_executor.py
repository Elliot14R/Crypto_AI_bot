import os, json, time, logging, requests, joblib
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from config import SYMBOLS, ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT, MODEL_FILE, LOG_FILE, get_tier, TIMEFRAME_ENTRY, TIMEFRAME_CONFIRM
from deribit_client import DeribitClient
from feature_engineering import add_indicators
from smart_scheduler import should_scan, get_mode_thresholds, get_effective_risk

TRADES_FILE = "trades.json"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
log = logging.getLogger(__name__)

def load_json(path, default):
    return json.load(open(path)) if Path(path).exists() else default

def save_json(path, data):
    json.dump(data, open(path, "w"), indent=2, default=str)

def get_data(symbol: str, interval: str) -> pd.DataFrame:
    """Hardened data fetcher to fix KeyError 'open'"""
    url = "https://api.binance.com/api/v3/klines"
    for _ in range(3):
        try:
            r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": 100}, timeout=10)
            if r.status_code == 200:
                df = pd.DataFrame(r.json()).iloc[:, :6]
                df.columns = ["open_time","open","high","low","close","volume"]
                # Convert to numeric immediately to satisfy feature_engineering
                for c in ["open","high","low","close","volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                return df
        except: time.sleep(1)
    return pd.DataFrame()

def _quality_score(row, r1h, signal, conf):
    score, reasons = 0, []
    if conf >= 65: score += 1; reasons.append(f"High conf ({conf:.0f}%)")
    adx = float(row.get("adx", 0))
    if adx > 20: score += 1; reasons.append(f"Strong ADX {adx:.0f}")
    rsi = float(row.get("rsi", 50))
    if signal == "BUY" and rsi < 50: score += 1; reasons.append("RSI bullish")
    elif signal == "SELL" and rsi > 50: score += 1; reasons.append("RSI bearish")
    return score, reasons

def generate_signal(symbol, pipeline, thresholds):
    try:
        raw_entry = get_data(symbol, TIMEFRAME_ENTRY)
        raw_confirm = get_data(symbol, TIMEFRAME_CONFIRM)
        
        if raw_entry.empty or len(raw_entry) < 30: return None
        
        df15 = add_indicators(raw_entry)
        df1h = add_indicators(raw_confirm)

        row = df15.iloc[-1].copy()
        r1h = df1h.iloc[-1] if not df1h.empty else pd.Series(dtype=float)
        
        af = pipeline["all_features"]
        row["rsi_1h"], row["adx_1h"], row["trend_1h"] = float(r1h.get("rsi", 50)), float(r1h.get("adx", 0)), float(r1h.get("trend", 0))
        
        X = pd.DataFrame([row[af].values], columns=af)
        Xs = pipeline["selector"].transform(X)
        prob = pipeline["ensemble"].predict_proba(Xs)[0]
        sig = {0: "BUY", 1: "SELL", 2: "NO_TRADE"}[pipeline["ensemble"].predict(Xs)[0]]
        conf = round(float(max(prob)) * 100, 1)

        # 🟢 RESTORED DETAILED LOGS
        log.info(f"      ML: {sig} {conf}% (need ≥{thresholds['min_confidence']}%)")
        if sig == "NO_TRADE" or conf < thresholds["min_confidence"]: return None

        adx = float(row.get("adx", 0))
        log.info(f"      ADX: {adx:.1f} (need ≥{thresholds['min_adx']})")
        if adx < thresholds["min_adx"]: return None

        score, reasons = _quality_score(row, r1h, sig, conf)
        log.info(f"      Score: {score} (need ≥{thresholds['min_score']})")
        if score < thresholds["min_score"]: return None

        return {"symbol": symbol, "signal": sig, "confidence": conf, "score": score, "entry": float(row["close"]), "atr": float(row["atr"]), "reasons": reasons}
    except Exception as e:
        log.error(f"      Signal Error: {e}")
        return None

def execute_trade(deribit, symbol, signal, entry, atr, confidence, score, reasons, risk_mult, balance):
    trades = load_json(TRADES_FILE, {})
    if symbol in trades: return False
    
    side, sl_side, tp_side = ("BUY", "SELL", "SELL") if signal == "BUY" else ("SELL", "BUY", "BUY")
    target_q = deribit.calc_contracts(symbol, balance, entry, entry - (atr*ATR_STOP_MULT if signal=="BUY" else -atr*ATR_STOP_MULT), risk_mult)

    try:
        res = deribit.place_market_order(symbol, signal, target_q)
        order = res.get("order", res)
        filled = float(order.get("filled_amount", 0))
        
        if filled > 0:
            actual = float(order.get("average_price", entry))
            stop = deribit.round_price(symbol, actual - (atr*ATR_STOP_MULT if signal=="BUY" else -atr*ATR_STOP_MULT))
            tp1 = deribit.round_price(symbol, actual + (atr*ATR_TARGET1_MULT if signal=="BUY" else -atr*ATR_TARGET1_MULT))
            
            sl_res = deribit.place_limit_order(symbol, sl_side, filled, stop, stop_price=stop)
            q1, q2 = deribit.split_amount(symbol, filled)
            tp1_res = deribit.place_limit_order(symbol, tp_side, q1, tp1)

            trades[symbol] = {"symbol": symbol, "signal": signal, "entry": actual, "qty": filled, "tp1_hit": False}
            save_json(TRADES_FILE, trades)
            log.info(f"      ✅ Trade Live: {symbol} @ {actual}")
            return True
    except Exception as e: log.error(f"      ❌ Trade Failed: {e}")
    return False

def run_execution_scan():
    log.info(f"\n{'═'*56}\nSCAN START — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n{'═'*56}")
    run, mode, vol, reason = should_scan()
    if not run: log.info(f"  SKIPPED: {reason}"); return

    deribit = DeribitClient(os.getenv("DERIBIT_CLIENT_ID"), os.getenv("DERIBIT_CLIENT_SECRET"))
    pipeline = joblib.load(MODEL_FILE)
    thresholds, risk_mult, balance = get_mode_thresholds(mode), get_effective_risk(mode, vol), deribit.get_total_equity_usd()

    log.info(f"Scanning {len(SYMBOLS)} coins | Open: {len(load_json(TRADES_FILE, {}))}/3")

    for symbol in SYMBOLS:
        if len(load_json(TRADES_FILE, {})) >= 3: break
        log.info(f"\n  ── Checking {symbol} ({get_tier(symbol)}) ──")
        sig = generate_signal(symbol, pipeline, thresholds)
        if sig:
            execute_trade(deribit, sig["symbol"], sig["signal"], sig["entry"], sig["atr"], sig["confidence"], sig["score"], sig["reasons"], risk_mult, balance)
            time.sleep(2)
        else:
            time.sleep(0.5)

    log.info(f"\n{'═'*56}\nSCAN COMPLETE\n{'═'*56}")

if __name__ == "__main__":
    run_execution_scan()
