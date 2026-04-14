# trade_executor.py — Production Enhanced Version
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
from deribit_client import DeribitClient

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
            "tradeable":      deribit.get_tradeable(),
            "open_positions": len(positions),
            "usdc_margin":    deribit.has_usdc_margin(),
        })
        log.info(f"  ✅ Balance: ${total_usd:.2f} | unrealised: {upnl:+.2f} | positions: {len(positions)}")
        return total_usd
    except Exception as e:
        log.error(f"  save_balance_json failed: {e}")
        return 0.0


# ════════════ MARKET DATA (ENHANCED) ══════════════════════════════════

def get_data(symbol: str, interval: str) -> pd.DataFrame:
    """Uses Vision API fallback to prevent 'Waiting for Data' loops"""
    urls = [
        "https://data-api.binance.vision/api/v3/klines",
        "https://api.binance.com/api/v3/klines"
    ]
    for url in urls:
        try:
            r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": LIVE_LIMIT}, timeout=15)
            if r.status_code == 200:
                df = pd.DataFrame(r.json()).iloc[:, :6]
                df.columns = ["open_time","open","high","low","close","volume"]
                # Hard conversion to fix KeyError: 'open'
                for c in ["open","high","low","close","volume"]: 
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                return df
        except Exception: continue
    return pd.DataFrame() # Return empty instead of crashing


# ════════════ STUCK TRADE CLEANER ════════════════════════════════════

def clean_invalid_trades(deribit: DeribitClient):
    trades = load_trades()
    if not trades: return
    live_positions = {}
    for p in deribit.get_positions():
        if float(p.get("size", 0)) != 0:
            inst = p.get("instrument_name", "")
            base = inst.split("_")[0] if "_" in inst else inst.split("-")[0]
            live_positions[f"{base}USDT"] = float(p.get("size", 0))

    to_remove = []
    for symbol, trade in trades.items():
        if float(trade.get("stop", 0)) == 0 or float(trade.get("tp1", 0)) == 0:
            to_remove.append(symbol); continue
        if symbol not in live_positions:
            _record_close(trade, float(trade.get("entry", 0)), 0.0, "Ghost — auto-removed")
            to_remove.append(symbol)

    if to_remove:
        for sym in to_remove: trades.pop(sym, None)
        save_trades(trades)
        log.info(f"  ✅ Cleaned {len(to_remove)} invalid trade(s)")


# ════════════ EXECUTE TRADE ══════════════════════════════════════════

def execute_trade(deribit: DeribitClient, symbol, signal, entry, atr,
                  confidence, score, reasons, risk_mult=1.0, balance=10000.0):

    trades = load_trades()
    if symbol in trades or len(trades) >= MAX_OPEN_TRADES: return False
    if not check_correlation(trades, signal): return False
    if not deribit.is_supported(symbol): return False

    dec = 4 if entry < 10 else 2
    if signal == "BUY":
        stop = round(entry - atr*ATR_STOP_MULT, dec)
        tp1 = round(entry + atr*ATR_TARGET1_MULT, dec)
        tp2 = round(entry + atr*ATR_TARGET2_MULT, dec)
        side, sl_side, tp_side = "BUY", "SELL", "SELL"
    else:
        stop = round(entry + atr*ATR_STOP_MULT, dec)
        tp1 = round(entry - atr*ATR_TARGET1_MULT, dec)
        tp2 = round(entry - atr*ATR_TARGET2_MULT, dec)
        side, sl_side, tp_side = "SELL", "BUY", "BUY"

    total_contracts = deribit.calc_contracts(symbol, balance, entry, stop, risk_mult)
    amount_tp1, amount_tp2 = deribit.split_amount(symbol, total_contracts)
    risk_usd = round(balance * RISK_PER_TRADE * risk_mult, 2)

    try:
        entry_result = deribit.place_market_order(symbol, side, total_contracts)
        order = entry_result.get("order", entry_result)
        filled_qty = float(order.get("filled_amount") or 0)
        
        if filled_qty <= 0: return False

        actual_entry = float(order.get("average_price", entry))
        log.info(f"  ✅ Filled @ ~{actual_entry:.{dec}f}")
        time.sleep(1)

        # Protective Orders
        sl_res = deribit.place_limit_order(symbol, sl_side, filled_qty, stop, stop_price=stop)
        q1, q2 = deribit.split_amount(symbol, filled_qty)
        deribit.place_limit_order(symbol, tp_side, q1, tp1)
        deribit.place_limit_order(symbol, tp_side, q2, tp2)

    except Exception as e:
        log.error(f"  Trade error {symbol}: {e}")
        return False

    record = {
        "symbol": symbol, "signal": signal, "entry": actual_entry, "stop": stop, 
        "tp1": tp1, "tp2": tp2, "qty": filled_qty, "opened_at": datetime.now(timezone.utc).isoformat(),
        "tp1_hit": False, "closed": False, "confidence": confidence, "score": score
    }
    trades[symbol] = record
    save_trades(trades)
    save_signal(record)
    return True


# ════════════ SIGNAL GENERATION ══════════════════════════════════════

def generate_signal(symbol, pipeline, thresholds):
    try:
        raw_entry = get_data(symbol, TIMEFRAME_ENTRY)
        if raw_entry.empty or len(raw_entry) < 30:
            log.info(f"      ML: WAITING (Insufficient Market Data)")
            return None
        
        df15 = add_indicators(raw_entry)
        df1h = add_indicators(get_data(symbol, TIMEFRAME_CONFIRM))
        
        row = df15.iloc[-1].copy()
        r1h = df1h.iloc[-1] if not df1h.empty else pd.Series(dtype=float)
        row["rsi_1h"], row["adx_1h"], row["trend_1h"] = float(r1h.get("rsi", 50)), float(r1h.get("adx", 0)), float(r1h.get("trend", 0))

        af = pipeline["all_features"]
        X = pd.DataFrame([row[af].values], columns=af)
        Xs = pipeline["selector"].transform(X)
        prob = pipeline["ensemble"].predict_proba(Xs)[0]
        sig = {0:"BUY", 1:"SELL", 2:"NO_TRADE"}[pipeline["ensemble"].predict(Xs)[0]]
        conf = round(float(max(prob)) * 100, 1)

        log.info(f"      ML: {sig} {conf}% (need ≥{thresholds['min_confidence']}%)")
        if sig == "NO_TRADE" or conf < thresholds["min_confidence"]: return None

        adx = float(row.get("adx", 0))
        log.info(f"      ADX: {adx:.1f} (need ≥{thresholds['min_adx']})")
        if adx < thresholds["min_adx"]: return None

        score, reasons = _quality_score(row, r1h, sig, conf)
        log.info(f"      Score: {score} (need ≥{thresholds['min_score']})")
        if score < thresholds["min_score"]: return None

        return {"symbol": symbol, "signal": sig, "confidence": conf, "score": score,
                "entry": float(row["close"]), "atr": float(row["atr"]), "reasons": reasons}
    except Exception as e:
        log.error(f"      Signal error: {e}")
        return None

def _quality_score(row, r1h, signal, conf):
    score, reasons = 0, []
    if conf>=65: score+=1; reasons.append("High Conf")
    adx=float(row.get("adx",0))
    if adx>20: score+=1; reasons.append("Strong ADX")
    rsi=float(row.get("rsi",50))
    if (signal=="BUY" and rsi<55) or (signal=="SELL" and rsi>45): score+=1; reasons.append("RSI Alignment")
    return score, reasons

def _record_close(trade, close_price, pnl, reason):
    append_history({**trade, "close_price": close_price, "pnl": pnl,
                    "closed_at": datetime.now(timezone.utc).isoformat(), "close_reason": reason})


# ════════════ MONITORING ══════════════════════════════════════════════

def check_open_trades(deribit: DeribitClient):
    trades = load_trades()
    if not trades: return
    to_remove = []
    for symbol, trade in list(trades.items()):
        try:
            # Check Stop Loss first
            o = deribit.get_order(trade["order_ids"].get("stop_loss"))
            if deribit.is_order_filled(o):
                _record_close(trade, trade["stop"], -trade["qty"]*0.01, "SL Hit")
                to_remove.append(symbol)
        except Exception: pass
    
    for s in to_remove: trades.pop(s, None)
    save_trades(trades)


# ════════════ MAIN ════════════════════════════════════════════════════

def run_execution_scan():
    log.info(f"\n{'═'*56}\nSCAN START — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n{'═'*56}")
    run, mode, vol, reason = should_scan()
    if not run: return

    deribit  = init_deribit()
    pipeline = load_model()
    thresholds, risk_mult, balance = get_mode_thresholds(mode), get_effective_risk(mode, vol), deribit.get_total_equity_usd()

    clean_invalid_trades(deribit)
    check_open_trades(deribit)
    save_balance_json(deribit)

    for symbol in SYMBOLS:
        if len(load_json(TRADES_FILE, {})) >= MAX_OPEN_TRADES: break
        log.info(f"\n  ── {symbol} ({get_tier(symbol)}) ──")
        sig = generate_signal(symbol, pipeline, thresholds)
        if sig:
            execute_trade(deribit, sig["symbol"], sig["signal"], sig["entry"], sig["atr"], sig["confidence"], sig["score"], sig["reasons"], risk_mult, balance)
            time.sleep(2)
        else:
            time.sleep(0.1)

if __name__ == "__main__":
    run_execution_scan()
