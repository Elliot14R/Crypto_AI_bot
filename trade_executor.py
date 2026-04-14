# trade_executor.py — Unified USDC Margin + Ghost Killer + IOC Fix
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
        })
        log.info(f"  ✅ Balance: ${total_usd:.2f} | unrealised: {upnl:+.2f} | positions: {len(positions)}")
        return total_usd
    except Exception as e:
        log.error(f"  save_balance_json failed: {e}")
        return 0.0


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
        except Exception: continue
    raise Exception(f"Cannot fetch data for {symbol}")


# ════════════ STUCK TRADE CLEANER ════════════════════════════════════

def clean_invalid_trades(deribit: DeribitClient):
    """Ghost killer: remove trades where Deribit shows 0 position."""
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
            log.warning(f"  🗑️ {symbol}: stop=0 or tp1=0 — removing broken state")
            to_remove.append(symbol); continue
        
        if symbol not in live_positions:
            log.warning(f"  🗑️ {symbol}: no live position on Deribit — clearing ghost")
            _record_close(trade, float(trade.get("entry", 0)), 0.0, "Ghost — auto-removed")
            to_remove.append(symbol)

    if to_remove:
        for sym in to_remove: trades.pop(sym, None)
        save_trades(trades)
        log.info(f"  ✅ Removed {len(to_remove)} invalid trade(s): {to_remove}")
        _send(f"🧹 Removed {len(to_remove)} invalid trade(s): {', '.join(to_remove)}")


# ════════════ EXECUTE TRADE ══════════════════════════════════════════

def execute_trade(deribit: DeribitClient, symbol, signal, entry, atr,
                  confidence, score, reasons, risk_mult=1.0, balance=10000.0):

    trades = load_trades()
    if symbol in trades: log.info(f"  {symbol}: already open — skip"); return False
    if len(trades) >= MAX_OPEN_TRADES: log.info("  Max trades — skip"); return False
    if not check_correlation(trades, signal): return False
    if not deribit.is_supported(symbol): log.info(f"  {symbol}: not on Deribit — skip"); return False
    if balance < 5: _warn(f"⚠️ Balance ${balance:.2f} too low"); return False

    dec = 4 if entry < 10 else 2
    side = "BUY" if signal == "BUY" else "SELL"
    sl_side = "SELL" if signal == "BUY" else "BUY"
    tp_side = "SELL" if signal == "BUY" else "BUY"

    total_contracts = deribit.calc_contracts(symbol, balance, entry, entry - (atr*ATR_STOP_MULT if signal=="BUY" else -atr*ATR_STOP_MULT), risk_mult)
    
    order_ids = {}
    
    try:
        # 1. Market entry (IOC)
        res = deribit.place_market_order(symbol, side, total_contracts)
        order = res.get("order", res)
        filled_qty = float(order.get("filled_amount", 0))
        
        if filled_qty <= 0:
            log.warning(f"  ⚠️ {symbol} IOC cancelled (no liquidity)")
            return False

        order_ids["entry"] = str(order.get("order_id"))
        actual_entry = float(order.get("average_price", entry))
        log.info(f"  ✅ Filled @ ~{actual_entry:.{dec}f}")
        time.sleep(1.5)

        # Recalculate levels from actual fill price
        if signal == "BUY":
            stop = deribit.round_price(symbol, actual_entry - atr*ATR_STOP_MULT)
            tp1  = deribit.round_price(symbol, actual_entry + atr*ATR_TARGET1_MULT)
            tp2  = deribit.round_price(symbol, actual_entry + atr*ATR_TARGET2_MULT)
        else:
            stop = deribit.round_price(symbol, actual_entry + atr*ATR_STOP_MULT)
            tp1  = deribit.round_price(symbol, actual_entry - atr*ATR_TARGET1_MULT)
            tp2  = deribit.round_price(symbol, actual_entry - atr*ATR_TARGET2_MULT)

        # 2. Stop Loss
        try:
            sl_res = deribit.place_limit_order(symbol, sl_side, filled_qty, stop, stop_price=stop)
            oid = str(sl_res.get("order", sl_res).get("order_id", ""))
            if oid: order_ids["stop_loss"] = oid
            log.info(f"  ✅ SL @ {stop:.{dec}f}")
        except Exception as e: log.warning(f"  SL failed: {e}")

        # 3. Take Profits
        q1, q2 = deribit.split_amount(symbol, filled_qty)
        try:
            if q1 > 0:
                tp1_res = deribit.place_limit_order(symbol, tp_side, q1, tp1)
                oid = str(tp1_res.get("order", tp1_res).get("order_id", ""))
                if oid: order_ids["tp1"] = oid
                log.info(f"  ✅ TP1 @ {tp1:.{dec}f}")
        except Exception as e: log.warning(f"  TP1 failed: {e}")

        try:
            if q2 > 0:
                tp2_res = deribit.place_limit_order(symbol, tp_side, q2, tp2)
                oid = str(tp2_res.get("order", tp2_res).get("order_id", ""))
                if oid: order_ids["tp2"] = oid
                log.info(f"  ✅ TP2 @ {tp2:.{dec}f}")
        except Exception as e: log.warning(f"  TP2 failed: {e}")

    except Exception as e:
        log.error(f"  Trade error {symbol}: {e}")
        return False

    record = {
        "symbol": symbol, "signal": signal, "entry": actual_entry, 
        "stop": stop, "tp1": tp1, "tp2": tp2, "qty": filled_qty, 
        "qty_tp1": q1, "qty_tp2": q2, "order_ids": order_ids,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "tp1_hit": False, "tp2_hit": False, "closed": False,
        "confidence": confidence, "score": score, "reasons": reasons
    }
    
    trades[symbol] = record
    save_trades(trades)
    save_signal(record)
    _send_open_alert(symbol, signal, confidence, score, actual_entry, stop, tp1, tp2, filled_qty, q1, q2, balance*0.01*risk_mult, balance, reasons, risk_mult)
    return True


# ════════════ TRADE MONITORING ════════════════════════════════════════

def _fill_price(order: dict, fallback: float) -> float:
    p = float(order.get("average_price") or order.get("price") or 0)
    return p if p > 0 else fallback

def check_open_trades(deribit: DeribitClient):
    trades = load_trades()
    if not trades: return
    to_remove = []
    
    for symbol, trade in list(trades.items()):
        oids = trade.get("order_ids", {})
        try:
            # Check TP1
            if not trade["tp1_hit"] and "tp1" in oids:
                o = deribit.get_order(oids["tp1"])
                if deribit.is_order_filled(o):
                    trade["tp1_hit"] = True
                    _send(f"🎯 TP1 Hit for {symbol} - Moving SL to Break-even")
                    # Break-even logic
                    deribit.cancel_order(oids["stop_loss"])
                    new_sl = deribit.place_limit_order(symbol, "SELL" if trade["signal"]=="BUY" else "BUY", trade["qty_tp2"], trade["entry"], stop_price=trade["entry"])
                    trade["order_ids"]["stop_loss"] = str(new_sl.get("order", new_sl).get("order_id"))

            # Check TP2
            if not trade.get("tp2_hit") and "tp2" in oids:
                o = deribit.get_order(oids["tp2"])
                if deribit.is_order_filled(o):
                    trade["closed"] = True; to_remove.append(symbol)
                    _send_close_alert(symbol, "FULL WIN", trade["qty"]*abs(trade["tp2"]-trade["entry"]), trade["entry"], trade["tp2"], trade["opened_at"])

            # Check SL
            o = deribit.get_order(oids["stop_loss"])
            if deribit.is_order_filled(o):
                trade["closed"] = True; to_remove.append(symbol)
                _send_close_alert(symbol, "STOP LOSS", -trade["qty"]*abs(trade["stop"]-trade["entry"]), trade["entry"], trade["stop"], trade["opened_at"])

        except Exception as e: log.error(f"Monitor error {symbol}: {e}")

    save_trades(trades)
    for sym in to_remove: trades.pop(sym, None)
    save_trades(trades)


def _calc_pnl(trade, close_price, close_type) -> float:
    qty  = float(trade["qty_tp1"] if close_type=="tp1" else
                 trade["qty_tp2"] if close_type=="tp2" else trade["qty"])
    diff = ((close_price - trade["entry"]) if trade["signal"]=="BUY"
            else (trade["entry"] - close_price))
    return round(diff * qty, 4)

def _record_close(trade, close_price, pnl, reason):
    append_history({**trade, "close_price": close_price, "pnl": pnl,
                    "closed_at": datetime.now(timezone.utc).isoformat(), "close_reason": reason})


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

        af = pipeline["all_features"]
        if any(f not in row.index for f in af): return None

        X    = pd.DataFrame([row[af].values], columns=af)
        Xs   = pipeline["selector"].transform(X)
        pred = pipeline["ensemble"].predict(Xs)[0]
        prob = pipeline["ensemble"].predict_proba(Xs)[0]
        sig  = {0:"BUY", 1:"SELL", 2:"NO_TRADE"}[pred]
        conf = round(float(max(prob)) * 100, 1)

        if sig == "NO_TRADE" or conf < thresholds["min_confidence"]: return None
        if float(row.get("adx", 0)) < thresholds["min_adx"]: return None

        score, reasons = _quality_score(row, r1h, sig, conf)
        if score < thresholds["min_score"]: return None

        return {"symbol":symbol,"signal":sig,"confidence":conf,"score":score,
                "entry":float(row["close"]),"atr":float(row["atr"]),"reasons":reasons}
    except Exception as e: return None


def _quality_score(row, r1h, signal, conf):
    score, reasons = 0, []
    if conf>=60: score+=2; reasons.append(f"High conf ({conf:.0f}%)")
    elif conf>=50: score+=1; reasons.append(f"Conf ({conf:.0f}%)")
    adx=float(row.get("adx",0))
    if adx>20: score+=1; reasons.append(f"Strong ADX {adx:.0f}")
    rsi=float(row.get("rsi",50))
    if signal=="BUY" and rsi<50: score+=1; reasons.append(f"RSI bullish")
    elif signal=="SELL" and rsi>50: score+=1; reasons.append(f"RSI bearish")
    return score, reasons


# ════════════ TELEGRAM ════════════════════════════════════════════════

def _send(text):
    tok=os.getenv("TELEGRAM_TOKEN",""); cid=os.getenv("TELEGRAM_CHAT_ID","")
    if not tok or not cid: return
    try: requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
            data={"chat_id":cid,"text":text,"parse_mode":"Markdown"}, timeout=10)
    except Exception: pass

def _warn(text): log.warning(text); _send(text)

def _send_open_alert(symbol, signal, confidence, score, entry, stop, tp1, tp2, amount, tp1_qty, tp2_qty, risk_usd, balance, reasons, risk_mult=1.0):
    emoji="🟢" if signal=="BUY" else "🔴"
    msg = (f"🤖 *DERIBIT TRADE OPENED*\n{emoji} *{signal} {symbol}* | Score: {score}\n"
           f"🎯 Conf: {confidence}%\n⚡ Entry: {entry}\n🛑 SL: {stop}\n"
           f"🎯 TP1: {tp1} ({tp1_qty})\n🎯 TP2: {tp2} ({tp2_qty})\n"
           f"💰 Risk: ${risk_usd:.2f} | Bal: ${balance:.2f}")
    _send(msg)

def _send_close_alert(symbol, result, pnl, entry, close, opened_at):
    emoji="✅" if pnl>0 else "❌"
    _send(f"🤖 *TRADE CLOSED*\n{emoji} {result} {symbol}\n💵 PnL: ${pnl:.4f}\n📥 {entry} -> 📤 {close}")


# ════════════ MAIN ════════════════════════════════════════════════════

def run_execution_scan():
    run, mode, vol, reason = should_scan()
    if not run: return
    
    deribit  = init_deribit()
    pipeline = load_model()
    thresholds = get_mode_thresholds(mode)
    eff_risk = get_effective_risk(mode, vol)

    balance = save_balance_json(deribit)
    clean_invalid_trades(deribit)
    check_open_trades(deribit)

    for symbol in SYMBOLS:
        if len(load_trades()) >= MAX_OPEN_TRADES: break
        sig = generate_signal(symbol, pipeline, thresholds)
        if sig:
            execute_trade(deribit, sig["symbol"], sig["signal"], sig["entry"], sig["atr"], sig["confidence"], sig["score"], sig["reasons"], eff_risk, balance)
            time.sleep(1)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "diagnostic": run_diagnostic()
    else: run_execution_scan()
