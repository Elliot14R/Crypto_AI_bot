# trade_executor.py — Delta Exchange + Paper Trading Fallback
# ══════════════════════════════════════════════════════════════════════
# STRATEGY: Try Delta Exchange first. If IP not whitelisted (401), 
# automatically fall back to Paper Trading so bot never stops working.
#
# HOW TO GET DELTA WORKING:
#   Delta Exchange requires IP whitelisting for Trading API keys.
#   GitHub Actions uses rotating IPs — you CANNOT whitelist them all.
#   Solutions:
#     1. Paper Trading (current fallback) — works perfectly, zero setup
#     2. Render.com paid tier (static IP) — whitelist once, works forever
#     3. VPS with static IP — best for production
#
# Market data:  Binance public API (never geo-blocked anywhere)
# Orders:       Delta Exchange testnet OR Paper Trading (auto-detected)
# ══════════════════════════════════════════════════════════════════════

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

TRADES_FILE     = "trades.json"
HISTORY_FILE    = "trade_history.json"
SIGNALS_FILE    = "signals.json"
MODE_FILE       = "scan_mode.json"
BALANCE_FILE    = "balance.json"
MAX_OPEN_TRADES = 3

# ── Trading mode detection ───────────────────────────────────────────
# AUTO: tries Delta first, falls back to Paper if IP not whitelisted
# PAPER: forces paper trading (set TRADING_MODE=paper in .env)
# DELTA: forces Delta (will error if IP not whitelisted)
TRADING_MODE = os.getenv("TRADING_MODE", "auto").lower()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
log = logging.getLogger(__name__)


# ════════════ HELPERS ════════════════════════════════════════════════

def load_json(p, d):
    try:
        for path in [Path(p), Path("data") / p]:
            if path.exists():
                with open(path) as f: return json.load(f)
    except: pass
    return d

def save_json(p, data):
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as f: json.dump(data, f, indent=2, default=str)
    os.replace(tmp, p)

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


# ════════════ EXCHANGE SETUP ═════════════════════════════════════════

def setup_exchange():
    """
    Returns (client, mode_str) where mode_str is 'delta' or 'paper'.
    Auto-detects which mode works based on IP whitelist status.
    """
    global TRADING_MODE

    if TRADING_MODE == "paper":
        log.info("📝 PAPER TRADING MODE (forced by TRADING_MODE=paper)")
        from paper_trader import PaperTrader
        return PaperTrader(), "paper"

    # Try Delta first (or if forced)
    try:
        from delta_client import DeltaClient
        key    = (os.getenv("DELTA_API_KEY", "")    or os.getenv("BINANCE_API_KEY", ""))
        secret = (os.getenv("DELTA_API_SECRET", "") or os.getenv("BINANCE_SECRET",  ""))

        if not key or not secret:
            raise ValueError("No API keys found — check DELTA_API_KEY and DELTA_API_SECRET")

        client = DeltaClient(key, secret)

        # Test connection — this will raise if IP not whitelisted
        bal = client.get_usdt_balance()
        log.info(f"✅ DELTA EXCHANGE TESTNET — Balance: {bal:.2f} USDT")
        return client, "delta"

    except Exception as e:
        err_str = str(e).lower()
        if "ip_not_whitelisted" in err_str or "401" in err_str:
            log.warning(
                f"⚠️ Delta Exchange IP not whitelisted ({e})\n"
                f"   Falling back to PAPER TRADING automatically.\n"
                f"   To use Delta: whitelist your server's static IP in Delta API settings."
            )
            _send(
                f"⚠️ *Delta IP not whitelisted*\n"
                f"Falling back to Paper Trading automatically.\n"
                f"Bot continues running with real market data."
            )
        elif TRADING_MODE == "delta":
            raise   # If forced delta mode, re-raise the error
        else:
            log.warning(f"Delta connection failed: {e} — using Paper Trading")

    # Fallback to paper trading
    log.info("📝 PAPER TRADING MODE (Delta unavailable — IP not whitelisted)")
    from paper_trader import PaperTrader
    return PaperTrader(), "paper"


# ════════════ BALANCE ════════════════════════════════════════════════

def fetch_and_save_balance(client, mode: str) -> float:
    """Fetch balance from Delta or Paper Trader and save to balance.json."""
    try:
        if mode == "delta":
            from delta_client import DeltaClient
            balances = client.get_wallet_balance()
            usdt     = float(balances.get("USDT", 0))
            assets   = [
                {"asset": a, "free": round(float(v), 4), "total": round(float(v), 4)}
                for a, v in balances.items() if float(v) > 0
            ]
            save_json(BALANCE_FILE, {
                "usdt":       round(usdt, 2),
                "equity":     round(usdt, 2),
                "unrealised": 0.0,
                "assets":     assets,
                "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "mode":       "delta_testnet",
                "source":     "delta_exchange_india",
            })
            log.info(f"✓ Balance: {usdt:.2f} USDT (Delta Testnet)")

        else:  # paper
            usdt = client.save_balance_snapshot()

        return round(usdt, 2)

    except Exception as e:
        log.error(f"Balance fetch failed: {e}")
        # Write error state so dashboard shows something useful
        bal_now = load_json(BALANCE_FILE, {})
        if not bal_now or bal_now.get("usdt") is None:
            save_json(BALANCE_FILE, {
                "usdt":       10000.0,
                "equity":     10000.0,
                "unrealised": 0.0,
                "assets":     [{"asset":"USDT","free":"10000.0","total":"10000.0"}],
                "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "mode":       "paper_trading",
                "note":       "Starting balance — Delta exchange unavailable",
            })
            return 10000.0
        return float(bal_now.get("usdt") or 0)


# ════════════ MARKET DATA ════════════════════════════════════════════

def get_data(symbol, interval):
    """Binance public klines — never geo-blocked."""
    endpoints = [
        "https://data-api.binance.vision/api/v3/klines",
        "https://api.binance.com/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
    ]
    params   = {"symbol": symbol, "interval": interval, "limit": LIVE_LIMIT}
    last_err = None
    for url in endpoints:
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            df = pd.DataFrame(r.json()).iloc[:, :6]
            df.columns = ["open_time","open","high","low","close","volume"]
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c])
            return df
        except Exception as e:
            last_err = e
    raise last_err


def load_model():
    p = joblib.load(MODEL_FILE)
    for k in ["ensemble", "selector", "all_features", "label_map"]:
        if k not in p: raise ValueError(f"Model missing key: {k}")
    log.info(f"✓ Model: {len(p['all_features'])} features")
    return p


def calc_pos_size(balance, entry, stop, risk_mult=1.0):
    effective_risk = RISK_PER_TRADE * risk_mult
    dist           = abs(entry - stop)
    if dist <= 0: return 0.0, 0.0
    qty     = (balance * effective_risk) / dist
    max_usd = balance * 0.20
    if qty * entry > max_usd:
        qty = max_usd / entry
    return round(qty, 6), round(balance * effective_risk, 2)


# ════════════ EXECUTE TRADE ══════════════════════════════════════════

def execute_trade(client, mode: str, symbol, signal, entry, atr,
                  confidence, score, reasons, risk_mult=1.0):

    trades = load_trades()
    if symbol in trades:
        log.info(f"  {symbol} already open — skip"); return False
    if len(trades) >= MAX_OPEN_TRADES:
        log.info("  Max open trades reached — skip"); return False
    if not check_correlation(trades, signal):
        return False

    dec = 4 if entry < 10 else 2
    if signal == "BUY":
        stop    = round(entry - atr * ATR_STOP_MULT,    dec)
        tp1     = round(entry + atr * ATR_TARGET1_MULT, dec)
        tp2     = round(entry + atr * ATR_TARGET2_MULT, dec)
        side    = "buy";  sl_side = "sell"; tp_side = "sell"
    else:
        stop    = round(entry + atr * ATR_STOP_MULT,    dec)
        tp1     = round(entry - atr * ATR_TARGET1_MULT, dec)
        tp2     = round(entry - atr * ATR_TARGET2_MULT, dec)
        side    = "sell"; sl_side = "buy";  tp_side = "buy"

    balance = fetch_and_save_balance(client, mode)
    if balance < 10:
        _warn(f"⚠️ Balance {balance:.2f} USDT too low — skip {symbol}")
        return False

    qty, risk_usd = calc_pos_size(balance, entry, stop, risk_mult)
    if qty <= 0:
        log.warning(f"  Position size zero for {symbol}"); return False

    qty_tp1 = round(qty * 0.5, 6)
    qty_tp2 = round(qty - qty_tp1, 6)

    log.info(f"  [{mode.upper()}] {signal} {symbol} qty={qty} "
             f"SL={stop:.{dec}f} TP1={tp1:.{dec}f} TP2={tp2:.{dec}f}")

    order_ids   = {}
    actual_entry = entry

    try:
        if mode == "delta":
            contracts     = client.calc_contracts(balance, entry, stop, risk_mult)
            contracts_tp1 = max(1, contracts // 2)
            contracts_tp2 = max(1, contracts - contracts_tp1)

            # Entry
            eo = client.place_market_order(symbol, side, contracts)
            if not eo: log.error(f"  Entry failed {symbol}"); return False
            order_ids["entry"] = str(eo.get("id", ""))
            actual_entry = float(eo.get("average_fill_price", entry) or entry) or entry
            log.info(f"  ✅ Entry @ ~{actual_entry:.{dec}f}")
            time.sleep(1.5)

            # SL
            try:
                sl_o = client.place_limit_order(symbol, sl_side, contracts,
                                                price=stop, stop_price=stop)
                if sl_o: order_ids["stop_loss"] = str(sl_o.get("id",""))
                log.info(f"  ✅ SL @ {stop:.{dec}f}")
            except Exception as e:
                log.warning(f"  SL failed: {e}")

            # TP1
            try:
                tp1_o = client.place_limit_order(symbol, tp_side, contracts_tp1, price=tp1)
                if tp1_o: order_ids["tp1"] = str(tp1_o.get("id",""))
                log.info(f"  ✅ TP1 @ {tp1:.{dec}f}")
            except Exception as e:
                log.warning(f"  TP1 failed: {e}")

            # TP2
            try:
                tp2_o = client.place_limit_order(symbol, tp_side, contracts_tp2, price=tp2)
                if tp2_o: order_ids["tp2"] = str(tp2_o.get("id",""))
                log.info(f"  ✅ TP2 @ {tp2:.{dec}f}")
            except Exception as e:
                log.warning(f"  TP2 failed: {e}")

        else:  # paper
            # Market entry
            eo = client.place_market_order(symbol, side.upper(), qty)
            order_ids["entry"] = str(eo.get("orderId", ""))
            actual_entry = float(eo.get("paper_fill", entry) or entry)
            log.info(f"  📝 PAPER entry @ {actual_entry:.{dec}f}")

            # SL and TP as limit orders (checked against live price each scan)
            sl_o  = client.place_limit_order(symbol, sl_side.upper(), qty_tp1+qty_tp2,
                                              price=stop, stop_price=stop)
            tp1_o = client.place_limit_order(symbol, tp_side.upper(), qty_tp1, price=tp1)
            tp2_o = client.place_limit_order(symbol, tp_side.upper(), qty_tp2, price=tp2)
            order_ids["stop_loss"] = str(sl_o.get("orderId",  ""))
            order_ids["tp1"]       = str(tp1_o.get("orderId", ""))
            order_ids["tp2"]       = str(tp2_o.get("orderId", ""))

    except Exception as e:
        log.error(f"  Trade error {symbol}: {e}")
        _warn(f"⚠️ Trade error {symbol}: {e}")
        return False

    record = {
        "symbol": symbol, "signal": signal,
        "entry": actual_entry, "stop": stop, "tp1": tp1, "tp2": tp2,
        "qty": qty, "qty_tp1": qty_tp1, "qty_tp2": qty_tp2,
        "risk_usd": risk_usd, "risk_mult": risk_mult,
        "balance_at_open": balance,
        "order_ids": order_ids,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "tp1_hit": False, "tp2_hit": False, "closed": False,
        "confidence": confidence, "score": score,
        "reasons": reasons, "tier": get_tier(symbol),
        "exchange": mode,
    }
    trades[symbol] = record
    save_trades(trades)
    save_signal(record)
    _send_open_alert(symbol, signal, confidence, score, actual_entry,
                     stop, tp1, tp2, qty, risk_usd, balance, reasons, mode)
    log.info(f"  ✅✅ TRADE OPENED: {symbol} {signal} [{mode.upper()}]")
    return True


# ════════════ MONITOR OPEN TRADES ════════════════════════════════════

def check_open_trades(client, mode: str):
    trades    = load_trades()
    if not trades:
        log.info("  No open trades"); return

    to_remove = []
    log.info(f"  Monitoring {len(trades)} open trade(s) [{mode.upper()}]")

    for symbol, trade in list(trades.items()):
        if trade.get("closed"):
            to_remove.append(symbol); continue

        oids  = trade.get("order_ids", {})
        entry = float(trade["entry"])
        dec   = 4 if entry < 10 else 2

        try:
            if mode == "paper":
                # Paper: use PaperTrader.get_order which checks live price
                def get_o(key):
                    if key not in oids: return {}
                    return client.get_order(symbol, oids[key])
            else:
                # Delta: fetch real order status
                def get_o(key):
                    if key not in oids: return {}
                    o = client.get_order(oids[key])
                    return o

            # ── Check TP1 ─────────────────────────────────────────────
            if not trade["tp1_hit"] and "tp1" in oids:
                o      = get_o("tp1")
                status = o.get("state", o.get("status", ""))
                if status in ("filled", "closed", "FILLED"):
                    trade["tp1_hit"] = True
                    avg = float(o.get("average_fill_price",
                                     o.get("price", trade["tp1"])) or trade["tp1"])
                    pnl = _pnl(trade, avg, "tp1")
                    log.info(f"  🎯 TP1 HIT {symbol} @ {avg:.{dec}f} | pnl={pnl:+.4f}")
                    _send_close_alert(symbol, "TP1 HIT 🎯", pnl, entry, avg,
                                      trade["opened_at"], mode)
                    if mode == "paper":
                        client.update_balance_after_close(pnl)

                    # Move SL to breakeven
                    if "stop_loss" in oids and mode != "paper":
                        try:
                            client.cancel_order(symbol, oids["stop_loss"])
                            sl_side = "sell" if trade["signal"] == "BUY" else "buy"
                            qty_rem = max(1, trade.get("contracts", 1) // 2)
                            new_sl  = client.place_limit_order(symbol, sl_side, qty_rem,
                                                               price=entry, stop_price=entry)
                            if new_sl:
                                trade["order_ids"]["stop_loss"] = str(new_sl.get("id",""))
                                trade["stop"] = entry
                                log.info(f"  🛡️ SL → breakeven @ {entry:.{dec}f}")
                                _send(f"🛡️ *{symbol} RISK-FREE!*\nSL moved to entry `{entry:.{dec}f}`")
                        except Exception as e:
                            log.warning(f"  SL move failed: {e}")

            # ── Check TP2 ─────────────────────────────────────────────
            if trade["tp1_hit"] and not trade["tp2_hit"] and "tp2" in oids:
                o      = get_o("tp2")
                status = o.get("state", o.get("status", ""))
                if status in ("filled", "closed", "FILLED"):
                    trade["tp2_hit"] = True
                    trade["closed"]  = True
                    avg = float(o.get("average_fill_price",
                                     o.get("price", trade["tp2"])) or trade["tp2"])
                    pnl = _pnl(trade, avg, "tp2")
                    log.info(f"  ✅ TP2 HIT {symbol} @ {avg:.{dec}f} | pnl={pnl:+.4f}")
                    _send_close_alert(symbol, "✅ FULL WIN (TP2)", pnl, entry, avg,
                                      trade["opened_at"], mode)
                    _record_close(trade, avg, pnl, "TP2 hit")
                    if mode == "paper": client.update_balance_after_close(pnl)
                    to_remove.append(symbol)

            # ── Check SL ──────────────────────────────────────────────
            if not trade.get("closed") and "stop_loss" in oids:
                o      = get_o("stop_loss")
                status = o.get("state", o.get("status", ""))
                if status in ("filled", "closed", "FILLED"):
                    trade["closed"] = True
                    avg = float(o.get("average_fill_price",
                                     o.get("price", trade["stop"])) or trade["stop"])
                    pnl = _pnl(trade, avg, "sl")
                    log.info(f"  ❌ SL HIT {symbol} @ {avg:.{dec}f} | pnl={pnl:+.4f}")
                    _send_close_alert(symbol, "❌ STOPPED OUT", pnl, entry, avg,
                                      trade["opened_at"], mode)
                    _record_close(trade, avg, pnl, "SL hit")
                    if mode == "paper": client.update_balance_after_close(pnl)
                    for k in ("tp1","tp2"):
                        if k in oids and not trade.get(f"{k}_hit"):
                            try: client.cancel_order(symbol, oids[k])
                            except: pass
                    to_remove.append(symbol)

        except Exception as e:
            log.error(f"  Monitor error {symbol}: {e}")

    save_trades(trades)
    for sym in set(to_remove):
        trades.pop(sym, None)
    save_trades(trades)


def clear_stuck_trades(client, mode: str):
    """Clear trades.json entries with no live orders."""
    trades = load_trades()
    if not trades: return

    log.info(f"  🔄 Checking {len(trades)} trade(s) for stuck orders...")
    cleared = 0

    for symbol in list(trades.keys()):
        trade    = trades[symbol]
        oids     = trade.get("order_ids", {})
        found    = False

        for key, oid in oids.items():
            if key == "entry": continue
            try:
                if mode == "delta":
                    o      = client.get_order(oid)
                    status = o.get("state", o.get("status", ""))
                    if status in ("open", "pending"):
                        found = True; break
                else:
                    o      = client.get_order(symbol, oid)
                    status = o.get("status", "")
                    if status == "NEW":
                        found = True; break
            except Exception:
                pass

        if not found and oids:
            log.warning(f"  ⚠️ {symbol}: no live orders — clearing")
            _record_close(trade, float(trade.get("entry",0)), 0.0, "Auto-cleared")
            trades.pop(symbol)
            cleared += 1

    if cleared:
        save_trades(trades)
        log.info(f"  ✅ Cleared {cleared} stuck trade(s)")
        _send(f"🧹 *{cleared} stuck trade(s) cleared*")
    else:
        log.info("  ✓ No stuck trades")


def _pnl(trade, close_price, close_type):
    entry = trade["entry"]
    qty   = (trade["qty_tp1"] if close_type == "tp1" else
             trade["qty_tp2"] if close_type == "tp2" else trade["qty"])
    return round(
        (close_price - entry) * qty if trade["signal"] == "BUY"
        else (entry - close_price) * qty, 4
    )

def _record_close(trade, close_price, pnl, reason):
    append_history({
        **trade, "close_price": close_price, "pnl": pnl,
        "closed_at": datetime.now(timezone.utc).isoformat(), "close_reason": reason,
    })


# ════════════ SIGNAL GENERATION ══════════════════════════════════════

def generate_signal(symbol, pipeline, thresholds):
    try:
        df_e = add_indicators(get_data(symbol, TIMEFRAME_ENTRY))
        df_c = add_indicators(get_data(symbol, TIMEFRAME_CONFIRM))
        if df_e.empty or len(df_e) < 50: return None

        row_e = df_e.iloc[-1].copy()
        row_c = df_c.iloc[-1] if not df_c.empty else pd.Series(dtype=float)
        row_e["rsi_1h"]   = float(row_c.get("rsi",  50))
        row_e["adx_1h"]   = float(row_c.get("adx",   0))
        row_e["trend_1h"] = float(row_c.get("trend", 0))

        af   = pipeline["all_features"]
        miss = [f for f in af if f not in row_e.index]
        if miss: log.warning(f"    Missing features: {miss[:3]}"); return None

        X    = pd.DataFrame([row_e[af].values], columns=af)
        Xs   = pipeline["selector"].transform(X)
        pred = pipeline["ensemble"].predict(Xs)[0]
        prob = pipeline["ensemble"].predict_proba(Xs)[0]
        sig  = {0:"BUY", 1:"SELL", 2:"NO_TRADE"}[pred]
        conf = round(float(max(prob)) * 100, 1)

        log.info(f"    ML: {sig} {conf:.1f}% (need ≥{thresholds['min_confidence']}%)")
        if sig == "NO_TRADE": return None
        if conf < thresholds["min_confidence"]: return None

        adx = float(row_e.get("adx", 0))
        log.info(f"    ADX: {adx:.1f} (need ≥{thresholds['min_adx']})")
        if adx < thresholds["min_adx"]: return None

        score, reasons = _quality_score(row_e, row_c, sig, conf)
        log.info(f"    Score: {score}/5 (need ≥{thresholds['min_score']})")

        entry = float(row_e["close"])
        atr   = float(row_e["atr"])
        dec   = 4 if entry < 10 else 2

        if sig == "BUY":
            stop = round(entry - atr * ATR_STOP_MULT,    dec)
            tp1  = round(entry + atr * ATR_TARGET1_MULT, dec)
            tp2  = round(entry + atr * ATR_TARGET2_MULT, dec)
        else:
            stop = round(entry + atr * ATR_STOP_MULT,    dec)
            tp1  = round(entry - atr * ATR_TARGET1_MULT, dec)
            tp2  = round(entry - atr * ATR_TARGET2_MULT, dec)

        if score < thresholds["min_score"]:
            save_signal({"symbol": symbol, "signal": sig, "confidence": conf,
                "score": score, "entry": entry, "atr": atr,
                "stop": stop, "tp1": tp1, "tp2": tp2, "reasons": reasons,
                "rejected": True, "reject_reason": f"score {score} < {thresholds['min_score']}"})
            return None

        return {"symbol": symbol, "signal": sig, "confidence": conf,
                "score": score, "entry": entry, "atr": atr,
                "stop": stop, "tp1": tp1, "tp2": tp2, "reasons": reasons}

    except requests.exceptions.HTTPError as e:
        log.warning(f"    HTTP error {symbol}: {e}"); return None
    except Exception as e:
        log.error(f"    Signal error {symbol}: {e}"); return None


def _quality_score(row_e, row_c, signal, confidence):
    s, r = 0, []
    if confidence >= 70:   s+=1; r.append(f"High confidence ({confidence:.0f}%)")
    elif confidence >= 55: s+=1; r.append(f"AI confidence ({confidence:.0f}%)")

    adx = float(row_e.get("adx", 0))
    if adx > 20:   s+=1; r.append(f"Strong trend ADX {adx:.0f}")
    elif adx > 15: s+=1; r.append(f"Moderate trend ADX {adx:.0f}")

    rsi = float(row_e.get("rsi", 50))
    if signal == "BUY"  and rsi < 50: s+=1; r.append(f"RSI bullish ({rsi:.0f})")
    elif signal == "SELL" and rsi > 50: s+=1; r.append(f"RSI bearish ({rsi:.0f})")

    e20, e50 = float(row_e.get("ema20",0)), float(row_e.get("ema50",0))
    if signal == "BUY"  and e20 > e50: s+=1; r.append("EMA20 > EMA50")
    elif signal == "SELL" and e20 < e50: s+=1; r.append("EMA20 < EMA50")

    c20, c50 = float(row_c.get("ema20",0)), float(row_c.get("ema50",0))
    if signal == "BUY"  and c20 > c50: s+=1; r.append("1h confirms")
    elif signal == "SELL" and c20 < c50: s+=1; r.append("1h confirms")

    if not r: r.append(f"ML signal {confidence:.0f}%")
    return s, r


# ════════════ TELEGRAM ═══════════════════════════════════════════════

def _send(text):
    tok = os.getenv("TELEGRAM_TOKEN", ""); cid = os.getenv("TELEGRAM_CHAT_ID", "")
    if not tok or not cid: return
    try:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
            data={"chat_id": cid, "text": text, "parse_mode": "Markdown"}, timeout=10)
    except: pass

def _warn(text): log.warning(text); _send(text)

def check_mode_switch(mode_dict):
    last = load_json(MODE_FILE, {})
    if last.get("mode") != mode_dict["mode"]:
        msgs = {"active":"📈 *Active hours* — conf≥55%",
                "quiet":"🌙 *Quiet hours* — conf≥62%",
                "weekend":"📅 *Weekend* — conf≥58%"}
        _send(msgs.get(mode_dict["mode"], "Mode changed"))
        save_json(MODE_FILE, {"mode": mode_dict["mode"],
                              "since": datetime.now(timezone.utc).isoformat()})

def _send_open_alert(symbol, signal, confidence, score, entry,
                     stop, tp1, tp2, qty, risk_usd, balance, reasons, mode):
    emoji  = "🟢" if signal == "BUY" else "🔴"
    stars  = "⭐" * min(score, 5)
    dec    = 4 if entry < 10 else 2
    fp     = lambda v: f"{v:.{dec}f}"
    sl_pct = abs((stop-entry)/entry*100)
    t1_pct = abs((tp1-entry)/entry*100)
    t2_pct = abs((tp2-entry)/entry*100)
    rlines = "\n".join([f"  • {r}" for r in reasons])
    exch   = "Delta Testnet" if mode == "delta" else "Paper Trading"
    _send(
        f"🤖 *TRADE OPENED — {exch}*\n━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{emoji} *{signal} — {symbol}* {stars}\n"
        f"🏷️ _{get_tier(symbol)}_\n"
        f"🎯 Conf: *{confidence:.1f}%* · Score: *{score}/5*\n\n"
        f"⚡ *ENTRY:* `{fp(entry)}`\n"
        f"🛑 *STOP:*  `{fp(stop)}`  (-{sl_pct:.1f}%)\n"
        f"🎯 *TP1:*   `{fp(tp1)}`  (+{t1_pct:.1f}%)\n"
        f"🎯 *TP2:*   `{fp(tp2)}`  (+{t2_pct:.1f}%)\n\n"
        f"💰 `{round(qty*entry,2):.2f} USDT` pos | "
        f"⚠️ `{risk_usd:.2f} USDT` risk\n💼 Balance: `{balance:.2f} USDT`\n\n"
        f"📊 *Reasons:*\n{rlines}\n\n━━━━━━━━━━━━━━━━━━━━\n_{exch}_"
    )

def _send_close_alert(symbol, result, pnl, entry, close_price, opened_at, mode):
    emoji = "✅" if pnl > 0 else "❌"
    dec   = 4 if entry < 10 else 2
    try: dur = str(datetime.now(timezone.utc)-datetime.fromisoformat(opened_at)).split(".")[0]
    except: dur = "—"
    exch = "Delta Testnet" if mode == "delta" else "Paper Trading"
    _send(
        f"🤖 *TRADE CLOSED — {exch}*\n━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{emoji} *{result} — {symbol}*\n\n"
        f"📥 Entry: `{entry:.{dec}f}`\n📤 Close: `{close_price:.{dec}f}`\n"
        f"💵 *PnL: `{pnl:+.4f} USDT`*\n⏱️ {dur}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n_{exch}_"
    )


# ════════════ DIAGNOSTIC ═════════════════════════════════════════════

def run_diagnostic():
    from smart_scheduler import get_scan_mode, check_btc_volatility
    mode = get_scan_mode(); vol = check_btc_volatility()
    client, exch_mode = setup_exchange()
    bal = fetch_and_save_balance(client, exch_mode)
    trades = load_trades()

    lines = [
        f"🔍 *Bot Diagnostic*",
        f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Exchange: {'✅ Delta Testnet (real)' if exch_mode == 'delta' else '📝 Paper Trading (IP not whitelisted)'}",
        "━━━━━━━━━━━━━━━━━━━━",
        f"Mode: *{mode['label']}*  conf≥{mode['min_confidence']}%",
        f"BTC ATR: *{vol['atr_pct']:.3f}%* ({vol['status']})",
        f"💰 Balance: *{bal:.2f} USDT*",
        f"📂 Open trades: *{len(trades)}*",
    ]
    try:
        p = load_model(); lines.append(f"🤖 Model: ✅ {len(p['all_features'])} features")
    except Exception as e: lines.append(f"❌ Model: {e}")
    lines.append(f"\nScanning {len(SYMBOLS)} coins next run")
    _send("\n".join(lines))


# ════════════ MAIN ════════════════════════════════════════════════════

def run_execution_scan():
    log.info(f"\n{'═'*56}")
    log.info(f"SCAN — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    log.info(f"{'═'*56}")

    run, mode, vol, reason = should_scan()
    check_mode_switch(mode)
    if not run:
        log.info(f"SKIPPED: {reason}"); return

    effective_risk = get_effective_risk(mode, vol)
    vol_warn       = vol["message"] if vol.get("warn") else None

    # Auto-detect exchange mode
    client, exch_mode = setup_exchange()

    pipeline   = load_model()
    thresholds = get_mode_thresholds(mode)

    log.info(f"\n[0] Fetching balance ({exch_mode.upper()})...")
    balance = fetch_and_save_balance(client, exch_mode)
    log.info(f"    Balance: {balance:.2f} USDT")

    log.info(f"\n[1] Clearing stuck trades...")
    clear_stuck_trades(client, exch_mode)

    log.info(f"\n[2] Checking open trades...")
    check_open_trades(client, exch_mode)

    trades = load_trades()
    log.info(f"\n[3] Scanning {len(SYMBOLS)} coins | Open:{len(trades)}/{MAX_OPEN_TRADES}")
    log.info(f"    conf≥{thresholds['min_confidence']}% | score≥{thresholds['min_score']} | "
             f"ADX≥{thresholds['min_adx']} | risk:{effective_risk:.2f} | {exch_mode.upper()}")

    found = 0
    for symbol in SYMBOLS:
        if len(load_trades()) >= MAX_OPEN_TRADES:
            log.info("  Max trades reached — stopping"); break

        log.info(f"\n  ── {symbol} ({get_tier(symbol)}) ──")
        sig = generate_signal(symbol, pipeline, thresholds)
        if sig is None:
            time.sleep(0.3); continue

        found += 1
        if vol_warn: sig["reasons"] = list(sig.get("reasons",[])) + [f"⚠️ {vol_warn}"]

        execute_trade(client, exch_mode, **{k: sig[k] for k in
            ["symbol","signal","entry","atr","confidence","score","reasons"]},
            risk_mult=effective_risk)
        time.sleep(1)

    fetch_and_save_balance(client, exch_mode)

    log.info(f"\n{'═'*56}")
    log.info(f"DONE — {found} signal(s) | {exch_mode.upper()} | Balance: {balance:.2f} USDT")
    log.info(f"{'═'*56}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "diagnostic":
        run_diagnostic()
    elif len(sys.argv) > 1 and sys.argv[1] == "clear_stuck":
        client, exch_mode = setup_exchange()
        clear_stuck_trades(client, exch_mode)
    else:
        run_execution_scan()
