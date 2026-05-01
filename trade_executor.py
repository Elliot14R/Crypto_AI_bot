# trade_executor.py — FIXED: dangerous hard-fallback SL removed, all other fixes retained
#
# FIXES APPLIED:
#   FIX 3 (this file): Removed the hard-fallback SL check that fires when
#          o_state != "untriggered". That condition catches empty strings from
#          API errors and was phantom-closing trades that had just had their
#          SL replaced with a breakeven order (state briefly = "").
#          is_sl_triggered() in deribit_client.py now handles all real cases safely.
#
#   Previously fixed (retained here):
#   1. Removed daily loss circuit breaker from scan path
#   2. Removed volume_ratio check
#   3. Lowered score requirement to 1 during active hours
#   4. Ghost trade recovery uses Deribit trade history before falling back to $0
#   5. Stop Loss execution fixed via is_sl_triggered

import os, json, time, logging, requests, joblib
import pandas as pd, numpy as np
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from config import (
    SYMBOLS, ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT,
    RISK_PER_TRADE, MODEL_FILE, LOG_FILE, get_tier,
    TIMEFRAME_ENTRY, TIMEFRAME_CONFIRM, LIVE_LIMIT, MAX_TRADE_AGE_HOURS
)
from deribit_client import DeribitClient, TRADEABLE_SYMBOLS
from feature_engineering import add_indicators
from smart_scheduler import (
    should_scan, get_mode_thresholds, get_effective_risk, check_correlation
)

TRADES_FILE     = "trades.json"
HISTORY_FILE    = "trade_history.json"
SIGNALS_FILE    = "signals.json"
BALANCE_FILE    = "balance.json"
MAX_OPEN_TRADES = 4

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
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
            with open(tmp,"w") as f: json.dump(data, f, indent=2, default=str)
            os.replace(tmp, str(dest))
        except Exception as e: log.error(f"save_json {dest}: {e}")

load_trades  = lambda: load_json(TRADES_FILE,  {})
save_trades  = lambda d: save_json(TRADES_FILE, d)
load_history = lambda: load_json(HISTORY_FILE, [])

def append_history(rec):
    h = load_history(); h.append(rec); save_json(HISTORY_FILE, h)

def save_signal(sig):
    s = load_json(SIGNALS_FILE, [])
    s.append({**sig, "generated_at": datetime.now(timezone.utc).isoformat()})
    save_json(SIGNALS_FILE, s[-500:])


# ════════════ BALANCE ════════════════════════════════════════════════

def save_balance(deribit: DeribitClient) -> float:
    try:
        bals  = deribit.get_all_balances()
        total = deribit.get_total_equity_usd()
        pos   = deribit.get_positions()
        upnl  = sum(float(p.get("floating_profit_loss_usd") or
                          p.get("floating_profit_loss") or 0) for p in pos)
        assets = [{"asset":c,"free":str(round(i["available"],6)),
                   "total":str(round(i["equity_usd"],2))} for c,i in bals.items()]
        save_json(BALANCE_FILE, {
            "usdt":round(total,2),"equity":round(total+upnl,2),
            "unrealised":round(upnl,4),"assets":assets,
            "updated_at":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "mode":"deribit_testnet","exchange":"Deribit(by Coinbase) Testnet",
            "open_positions":len(pos),
        })
        log.info(f"  Balance: ${total:.2f} | upnl:{upnl:+.2f} | positions:{len(pos)}")
        return total
    except Exception as e:
        log.error(f"  save_balance: {e}"); return 0.0


# ════════════ MARKET DATA ════════════════════════════════════════════

def get_data(symbol: str, interval: str) -> pd.DataFrame:
    for url in ["https://data-api.binance.vision/api/v3/klines",
                "https://api.binance.com/api/v3/klines"]:
        try:
            r = requests.get(url,
                params={"symbol":symbol,"interval":interval,"limit":LIVE_LIMIT},
                timeout=10)
            if r.status_code == 200:
                df = pd.DataFrame(r.json()).iloc[:,:6]
                df.columns = ["open_time","open","high","low","close","volume"]
                for c in ["open","high","low","close","volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                return df
        except Exception: continue
    return pd.DataFrame()


# ════════════ SIGNAL GENERATION ══════════════════════════════════════

def generate_signal(symbol, pipeline, thresholds):
    try:
        df15 = add_indicators(get_data(symbol, TIMEFRAME_ENTRY)).fillna(0)
        df1h_raw = get_data(symbol, TIMEFRAME_CONFIRM)
        df1h = add_indicators(df1h_raw).fillna(0) if not df1h_raw.empty else pd.DataFrame()
        if df15.empty or len(df15) < 30: return None

        row = df15.iloc[-1].copy()
        r1h = df1h.iloc[-1] if not df1h.empty else pd.Series(0, index=df15.columns)
        row["rsi_1h"]   = float(r1h.get("rsi",  50))
        row["adx_1h"]   = float(r1h.get("adx",   0))
        row["trend_1h"] = float(r1h.get("trend", 0))

        af  = pipeline["all_features"]
        X   = pd.DataFrame([row[af].values], columns=af).replace([np.inf,-np.inf],0).fillna(0)
        Xs  = pipeline["selector"].transform(X)
        pred = pipeline["ensemble"].predict(Xs)[0]
        prob = pipeline["ensemble"].predict_proba(Xs)[0]
        sig  = {0:"BUY",1:"SELL",2:"NO_TRADE"}[pred]
        conf = round(float(max(prob))*100, 1)

        log.info(f"    ML: {sig} {conf:.1f}% (need ≥{thresholds['min_confidence']}%)")
        if sig == "NO_TRADE" or conf < thresholds["min_confidence"]: return None

        adx = float(row.get("adx", 0))
        log.info(f"    ADX: {adx:.1f} (need ≥{thresholds['min_adx']})")
        if adx < thresholds["min_adx"]: return None

        # BTC Trend Gate - Do not short if BTC is pumping
        if sig == "SELL":
            btc_df = add_indicators(get_data("BTCUSDT", TIMEFRAME_ENTRY))
            if not btc_df.empty:
                b_row = btc_df.iloc[-1]
                if b_row["ema20"] > b_row["ema50"] and b_row["close"] > b_row["ema20"]:
                    log.info(f"    BTC uptrend detected — skipping SELL signal for {symbol}")
                    return None

        # Quality score
        score = 0; reasons = []
        if conf >= 70:   score+=1; reasons.append(f"High conf ({conf:.0f}%)")
        elif conf >= 60: score+=1; reasons.append(f"Conf ({conf:.0f}%)")

        adx_val = adx
        if adx_val > 25:   score+=1; reasons.append(f"Strong ADX {adx_val:.0f}")
        elif adx_val > 18: score+=1; reasons.append(f"ADX {adx_val:.0f}")

        rsi = float(row.get("rsi", 50))
        if sig=="BUY"  and rsi < 50: score+=1; reasons.append(f"RSI bullish ({rsi:.0f})")
        elif sig=="SELL" and rsi > 50: score+=1; reasons.append(f"RSI bearish ({rsi:.0f})")

        e20=float(row.get("ema20",0)); e50=float(row.get("ema50",0))
        if sig=="BUY"  and e20>e50: score+=1; reasons.append("EMA bullish")
        elif sig=="SELL" and e20<e50: score+=1; reasons.append("EMA bearish")

        c20=float(r1h.get("ema20",0)); c50=float(r1h.get("ema50",0))
        if sig=="BUY"  and c20>c50: score+=1; reasons.append("1h confirms")
        elif sig=="SELL" and c20<c50: score+=1; reasons.append("1h confirms")

        log.info(f"    Score: {score} (need ≥{thresholds['min_score']})")
        if score < thresholds["min_score"]:
            save_signal({"symbol":symbol,"signal":sig,"confidence":conf,"score":score,
                "reasons":reasons,"rejected":True,
                "reject_reason":f"score {score}<{thresholds['min_score']}"})
            return None

        if not reasons: reasons.append(f"ML {conf:.0f}%")

        entry = float(row["close"]); atr = float(row["atr"])
        dec   = 4 if entry < 10 else 2
        if sig == "BUY":
            stop=round(entry-atr*ATR_STOP_MULT,dec)
            tp1 =round(entry+atr*ATR_TARGET1_MULT,dec)
            tp2 =round(entry+atr*ATR_TARGET2_MULT,dec)
        else:
            stop=round(entry+atr*ATR_STOP_MULT,dec)
            tp1 =round(entry-atr*ATR_TARGET1_MULT,dec)
            tp2 =round(entry-atr*ATR_TARGET2_MULT,dec)

        return {"symbol":symbol,"signal":sig,"confidence":conf,"score":score,
                "entry":entry,"atr":atr,"stop":stop,"tp1":tp1,"tp2":tp2,"reasons":reasons}
    except Exception as e:
        log.error(f"    Signal {symbol}: {e}"); return None


# ════════════ EXECUTE TRADE ══════════════════════════════════════════

def execute_trade(deribit: DeribitClient, sig: dict, risk_mult: float, balance: float) -> bool:
    symbol=sig["symbol"]; signal=sig["signal"]
    entry=sig["entry"]; atr=sig["atr"]
    stop=sig["stop"]; tp1=sig["tp1"]; tp2=sig["tp2"]

    trades     = load_trades()
    open_count = len([t for t in trades.values() if not t.get("closed",False)])
    if open_count >= MAX_OPEN_TRADES:
        log.info(f"  🛑 MAX TRADES ({MAX_OPEN_TRADES}) — skip {symbol}"); return False
    if symbol in trades and not trades[symbol].get("closed",False):
        log.info(f"  {symbol}: already open — skip"); return False
    if not deribit.is_supported(symbol):
        log.info(f"  {symbol}: not on Deribit — skip"); return False
    if not check_correlation(trades, signal): return False

    dec=4 if entry<10 else 2
    side   ="BUY"  if signal=="BUY" else "SELL"
    sl_side="SELL" if signal=="BUY" else "BUY"
    tp_side="SELL" if signal=="BUY" else "BUY"

    total_q          = deribit.calc_contracts(symbol, balance, entry, stop, risk_mult)
    qty_tp1, qty_tp2 = deribit.split_amount(symbol, total_q)
    risk_usd         = round(balance * RISK_PER_TRADE * risk_mult, 2)

    log.info(f"  {signal} {symbol} total={total_q} risk=${risk_usd:.2f}")
    log.info(f"  SL={stop:.{dec}f} TP1={tp1:.{dec}f} TP2={tp2:.{dec}f}")

    order_ids = {}; actual_entry = entry
    try:
        er = deribit.place_market_order(symbol, side, total_q)
        eo = er.get("order", er)
        order_ids["entry"] = str(eo.get("order_id",""))
        actual_entry = deribit.get_fill_price(er, entry) or entry
        if eo.get("order_state","") == "cancelled" and float(eo.get("filled_amount",0) or 0) == 0:
            log.warning(f"  Market cancelled (thin book) — skip"); return False
        log.info(f"  ✅ Entry @ {actual_entry:.{dec}f}"); time.sleep(1.5)

        if signal=="BUY":
            stop=deribit.round_price(symbol,actual_entry-atr*ATR_STOP_MULT)
            tp1 =deribit.round_price(symbol,actual_entry+atr*ATR_TARGET1_MULT)
            tp2 =deribit.round_price(symbol,actual_entry+atr*ATR_TARGET2_MULT)
        else:
            stop=deribit.round_price(symbol,actual_entry+atr*ATR_STOP_MULT)
            tp1 =deribit.round_price(symbol,actual_entry-atr*ATR_TARGET1_MULT)
            tp2 =deribit.round_price(symbol,actual_entry-atr*ATR_TARGET2_MULT)

        tick = deribit.get_tick_size(symbol)
        sl_limit = deribit.round_price(symbol, stop - tick if signal=="BUY" else stop + tick)

        for label, qty, price, sl_p, key in [
            ("SL",  total_q, sl_limit, stop, "stop_loss"),
            ("TP1", qty_tp1, tp1,  None, "tp1"),
            ("TP2", qty_tp2, tp2,  None, "tp2"),
        ]:
            if qty <= 0: continue
            try:
                res = deribit.place_limit_order(
                    symbol, sl_side if label=="SL" else tp_side,
                    qty, price, stop_price=sl_p)
                o   = res.get("order", res)
                oid = str(o.get("order_id",""))
                if oid: order_ids[key] = oid
                log.info(f"  ✅ {label} @ {price:.{dec}f} × {qty} id:{oid or 'MISSING'}")
            except Exception as e: log.warning(f"  {label} failed: {e}")

    except Exception as e:
        log.error(f"  Trade error {symbol}: {e}"); _send(f"⚠️ {symbol}: {e}"); return False

    record = {
        "symbol":symbol,"signal":signal,"entry":actual_entry,
        "stop":stop,"tp1":tp1,"tp2":tp2,
        "qty":total_q,"qty_tp1":qty_tp1,"qty_tp2":qty_tp2,
        "risk_usd":risk_usd,"balance_at_open":balance,"risk_mult":risk_mult,
        "order_ids":order_ids,
        "opened_at":datetime.now(timezone.utc).isoformat(),
        "tp1_hit":False,"tp2_hit":False,"closed":False,
        "confidence":sig["confidence"],"score":sig["score"],
        "reasons":sig.get("reasons",[]),"tier":get_tier(symbol),
        "exchange":"deribit_testnet",
    }
    trades[symbol] = record
    save_trades(trades)
    save_signal({**record,"type":"executed"})
    _send_open_alert(symbol, signal, sig["confidence"], sig["score"],
                     actual_entry, stop, tp1, tp2, total_q, qty_tp1, qty_tp2, risk_usd, balance)
    log.info(f"  ✅✅ TRADE OPENED: {symbol} {signal}")
    return True


# ════════════ MONITOR OPEN TRADES ════════════════════════════════════

def _replace_missing_orders(deribit: DeribitClient, symbol: str, trade: dict):
    """Re-place any SL or TP orders that are missing from an open trade."""
    oids   = trade.get("order_ids", {})
    signal = trade["signal"]
    entry  = float(trade["entry"])
    stop   = float(trade.get("stop", 0))
    tp1    = float(trade.get("tp1", 0))
    tp2    = float(trade.get("tp2", 0))
    qty    = float(trade.get("qty", 0))
    qty_t1 = float(trade.get("qty_tp1", 0))
    qty_t2 = float(trade.get("qty_tp2", 0))
    dec    = 4 if entry < 10 else 2
    sl_side = "SELL" if signal == "BUY" else "BUY"
    tp_side = "SELL" if signal == "BUY" else "BUY"
    changed = False

    # Re-place SL
    sl_oid = str(oids.get("stop_loss", ""))
    if stop > 0 and qty > 0 and (not sl_oid or sl_oid in ("", "None")):
        try:
            # Use limit price slightly worse than trigger to guarantee fill
            tick = deribit.get_tick_size(symbol)
            sl_limit = deribit.round_price(symbol, stop - tick if signal == "BUY" else stop + tick)
            res = deribit.place_limit_order(symbol, sl_side, qty, sl_limit, stop_price=stop)
            o   = res.get("order", res)
            oid = str(o.get("order_id", ""))
            if oid:
                trade["order_ids"]["stop_loss"] = oid
                log.info(f"  🛠 Re-placed SL for {symbol} @ limit {sl_limit:.{dec}f} / trigger {stop:.{dec}f} id:{oid}")
                changed = True
        except Exception as e:
            log.warning(f"  SL re-place {symbol}: {e}")

    # Re-place TP1
    tp1_oid = str(oids.get("tp1", ""))
    if tp1 > 0 and qty_t1 > 0 and not trade.get("tp1_hit") and (not tp1_oid or tp1_oid in ("", "None")):
        try:
            res = deribit.place_limit_order(symbol, tp_side, qty_t1, tp1)
            o   = res.get("order", res)
            oid = str(o.get("order_id", ""))
            if oid:
                trade["order_ids"]["tp1"] = oid
                log.info(f"  🛠 Re-placed TP1 for {symbol} @ {tp1:.{dec}f} id:{oid}")
                changed = True
        except Exception as e:
            log.warning(f"  TP1 re-place {symbol}: {e}")

    # Re-place TP2
    tp2_oid = str(oids.get("tp2", ""))
    if tp2 > 0 and qty_t2 > 0 and not trade.get("tp2_hit") and (not tp2_oid or tp2_oid in ("", "None")):
        try:
            res = deribit.place_limit_order(symbol, tp_side, qty_t2, tp2)
            o   = res.get("order", res)
            oid = str(o.get("order_id", ""))
            if oid:
                trade["order_ids"]["tp2"] = oid
                log.info(f"  🛠 Re-placed TP2 for {symbol} @ {tp2:.{dec}f} id:{oid}")
                changed = True
        except Exception as e:
            log.warning(f"  TP2 re-place {symbol}: {e}")

    return changed


def fp(o, fb):
    p = float(o.get("average_price") or o.get("last_price") or o.get("price") or 0)
    return p if p > 0 else fb

def _pnl(t, cp, ct):
    qty  = float(t["qty_tp1"] if ct=="tp1" else t["qty_tp2"] if ct=="tp2" else t["qty"])
    diff = (cp-t["entry"]) if t["signal"]=="BUY" else (t["entry"]-cp)
    return round(diff*qty, 4)

def _close_record(t, cp, pnl, reason):
    append_history({**t,"close_price":cp,"pnl":pnl,
        "closed_at":datetime.now(timezone.utc).isoformat(),"close_reason":reason})

def check_open_trades(deribit: DeribitClient):
    trades = load_trades()
    if not trades: log.info("  No open trades"); return
    to_remove = []
    log.info(f"  Monitoring {len(trades)} trade(s)")

    live_positions = set()
    try:
        for p in deribit.get_positions():
            if float(p.get("size", 0)) != 0:
                inst = p.get("instrument_name","")
                base = inst.split("_")[0] if "_" in inst else inst.split("-")[0]
                live_positions.add(f"{base}USDT")
    except Exception as e:
        log.warning(f"  Could not fetch live positions: {e}")

    for symbol, trade in list(trades.items()):
        if trade.get("closed"): to_remove.append(symbol); continue

        # Re-place any missing SL/TP orders before monitoring
        if _replace_missing_orders(deribit, symbol, trade):
            save_trades(trades)  # persist new order IDs immediately

        oids  = trade.get("order_ids", {})
        entry = float(trade["entry"])
        dec   = 4 if entry < 10 else 2

        def get_o(key):
            if key not in oids or not oids[key] or str(oids[key]) in ("","None"): return {}
            return deribit.get_order(str(oids[key]))

        try:
            # ── TP1 ─────────────────────────────────────────────────
            if not trade.get("tp1_hit") and "tp1" in oids:
                o = get_o("tp1")
                if deribit.is_order_filled(o):
                    trade["tp1_hit"] = True
                    fill = fp(o, trade["tp1"])
                    pnl  = _pnl(trade, fill, "tp1")
                    log.info(f"  🎯 TP1 {symbol} @ {fill:.{dec}f} pnl≈{pnl:+.4f}")
                    _send(f"🎯 *TP1 HIT — {symbol}*\n@ `{fill:.{dec}f}` | PnL ≈ `{pnl:+.4f}`")
                    if oids.get("stop_loss") and trade.get("qty_tp2",0) > 0:
                        try:
                            deribit.cancel_order(oids["stop_loss"])
                            sl_s = "SELL" if trade["signal"]=="BUY" else "BUY"
                            be   = deribit.place_limit_order(symbol,sl_s,trade["qty_tp2"],
                                       entry,stop_price=entry)
                            be_o = be.get("order",be); nid=str(be_o.get("order_id",""))
                            if nid: trade["order_ids"]["stop_loss"]=nid; trade["stop"]=entry
                            _send(f"🛡️ *{symbol} RISK-FREE* SL→entry `{entry:.{dec}f}`")
                        except Exception as be: log.warning(f"  BE SL: {be}")

            # ── Trailing stop ────────────────────────────────────────
            if trade.get("tp1_hit") and not trade.get("tp2_hit") and oids.get("stop_loss"):
                live = deribit.get_live_price(symbol)
                if live > 0:
                    halfway = (entry+float(trade["tp2"]))/2
                    at_t = (trade["signal"]=="BUY" and live>=halfway) or \
                           (trade["signal"]=="SELL" and live<=halfway)
                    sl_be = abs(float(trade.get("stop",0))-entry) < entry*0.001
                    if at_t and sl_be and trade.get("qty_tp2",0) > 0:
                        try:
                            deribit.cancel_order(oids["stop_loss"])
                            sl_s = "SELL" if trade["signal"]=="BUY" else "BUY"
                            sl_r = deribit.place_limit_order(symbol,sl_s,trade["qty_tp2"],
                                       trade["tp1"],stop_price=trade["tp1"])
                            sl_o=sl_r.get("order",sl_r); nid=str(sl_o.get("order_id",""))
                            if nid: trade["order_ids"]["stop_loss"]=nid; trade["stop"]=trade["tp1"]
                            _send(f"🚀 *{symbol}* Trail SL→TP1 `{trade['tp1']:.{dec}f}` locked!")
                        except Exception as e: log.warning(f"  Trail SL: {e}")

            # ── TP2 ─────────────────────────────────────────────────
            if trade.get("tp1_hit") and not trade.get("tp2_hit") and "tp2" in oids:
                o = get_o("tp2")
                if deribit.is_order_filled(o):
                    trade["tp2_hit"]=True; trade["closed"]=True
                    fill=fp(o,trade["tp2"]); pnl=_pnl(trade,fill,"tp2")
                    log.info(f"  ✅ TP2 {symbol} @ {fill:.{dec}f} pnl≈{pnl:+.4f}")
                    _send(f"✅ *FULL WIN — {symbol}*\nTP2 @ `{fill:.{dec}f}` | PnL ≈ `{pnl:+.4f}`")
                    _close_record(trade,fill,pnl,"TP2 hit"); to_remove.append(symbol)

            # ── SL ──────────────────────────────────────────────────
            # FIX 3: Removed the dangerous hard-fallback price-vs-SL check.
            #
            # The old fallback fired when o_state != "untriggered", which
            # catches empty strings from API errors or the brief window when
            # a new breakeven SL order hasn't been fetched yet. This caused
            # trades to be closed immediately after TP1 hit.
            #
            # is_sl_triggered() in deribit_client.py now safely handles all
            # real SL states (triggered, filled, cancelled-with-fill).
            # The ghost cleaner handles positions that vanish entirely.
            if not trade.get("closed") and oids.get("stop_loss"):
                o       = get_o("stop_loss")
                o_state = o.get("order_state", "").lower()
                sl_hit  = deribit.is_sl_triggered(o)

                if sl_hit:
                    trade["closed"] = True
                    fill = fp(o, trade["stop"])
                    if fill == trade["stop"] or fill == 0:
                        live = deribit.get_live_price(symbol)
                        if live > 0: fill = live

                    pnl  = _pnl(trade, fill, "sl")
                    lbl  = "BREAK-EVEN ⚖️" if abs(fill-entry) < entry*0.002 else "STOPPED OUT ❌"
                    log.info(f"  ❌ SL {symbol} @ {fill:.{dec}f} pnl≈{pnl:+.4f} (state={o_state})")
                    _send(f"{'⚖️' if 'BREAK' in lbl else '❌'} *{lbl} — {symbol}*\nPnL ≈ `{pnl:+.4f}`")
                    _close_record(trade, fill, pnl, lbl)

                    for k in ("tp1","tp2"):
                        if oids.get(k) and not trade.get(f"{k}_hit"):
                            try: deribit.cancel_order(oids[k])
                            except Exception: pass
                    to_remove.append(symbol)

        except Exception as e: log.error(f"  Monitor {symbol}: {e}")

    save_trades(trades)
    for sym in set(to_remove): trades.pop(sym,None)
    save_trades(trades)


def check_stale_trades(deribit: DeribitClient):
    trades=load_trades(); now=datetime.now(timezone.utc); to_remove=[]
    for symbol, trade in trades.items():
        if trade.get("closed") or trade.get("tp1_hit"): continue
        try:
            age_h = (now-datetime.fromisoformat(
                trade.get("opened_at","").replace("Z",""))).total_seconds()/3600
        except Exception: continue
        if age_h > MAX_TRADE_AGE_HOURS:
            log.warning(f"  ⏰ {symbol}: {age_h:.0f}h — time-based exit")
            try:
                for k,oid in trade.get("order_ids",{}).items():
                    if k!="entry" and oid:
                        try: deribit.cancel_order(oid)
                        except Exception: pass
                live=deribit.get_live_price(symbol)
                pnl=_pnl(trade,live if live>0 else trade["entry"],"sl")
                _close_record(trade,live if live>0 else trade["entry"],pnl,
                              f"Time exit ({age_h:.0f}h)")
                _send(f"⏰ *TIME EXIT — {symbol}*\n{age_h:.0f}h without TP1 | PnL≈`{pnl:+.4f}`")
                to_remove.append(symbol)
            except Exception as e: log.error(f"  Time exit {symbol}: {e}")
    if to_remove:
        for sym in to_remove: trades.pop(sym,None)
        save_trades(trades)


def clean_ghost_trades(deribit: DeribitClient):
    """Recover real PnL before marking as ghost."""
    trades=load_trades()
    if not trades: return
    live_pos = {}
    for p in deribit.get_positions():
        if float(p.get("size",0))!=0:
            inst=p.get("instrument_name","")
            base=inst.split("_")[0] if "_" in inst else inst.split("-")[0]
            live_pos[f"{base}USDT"]=True

    to_remove=[]
    for symbol, trade in trades.items():
        if float(trade.get("stop",0))==0 or float(trade.get("tp1",0))==0:
            log.warning(f"  🗑️ {symbol}: broken state — remove")
            to_remove.append(symbol); continue
        if symbol not in live_pos:
            log.warning(f"  🕵️ {symbol}: no position — recovering PnL...")
            real_pnl=None; real_close=None; reason="Closed on exchange"
            try:
                fills=deribit.get_trade_history_for_instrument(symbol,count=20)
                entry=float(trade["entry"]); entry_dir=trade["signal"]
                close_fills=[f for f in fills
                    if (entry_dir=="BUY" and f.get("direction")=="sell") or
                       (entry_dir=="SELL" and f.get("direction")=="buy")]
                if close_fills:
                    latest=close_fills[0]; real_close=float(latest.get("price",0) or 0)
                    qty=float(trade.get("qty",0))
                    if real_close>0 and qty>0:
                        diff=(real_close-entry) if entry_dir=="BUY" else (entry-real_close)
                        real_pnl=round(diff*qty,4)

                        tp1_p=float(trade.get("tp1",0)); tp2_p=float(trade.get("tp2",0))

                        if entry_dir=="BUY":
                            if real_close >= tp2_p*0.998: reason="TP2 hit"
                            elif real_close >= tp1_p*0.998: reason="TP1 hit"
                            else: reason="SL hit"
                        else:
                            if real_close <= tp2_p*1.002: reason="TP2 hit"
                            elif real_close <= tp1_p*1.002: reason="TP1 hit"
                            else: reason="SL hit"

                        log.info(f"  ✅ Recovered {symbol}: close={real_close:.4f} pnl={real_pnl:+.4f} ({reason})")
            except Exception as e: log.warning(f"  PnL recovery {symbol}: {e}")

            if real_pnl is not None and real_close is not None:
                _close_record(trade,real_close,real_pnl,reason)
                _send(f"{'✅' if real_pnl>0 else '❌'} *{reason} — {symbol}*\nPnL:`{real_pnl:+.4f}`")
            else:
                _close_record(trade,float(trade.get("entry",0)),0.0,"Ghost — PnL unrecoverable")
            to_remove.append(symbol)

    if to_remove:
        for sym in to_remove: trades.pop(sym,None)
        save_trades(trades)
        log.info(f"  Processed {len(to_remove)} ghost/closed trade(s)")


# ════════════ TELEGRAM ════════════════════════════════════════════════

def _send(text):
    tok=os.getenv("TELEGRAM_TOKEN",""); cid=os.getenv("TELEGRAM_CHAT_ID","")
    if not tok or not cid: return
    try: requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
            data={"chat_id":cid,"text":text,"parse_mode":"Markdown"},timeout=10)
    except Exception: pass

def _send_open_alert(sym,sig,conf,score,entry,stop,tp1,tp2,qty,q1,q2,risk,bal):
    e="🟢" if sig=="BUY" else "🔴"; d=4 if entry<10 else 2
    sp=abs((stop-entry)/entry*100); t1=abs((tp1-entry)/entry*100); t2=abs((tp2-entry)/entry*100)
    _send(f"🤖 *DERIBIT TRADE*\n━━━━━━━━━━━━━━━━━━━━\n"
          f"{e} *{sig} — {sym}* ⭐×{score}\n🎯 {conf:.1f}% conf\n\n"
          f"⚡ Entry: `{entry:.{d}f}`\n"
          f"🛑 SL:    `{stop:.{d}f}` (-{sp:.1f}%)\n"
          f"🎯 TP1:   `{tp1:.{d}f}` (+{t1:.1f}%) × {q1}\n"
          f"🎯 TP2:   `{tp2:.{d}f}` (+{t2:.1f}%) × {q2}\n"
          f"📦 {qty} contracts · Risk: ${risk:.2f} · ${bal:.2f}\n"
          f"━━━━━━━━━━━━━━━━━━━━")


# ════════════ MAIN SCAN ═══════════════════════════════════════════════

def run_execution_scan():
    log.info(f"\n{'═'*56}\nSCAN — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n{'═'*56}")

    run, mode, vol, reason = should_scan()
    if not run:
        log.info(f"  Scan skipped: {reason}"); return

    deribit    = DeribitClient(os.getenv("DERIBIT_CLIENT_ID",""), os.getenv("DERIBIT_CLIENT_SECRET",""))
    deribit.test_connection()
    pipeline   = joblib.load(MODEL_FILE)
    thresholds = get_mode_thresholds(mode)
    risk_mult  = get_effective_risk(mode, vol)

    log.info(f"  {mode['label']} | conf≥{thresholds['min_confidence']}% "
             f"| score≥{thresholds['min_score']} | ADX≥{thresholds['min_adx']} | risk:{risk_mult:.2f}")

    log.info("\n[0] Balance..."); balance = save_balance(deribit)
    log.info("\n[1] Monitor trades..."); check_open_trades(deribit)
    log.info("\n[2] Stale trade check..."); check_stale_trades(deribit)
    log.info("\n[3] Ghost trade recovery..."); clean_ghost_trades(deribit)
    save_balance(deribit)

    open_count = len([t for t in load_trades().values() if not t.get("closed",False)])
    log.info(f"\n[4] Scanning {len(SYMBOLS)} coins | Open:{open_count}/{MAX_OPEN_TRADES}")

    found = 0
    for symbol in SYMBOLS:
        log.info(f"\n  ── {symbol} ({get_tier(symbol)}) ──")
        sig = generate_signal(symbol, pipeline, thresholds)
        if sig is None: time.sleep(0.2); continue
        found += 1
        if execute_trade(deribit, sig, risk_mult, balance): time.sleep(1.5)

    save_balance(deribit)
    log.info(f"\n{'═'*56}\nDONE — {found} signal(s) | ${balance:.2f}\n{'═'*56}")


if __name__ == "__main__":
    run_execution_scan()
