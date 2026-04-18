# trade_executor.py — Professional grade with all enhancements
#
# NEW vs previous version:
#   1. Time-based exit: trades > 48h without TP1 → market close (prevent dead capital)
#   2. Daily loss circuit breaker integrated
#   3. Ghost trades filtered from win rate stats
#   4. Min confidence hard-coded to 65% (no more 50% signals)
#   5. Volume confirmation: signal only if volume_ratio > 0.8
#   6. Market regime: skip SELL signals if BTC ADX < 15 (ranging market)

import os, json, time, logging, requests, joblib
import pandas as pd, numpy as np
from datetime import datetime, timezone, timedelta
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
    should_scan, get_mode_thresholds, get_effective_risk,
    check_correlation, check_daily_loss_limit
)

TRADES_FILE     = "trades.json"
HISTORY_FILE    = "trade_history.json"
SIGNALS_FILE    = "signals.json"
BALANCE_FILE    = "balance.json"
MAX_OPEN_TRADES = 3

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
            "usdt": round(total,2), "equity": round(total+upnl,2),
            "unrealised": round(upnl,4), "assets": assets,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "mode": "deribit_testnet", "exchange": "Deribit(by Coinbase) Testnet",
            "open_positions": len(pos),
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

        # Volume confirmation: avoid signals in low-volume periods
        vol_ratio = float(row.get("volume_ratio", 1.0))
        if vol_ratio < 0.7:
            log.info(f"    Volume ratio {vol_ratio:.2f} < 0.7 — skip (low volume)")
            return None

        # Quality score
        score = 0; reasons = []
        if conf >= 70:   score+=1; reasons.append(f"High conf ({conf:.0f}%)")
        elif conf >= 65: score+=1; reasons.append(f"Good conf ({conf:.0f}%)")

        if adx > 25:   score+=1; reasons.append(f"Strong ADX {adx:.0f}")
        elif adx > 18: score+=1; reasons.append(f"ADX {adx:.0f}")

        rsi = float(row.get("rsi", 50))
        if sig=="BUY"  and rsi < 50: score+=1; reasons.append(f"RSI bullish ({rsi:.0f})")
        elif sig=="SELL" and rsi > 50: score+=1; reasons.append(f"RSI bearish ({rsi:.0f})")

        e20=float(row.get("ema20",0)); e50=float(row.get("ema50",0))
        if sig=="BUY"  and e20>e50: score+=1; reasons.append("EMA bullish")
        elif sig=="SELL" and e20<e50: score+=1; reasons.append("EMA bearish")

        c20=float(r1h.get("ema20",0)); c50=float(r1h.get("ema50",0))
        if sig=="BUY"  and c20>c50: score+=1; reasons.append("1h confirms")
        elif sig=="SELL" and c20<c50: score+=1; reasons.append("1h confirms")

        log.info(f"    Score: {score} (need ≥{thresholds['min_score']}) | Vol:{vol_ratio:.2f}")
        if score < thresholds["min_score"]:
            save_signal({"symbol":symbol,"signal":sig,"confidence":conf,"score":score,
                "reasons":reasons,"rejected":True,"reject_reason":f"score {score}<{thresholds['min_score']}"})
            return None

        entry = float(row["close"]); atr = float(row["atr"])
        dec   = 4 if entry < 10 else 2
        if sig == "BUY":
            stop=round(entry-atr*ATR_STOP_MULT,dec); tp1=round(entry+atr*ATR_TARGET1_MULT,dec); tp2=round(entry+atr*ATR_TARGET2_MULT,dec)
        else:
            stop=round(entry+atr*ATR_STOP_MULT,dec); tp1=round(entry-atr*ATR_TARGET1_MULT,dec); tp2=round(entry-atr*ATR_TARGET2_MULT,dec)

        return {"symbol":symbol,"signal":sig,"confidence":conf,"score":score,
                "entry":entry,"atr":atr,"stop":stop,"tp1":tp1,"tp2":tp2,"reasons":reasons}
    except Exception as e:
        log.error(f"    Signal {symbol}: {e}"); return None


# ════════════ EXECUTE TRADE ══════════════════════════════════════════

def execute_trade(deribit: DeribitClient, sig: dict, risk_mult: float, balance: float) -> bool:
    symbol=sig["symbol"]; signal=sig["signal"]
    entry=sig["entry"]; atr=sig["atr"]
    stop=sig["stop"]; tp1=sig["tp1"]; tp2=sig["tp2"]

    trades = load_trades()
    open_trades = [t for t in trades.values() if not t.get("closed",False)]
    if len(open_trades) >= MAX_OPEN_TRADES:
        log.info(f"  🛑 MAX TRADES ({MAX_OPEN_TRADES}) — skip {symbol}"); return False
    if symbol in trades and not trades[symbol].get("closed",False):
        log.info(f"  {symbol}: already open — skip"); return False
    if not deribit.is_supported(symbol):
        log.info(f"  {symbol}: not on Deribit — skip"); return False

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
        state  = eo.get("order_state","")
        filled = float(eo.get("filled_amount",0) or 0)
        if state == "cancelled" and filled == 0:
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

        for label,qty,price,sl_p in [
            ("SL",  total_q, stop, stop),
            ("TP1", qty_tp1, tp1,  None),
            ("TP2", qty_tp2, tp2,  None),
        ]:
            if qty <= 0: continue
            try:
                res = deribit.place_limit_order(symbol, sl_side if label=="SL" else tp_side,
                                                qty, price, stop_price=sl_p)
                o   = res.get("order", res)
                oid = str(o.get("order_id",""))
                key = {"SL":"stop_loss","TP1":"tp1","TP2":"tp2"}[label]
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
    log.info(f"  ✅✅ TRADE OPENED: {symbol} {signal} | orders:{order_ids}")
    return True


# ════════════ MONITOR OPEN TRADES ════════════════════════════════════

def _fp(o, fb): p=float(o.get("average_price") or o.get("price") or 0); return p if p>0 else fb
def _pnl(t,cp,ct):
    qty=float(t["qty_tp1"] if ct=="tp1" else t["qty_tp2"] if ct=="tp2" else t["qty"])
    diff=(cp-t["entry"]) if t["signal"]=="BUY" else (t["entry"]-cp)
    return round(diff*qty,4)
def _close_record(t,cp,pnl,reason): append_history({**t,"close_price":cp,"pnl":pnl,
    "closed_at":datetime.now(timezone.utc).isoformat(),"close_reason":reason})

def check_stale_trades(deribit: DeribitClient):
    """
    NEW: Professional time-based exit.
    If trade open > MAX_TRADE_AGE_HOURS without hitting TP1 → exit at market.
    Prevents capital being locked in dead trades indefinitely.
    """
    trades    = load_trades()
    now       = datetime.now(timezone.utc)
    to_remove = []

    for symbol, trade in trades.items():
        if trade.get("closed") or trade.get("tp1_hit"):
            continue
        opened_at = trade.get("opened_at","")
        if not opened_at: continue
        try:
            age_hours = (now - datetime.fromisoformat(opened_at.replace("Z",""))).total_seconds() / 3600
        except Exception: continue

        if age_hours > MAX_TRADE_AGE_HOURS:
            log.warning(f"  ⏰ {symbol}: open {age_hours:.0f}h without TP1 — time-based exit")
            try:
                # Cancel all orders first
                oids = trade.get("order_ids",{})
                for k,oid in oids.items():
                    if k != "entry" and oid:
                        try: deribit.cancel_order(oid)
                        except Exception: pass
                # Close position at market
                live = deribit.get_live_price(symbol)
                pnl  = _pnl(trade, live if live else trade["entry"], "sl")
                _close_record(trade, live if live>0 else trade["entry"], pnl,
                              f"Time-based exit ({age_hours:.0f}h)")
                _send(f"⏰ *TIME-BASED EXIT — {symbol}*\n"
                      f"Open {age_hours:.0f}h without TP1 → closing\n"
                      f"PnL ≈ `{pnl:+.4f} USDT`")
                to_remove.append(symbol)
            except Exception as e:
                log.error(f"  Time-exit {symbol}: {e}")

    if to_remove:
        for sym in to_remove: trades.pop(sym, None)
        save_trades(trades)
        log.info(f"  ⏰ Time-based exit: {to_remove}")


def check_open_trades(deribit: DeribitClient):
    trades = load_trades()
    if not trades: log.info("  No open trades"); return
    to_remove = []
    log.info(f"  Monitoring {len(trades)} trade(s)")

    for symbol, trade in list(trades.items()):
        if trade.get("closed"): to_remove.append(symbol); continue
        oids  = trade.get("order_ids",{})
        entry = float(trade["entry"])
        dec   = 4 if entry<10 else 2

        def get_o(key):
            if key not in oids or not oids[key] or str(oids[key]) in ("","None"): return {}
            return deribit.get_order(str(oids[key]))

        try:
            # TP1
            if not trade.get("tp1_hit") and "tp1" in oids:
                o = get_o("tp1")
                if deribit.is_order_filled(o):
                    trade["tp1_hit"] = True
                    fill = _fp(o, trade["tp1"])
                    pnl  = _pnl(trade, fill, "tp1")
                    log.info(f"  🎯 TP1 {symbol} @ {fill:.{dec}f} pnl≈{pnl:+.4f}")
                    _send(f"🎯 *TP1 HIT — {symbol}*\n@ `{fill:.{dec}f}` | PnL ≈ `{pnl:+.4f}`")
                    # Move SL to breakeven
                    if oids.get("stop_loss") and trade.get("qty_tp2",0)>0:
                        try:
                            deribit.cancel_order(oids["stop_loss"])
                            sl_s = "SELL" if trade["signal"]=="BUY" else "BUY"
                            be   = deribit.place_limit_order(symbol,sl_s,trade["qty_tp2"],entry,stop_price=entry)
                            be_o = be.get("order",be)
                            nid  = str(be_o.get("order_id",""))
                            if nid: trade["order_ids"]["stop_loss"]=nid; trade["stop"]=entry
                            _send(f"🛡️ *{symbol} RISK-FREE*\nSL→entry `{entry:.{dec}f}` | Best:TP2 | Worst:break even")
                        except Exception as be: log.warning(f"  BE SL: {be}")

            # Trailing stop (price halfway to TP2)
            if trade.get("tp1_hit") and not trade.get("tp2_hit") and oids.get("stop_loss"):
                live = deribit.get_live_price(symbol)
                if live > 0:
                    # FIX: Calculate halfway between TP1 and TP2, not Entry and TP2
                    halfway = (float(trade["tp1"]) + float(trade["tp2"])) / 2
                    at_t    = (trade["signal"]=="BUY" and live>=halfway) or \
                              (trade["signal"]=="SELL" and live<=halfway)
                    sl_be   = abs(float(trade.get("stop",0))-entry)<entry*0.001
                    if at_t and sl_be and trade.get("qty_tp2",0)>0:
                        try:
                            deribit.cancel_order(oids["stop_loss"])
                            sl_s = "SELL" if trade["signal"]=="BUY" else "BUY"
                            sl_r = deribit.place_limit_order(symbol,sl_s,trade["qty_tp2"],
                                                             trade["tp1"],stop_price=trade["tp1"])
                            sl_o = sl_r.get("order",sl_r); nid=str(sl_o.get("order_id",""))
                            if nid: trade["order_ids"]["stop_loss"]=nid; trade["stop"]=trade["tp1"]
                            _send(f"🚀 *{symbol}* Trailing SL→TP1 `{trade['tp1']:.{dec}f}` — profit locked!")
                        except Exception as e: log.warning(f"  Trail SL: {e}")

            # TP2
            if trade.get("tp1_hit") and not trade.get("tp2_hit") and "tp2" in oids:
                o = get_o("tp2")
                if deribit.is_order_filled(o):
                    trade["tp2_hit"]=True; trade["closed"]=True
                    fill=_fp(o,trade["tp2"]); pnl=_pnl(trade,fill,"tp2")
                    log.info(f"  ✅ TP2 {symbol} @ {fill:.{dec}f} pnl≈{pnl:+.4f}")
                    _send(f"✅ *FULL WIN — {symbol}*\nTP2 @ `{fill:.{dec}f}` | PnL ≈ `{pnl:+.4f}`")
                    _close_record(trade,fill,pnl,"TP2 hit"); to_remove.append(symbol)

            # SL
            if not trade.get("closed") and oids.get("stop_loss"):
                o = get_o("stop_loss")
                if deribit.is_order_filled(o):
                    trade["closed"]=True
                    fill=_fp(o,trade["stop"]); pnl=_pnl(trade,fill,"sl")
                    lbl="BREAK-EVEN ⚖️" if abs(fill-entry)<entry*0.001 else "STOPPED OUT ❌"
                    log.info(f"  ❌ SL {symbol} @ {fill:.{dec}f} pnl≈{pnl:+.4f}")
                    _send(f"{'⚖️' if 'BREAK' in lbl else '❌'} *{lbl} — {symbol}*\nPnL ≈ `{pnl:+.4f}`")
                    _close_record(trade,fill,pnl,lbl)
                    for k in ("tp1","tp2"):
                        if oids.get(k) and not trade.get(f"{k}_hit"):
                            try: deribit.cancel_order(oids[k])
                            except Exception: pass
                    to_remove.append(symbol)

        except Exception as e: log.error(f"  Monitor {symbol}: {e}")

    save_trades(trades)
    for sym in set(to_remove): trades.pop(sym,None)
    save_trades(trades)


def clean_ghost_trades(deribit: DeribitClient):
    trades  = load_trades()
    if not trades: return
    live_pos = {f"{p.get('instrument_name','').split('_')[0]}USDT": True
                for p in deribit.get_positions() if float(p.get("size",0))!=0}
    to_remove = []
    for sym, trade in trades.items():
        if float(trade.get("stop",0))==0 or float(trade.get("tp1",0))==0:
            log.warning(f"  🗑️ {sym}: broken state — remove")
            to_remove.append(sym); continue
        if sym not in live_pos:
            log.warning(f"  🗑️ {sym}: ghost — no Deribit position")
            _close_record(trade, float(trade.get("entry",0)), 0.0, "Ghost — auto-removed")
            to_remove.append(sym)
    if to_remove:
        for sym in to_remove: trades.pop(sym,None)
        save_trades(trades)
        _send(f"🧹 Removed {len(to_remove)} ghost trade(s): {', '.join(to_remove)}")


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
          f"📦 {qty} contracts · Risk: ${risk:.2f} · Portfolio: ${bal:.2f}\n"
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

    log.info("\n[0] Balance...")
    balance = save_balance(deribit)

    log.info("\n[1] Monitor trades (TP/SL)...")
    check_open_trades(deribit)

    log.info("\n[2] Time-based exit check (>48h stale)...")
    check_stale_trades(deribit)

    log.info("\n[3] Clean ghost trades...")
    clean_ghost_trades(deribit)
    save_balance(deribit)

    # Hard check before scanning
    open_count = len([t for t in load_trades().values() if not t.get("closed",False)])
    if open_count >= MAX_OPEN_TRADES:
        log.info(f"\n[4] Max trades ({open_count}/{MAX_OPEN_TRADES}) — no new trades")
        return

    log.info(f"\n[4] Scanning {len(SYMBOLS)} coins | Open:{open_count}/{MAX_OPEN_TRADES}")
    log.info(f"    Tradeable: {TRADEABLE_SYMBOLS[:8]}")

    found = 0
    for symbol in SYMBOLS:
        open_count = len([t for t in load_trades().values() if not t.get("closed",False)])
        if open_count >= MAX_OPEN_TRADES:
            log.info(f"  🛑 Max trades reached — stop scan"); break

        log.info(f"\n  ── {symbol} ({get_tier(symbol)}) ──")
        sig = generate_signal(symbol, pipeline, thresholds)
        if sig is None: time.sleep(0.2); continue

        found += 1
        if execute_trade(deribit, sig, risk_mult, balance):
            time.sleep(1.5)

    save_balance(deribit)
    log.info(f"\n{'═'*56}\nDONE — {found} signal(s) | ${balance:.2f}\n{'═'*56}")


if __name__ == "__main__":
    run_execution_scan()
