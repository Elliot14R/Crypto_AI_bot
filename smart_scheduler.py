# smart_scheduler.py — Fixed: removed blocking circuit breaker from scan path
#
# ROOT CAUSE OF NO TRADES FOR 1.5 DAYS:
#   check_daily_loss_limit() was called inside should_scan().
#   If balance.json was missing, stale, or showed 0 USDT:
#     limit = 0 * 5% = 0
#     any negative pnl today_pl < 0 → triggered = True → EVERY SCAN BLOCKED
#
# FIX: Daily loss check is now advisory only (logs warning, never blocks).
#      Thresholds also simplified — fewer combined conditions to pass.

import logging, requests, json
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

ATR_VERY_HIGH = 4.0
ATR_HIGH_PCT  = 2.0
ATR_DEAD_PCT  = 0.05   # very conservative — only skip truly dead markets


def get_scan_mode() -> dict:
    now        = datetime.now(timezone.utc)
    hour       = now.hour
    is_weekend = now.weekday() >= 5
    is_active  = 8 <= hour < 20

    if is_weekend:
        return {
            "mode": "weekend", "label": "WEEKEND MODE", "emoji": "📅",
            "min_confidence": 60, "min_score": 1, "min_adx": 15,
            "interval_min": 15, "risk_mult": 0.75,
        }
    if is_active:
        return {
            "mode": "active", "label": "ACTIVE HOURS", "emoji": "📈",
            "min_confidence": 60, "min_score": 1, "min_adx": 15,
            "interval_min": 15, "risk_mult": 1.0,
        }
    return {
        "mode": "quiet", "label": "QUIET HOURS", "emoji": "🌙",
        "min_confidence": 65, "min_score": 2, "min_adx": 18,
        "interval_min": 30, "risk_mult": 0.5,
    }


def check_btc_volatility() -> dict:
    try:
        r = requests.get(
            "https://data-api.binance.vision/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "15m", "limit": 30},
            timeout=10
        )
        data = r.json()
        df   = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_vol","trades","tb_base","tb_quote","ignore"
        ])
        for c in ["high","low","close"]:
            df[c] = pd.to_numeric(df[c])
        prev_c  = df["close"].shift(1)
        df["tr"] = pd.concat([
            df["high"]-df["low"],
            (df["high"]-prev_c).abs(),
            (df["low"]-prev_c).abs()
        ], axis=1).max(axis=1)
        atr     = df["tr"].rolling(14).mean().iloc[-1]
        price   = df["close"].iloc[-1]
        atr_pct = atr / price * 100

        if atr_pct > ATR_VERY_HIGH:
            return {"status":"VERY_HIGH","atr":round(atr,2),"atr_pct":round(atr_pct,3),
                    "price":round(price,2),"skip":False,"warn":True,"risk_mult":0.25,
                    "message":f"🚨 EXTREME VOL {atr_pct:.2f}% — 25% position"}
        elif atr_pct > ATR_HIGH_PCT:
            return {"status":"HIGH","atr":round(atr,2),"atr_pct":round(atr_pct,3),
                    "price":round(price,2),"skip":False,"warn":True,"risk_mult":0.5,
                    "message":f"⚠️ HIGH VOL {atr_pct:.2f}% — 50% position"}
        elif atr_pct < ATR_DEAD_PCT:
            return {"status":"DEAD","atr":round(atr,2),"atr_pct":round(atr_pct,3),
                    "price":round(price,2),"skip":True,"warn":False,"risk_mult":0.0,
                    "message":f"😴 Dead market {atr_pct:.2f}% — skip scan"}
        else:
            return {"status":"NORMAL","atr":round(atr,2),"atr_pct":round(atr_pct,3),
                    "price":round(price,2),"skip":False,"warn":False,"risk_mult":1.0,
                    "message":f"✓ Normal vol — BTC ATR {atr_pct:.2f}%"}
    except Exception as e:
        log.warning(f"Volatility check failed: {e}")
        # Default to normal — never block a scan due to API error
        return {"status":"UNKNOWN","atr":0,"atr_pct":0,"price":0,
                "skip":False,"warn":False,"risk_mult":1.0,
                "message":"Vol check failed — scanning anyway"}


def check_daily_pnl_advisory() -> str:
    """
    Advisory only — logs daily PnL status but NEVER blocks a scan.
    Returns a warning string if losses are high, empty string otherwise.
    """
    try:
        bal = {}
        for p in [Path("balance.json"), Path("data/balance.json")]:
            if p.exists():
                with open(p) as f: bal = json.load(f); break
        current_balance = float(bal.get("usdt", 0) or 0)
        if current_balance <= 0:
            return ""   # Can't compute, skip advisory

        hist = []
        for p in [Path("trade_history.json"), Path("data/trade_history.json")]:
            if p.exists():
                with open(p) as f: hist = json.load(f); break

        today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_pl = sum(
            float(h.get("pnl", 0) or 0) for h in hist
            if (h.get("closed_at","") or h.get("opened_at",""))[:10] == today
            and "Ghost" not in (h.get("close_reason",""))
            and "auto-removed" not in (h.get("close_reason",""))
        )

        limit = current_balance * 5.0 / 100
        if today_pl < -limit:
            msg = (f"⚠️ Daily loss advisory: {today_pl:.2f} USDT today "
                   f"(>{5.0}% of ${current_balance:.0f}) — consider manual review")
            log.warning(msg)
            return msg
        return ""
    except Exception:
        return ""


def check_correlation(trades: dict, new_signal: str) -> bool:
    """Max 2 trades in the same direction."""
    same = sum(1 for t in trades.values()
               if t.get("signal") == new_signal and not t.get("closed", False))
    if same >= 2:
        log.info(f"  Correlation filter: {same} {new_signal} already open — skip")
        return False
    return True


def should_scan() -> tuple:
    """
    Returns (run: bool, mode: dict, vol: dict, reason: str).
    Only blocks on genuine market conditions (dead market).
    Never blocks due to balance/PnL file issues.
    """
    mode = get_scan_mode()
    vol  = check_btc_volatility()

    log.info(f"  {mode['label']} | conf≥{mode['min_confidence']}% "
             f"| score≥{mode['min_score']} | ADX≥{mode['min_adx']} | {vol['message']}")

    # Only block on truly dead market (ATR < 0.05%)
    if vol.get("skip"):
        return False, mode, vol, vol["message"]

    # Log daily PnL advisory (never blocks)
    advisory = check_daily_pnl_advisory()
    if advisory:
        log.warning(advisory)

    return True, mode, vol, f"{mode['label']}"


def get_mode_thresholds(mode: dict) -> dict:
    return {
        "min_confidence": mode["min_confidence"],
        "min_score":      mode["min_score"],
        "min_adx":        mode["min_adx"],
        "risk_mult":      mode.get("risk_mult", 1.0),
    }


def get_effective_risk(mode: dict, vol: dict) -> float:
    combined = mode.get("risk_mult", 1.0) * vol.get("risk_mult", 1.0)
    return max(combined, 0.25)
