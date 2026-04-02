# smart_scheduler.py — Fixed version
# Key fixes:
#   - ATR low threshold lowered (was skipping too many scans)
#   - Quiet hours threshold less aggressive (was blocking all trades)
#   - Better logging so you can see exactly why scans are skipped

import logging
import requests
import pandas as pd
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ── Base thresholds ───────────────────────────────────────────────
BASE_CONFIDENCE = 60   # relaxed from 65
BASE_SCORE      = 2    # relaxed from 3
BASE_ADX        = 18   # relaxed from 20

# ── Volatility thresholds ─────────────────────────────────────────
# FIXED: was ATR_LOW_PCT=0.3 which skipped most scans (BTC ATR is often 0.3-0.8%)
ATR_HIGH_PCT = 3.0   # above 3% = high vol warning
ATR_LOW_PCT  = 0.1   # below 0.1% = truly dead market, skip (was 0.3 — too aggressive)


def get_scan_mode() -> dict:
    """Returns scan mode based on UTC time + day of week."""
    now        = datetime.now(timezone.utc)
    hour       = now.hour
    weekday    = now.weekday()   # 0=Mon, 5=Sat, 6=Sun
    is_weekend = weekday >= 5
    is_active  = (8 <= hour < 20)

    if is_weekend:
        return {
            "mode":           "weekend",
            "label":          "WEEKEND MODE",
            "emoji":          "📅",
            "min_confidence": 65,    # raised by 5 on weekends
            "min_score":      2,
            "min_adx":        18,
            "interval_min":   15 if is_active else 30,
            "description":    "Weekend — confidence raised to 65%",
        }

    if is_active:
        return {
            "mode":           "active",
            "label":          "ACTIVE HOURS",
            "emoji":          "📈",
            "min_confidence": BASE_CONFIDENCE,
            "min_score":      BASE_SCORE,
            "min_adx":        BASE_ADX,
            "interval_min":   15,
            "description":    "Active hours 08:00–20:00 UTC",
        }

    # Quiet hours — FIXED: was 72% conf which basically blocked everything
    return {
        "mode":           "quiet",
        "label":          "QUIET HOURS",
        "emoji":          "🌙",
        "min_confidence": 68,   # was 72 — that was too high
        "min_score":      3,    # was 4
        "min_adx":        22,   # was 30 — way too strict
        "interval_min":   30,
        "description":    "Quiet hours 00:00–08:00 UTC",
    }


def check_btc_volatility() -> dict:
    """Fetch BTC 15m data and check ATR as % of price."""
    try:
        url    = "https://data-api.binance.vision/api/v3/klines"
        params = {"symbol": "BTCUSDT", "interval": "15m", "limit": 30}
        resp   = requests.get(url, params=params, timeout=10)
        data   = resp.json()

        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"
        ])
        for c in ["high","low","close"]:
            df[c] = pd.to_numeric(df[c])

        df["prev_close"] = df["close"].shift(1)
        df["tr"] = df.apply(
            lambda r: max(r["high"]-r["low"],
                          abs(r["high"]-r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
                          abs(r["low"] -r["prev_close"]) if pd.notna(r["prev_close"]) else 0), axis=1
        )
        atr     = df["tr"].rolling(14).mean().iloc[-1]
        price   = df["close"].iloc[-1]
        atr_pct = atr / price * 100

        if atr_pct > ATR_HIGH_PCT:
            return {"status":"HIGH",   "atr":round(atr,2),"atr_pct":round(atr_pct,3),
                    "price":round(price,2),"skip":False,"warn":True,
                    "message":f"⚠️ HIGH VOLATILITY — BTC ATR {atr_pct:.2f}% — reduce position size!"}
        elif atr_pct < ATR_LOW_PCT:
            return {"status":"LOW",    "atr":round(atr,2),"atr_pct":round(atr_pct,3),
                    "price":round(price,2),"skip":True,"warn":False,
                    "message":f"😴 Market dead — BTC ATR {atr_pct:.2f}% — scan skipped"}
        else:
            return {"status":"NORMAL", "atr":round(atr,2),"atr_pct":round(atr_pct,3),
                    "price":round(price,2),"skip":False,"warn":False,
                    "message":f"✓ Normal vol — BTC ATR {atr_pct:.2f}%"}

    except Exception as e:
        log.warning(f"  Volatility check failed: {e} — defaulting to NORMAL (will scan)")
        # FIXED: on error, default to running the scan rather than skipping
        return {"status":"UNKNOWN","atr":0,"atr_pct":0,"price":0,
                "skip":False,"warn":False,"message":"Volatility check failed — scan running anyway"}


def should_scan() -> tuple:
    """
    Returns (should_run, mode, vol, reason).
    Only skips if ATR is truly dead (< 0.1%).
    """
    mode = get_scan_mode()
    vol  = check_btc_volatility()

    log.info(f"  Mode: {mode['label']} | conf≥{mode['min_confidence']}% "
             f"| score≥{mode['min_score']} | ADX≥{mode['min_adx']} | {vol['message']}")

    if vol["skip"]:
        return False, mode, vol, vol["message"]

    return True, mode, vol, f"{mode['label']} — ATR {vol['status']}"


def get_mode_thresholds(mode: dict) -> dict:
    return {
        "min_confidence": mode["min_confidence"],
        "min_score":      mode["min_score"],
        "min_adx":        mode["min_adx"],
    }
