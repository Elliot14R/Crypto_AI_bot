# smart_scheduler.py
# Smart scanning rules — active/quiet/weekend + BTC volatility check
# Import this in trade_executor.py or run standalone

import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ── Base thresholds (overridden by mode) ───────────────────────
BASE_CONFIDENCE = 65
BASE_SCORE      = 3
BASE_ADX        = 20

# ── Volatility thresholds ──────────────────────────────────────
ATR_HIGH_PCT    = 2.0    # above 2% = high vol → warn, reduce size
ATR_LOW_PCT     = 0.3    # below 0.3% = low vol → skip scan entirely


def get_scan_mode() -> dict:
    """
    Returns current scan mode based on UTC time + day of week.

    Modes:
      active  — Mon-Fri 08:00–20:00 UTC — scan every 15 min, conf 65%
      quiet   — Mon-Fri 00:00–08:00 UTC — scan every 30 min, conf 72%, ADX 30+
      weekend — Sat/Sun                 — conf raised to 70%
    """
    now     = datetime.now(timezone.utc)
    hour    = now.hour
    weekday = now.weekday()   # 0=Mon, 5=Sat, 6=Sun
    is_weekend = weekday >= 5

    if is_weekend:
        active = (8 <= hour < 20)
        return {
            "mode":        "weekend",
            "label":       "WEEKEND MODE",
            "emoji":       "📅",
            "min_confidence": 70,
            "min_score":   4,
            "min_adx":     BASE_ADX,
            "interval_min": 15 if active else 30,
            "extra_adx":   False,
            "description": f"Weekend — confidence raised to 70% (low volume)",
        }

    if 8 <= hour < 20:
        return {
            "mode":        "active",
            "label":       "ACTIVE HOURS",
            "emoji":       "📈",
            "min_confidence": BASE_CONFIDENCE,
            "min_score":   BASE_SCORE,
            "min_adx":     BASE_ADX,
            "interval_min": 15,
            "extra_adx":   False,
            "description": "Active trading hours (08:00–20:00 UTC) — full scan",
        }

    # Quiet hours
    return {
        "mode":        "quiet",
        "label":       "QUIET HOURS",
        "emoji":       "🌙",
        "min_confidence": 72,
        "min_score":   4,
        "min_adx":     30,
        "interval_min": 30,
        "extra_adx":   True,
        "description": "Quiet hours (00:00–08:00 UTC) — higher threshold",
    }


def check_btc_volatility() -> dict:
    """
    Fetch BTC 15m ATR and classify as HIGH / NORMAL / LOW.
    Uses Binance 15m OHLCV to compute ATR(14).
    """
    try:
        url    = "https://api.binance.com/api/v3/klines"
        params = {"symbol": "BTCUSDT", "interval": "15m", "limit": 30}
        resp   = requests.get(url, params=params, timeout=10)
        data   = resp.json()

        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"
        ])
        for c in ["high","low","close"]:
            df[c] = pd.to_numeric(df[c])

        # Simple ATR(14) = rolling mean of true range
        df["prev_close"] = df["close"].shift(1)
        df["tr"] = df[["high","low","prev_close"]].apply(
            lambda r: max(r.high - r.low,
                          abs(r.high - r.prev_close),
                          abs(r.low  - r.prev_close)), axis=1
        )
        atr       = df["tr"].rolling(14).mean().iloc[-1]
        price     = df["close"].iloc[-1]
        atr_pct   = atr / price * 100

        if atr_pct > ATR_HIGH_PCT:
            status = "HIGH"
            skip   = False
            warn   = True
            msg    = f"⚠️ HIGH VOLATILITY — BTC ATR {atr_pct:.2f}% (>{ATR_HIGH_PCT}%) — reduce position size!"
        elif atr_pct < ATR_LOW_PCT:
            status = "LOW"
            skip   = True
            warn   = False
            msg    = f"😴 LOW VOLATILITY — BTC ATR {atr_pct:.2f}% (<{ATR_LOW_PCT}%) — market ranging, scan skipped"
        else:
            status = "NORMAL"
            skip   = False
            warn   = False
            msg    = f"✓ Normal volatility — BTC ATR {atr_pct:.2f}%"

        log.info(f"  Volatility: {status} | ATR {atr:.2f} | ATR% {atr_pct:.2f}%")

        return {
            "status":   status,
            "atr":      round(atr, 2),
            "atr_pct":  round(atr_pct, 3),
            "price":    round(price, 2),
            "skip":     skip,
            "warn":     warn,
            "message":  msg,
        }

    except Exception as e:
        log.warning(f"  Volatility check failed: {e}")
        return {"status": "UNKNOWN", "atr": 0, "atr_pct": 0, "skip": False, "warn": False, "message": ""}


def should_scan() -> tuple:
    """
    Master decision function.
    Returns (should_run: bool, mode: dict, vol: dict, reason: str)
    """
    mode = get_scan_mode()
    vol  = check_btc_volatility()

    log.info(f"\n  Mode: {mode['label']} | Min conf: {mode['min_confidence']}% | "
             f"Min score: {mode['min_score']} | Interval: {mode['interval_min']}min")

    if vol["skip"]:
        return False, mode, vol, vol["message"]

    return True, mode, vol, f"{mode['label']} — {vol['status']} volatility"


def get_mode_thresholds(mode: dict) -> dict:
    """Return the confidence/score/adx thresholds for the current mode."""
    return {
        "min_confidence": mode["min_confidence"],
        "min_score":      mode["min_score"],
        "min_adx":        mode["min_adx"],
    }
