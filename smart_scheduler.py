# smart_scheduler.py — Relaxed thresholds to allow trades
#
# KEY INSIGHT: The model predicts BUY/SELL at ~36-43% confidence because
# crypto markets in April 2026 are ranging/choppy. The 65% threshold
# blocks ALL signals. Lowering to 55% active / 60% quiet lets trades fire.
#
# THRESHOLDS LOGIC:
#   - Model is 73% accurate at test time, but live confidence varies
#   - In ranging markets, even 55% confidence = real edge over random
#   - Score ≥ 2 means at least 2 of 5 quality checks passed
#   - ADX ≥ 15 means SOME trend exists (was 20, too strict for current market)

import logging
import requests
import pandas as pd
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ── Volatility thresholds ──────────────────────────────────────────
ATR_HIGH_PCT = 3.0   # above 3% = high vol, half position size
ATR_LOW_PCT  = 0.08  # below 0.08% = dead market, skip scan


def get_scan_mode() -> dict:
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
            "min_confidence": 58,   # RELAXED: was 65
            "min_score":      2,
            "min_adx":        15,   # RELAXED: was 18
            "interval_min":   15 if is_active else 30,
            "description":    "Weekend — relaxed thresholds, smaller positions",
        }

    if is_active:
        return {
            "mode":           "active",
            "label":          "ACTIVE HOURS",
            "emoji":          "📈",
            "min_confidence": 55,   # RELAXED: was 60/65 — allows real signals through
            "min_score":      2,    # need 2/5 quality checks
            "min_adx":        15,   # RELAXED: was 18/20 — current market is ranging
            "interval_min":   15,
            "description":    "Active hours 08:00–20:00 UTC",
        }

    # Quiet hours (00:00-08:00 UTC)
    return {
        "mode":           "quiet",
        "label":          "QUIET HOURS",
        "emoji":          "🌙",
        "min_confidence": 62,   # RELAXED: was 68/72 — still higher than active
        "min_score":      2,
        "min_adx":        18,   # RELAXED: was 22/30
        "interval_min":   30,
        "description":    "Quiet hours 00:00–08:00 UTC",
    }


def check_btc_volatility() -> dict:
    """Check BTC ATR as % of price to assess market conditions."""
    try:
        # Try CDN mirror first (less restricted)
        for base_url in [
            "https://data-api.binance.vision/api/v3/klines",
            "https://api.binance.com/api/v3/klines",
        ]:
            try:
                resp = requests.get(base_url,
                    params={"symbol": "BTCUSDT", "interval": "15m", "limit": 30},
                    timeout=10)
                if resp.ok:
                    data = resp.json()
                    break
            except Exception:
                continue
        else:
            raise Exception("All endpoints failed")

        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"
        ])
        for c in ["high","low","close"]:
            df[c] = pd.to_numeric(df[c])

        df["prev_close"] = df["close"].shift(1)
        df["tr"] = df.apply(
            lambda r: max(
                r["high"] - r["low"],
                abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
                abs(r["low"]  - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
            ), axis=1
        )
        atr     = df["tr"].rolling(14).mean().iloc[-1]
        price   = df["close"].iloc[-1]
        atr_pct = atr / price * 100

        if atr_pct > ATR_HIGH_PCT:
            return {
                "status": "HIGH", "atr": round(atr, 2), "atr_pct": round(atr_pct, 3),
                "price": round(price, 2), "skip": False, "warn": True,
                "message": f"⚠️ HIGH VOLATILITY — BTC ATR {atr_pct:.2f}% — half position size",
            }
        elif atr_pct < ATR_LOW_PCT:
            return {
                "status": "LOW", "atr": round(atr, 2), "atr_pct": round(atr_pct, 3),
                "price": round(price, 2), "skip": True, "warn": False,
                "message": f"😴 Dead market — BTC ATR {atr_pct:.2f}% < {ATR_LOW_PCT}% — scan skipped",
            }
        else:
            return {
                "status": "NORMAL", "atr": round(atr, 2), "atr_pct": round(atr_pct, 3),
                "price": round(price, 2), "skip": False, "warn": False,
                "message": f"✓ Normal vol — BTC ATR {atr_pct:.2f}%",
            }

    except Exception as e:
        log.warning(f"Volatility check failed: {e} — defaulting to NORMAL")
        return {
            "status": "UNKNOWN", "atr": 0, "atr_pct": 0.5, "price": 0,
            "skip": False, "warn": False,
            "message": "Volatility check failed — scanning anyway",
        }


def should_scan() -> tuple:
    """Returns (should_run, mode, vol, reason)."""
    mode = get_scan_mode()
    vol  = check_btc_volatility()

    log.info(
        f"  Mode: {mode['label']} | "
        f"conf≥{mode['min_confidence']}% | "
        f"score≥{mode['min_score']} | "
        f"ADX≥{mode['min_adx']} | "
        f"{vol['message']}"
    )

    if vol["skip"]:
        return False, mode, vol, vol["message"]

    return True, mode, vol, f"{mode['label']} — ATR {vol['status']}"


def get_mode_thresholds(mode: dict) -> dict:
    return {
        "min_confidence": mode["min_confidence"],
        "min_score":      mode["min_score"],
        "min_adx":        mode["min_adx"],
    }


def check_correlation(open_trades: dict, new_signal: str) -> bool:
    """Max 2 trades in same direction at once."""
    try:
        from config import MAX_SAME_DIRECTION
    except ImportError:
        MAX_SAME_DIRECTION = 2

    same_dir = sum(
        1 for t in open_trades.values()
        if t.get("signal") == new_signal and not t.get("closed")
    )
    if same_dir >= MAX_SAME_DIRECTION:
        log.info(f"  Correlation filter: {same_dir} {new_signal} trades already open — skip")
        return False
    return True


def get_effective_risk(mode: dict, vol: dict) -> float:
    """
    Returns risk multiplier (0.5 – 1.0) based on mode and volatility.
    Applied to RISK_PER_TRADE (1% of balance).
    """
    if vol.get("status") == "HIGH":
        return 0.5    # high vol → half position

    mode_name = mode.get("mode", "active")
    if mode_name == "quiet":
        return 0.75   # quiet hours → 75% size
    if mode_name == "weekend":
        return 0.75   # weekend → 75% size

    return 1.0        # active + normal vol → full 1%
