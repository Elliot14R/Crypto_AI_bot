# smart_scheduler.py — Complete version with all required exports

import logging
import requests
import pandas as pd
from datetime import datetime, timezone

log = logging.getLogger(__name__)

ATR_VERY_HIGH = 4.0
ATR_HIGH_PCT  = 2.0
ATR_LOW_PCT   = 0.1


def get_scan_mode() -> dict:
    now        = datetime.now(timezone.utc)
    hour       = now.hour
    is_weekend = now.weekday() >= 5
    is_active  = 8 <= hour < 20

    if is_weekend:
        return {
            "mode": "weekend", "label": "WEEKEND MODE", "emoji": "📅",
            "min_confidence": 65, "min_score": 3, "min_adx": 20,
            "interval_min": 15 if is_active else 30, "risk_mult": 0.75,
            "description": "Weekend — reduced risk",
        }
    if is_active:
        return {
            "mode": "active", "label": "ACTIVE HOURS", "emoji": "📈",
            "min_confidence": 65, "min_score": 3, "min_adx": 20,
            "interval_min": 15, "risk_mult": 1.0,
            "description": "Active hours 08:00–20:00 UTC",
        }
    return {
        "mode": "quiet", "label": "QUIET HOURS", "emoji": "🌙",
        "min_confidence": 72, "min_score": 4, "min_adx": 25,
        "interval_min": 30, "risk_mult": 0.5,
        "description": "Quiet hours 00:00–08:00 UTC",
    }


def check_btc_volatility() -> dict:
    try:
        url    = "https://data-api.binance.vision/api/v3/klines"
        params = {"symbol": "BTCUSDT", "interval": "15m", "limit": 30}
        resp   = requests.get(url, params=params, timeout=10)
        data   = resp.json()

        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_vol","trades","tb_base","tb_quote","ignore"
        ])
        for col in ["high","low","close"]:
            df[col] = pd.to_numeric(df[col])

        prev_c = df["close"].shift(1)
        df["tr"] = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_c).abs(),
            (df["low"]  - prev_c).abs()
        ], axis=1).max(axis=1)

        atr     = df["tr"].rolling(14).mean().iloc[-1]
        price   = df["close"].iloc[-1]
        atr_pct = atr / price * 100

        if atr_pct > ATR_VERY_HIGH:
            return {"status":"VERY_HIGH","atr":round(atr,2),"atr_pct":round(atr_pct,3),
                    "price":round(price,2),"skip":False,"warn":True,"risk_mult":0.25,
                    "message":f"🚨 EXTREME VOLATILITY — BTC ATR {atr_pct:.2f}% — 25% position size"}
        elif atr_pct > ATR_HIGH_PCT:
            return {"status":"HIGH","atr":round(atr,2),"atr_pct":round(atr_pct,3),
                    "price":round(price,2),"skip":False,"warn":True,"risk_mult":0.5,
                    "message":f"⚠️ HIGH VOLATILITY — BTC ATR {atr_pct:.2f}% — 50% position size"}
        elif atr_pct < ATR_LOW_PCT:
            return {"status":"LOW","atr":round(atr,2),"atr_pct":round(atr_pct,3),
                    "price":round(price,2),"skip":True,"warn":False,"risk_mult":0.0,
                    "message":f"😴 Dead market — BTC ATR {atr_pct:.2f}% — skipping scan"}
        else:
            return {"status":"NORMAL","atr":round(atr,2),"atr_pct":round(atr_pct,3),
                    "price":round(price,2),"skip":False,"warn":False,"risk_mult":1.0,
                    "message":f"✓ Normal vol — BTC ATR {atr_pct:.2f}%"}

    except Exception as e:
        log.warning(f"Volatility check failed: {e}")
        return {"status":"UNKNOWN","atr":0,"atr_pct":0,"price":0,
                "skip":False,"warn":False,"risk_mult":1.0,
                "message":"Volatility check failed — scanning anyway"}


def check_correlation(trades: dict, new_signal: str) -> bool:
    """Returns True if OK to place trade. False if too many same-direction trades."""
    if not trades:
        return True
    same = sum(1 for t in trades.values()
               if t.get("signal") == new_signal and not t.get("closed", False))
    if same >= 2:
        log.info(f"  Correlation filter: already {same} {new_signal} trades — skip")
        return False
    return True


def should_scan() -> tuple:
    """Returns (should_run, mode, vol, reason)."""
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
        "risk_mult":      mode.get("risk_mult", 1.0),
    }


def get_effective_risk(mode: dict, vol: dict) -> float:
    """Combined risk multiplier: mode × volatility. Min 0.25."""
    return max(mode.get("risk_mult", 1.0) * vol.get("risk_mult", 1.0), 0.25)
