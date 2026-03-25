# run_scan_once.py
# GitHub Actions version of live_scanner.py
# Runs ONE scan and exits — GitHub Actions handles the scheduling

import requests
import pandas as pd
import joblib
import time
import logging
from datetime import datetime
from feature_engineering import add_indicators
from telegram_alert import send_signal
from news_sentiment import get_news_sentiment, get_market_conditions
from config import (
    SYMBOLS, FEATURES, TIMEFRAME_ENTRY, TIMEFRAME_CONFIRM,
    TIMEFRAME_TREND, MIN_CONFIDENCE, MIN_ADX, MIN_SCORE,
    ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT,
    MODEL_FILE, LIVE_LIMIT, LOG_FILE
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
log = logging.getLogger(__name__)


def load_model():
    try:
        pipeline = joblib.load(MODEL_FILE)
        log.info(f"Model loaded: {MODEL_FILE}")
        return pipeline
    except FileNotFoundError:
        log.error("Model file not found! Upload pro_crypto_ai_model.pkl to your repo.")
        raise


def get_data(symbol, interval):
    # Using Binance's public data endpoint to bypass GitHub Actions US geo-blocks
    url    = "https://data-api.binance.vision/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": LIVE_LIMIT}
    resp   = requests.get(url, params=params, timeout=15)
    data   = resp.json()

    # Safety Catch: If Binance returns an error dictionary instead of price data
    if isinstance(data, dict):
        raise ValueError(f"Binance API Error/Block: {data}")

    df     = pd.DataFrame(data).iloc[:, :6]
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c])
    return df


def quality_score(row_entry, row_confirm, row_trend, signal, confidence):
    score   = 0
    reasons = []

    if confidence >= 75:
        score += 1
        reasons.append(f"High AI confidence ({confidence:.0f}%)")
    elif confidence >= 65:
        reasons.append(f"AI confidence ({confidence:.0f}%)")

    adx = row_entry.get("adx", 0)
    if adx > 25:
        score += 1
        reasons.append(f"Strong trend (ADX {adx:.0f})")
    elif adx > 20:
        score += 1
        reasons.append(f"Moderate trend (ADX {adx:.0f})")

    rsi = row_entry.get("rsi", 50)
    if signal == "BUY" and rsi < 40:
        score += 1
        reasons.append(f"RSI oversold ({rsi:.0f})")
    elif signal == "SELL" and rsi > 60:
        score += 1
        reasons.append(f"RSI overbought ({rsi:.0f})")

    e20  = row_entry.get("ema20",  0)
    e50  = row_entry.get("ema50",  0)
    e200 = row_entry.get("ema200", 0)
    if signal == "BUY" and e20 > e50 > e200:
        score += 1
        reasons.append("EMA uptrend (20 > 50 > 200)")
    elif signal == "SELL" and e20 < e50 < e200:
        score += 1
        reasons.append("EMA downtrend (20 < 50 < 200)")

    if signal == "BUY" and row_confirm.get("ema20", 0) > row_confirm.get("ema50", 0):
        score += 1
        reasons.append(f"{TIMEFRAME_CONFIRM} EMA confirms uptrend")
    elif signal == "SELL" and row_confirm.get("ema20", 0) < row_confirm.get("ema50", 0):
        score += 1
        reasons.append(f"{TIMEFRAME_CONFIRM} EMA confirms downtrend")

    if signal == "BUY" and row_trend.get("ema20", 0) > row_trend.get("ema50", 0):
        score += 1
        reasons.append(f"{TIMEFRAME_TREND} EMA confirms uptrend")
    elif signal == "SELL" and row_trend.get("ema20", 0) < row_trend.get("ema50", 0):
        score += 1
        reasons.append(f"{TIMEFRAME_TREND} EMA confirms downtrend")

    return score, reasons


def scan_symbol(symbol, pipeline):
    try:
        df_entry   = add_indicators(get_data(symbol, TIMEFRAME_ENTRY))
        df_confirm = add_indicators(get_data(symbol, TIMEFRAME_CONFIRM))
        df_trend   = add_indicators(get_data(symbol, TIMEFRAME_TREND))

        if df_entry.empty or len(df_entry) < 10:
            return

        # Use .copy() so we can safely attach new columns to this row
        row_entry   = df_entry.iloc[-1].copy()
        row_confirm = df_confirm.iloc[-1] if not df_confirm.empty else pd.Series(dtype=float)
        row_trend   = df_trend.iloc[-1]   if not df_trend.empty   else pd.Series(dtype=float)

        # Attach the missing 1-hour features the AI model requires
        row_entry['rsi_1h']   = row_confirm.get('rsi', 50)
        row_entry['adx_1h']   = row_confirm.get('adx', 0)
        row_entry['trend_1h'] = row_confirm.get('trend', 0)

        all_features = pipeline["all_features"]
        selector     = pipeline["selector"]
        ensemble     = pipeline["ensemble"]

        missing = [f for f in all_features if f not in df_entry.columns and f not in row_entry.index]
        if missing:
            log.warning(f"  {symbol}: Missing features {missing}")
            return

        X_raw = pd.DataFrame([row_entry[all_features].values], columns=all_features)
        X_sel = selector.transform(X_raw)

        pred       = ensemble.predict(X_sel)[0]
        prob       = ensemble.predict_proba(X_sel)[0]
        labels     = {0: "BUY", 1: "SELL", 2: "NO_TRADE"}
        signal     = labels[pred]
        confidence = round(float(max(prob)) * 100, 1)

        if signal == "NO_TRADE" or confidence < MIN_CONFIDENCE:
            log.info(f"  {symbol}: {signal} {confidence}% — skipped")
            return

        adx_val = float(row_entry.get("adx", 0))
        if adx_val < MIN_ADX:
            log.info(f"  {symbol}: ADX {adx_val:.0f} too low — skipped")
            return

        score, reasons = quality_score(row_entry, row_confirm, row_trend, signal, confidence)

        if score < MIN_SCORE:
            log.info(f"  {symbol}: Score {score}/6 too low — skipped")
            return

        entry = float(row_entry["close"])
        atr   = float(row_entry["atr"])
        dec   = 4 if entry < 10 else 2

        if signal == "BUY":
            stop = round(entry - atr * ATR_STOP_MULT,    dec)
            t1   = round(entry + atr * ATR_TARGET1_MULT, dec)
            t2   = round(entry + atr * ATR_TARGET2_MULT, dec)
        else:
            stop = round(entry + atr * ATR_STOP_MULT,    dec)
            t1   = round(entry - atr * ATR_TARGET1_MULT, dec)
            t2   = round(entry - atr * ATR_TARGET2_MULT, dec)

        # News sentiment
        try:
            market = get_market_conditions()
            news   = get_news_sentiment(symbol)
            if news["sentiment"] == "BULLISH" and signal == "BUY":
                score += 1
                reasons.append(f"News BULLISH (score: {news['score']:+d})")
            elif news["sentiment"] == "BEARISH" and signal == "SELL":
                score += 1
                reasons.append(f"News BEARISH (score: {news['score']:+d})")
            elif news["sentiment"] == "BULLISH" and signal == "SELL":
                reasons.append("News BULLISH but signal SELL — caution")
            elif news["sentiment"] == "BEARISH" and signal == "BUY":
                reasons.append("News BEARISH but signal BUY — caution")
            fg = market.get("fear_greed", 50)
            reasons.append(f"Fear & Greed: {fg} — {market.get('label', '')}")
        except Exception as news_err:
            log.warning(f"  {symbol}: News check failed — {news_err}")

        log.info(f"  ✅ SIGNAL: {symbol} {signal} | {confidence}% | score {score}/6")
        send_signal(
            symbol=symbol, signal=signal, confidence=confidence,
            entry=round(entry, dec), stop=stop, t1=t1, t2=t2,
            reasons=reasons, score=score,
        )

    except Exception as e:
        log.error(f"  {symbol}: Error — {e}")


def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    log.info(f"\n{'─'*50}")
    log.info(f"GitHub Actions Scan — {now}")
    log.info(f"Scanning {len(SYMBOLS)} symbols")
    log.info(f"{'─'*50}")

    pipeline = load_model()

    for symbol in SYMBOLS:
        scan_symbol(symbol, pipeline)
        time.sleep(0.5)   # gentle rate limiting

    log.info("Scan complete.")


if __name__ == "__main__":
    main()
