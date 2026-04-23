# feature_engineering.py — Enhanced features + ImportanceSelector
# ImportanceSelector MUST live here so joblib.load() can find it
# from both train_model.py and trade_executor.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# ── ImportanceSelector ────────────────────────────────────────────────
# Defined here (not in train_model.py) so the pickle can be loaded
# from any script that imports feature_engineering.
class ImportanceSelector(BaseEstimator, TransformerMixin):
    """Picklable feature selector — stores selected feature names."""
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names].values
        # Already a numpy array (during live prediction)
        return X


# ── Indicators ────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or len(df) < 20:
        return df

    df = df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]
    o = df["open"];  v = df["volume"]

    # EMAs
    df["ema9"]   = c.ewm(span=9,   adjust=False).mean()
    df["ema20"]  = c.ewm(span=20,  adjust=False).mean()
    df["ema50"]  = c.ewm(span=50,  adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    df["ema20_slope"]    = df["ema20"].diff(3) / df["ema20"].shift(3) * 100
    df["ema50_slope"]    = df["ema50"].diff(3) / df["ema50"].shift(3) * 100
    df["price_vs_ema20"] = (c - df["ema20"])  / df["ema20"]  * 100
    df["price_vs_ema50"] = (c - df["ema50"])  / df["ema50"]  * 100
    df["price_vs_ema200"]= (c - df["ema200"]) / df["ema200"] * 100
    df["ema20_vs_ema50"] = (df["ema20"] - df["ema50"]) / df["ema50"] * 100

    # RSI 14
    delta  = c.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(com=13, adjust=False).mean()
    avg_l  = loss.ewm(com=13, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(50)
    df["rsi_slope"] = df["rsi"].diff(3)

    # RSI 7 (fast)
    avg_g7 = gain.ewm(com=6,  adjust=False).mean()
    avg_l7 = loss.ewm(com=6,  adjust=False).mean()
    rs7    = avg_g7 / avg_l7.replace(0, np.nan)
    df["rsi_fast"] = (100 - 100 / (1 + rs7)).fillna(50)

    # Stochastic
    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df["stoch_k"] = 100 * (c - low14) / (high14 - low14 + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["macd_slope"]  = df["macd"].diff(3)

    # ATR
    prev_c = c.shift(1)
    tr     = pd.concat([h-l, (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    df["atr"]     = tr.ewm(span=14, adjust=False).mean()
    df["atr_pct"] = df["atr"] / c * 100

    # ADX
    dm_pos = (h.diff()).clip(lower=0)
    dm_neg = (-l.diff()).clip(lower=0)
    dm_pos = dm_pos.where(dm_pos > dm_neg, 0)
    dm_neg = dm_neg.where(dm_neg > dm_pos, 0)
    atr14  = tr.ewm(span=14, adjust=False).mean()
    di_pos = 100 * dm_pos.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan)
    di_neg = 100 * dm_neg.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan)
    dx     = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg + 1e-10)
    df["adx"]     = dx.ewm(span=14, adjust=False).mean()
    df["adx_pos"] = di_pos
    df["adx_neg"] = di_neg
    df["di_diff"] = di_pos - di_neg

    # Bollinger Bands
    sma20    = c.rolling(20).mean()
    std20    = c.rolling(20).std()
    bb_high  = sma20 + 2 * std20
    bb_low   = sma20 - 2 * std20
    bb_width = bb_high - bb_low
    df["bb_high"]  = bb_high
    df["bb_low"]   = bb_low
    df["bb_pct"]   = (c - bb_low) / (bb_width + 1e-10)
    df["bb_width"] = bb_width / sma20 * 100

    # Volume
    vol_ma20         = v.rolling(20).mean()
    df["volume_ratio"] = v / vol_ma20.replace(0, np.nan)
    df["volume_spike"] = (df["volume_ratio"] > 2.0).astype(int)
    obv              = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df["obv_slope"]  = obv.diff(5) / (vol_ma20 * 5 + 1e-10)

    # Price action
    df["price_change"]  = c.pct_change(1) * 100
    df["price_change3"] = c.pct_change(3) * 100
    df["price_change6"] = c.pct_change(6) * 100
    df["high_low_pct"]  = (h - l) / c * 100
    df["body_pct"]      = (c - o).abs() / (h - l + 1e-10)
    df["momentum"]      = c - c.shift(10)
    df["volatility"]    = c.rolling(14).std() / c * 100

    # Candlestick patterns
    body       = (c - o).abs()
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l
    rng        = h - l + 1e-10
    df["bullish_candle"] = ((c > o) & (body > rng * 0.6)).astype(int)
    df["doji"]           = (body < rng * 0.1).astype(int)
    df["hammer"]         = ((lower_wick > body * 2) & (upper_wick < body)).astype(int)

    # Trend
    df["trend"] = np.where(df["ema20"] > df["ema50"], 1,
                  np.where(df["ema20"] < df["ema50"], -1, 0))

    # 1h placeholders (filled by trade_executor)
    if "rsi_1h"   not in df.columns: df["rsi_1h"]   = 50.0
    if "adx_1h"   not in df.columns: df["adx_1h"]   = 0.0
    if "trend_1h" not in df.columns: df["trend_1h"] = 0.0

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    return df


# Full feature list (must match pipeline["all_features"])
ALL_FEATURES = [
    "ema9","ema20","ema50","ema200","ema20_slope","ema50_slope",
    "price_vs_ema20","price_vs_ema50","price_vs_ema200","ema20_vs_ema50",
    "rsi","rsi_slope","rsi_fast","stoch_k","stoch_d",
    "macd","macd_signal","macd_hist","macd_slope",
    "adx","adx_pos","adx_neg","di_diff","atr","atr_pct",
    "bb_high","bb_low","bb_pct","bb_width",
    "volume_ratio","volume_spike","obv_slope",
    "price_change","price_change3","price_change6",
    "high_low_pct","body_pct","momentum","volatility",
    "bullish_candle","doji","hammer",
    "rsi_1h","adx_1h","trend_1h",
]
