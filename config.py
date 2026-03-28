# config.py — All settings in one place
# 20 coins, all verified on Binance Spot/Testnet

# ── Coin Universe — 4 Tiers ───────────────────────────────────────
# Tier 1 — 👑 Big Three
TIER_BIG3    = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

# Tier 2 — ⚡ High-Liquidity L1s
TIER_LIQ1    = ["SOLUSDT", "AVAXUSDT", "NEARUSDT", "SUIUSDT", "APTUSDT"]

# Tier 3 — 🏛️ Institutional Alts
TIER_INST    = ["LINKUSDT", "DOTUSDT", "UNIUSDT", "AAVEUSDT", "XRPUSDT"]

# Tier 4 — 🤖 AI & Oracles + 🚀 Momentum
TIER_AI_MOM  = ["FETUSDT", "RENDERUSDT", "ADAUSDT", "INJUSDT", "ARBUSDT", "OPUSDT", "SEIUSDT"]

# All 20 coins
SYMBOLS = TIER_BIG3 + TIER_LIQ1 + TIER_INST + TIER_AI_MOM

COIN_TIERS = {
    "big3":   {"label": "👑 Big Three",          "coins": TIER_BIG3},
    "liq1":   {"label": "⚡ High-Liquidity L1s", "coins": TIER_LIQ1},
    "inst":   {"label": "🏛 Institutional Alts", "coins": TIER_INST},
    "ai_mom": {"label": "🤖 AI & Momentum",      "coins": TIER_AI_MOM},
}

def get_tier(symbol: str) -> str:
    for t in COIN_TIERS.values():
        if symbol in t["coins"]: return t["label"]
    return "Unknown"

# ── Timeframes ────────────────────────────────────────────────────
TIMEFRAME_ENTRY   = "15m"
TIMEFRAME_CONFIRM = "1h"
TIMEFRAME_TREND   = "4h"

DOWNLOAD_LIMIT    = 1500
LIVE_LIMIT        = 300

# ── Features (must match feature_engineering.py output) ──────────
FEATURES = [
    "ema9","ema20","ema50","ema200",
    "ema20_slope","ema50_slope",
    "price_vs_ema20","price_vs_ema50","price_vs_ema200","ema20_vs_ema50",
    "rsi","rsi_slope","rsi_fast",
    "stoch_k","stoch_d",
    "macd","macd_signal","macd_hist","macd_slope",
    "adx","adx_pos","adx_neg","di_diff",
    "atr","atr_pct",
    "bb_high","bb_low","bb_pct","bb_width",
    "volume_ratio","volume_spike","obv_slope",
    "price_change","price_change3","price_change6",
    "high_low_pct","body_pct","momentum","volatility",
    "bullish_candle","doji","hammer",
    "rsi_1h","adx_1h","trend_1h",
]

# ── Signal thresholds — BASE values, smart_scheduler overrides ────
# IMPORTANT: Kept deliberately relaxed so signals actually fire
MIN_CONFIDENCE    = 60   # lowered from 65 — model is 73% accurate, 60 gives more trades
MIN_ADX           = 18   # lowered from 20
MIN_SCORE         = 2    # lowered from 3 — score of 2/6 still valid

# ── Risk management ───────────────────────────────────────────────
ATR_STOP_MULT     = 1.5
ATR_TARGET1_MULT  = 2.0
ATR_TARGET2_MULT  = 3.0
RISK_PER_TRADE    = 0.01   # 1% of balance

# ── Training ──────────────────────────────────────────────────────
TARGET_FUTURE     = 6
TARGET_THRESHOLD  = 0.005
TEST_SPLIT        = 0.30
RANDOM_STATE      = 42

# ── Scheduling ────────────────────────────────────────────────────
SCAN_INTERVAL_MIN = 15

# ── Files ─────────────────────────────────────────────────────────
RAW_DATA_FILE  = "training_data.csv"
FEATURES_FILE  = "training_features.csv"
DATASET_FILE   = "training_dataset.csv"
TRAIN_FILE     = "train_data.csv"
TEST_FILE      = "test_data.csv"
MODEL_FILE     = "pro_crypto_ai_model.pkl"
LOG_FILE       = "bot.log"
