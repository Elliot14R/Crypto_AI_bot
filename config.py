# config.py — Complete settings file
# ALL imports that trade_executor.py and other files need are here

TIER_BIG3   = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
TIER_LIQ1   = ["SOLUSDT", "AVAXUSDT", "NEARUSDT", "SUIUSDT", "APTUSDT"]
TIER_INST   = ["LINKUSDT", "DOTUSDT", "UNIUSDT", "AAVEUSDT", "XRPUSDT"]
TIER_AI_MOM = ["FETUSDT", "RENDERUSDT", "ADAUSDT", "INJUSDT", "ARBUSDT", "OPUSDT", "SEIUSDT"]

SYMBOLS = TIER_BIG3 + TIER_LIQ1 + TIER_INST + TIER_AI_MOM

COIN_TIERS = {
    "big3":   {"label": "👑 Big Three",          "coins": TIER_BIG3},
    "liq1":   {"label": "⚡ High-Liquidity L1s", "coins": TIER_LIQ1},
    "inst":   {"label": "🏛 Institutional Alts", "coins": TIER_INST},
    "ai_mom": {"label": "🤖 AI & Momentum",      "coins": TIER_AI_MOM},
}

COIN_META = {
    "BTCUSDT":    {"name": "Bitcoin",   "short": "BTC",  "color": "#f7931a"},
    "ETHUSDT":    {"name": "Ethereum",  "short": "ETH",  "color": "#627eea"},
    "BNBUSDT":    {"name": "BNB",       "short": "BNB",  "color": "#f3ba2f"},
    "SOLUSDT":    {"name": "Solana",    "short": "SOL",  "color": "#9945ff"},
    "AVAXUSDT":   {"name": "Avalanche", "short": "AVAX", "color": "#e84142"},
    "NEARUSDT":   {"name": "NEAR",      "short": "NEAR", "color": "#00c08b"},
    "SUIUSDT":    {"name": "Sui",       "short": "SUI",  "color": "#4da2ff"},
    "APTUSDT":    {"name": "Aptos",     "short": "APT",  "color": "#00d4aa"},
    "LINKUSDT":   {"name": "Chainlink", "short": "LINK", "color": "#2a5ada"},
    "DOTUSDT":    {"name": "Polkadot",  "short": "DOT",  "color": "#e6007a"},
    "UNIUSDT":    {"name": "Uniswap",   "short": "UNI",  "color": "#ff007a"},
    "AAVEUSDT":   {"name": "Aave",      "short": "AAVE", "color": "#b6509e"},
    "XRPUSDT":    {"name": "XRP",       "short": "XRP",  "color": "#00aae4"},
    "FETUSDT":    {"name": "Fetch.AI",  "short": "FET",  "color": "#1a1f6e"},
    "RENDERUSDT": {"name": "Render",    "short": "RNDR", "color": "#f14c27"},
    "ADAUSDT":    {"name": "Cardano",   "short": "ADA",  "color": "#0033ad"},
    "INJUSDT":    {"name": "Injective", "short": "INJ",  "color": "#00b2ff"},
    "ARBUSDT":    {"name": "Arbitrum",  "short": "ARB",  "color": "#28a0f0"},
    "OPUSDT":     {"name": "Optimism",  "short": "OP",   "color": "#ff0420"},
    "SEIUSDT":    {"name": "Sei",       "short": "SEI",  "color": "#9d1ef9"},
}

# Also expose as COIN_CATEGORIES for dashboard_api
COIN_CATEGORIES = {
    "👑 Big Three":          TIER_BIG3,
    "⚡ High-Liquidity L1s": TIER_LIQ1,
    "🏛 Institutional Alts": TIER_INST,
    "🤖 AI & Momentum":      TIER_AI_MOM,
}

def get_tier(symbol: str) -> str:
    for t in COIN_TIERS.values():
        if symbol in t["coins"]:
            return t["label"]
    return "Unknown"

# Timeframes
TIMEFRAME_ENTRY   = "15m"
TIMEFRAME_CONFIRM = "1h"
TIMEFRAME_TREND   = "4h"
DOWNLOAD_LIMIT    = 1500
LIVE_LIMIT        = 300

# Features list — must match feature_engineering.add_indicators() output
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

# Signal thresholds (smart_scheduler overrides these at runtime)
MIN_CONFIDENCE     = 65
MIN_ADX            = 20
MIN_SCORE          = 3

# Correlation filter — max same-direction trades at once
MAX_SAME_DIRECTION = 2

# Risk management
ATR_STOP_MULT     = 1.5
ATR_TARGET1_MULT  = 2.0
ATR_TARGET2_MULT  = 3.0
RISK_PER_TRADE    = 0.01   # 1% of balance

# Training
TARGET_FUTURE    = 6
TARGET_THRESHOLD = 0.005
TEST_SPLIT       = 0.30
RANDOM_STATE     = 42

# Scheduling
SCAN_INTERVAL_MIN = 15

# Files
RAW_DATA_FILE  = "training_data.csv"
FEATURES_FILE  = "training_features.csv"
DATASET_FILE   = "training_dataset.csv"
TRAIN_FILE     = "train_data.csv"
TEST_FILE      = "test_data.csv"
MODEL_FILE     = "pro_crypto_ai_model.pkl"
LOG_FILE       = "bot.log"
