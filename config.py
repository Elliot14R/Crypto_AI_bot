# config.py - All settings in one place
# Updated for Automated Testnet Execution

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "XRPUSDT", "ADAUSDT", "AVAXUSDT", "SUIUSDT",
    "DOTUSDT", "LINKUSDT", "MATICUSDT", "NEARUSDT",
    "APTUSDT", "ARBUSDT",
]

TIMEFRAME_ENTRY   = "15m"           # 15m is far more reliable than 3m
TIMEFRAME_CONFIRM = "1h"
TIMEFRAME_TREND   = "4h"

DOWNLOAD_LIMIT    = 1500
LIVE_LIMIT        = 300

# These MUST exactly match what feature_engineering.py creates
FEATURES = [
    # EMA trend
    "ema9", "ema20", "ema50", "ema200",
    "ema20_slope", "ema50_slope",
    "price_vs_ema20", "price_vs_ema50", "price_vs_ema200",
    "ema20_vs_ema50",
    # Momentum
    "rsi", "rsi_slope", "rsi_fast",
    "stoch_k", "stoch_d",
    # MACD
    "macd", "macd_signal", "macd_hist", "macd_slope",
    # Trend strength
    "adx", "adx_pos", "adx_neg", "di_diff",
    # Volatility
    "atr", "atr_pct",
    "bb_high", "bb_low", "bb_pct", "bb_width",
    # Volume
    "volume_ratio", "volume_spike", "obv_slope",
    # Price action
    "price_change", "price_change3", "price_change6",
    "high_low_pct", "body_pct",
    "momentum", "volatility",
    # Candle patterns
    "bullish_candle", "doji", "hammer",
    # Higher timeframe
    "rsi_1h", "adx_1h", "trend_1h",
]

# ─── AI & SIGNAL FILTERS (Strict for Testnet) ───
MIN_CONFIDENCE    = 75       # Increased from 65 to ensure only A+ setups
MIN_ADX           = 25       # Increased from 20 to ensure strong momentum
MIN_SCORE         = 3

# ─── RISK MANAGEMENT & EXECUTION ───
MAX_OPEN_TRADES   = 3        # NEW: Prevents bot from over-exposing the account
ATR_STOP_MULT     = 1.5
ATR_TARGET1_MULT  = 1.5      # Changed to 1.5 for a 1:1 Risk/Reward on the first 50% take profit
ATR_TARGET2_MULT  = 3.0      # 1:2 Risk/Reward for the remaining runner
RISK_PER_TRADE    = 0.01     # 1% account risk per trade

# ─── ML TRAINING PARAMS ───
TARGET_FUTURE     = 6                # look 6 candles ahead on 15m = 90 min
TARGET_THRESHOLD  = 0.005            # 0.5% move = meaningful signal

TEST_SPLIT        = 0.30
RANDOM_STATE      = 42

SCAN_INTERVAL_MIN = 15              # scan every 15 min matching candle close

# ─── FILE PATHS ───
RAW_DATA_FILE     = "training_data.csv"
FEATURES_FILE     = "training_features.csv"
DATASET_FILE      = "training_dataset.csv"
TRAIN_FILE        = "train_data.csv"
TEST_FILE         = "test_data.csv"
MODEL_FILE        = "pro_crypto_ai_model.pkl"
LOG_FILE          = "bot.log"
