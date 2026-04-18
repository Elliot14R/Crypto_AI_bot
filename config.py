# config.py — Professional settings
# Key changes:
#   RISK_PER_TRADE = 1% (was inconsistent 1-2%)
#   ATR multipliers kept at 1.5/2.0/3.0 (good risk:reward = 1:2)
#   MAX_TRADE_AGE_HOURS = 48 (new: auto-exit stale trades)

# ── File Paths ────────────────────────────────────────────────────────
RAW_DATA_FILE = "data/crypto_historical_15m_expanded.csv"
FEATURES_FILE = "data/crypto_features_15m_expanded.csv"
MODEL_FILE    = "pro_crypto_ai_model.pkl"
LOG_FILE      = "bot.log"

# ── Features List ─────────────────────────────────────────────────────
FEATURES = [
    'rsi', 'macd', 'macd_signal', 'macd_hist', 'stoch_k', 'stoch_d',
    'adx', 'cci', 'atr', 'bb_upper', 'bb_middle', 'bb_lower',
    'ema20', 'ema50', 'sma200', 'volume_ratio', 'price_change',
    'rsi_1h', 'adx_1h', 'trend_1h'
]

# ── Coin Tiers ────────────────────────────────────────────────────────
TIER_BIG3   = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
TIER_LIQ1   = ["SOLUSDT", "AVAXUSDT", "XRPUSDT", "LINKUSDT", "NEARUSDT"]
TIER_INST   = ["DOTUSDT", "ADAUSDT", "INJUSDT", "ARBUSDT", "OPUSDT"]
TIER_AI_MOM = ["FETUSDT", "RENDERUSDT", "UNIUSDT", "AAVEUSDT", "SEIUSDT"]

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

# Timeframes
TIMEFRAME_ENTRY   = "15m"
TIMEFRAME_CONFIRM = "1h"
LIVE_LIMIT        = 300

# ── Risk management (professional grade) ─────────────────────────────
RISK_PER_TRADE     = 0.01   # 1% per trade — strict, never change this
ATR_STOP_MULT      = 1.5    # SL = entry ± 1.5 × ATR
ATR_TARGET1_MULT   = 2.0    # TP1 = entry ± 2.0 × ATR  (R:R = 1.33:1)
ATR_TARGET2_MULT   = 3.0    # TP2 = entry ± 3.0 × ATR  (R:R = 2:1)
MAX_SAME_DIRECTION = 2      # max 2 BUY or 2 SELL simultaneously

# ── NEW: Time-based exit ──────────────────────────────────────────────
# If a trade has been open for > 48 hours without hitting TP1,
# exit at market. Prevents capital being locked in stale positions.
MAX_TRADE_AGE_HOURS = 48
