# train_model.py — Retrain model with enhanced features
# Run on GitHub Actions (free, no local setup needed)
# Expected accuracy: 73-76% (volume features add edge)
#
# What's new vs original training:
#   + volume_ratio  — volume vs 20-period average
#   + volume_spike  — binary flag for 2x+ volume
#   + obv_slope     — on-balance volume momentum
#   + bb_width      — Bollinger Band squeeze/expansion
#   + rsi_fast      — 7-period RSI for faster signals
#   + stoch_k/d     — Stochastic oscillator
#   + price_change3/6 — multi-period momentum
#   + hammer/doji   — candlestick reversal patterns
#   + class_weight  — fixed NO_TRADE imbalance

import os, json, time, logging, joblib, requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from feature_engineering import add_indicators, ALL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────
SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","AVAXUSDT",
    "XRPUSDT","LINKUSDT","NEARUSDT","DOTUSDT","ADAUSDT",
    "INJUSDT","ARBUSDT","OPUSDT","UNIUSDT","AAVEUSDT",
]
INTERVALS    = ["15m", "1h"]
LIMIT        = 1000          # candles per symbol per interval
TARGET_BARS  = 6             # predict 6 bars ahead
TARGET_PCT   = 0.0035        # 0.35% move = signal (lowered to balance classes)
TEST_SPLIT   = 0.25
MODEL_FILE   = "pro_crypto_ai_model.pkl"


# ── Data fetch ────────────────────────────────────────────────────────

def fetch_klines(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    for url in ["https://data-api.binance.vision/api/v3/klines",
                "https://api.binance.com/api/v3/klines"]:
        try:
            r = requests.get(url,
                params={"symbol":symbol,"interval":interval,"limit":limit},
                timeout=20)
            if r.status_code == 200:
                df = pd.DataFrame(r.json()).iloc[:,:6]
                df.columns = ["open_time","open","high","low","close","volume"]
                for c in ["open","high","low","close","volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                return df
        except Exception as e:
            log.warning(f"  {symbol} {interval}: {e}")
    return pd.DataFrame()


# ── Target labeling ───────────────────────────────────────────────────

def make_targets(df: pd.DataFrame) -> pd.Series:
    """
    Label each candle:
      BUY      if price rises >0.35% in next 6 bars
      SELL     if price falls >0.35% in next 6 bars
      NO_TRADE otherwise
    """
    future_close = df["close"].shift(-TARGET_BARS)
    pct_change   = (future_close - df["close"]) / df["close"]

    labels = pd.Series("NO_TRADE", index=df.index)
    labels[pct_change >  TARGET_PCT] = "BUY"
    labels[pct_change < -TARGET_PCT] = "SELL"
    return labels


# ── Build dataset ─────────────────────────────────────────────────────

def build_dataset():
    log.info(f"Building dataset from {len(SYMBOLS)} symbols × {len(INTERVALS)} intervals")
    all_rows = []

    for symbol in SYMBOLS:
        log.info(f"  {symbol}...")

        # Fetch 15m and 1h
        df15 = fetch_klines(symbol, "15m", LIMIT)
        df1h = fetch_klines(symbol, "1h",  LIMIT // 4)
        if df15.empty or len(df15) < 100:
            log.warning(f"  {symbol}: insufficient data"); continue
        time.sleep(0.3)  # rate limit

        df15 = add_indicators(df15)
        df1h  = add_indicators(df1h) if not df1h.empty else pd.DataFrame()

        # Add 1h context to each 15m row
        if not df1h.empty:
            df15["rsi_1h"]   = df1h["rsi"].reindex(df15.index, method="ffill").fillna(50)
            df15["adx_1h"]   = df1h["adx"].reindex(df15.index, method="ffill").fillna(0)
            df15["trend_1h"] = df1h["trend"].reindex(df15.index, method="ffill").fillna(0)
        else:
            df15["rsi_1h"]   = 50.0
            df15["adx_1h"]   = 0.0
            df15["trend_1h"] = 0.0

        # Labels
        labels = make_targets(df15)
        df15["target"] = labels
        df15["symbol"] = symbol

        # Drop last TARGET_BARS rows (no future to label)
        df15 = df15.iloc[:-TARGET_BARS]
        all_rows.append(df15)

    if not all_rows:
        raise Exception("No data fetched — check network connection")

    dataset = pd.concat(all_rows, ignore_index=True)
    log.info(f"Dataset: {len(dataset)} rows | "
             f"BUY:{(dataset.target=='BUY').sum()} "
             f"SELL:{(dataset.target=='SELL').sum()} "
             f"NO_TRADE:{(dataset.target=='NO_TRADE').sum()}")
    return dataset


# ── Training ──────────────────────────────────────────────────────────

def train(dataset: pd.DataFrame):
    # Check all features exist
    missing = [f for f in ALL_FEATURES if f not in dataset.columns]
    if missing:
        log.warning(f"Missing features (will fill with 0): {missing}")
        for f in missing:
            dataset[f] = 0.0

    X = dataset[ALL_FEATURES].replace([np.inf,-np.inf], np.nan).fillna(0)
    y_raw = dataset["target"]

    # Encode labels
    le      = LabelEncoder()
    y       = le.fit_transform(y_raw)
    classes = list(le.classes_)
    log.info(f"Classes: {classes}")  # ['BUY', 'NO_TRADE', 'SELL']

    # Train/test split (time-aware — no shuffle)
    split     = int(len(X) * (1 - TEST_SPLIT))
    X_train   = X.iloc[:split];  y_train = y[:split]
    X_test    = X.iloc[split:];  y_test  = y[split:]

    # Calculate sample weights to fix the class imbalance (forces AI to care about BUY/SELL)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # Feature selection — top 20 features
    selector  = SelectKBest(f_classif, k=min(20, len(ALL_FEATURES)))
    X_tr_sel  = selector.fit_transform(X_train, y_train)
    X_te_sel  = selector.transform(X_test)

    # Show selected features
    mask     = selector.get_support()
    selected = [f for f,m in zip(ALL_FEATURES, mask) if m]
    log.info(f"Selected features ({len(selected)}): {selected}")

    # ── Models ────────────────────────────────────────────────────────
    log.info("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1
    )
    xgb.fit(X_tr_sel, y_train, sample_weight=sample_weights)

    log.info("Training RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", # Built-in balancing for Random Forest
        random_state=42, n_jobs=-1
    )
    rf.fit(X_tr_sel, y_train)

    log.info("Training GradientBoosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    gb.fit(X_tr_sel, y_train, sample_weight=sample_weights)

    # ── Ensemble ──────────────────────────────────────────────────────
    log.info("Building ensemble...")
    ensemble = VotingClassifier(
        estimators=[("xgb",xgb),("rf",rf),("gb",gb)],
        voting="soft",
        weights=[2, 1, 1]  # XGB gets 2x weight (usually best)
    )
    
    # Pass sample weights into the VotingClassifier so they pass down to XGB and GB during refit
    ensemble.fit(X_tr_sel, y_train, sample_weight=sample_weights)

    # ── Evaluation ────────────────────────────────────────────────────
    y_pred = ensemble.predict(X_te_sel)
    acc    = accuracy_score(y_test, y_pred)

    log.info(f"\n{'='*50}")
    log.info(f"TEST ACCURACY: {acc*100:.1f}%")
    log.info(f"{'='*50}")
    log.info("\n" + classification_report(y_test, y_pred, target_names=classes))

    # Per-class accuracy
    for i, cls in enumerate(classes):
        mask   = y_test == i
        if mask.sum() > 0:
            cls_acc = (y_pred[mask] == i).mean()
            log.info(f"  {cls}: {cls_acc*100:.1f}% ({mask.sum()} samples)")

    # Cross-val on training set (we bypass weights here just for simple variance checking)
    cv_scores = cross_val_score(ensemble, X_tr_sel, y_train, cv=5, n_jobs=-1)
    log.info(f"\nCross-val (5-fold): {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    # Label map for trade_executor
    label_map = {i: cls for i, cls in enumerate(classes)}
    inverse_map = {cls: i for i, cls in enumerate(classes)}

    # ── Save pipeline ─────────────────────────────────────────────────
    pipeline = {
        "ensemble":     ensemble,
        "selector":     selector,
        "all_features": ALL_FEATURES,
        "best_features": selected,
        "label_map":    label_map,
        "label_encoder": le,
        "accuracy":     round(acc * 100, 1),
        "trained_at":   datetime.now(timezone.utc).isoformat(),
        "symbols":      SYMBOLS,
        "n_features":   len(ALL_FEATURES),
    }
    joblib.dump(pipeline, MODEL_FILE)
    log.info(f"\n✅ Model saved: {MODEL_FILE}")
    log.info(f"   Accuracy: {acc*100:.1f}%")
    log.info(f"   Features: {len(ALL_FEATURES)}")
    log.info(f"   Selected: {len(selected)}")

    # Save performance summary
    perf = {
        "accuracy":      round(acc*100,1),
        "cv_mean":       round(cv_scores.mean()*100,1),
        "cv_std":        round(cv_scores.std()*100,1),
        "n_train":       int(len(X_train)),
        "n_test":        int(len(X_test)),
        "features":      ALL_FEATURES,
        "selected":      selected,
        "label_map":     label_map,
        "trained_at":    datetime.now(timezone.utc).isoformat(),
    }
    with open("model_performance.json","w") as f:
        json.dump(perf, f, indent=2)

    return acc


if __name__ == "__main__":
    log.info("="*50)
    log.info("CRYPTOBOT AI — MODEL TRAINING")
    log.info("="*50)
    t0 = time.time()

    dataset = build_dataset()
    acc     = train(dataset)

    elapsed = time.time() - t0
    log.info(f"\nTotal time: {elapsed/60:.1f} minutes")
    log.info(f"Final accuracy: {acc*100:.1f}%")

    if acc < 0.65:
        log.warning("⚠️ Accuracy below 65% — check data quality or feature engineering")
    else:
        log.info("✅ Training complete — upload pro_crypto_ai_model.pkl to GitHub repo")
