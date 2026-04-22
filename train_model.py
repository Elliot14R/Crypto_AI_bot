# train_model.py — Definitive version
# TARGET: BUY/SELL precision ≥ 65%, recall ≥ 40%
# KEY DECISIONS (with reasoning):
#   TARGET_PCT = 0.003 ✅ — labels 3× more BUY/SELL rows (other AI was right)
#   ImportanceSelector ✅ — XGB importance keeps volume features (over SelectKBest)
#   sample_weight on XGB ONLY ✅ — fixes imbalance without double-correcting
#   NO class_weight on RF/GB ✅ — that's what caused the 43% collapse
#   Stratified split ✅ — balanced across bull/bear/sideways
#   NO confidence threshold ✅ — applied at runtime via smart_scheduler, not here

import os, json, time, logging, joblib, requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from feature_engineering import add_indicators, ALL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


class ImportanceSelector(BaseEstimator, TransformerMixin):
    """Picklable feature selector using XGB importances (no lambdas)."""
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, X, y=None): return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame): return X[self.feature_names].values
        return X


SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","AVAXUSDT","NEARUSDT",
    "SUIUSDT","APTUSDT","MATICUSDT","ATOMUSDT","LINKUSDT","DOTUSDT",
    "UNIUSDT","AAVEUSDT","XRPUSDT","LTCUSDT","BCHUSDT","FETUSDT",
    "RENDERUSDT","ADAUSDT","INJUSDT","ARBUSDT","OPUSDT","SEIUSDT",
]
LIMIT       = 1500
TARGET_BARS = 6
TARGET_PCT  = 0.003   # 0.3% threshold → 3× more BUY/SELL labels
TEST_SPLIT  = 0.20
MODEL_FILE  = "pro_crypto_ai_model.pkl"
N_FEATURES  = 32


def fetch_klines(symbol, interval, limit=1500):
    for url in ["https://data-api.binance.vision/api/v3/klines","https://api.binance.com/api/v3/klines"]:
        try:
            r = requests.get(url, params={"symbol":symbol,"interval":interval,"limit":limit}, timeout=20)
            if r.status_code == 200:
                df = pd.DataFrame(r.json()).iloc[:,:6]
                df.columns = ["open_time","open","high","low","close","volume"]
                for c in ["open","high","low","close","volume"]: df[c] = pd.to_numeric(df[c], errors="coerce")
                return df
        except Exception as e: log.warning(f"  {symbol}: {e}")
    return pd.DataFrame()


def make_targets(df):
    pct = (df["close"].shift(-TARGET_BARS) - df["close"]) / df["close"]
    labels = pd.Series("NO_TRADE", index=df.index)
    labels[pct >  TARGET_PCT] = "BUY"
    labels[pct < -TARGET_PCT] = "SELL"
    return labels


def build_dataset():
    log.info(f"Building dataset from {len(SYMBOLS)} symbols")
    rows = []
    for symbol in SYMBOLS:
        log.info(f"  {symbol}...")
        df15 = fetch_klines(symbol, "15m", LIMIT)
        df1h = fetch_klines(symbol, "1h",  LIMIT//4)
        if df15.empty or len(df15) < 100: continue
        time.sleep(0.2)
        df15 = add_indicators(df15)
        if not df1h.empty:
            df1h = add_indicators(df1h)
            df15["rsi_1h"]   = df1h["rsi"].reindex(df15.index, method="ffill").fillna(50)
            df15["adx_1h"]   = df1h["adx"].reindex(df15.index, method="ffill").fillna(0)
            df15["trend_1h"] = df1h["trend"].reindex(df15.index, method="ffill").fillna(0)
        else:
            df15["rsi_1h"] = 50.0; df15["adx_1h"] = 0.0; df15["trend_1h"] = 0.0
        df15["target"] = make_targets(df15)
        rows.append(df15.iloc[:-TARGET_BARS])
    ds = pd.concat(rows, ignore_index=True)
    b=(ds.target=="BUY").sum(); s=(ds.target=="SELL").sum(); n=(ds.target=="NO_TRADE").sum()
    log.info(f"Dataset: {len(ds)} rows | BUY:{b}({b/len(ds)*100:.0f}%) SELL:{s}({s/len(ds)*100:.0f}%) NO_TRADE:{n}({n/len(ds)*100:.0f}%)")
    return ds


def train(ds):
    for f in ALL_FEATURES:
        if f not in ds.columns: ds[f] = 0.0

    X = ds[ALL_FEATURES].replace([np.inf,-np.inf], np.nan).fillna(0)
    le = LabelEncoder()
    y  = le.fit_transform(ds["target"])
    classes = list(le.classes_)
    log.info(f"Classes: {list(zip(range(len(classes)), classes))}")

    # Stratified split — essential so test set mirrors real distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, shuffle=True, stratify=y
    )

    # XGB importance scan — non-linear, keeps volume_ratio/volume_spike
    log.info("Importance scan...")
    scanner = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric="mlogloss")
    scanner.fit(X_train, y_train)
    top_idx = np.argsort(scanner.feature_importances_)[::-1]

    # Essential volume/volatility features always included
    essential = ["volume_ratio","volume_spike","obv_slope","bb_width","atr_pct","volatility"]
    selected  = [f for f in essential if f in ALL_FEATURES]
    for i in top_idx:
        f = ALL_FEATURES[i]
        if f not in selected: selected.append(f)
        if len(selected) >= N_FEATURES: break
    log.info(f"Top {len(selected)} features: {selected}")

    Xtr = X_train[selected].values
    Xte = X_test[selected].values

    # sample_weight: BUY/SELL get 2.5× weight vs NO_TRADE
    # Fixes imbalance WITHOUT double-correction (no class_weight on RF/GB)
    nt_idx   = classes.index("NO_TRADE") if "NO_TRADE" in classes else -1
    sw       = np.where(y_train == nt_idx, 1.0, 2.5)

    log.info("Training XGBoost...")
    xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.03,
                        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
                        gamma=0.05, eval_metric="mlogloss", random_state=42, n_jobs=-1)
    xgb.fit(Xtr, y_train, sample_weight=sw)

    log.info("Training RandomForest...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3,
                                max_features="sqrt", random_state=42, n_jobs=-1)
    rf.fit(Xtr, y_train)

    log.info("Training GradientBoosting...")
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.04,
                                    subsample=0.85, min_samples_leaf=3, random_state=42)
    gb.fit(Xtr, y_train)

    log.info("Building ensemble [XGB×3, RF×2, GB×1]...")
    ensemble = VotingClassifier(estimators=[("xgb",xgb),("rf",rf),("gb",gb)],
                                voting="soft", weights=[3,2,1])
    ensemble.fit(Xtr, y_train)

    # Raw accuracy (no threshold — bot applies threshold at runtime)
    y_pred = ensemble.predict(Xte)
    acc    = accuracy_score(y_test, y_pred)
    log.info(f"\n{'='*55}\nRAW ACCURACY: {acc*100:.1f}%\n{'='*55}")
    log.info("\n" + classification_report(y_test, y_pred, target_names=classes))

    # Calibration table — shows effect of each confidence threshold
    probas    = ensemble.predict_proba(Xte)
    buy_idx   = classes.index("BUY")  if "BUY"  in classes else 0
    sell_idx  = classes.index("SELL") if "SELL" in classes else 2
    real_buy  = (y_test == buy_idx).sum()
    real_sell = (y_test == sell_idx).sum()

    log.info("\n── Confidence Threshold Calibration ──────────────────────")
    log.info(f"{'Thresh':>7} {'Signals':>8} {'BUY prec':>10} {'SELL prec':>10} {'Recall':>8} {'PnL sim':>10}")
    log.info("-" * 58)

    best_thresh = 0.50
    best_score  = 0.0

    for thresh in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        yp = []
        for prob in probas:
            bc = np.argmax(prob)
            if bc != nt_idx and prob[bc] < thresh: yp.append(nt_idx)
            else: yp.append(bc)
        yp = np.array(yp)

        bm = yp == buy_idx;  sm = yp == sell_idx
        pb = (y_test[bm] == buy_idx).mean()  if bm.sum() > 0 else 0
        ps = (y_test[sm] == sell_idx).mean() if sm.sum() > 0 else 0
        rb = (yp[y_test==buy_idx]  == buy_idx).mean()  if real_buy  > 0 else 0
        rs = (yp[y_test==sell_idx] == sell_idx).mean() if real_sell > 0 else 0
        avg_prec   = (pb + ps) / 2
        avg_recall = (rb + rs) / 2
        n_signals  = int((bm | sm).sum())

        # Simulate PnL: $100 risk, 2:1 R:R
        wins   = n_signals * avg_prec
        losses = n_signals * (1 - avg_prec)
        pnl    = wins * 200 - losses * 100

        score = avg_prec * avg_recall
        if score > best_score and n_signals > 10:
            best_score = score; best_thresh = thresh

        log.info(f"  {thresh:.2f}    {n_signals:>7}    {pb:>8.1%}    {ps:>9.1%}   {avg_recall:>7.1%}   ${pnl:>8,.0f}")

    log.info(f"\n  → Best threshold: {best_thresh:.2f}")
    log.info(f"  → Set min_confidence={int(best_thresh*100)} in smart_scheduler.py")

    log.info("\n5-fold CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_sc = cross_val_score(ensemble, Xtr, y_train, cv=cv, n_jobs=-1)
    log.info(f"CV: {cv_sc.mean()*100:.1f}% ± {cv_sc.std()*100:.1f}%")

    pipeline = {
        "ensemble":              ensemble,
        "selector":              ImportanceSelector(selected),
        "all_features":          ALL_FEATURES,
        "best_features":         selected,
        "label_map":             {i:c for i,c in enumerate(classes)},
        "label_encoder":         le,
        "accuracy":              round(acc*100, 1),
        "trained_at":            datetime.now(timezone.utc).isoformat(),
        "symbols":               SYMBOLS,
        "n_features":            len(ALL_FEATURES),
        "recommended_threshold": best_thresh,
    }
    joblib.dump(pipeline, MODEL_FILE)
    log.info(f"\n✅ Saved: {MODEL_FILE}")

    perf = {
        "accuracy":              round(acc*100, 1),
        "test_accuracy":         round(acc, 4),
        "cv_mean":               round(cv_sc.mean()*100, 1),
        "cv_std":                round(cv_sc.std()*100, 1),
        "n_train":               int(len(X_train)),
        "n_test":                int(len(X_test)),
        "features":              ALL_FEATURES,
        "selected":              selected,
        "target_pct":            TARGET_PCT,
        "n_symbols":             len(SYMBOLS),
        "recommended_threshold": best_thresh,
    }
    with open("model_performance.json","w") as f: json.dump(perf, f, indent=2)
    return acc


if __name__ == "__main__":
    t0 = time.time()
    ds = build_dataset()
    acc = train(ds)
    log.info(f"\nDone in {(time.time()-t0)/60:.1f} min | Raw accuracy: {acc*100:.1f}%")
    log.info("See calibration table above — pick threshold for live trading")
