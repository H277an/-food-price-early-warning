import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

SEQ_LEN = 12


def prepare_features(commodity: str, regions: list[str]) -> pd.DataFrame:
    """Merge price + weather (averaged across the commodity's regions) into
    one monthly feature table, matching prepare_features.py's logic."""
    conn = sqlite3.connect("db/food_prices.db")

    prices = pd.read_sql(f"""
        SELECT date, index_value AS price
        FROM prices WHERE commodity = '{commodity}' ORDER BY date
    """, conn)

    region_list = "', '".join(regions)
    weather = pd.read_sql(f"""
        SELECT substr(date, 1, 7) AS date,
               AVG(precipitation_sum) AS avg_precipitation,
               AVG(temperature_2m_max) AS avg_max_temp
        FROM weather WHERE region IN ('{region_list}')
        GROUP BY substr(date, 1, 7)
    """, conn)
    conn.close()

    df = pd.merge(prices, weather, on="date", how="inner")
    for lag in [1, 2, 3, 6, 12]:
        df[f"price_lag_{lag}"] = df["price"].shift(lag)
    df["target_next_price"] = df["price"].shift(-1)
    df["target_pct_change"] = (df["target_next_price"] - df["price"]) / df["price"] * 100
    df = df.dropna().reset_index(drop=True)
    return df


def build_lstm_features(df: pd.DataFrame, commodity: str, test_size: int = 60) -> pd.DataFrame:
    """Train an LSTM feature extractor.

    Two fixes applied vs. the original version:
    1. Scaler is fit ONLY on rows that precede the test period, not the
       whole dataset — the original fit MinMaxScaler on all 427 rows
       including the 60 test months, which is leakage (the scaler's
       min/max reflected test-period values it should never see).
    2. Sequences now end at month i (inclusive) to predict the move
       FROM month i, instead of ending at i-1. The original built
       X = scaled[i-12:i] (through i-1) to predict target_pct_change[i]
       (the change from price[i] to price[i+1]) — meaning the model
       never saw the single most recent, most relevant month before
       making its forecast.
    """
    np.random.seed(42)
    import tensorflow as tf
    import os
    os.environ["PYTHONHASHSEED"] = "0"
    tf.random.set_seed(42)
    tf.config.experimental.enable_op_determinism()  # reduces (does not fully
    # eliminate) the run-to-run variance we saw across repeated Cereals runs
    # (AUC 0.635 -> 0.563 -> 0.563 for nominally the same method)

    seq_cols = ["price", "avg_precipitation", "avg_max_temp"]

    # --- Fix 2: corrected sequence construction ---
    # Sequence for target row i now spans raw rows [i-SEQ_LEN+1, i] inclusive
    # (12 months ending at the current month), predicting target_pct_change[i]
    # = the % move from price[i] to price[i+1]. This is the intended framing:
    # "using everything known through month i, forecast the move to i+1."
    n_rows = len(df)
    n_sequences = n_rows - SEQ_LEN + 1
    split_seq = n_sequences - test_size

    # --- Fix 1: scaler fit boundary ---
    # The earliest raw row referenced by any TEST sequence is row `split_seq`
    # (the first test sequence spans [split_seq, split_seq+SEQ_LEN-1]).
    # So the scaler must only see rows strictly before that.
    raw_fit_cutoff = split_seq
    if raw_fit_cutoff < SEQ_LEN:
        raise ValueError(
            f"[{commodity}] Not enough training data ({raw_fit_cutoff} rows) "
            f"to fit the scaler before the test period begins."
        )

    scaler = MinMaxScaler()
    scaler.fit(df[seq_cols].iloc[:raw_fit_cutoff])
    scaled = scaler.transform(df[seq_cols])

    X, y, idx = [], [], []
    for i in range(SEQ_LEN - 1, n_rows):
        X.append(scaled[i - SEQ_LEN + 1: i + 1])
        y.append(df["target_pct_change"].iloc[i])
        idx.append(i)
    X, y = np.array(X), np.array(y)

    X_train, X_test = X[:split_seq], X[split_seq:]
    y_train, y_test = y[:split_seq], y[split_seq:]

    inputs = layers.Input(shape=(SEQ_LEN, len(seq_cols)))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    feature_output = layers.Dense(16, activation="relu", name="feature_layer")(x)
    final_output = layers.Dense(1)(feature_output)
    model = keras.Model(inputs=inputs, outputs=final_output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stop = keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-5)
    model.fit(X_train, y_train, validation_split=0.15, epochs=150, batch_size=16,
              callbacks=[early_stop, reduce_lr], verbose=0)

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"  [{commodity}] LSTM Test MAE (pct change, corrected alignment): {test_mae:.3f}")

    feature_extractor = keras.Model(inputs=inputs, outputs=feature_output)
    lstm_features = feature_extractor.predict(X, verbose=0)

    feat_df = pd.DataFrame(lstm_features, columns=[f"lstm_feat_{i}" for i in range(16)])
    feat_df["date"] = df["date"].iloc[idx].values
    return feat_df

def train_random_forest(merged: pd.DataFrame, commodity: str) -> dict:
    """Train the final RF model, matching train_random_forest.py's logic."""
    feature_cols = [c for c in merged.columns if c not in ("date", "target_next_price", "target_pct_change")]
    X = merged[feature_cols]
    y = merged["target_pct_change"]

    split = len(merged) - 60
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                    random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds_pct = model.predict(X_test)

    current_prices = merged["price"].iloc[split:].values
    predicted_prices = current_prices * (1 + preds_pct / 100)
    actual_prices = merged["target_next_price"].iloc[split:].values

    mae = mean_absolute_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)
    print(f"  [{commodity}] RF Price-level Test MAE: {mae:.3f}, R²: {r2:.3f}")

    return {"commodity": commodity, "mae": mae, "r2": r2, "model": model}
def train_shock_classifier(merged: pd.DataFrame, commodity: str,
                             relative_threshold_quantile: float = 0.90) -> dict:
    """Shock classifier using a commodity-relative threshold and full-history
    walk-forward evaluation.

    Two fixes vs. the original version:
    1. The old flat >5% cutoff produced wildly different shock rates per
       commodity (Cereals 8.7%, Sugar 23.4%, Meat 1.9%) — not a consistent
       definition of "shock." Now each commodity's threshold is its OWN
       90th percentile of historical % change, so every commodity gets a
       comparable ~10% positive rate, and Meat is no longer skipped.
    2. The old TimeSeriesSplit(test_size=40, n_splits=5) only ever evaluated
       the most recent ~200 months, leaving the 2007-08 food price crisis
       permanently in training and never in a test fold. This now starts
       evaluation much earlier and steps forward in smaller increments.
    """
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score, precision_recall_curve

    df = merged.copy()
    df["pct_change"] = (df["target_next_price"] - df["price"]) / df["price"] * 100

    # Fix: per-commodity relative threshold instead of a flat 5%
    threshold_value = df["pct_change"].quantile(relative_threshold_quantile)
    df["is_shock"] = (df["pct_change"] > threshold_value).astype(int)
    print(f"  [{commodity}] Relative shock threshold: {threshold_value:.2f}% "
          f"(90th percentile of this commodity's own history)")

    feature_cols = [c for c in df.columns if c not in
                    ("date", "target_next_price", "target_pct_change", "pct_change", "is_shock")]
    X = df[feature_cols]
    y = df["is_shock"]

    total_shocks = y.sum()
    print(f"  [{commodity}] Total shocks: {total_shocks}/{len(y)} ({y.mean()*100:.1f}%)")

    # Fix: smaller test windows, more folds, starting much earlier in history
    # so early periods (including 2007-08) get evaluated, not just held out
    # as permanent training data.
    min_train_size = 100
    test_size = 20
    n_splits = (len(df) - min_train_size) // test_size - 1
    n_splits = max(n_splits, 3)

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    all_y_test, all_probs, fold_date_ranges = [], [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if len(train_idx) < min_train_size:
            continue
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        if pos == 0:
            continue
        scale_pos_weight = neg / pos

        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42, eval_metric="logloss",
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        test_dates = df["date"].iloc[test_idx]
        fold_date_ranges.append((test_dates.min(), test_dates.max(), y_test.sum()))

        all_y_test.extend(y_test)
        all_probs.extend(probs)

    print(f"  [{commodity}] {len(fold_date_ranges)} folds, covering:")
    for start, end, n_shocks in fold_date_ranges:
        print(f"    {start} to {end}: {n_shocks} shocks")

    if sum(all_y_test) == 0:
        print(f"  [{commodity}] No shocks appeared in any test fold — cannot evaluate")
        return {"commodity": commodity, "auc": None, "skipped": True}

    auc = roc_auc_score(all_y_test, all_probs)

    precisions, recalls, thresholds = precision_recall_curve(all_y_test, all_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    low_thresh_preds = [1 if p >= 0.2 else 0 for p in all_probs]
    tp = sum((pl == 1 and yt == 1) for pl, yt in zip(low_thresh_preds, all_y_test))
    fn = sum((pl == 0 and yt == 1) for pl, yt in zip(low_thresh_preds, all_y_test))
    low_thresh_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"  [{commodity}] ROC AUC: {auc:.3f} (evaluated across {len(all_y_test)} months, "
          f"{sum(all_y_test)} shocks) | best-F1 threshold: {best_threshold:.3f} | "
          f"recall@0.2: {low_thresh_recall:.2f}")

    return {
        "commodity": commodity,
        "auc": round(auc, 3),
        "threshold_value_pct": round(float(threshold_value), 2),
        "total_shocks": int(total_shocks),
        "test_months_evaluated": len(all_y_test),
        "test_shocks_evaluated": int(sum(all_y_test)),
        "best_threshold": round(float(best_threshold), 3),
        "recall_at_0.2": round(low_thresh_recall, 3),
        "skipped": False,
    }