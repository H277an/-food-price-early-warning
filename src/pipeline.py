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


def build_lstm_features(df: pd.DataFrame, commodity: str) -> pd.DataFrame:
    """Train an LSTM feature extractor, matching build_lstm.py's architecture."""
    np.random.seed(42)
    import tensorflow as tf
    tf.random.set_seed(42)

    seq_cols = ["price", "avg_precipitation", "avg_max_temp"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[seq_cols])

    X, y, idx = [], [], []
    for i in range(SEQ_LEN, len(df)):
        X.append(scaled[i - SEQ_LEN:i])
        y.append(df["target_pct_change"].iloc[i])
        idx.append(i)
    X, y = np.array(X), np.array(y)

    split = len(X) - 60
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

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
    print(f"  [{commodity}] LSTM Test MAE (pct change): {test_mae:.3f}")

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