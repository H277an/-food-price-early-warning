import pandas as pd
import numpy as np
import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

SEQ_LEN = 12  # look back 12 months to capture yearly seasonality

df = pd.read_csv("data/processed/cereals_features.csv")
df = df.sort_values("date").reset_index(drop=True)

# Raw sequence inputs for the LSTM — NOT the lag columns, since the LSTM
# learns temporal patterns itself from the raw series
seq_cols = ["cereals_price", "avg_precipitation", "avg_max_temp"]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[seq_cols])

X, y, idx = [], [], []
for i in range(SEQ_LEN, len(df)):
    X.append(scaled[i - SEQ_LEN:i])
    y.append(df["target_next_price"].iloc[i])
    idx.append(i)

X = np.array(X)
y = np.array(y)

# Time-respecting split — last 60 months as test, NEVER shuffle time series data
split = len(X) - 60
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")

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

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=150,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest MAE: {test_mae:.3f}")

# Extract the feature_layer output — this is what feeds into XGBoost next
feature_extractor = keras.Model(inputs=inputs, outputs=feature_output)
lstm_features = feature_extractor.predict(X)

feature_df = pd.DataFrame(lstm_features, columns=[f"lstm_feat_{i}" for i in range(16)])
feature_df["date"] = df["date"].iloc[idx].values
feature_df["target_next_price"] = y

feature_df.to_csv("data/processed/lstm_features.csv", index=False)
model.save("db/lstm_model.keras")

print(f"\nSaved {len(feature_df)} rows of LSTM-extracted features to data/processed/lstm_features.csv")
print(feature_df.head())