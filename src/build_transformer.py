import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

SEQ_LEN = 12

df = pd.read_csv("data/processed/cereals_features.csv")
df = df.sort_values("date").reset_index(drop=True)

seq_cols = ["cereals_price", "avg_precipitation", "avg_max_temp"]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[seq_cols])

X, y, idx = [], [], []
for i in range(SEQ_LEN, len(df)):
    X.append(scaled[i - SEQ_LEN:i])
    y.append(df["target_pct_change"].iloc[i])
    idx.append(i)

X = np.array(X)
y = np.array(y)

split = len(X) - 60
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")

# Small transformer encoder — kept intentionally compact given ~355 training
# sequences; a larger model would overfit before it could learn anything useful
inputs = layers.Input(shape=(SEQ_LEN, len(seq_cols)))

# Positional encoding — transformers have no inherent sense of order (unlike
# LSTM's recurrence), so we add a learned position signal explicitly
positions = layers.Embedding(input_dim=SEQ_LEN, output_dim=len(seq_cols))(tf.range(SEQ_LEN))
x = inputs + positions

attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=8, dropout=0.3)(x, x)
x = layers.LayerNormalization()(x + attn_output)

ff = layers.Dense(32, activation="relu")(x)
ff = layers.Dropout(0.3)(ff)
ff = layers.Dense(len(seq_cols))(ff)
x = layers.LayerNormalization()(x + ff)

x = layers.GlobalAveragePooling1D()(x)
feature_output = layers.Dense(16, activation="relu", name="feature_layer")(x)
final_output = layers.Dense(1)(feature_output)

model = keras.Model(inputs=inputs, outputs=final_output)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

early_stop = keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-5)

model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=150,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTransformer Test MAE (pct change): {test_mae:.3f}")
print(f"(Compare against LSTM: 2.498)")

feature_extractor = keras.Model(inputs=inputs, outputs=feature_output)
transformer_features = feature_extractor.predict(X)

feature_df = pd.DataFrame(transformer_features, columns=[f"transformer_feat_{i}" for i in range(16)])
feature_df["date"] = df["date"].iloc[idx].values
feature_df["target_pct_change"] = y

feature_df.to_csv("data/processed/transformer_features.csv", index=False)
model.save("db/transformer_model.keras")

print(f"\nSaved {len(feature_df)} rows to data/processed/transformer_features.csv")
print(feature_df.head())