import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

np.random.seed(42)

# Original lag/price features
base_df = pd.read_csv("data/processed/cereals_features.csv")

# LSTM-extracted features
lstm_df = pd.read_csv("data/processed/lstm_features.csv")

# Merge on date — this combines your hand-crafted lag features with the
# LSTM's learned temporal representation, which is the whole point of the hybrid
merged = pd.merge(base_df, lstm_df.drop(columns=["target_next_price"]), on="date")
merged = merged.sort_values("date").reset_index(drop=True)

feature_cols = [c for c in merged.columns if c not in ("date", "target_next_price")]
X = merged[feature_cols]
y = merged["target_next_price"]

# Same time-respecting split as the LSTM stage
split = len(merged) - 60
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"XGBoost Test MAE: {mae:.3f}")
print(f"XGBoost Test R²: {r2:.3f}")

# Feature importance — useful for your README's "what drives price" story
importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 features:")
print(importance.head(10))

merged.to_csv("data/processed/final_merged_features.csv", index=False)