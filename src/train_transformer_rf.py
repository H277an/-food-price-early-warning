import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

np.random.seed(42)

base_df = pd.read_csv("data/processed/cereals_features.csv")
transformer_df = pd.read_csv("data/processed/transformer_features.csv")

merged = pd.merge(base_df, transformer_df.drop(columns=["target_pct_change"]), on="date")
merged = merged.sort_values("date").reset_index(drop=True)

feature_cols = [c for c in merged.columns if c not in ("date", "target_next_price", "target_pct_change")]
X = merged[feature_cols]
y = merged["target_pct_change"]

split = len(merged) - 60
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

preds_pct = model.predict(X_test)

current_prices = merged["cereals_price"].iloc[split:].values
predicted_prices = current_prices * (1 + preds_pct / 100)
actual_prices = merged["target_next_price"].iloc[split:].values

price_mae = mean_absolute_error(actual_prices, predicted_prices)
price_r2 = r2_score(actual_prices, predicted_prices)

print(f"Transformer + RF — Price-level Test MAE: {price_mae:.3f}")
print(f"Transformer + RF — Price-level Test R²: {price_r2:.3f}")
print(f"\nCompare against naive baseline: MAE 3.085, R² 0.927")
print(f"Compare against LSTM+RF (Model 2): MAE 3.857, R² 0.899")
print(f"Compare against LSTM+XGBoost (Model 1): MAE 4.328, R² 0.886")

importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 features:")
print(importance.head(10))

merged.to_csv("data/processed/final_merged_features_v3.csv", index=False)