import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

np.random.seed(42)

# Same merged features file from the XGBoost stage — no need to rebuild anything
merged = pd.read_csv("data/processed/final_merged_features.csv")
merged = merged.sort_values("date").reset_index(drop=True)

feature_cols = [c for c in merged.columns if c not in ("date", "target_next_price", "target_pct_change")]
X = merged[feature_cols]
y = merged["target_pct_change"]

# Identical split to the XGBoost run — required for a fair comparison
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

print(f"Random Forest (via % change) — Price-level Test MAE: {price_mae:.3f}")
print(f"Random Forest (via % change) — Price-level Test R²: {price_r2:.3f}")
print(f"Compare against naive baseline: MAE 3.085, R² 0.927")
print(f"Compare against XGBoost: MAE 4.328, R² 0.886")
importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 features:")
print(importance.head(10))