import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

merged = pd.read_csv("data/processed/final_merged_features.csv")
merged = merged.sort_values("date").reset_index(drop=True)

split = len(merged) - 60
test = merged.iloc[split:]

# Naive baseline: predict next month = this month's price, no model at all
naive_preds = test["cereals_price"]
naive_mae = mean_absolute_error(test["target_next_price"], naive_preds)
naive_r2 = r2_score(test["target_next_price"], naive_preds)

print(f"Naive baseline MAE: {naive_mae:.3f}")
print(f"Naive baseline R²: {naive_r2:.3f}")
