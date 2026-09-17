from pipeline import prepare_features, build_lstm_features, train_random_forest
import pandas as pd

df = prepare_features("Cereals", ["us_midwest", "ukraine_wheat_belt", "india_punjab"])
print(f"Cereals features: {len(df)} rows")

lstm_feats = build_lstm_features(df, "Cereals")
merged = pd.merge(df, lstm_feats, on="date")

result = train_random_forest(merged, "Cereals")
print(f"\nExpected from earlier: MAE 3.857, R² 0.899")