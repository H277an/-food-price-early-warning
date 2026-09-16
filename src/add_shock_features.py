import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/final_merged_features.csv")
df = df.sort_values("date").reset_index(drop=True)
df["month_num"] = df["date"].str[5:7]

# Weather anomalies: deviation from that calendar month's historical average.
# This is what actually signals "unusual weather" rather than just "weather"
precip_climatology = df.groupby("month_num")["avg_precipitation"].transform("mean")
temp_climatology = df.groupby("month_num")["avg_max_temp"].transform("mean")
df["precip_anomaly"] = df["avg_precipitation"] - precip_climatology
df["temp_anomaly"] = df["avg_max_temp"] - temp_climatology

# Price volatility: rolling std of month-over-month % change.
# Rising volatility often precedes a shock, even before the shock itself shows up
df["pct_change_hist"] = df["cereals_price"].pct_change() * 100
for window in [3, 6, 12]:
    df[f"volatility_{window}m"] = df["pct_change_hist"].rolling(window).std()

# Momentum: cumulative % change over the trailing window, captures trend strength
for window in [3, 6]:
    df[f"momentum_{window}m"] = (
        df["cereals_price"] / df["cereals_price"].shift(window) - 1
    ) * 100

df = df.dropna().reset_index(drop=True)
df.to_csv("data/processed/shock_features_v2.csv", index=False)
print(f"Saved {len(df)} rows with {df.shape[1]} columns")
print(f"New columns: precip_anomaly, temp_anomaly, volatility_3m/6m/12m, momentum_3m/6m")
