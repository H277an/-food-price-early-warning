import sqlite3
import pandas as pd

conn = sqlite3.connect("db/food_prices.db")

# Monthly cereals price series
prices = pd.read_sql("""
    SELECT date, index_value AS cereals_price
    FROM prices
    WHERE commodity = 'Cereals'
    ORDER BY date
""", conn)

# Monthly weather aggregates, averaged across our 3 wheat-growing regions
weather = pd.read_sql("""
    SELECT
        substr(date, 1, 7) AS date,
        AVG(precipitation_sum) AS avg_precipitation,
        AVG(temperature_2m_max) AS avg_max_temp
    FROM weather
    GROUP BY substr(date, 1, 7)
""", conn)

conn.close()

# Merge on year-month
df = pd.merge(prices, weather, on="date", how="inner")

# Lag features — critical for both the LSTM sequence input and XGBoost
for lag in [1, 2, 3, 6, 12]:
    df[f"price_lag_{lag}"] = df["cereals_price"].shift(lag)

# Target: next month's price (what we're predicting)
df["target_next_price"] = df["cereals_price"].shift(-1)

df = df.dropna().reset_index(drop=True)

df.to_csv("data/processed/cereals_features.csv", index=False)
print(f"Saved {len(df)} rows with {df.shape[1]} columns")
print(df.head())
print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")