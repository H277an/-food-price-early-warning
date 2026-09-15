import requests
import pandas as pd
from pathlib import Path

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# region_name: (latitude, longitude) — major growing areas for wheat/maize
REGIONS = {
    "us_midwest": (41.6, -93.6),
    "ukraine_wheat_belt": (49.0, 32.0),
    "india_punjab": (30.9, 75.8),
}

START_DATE = "1990-01-01"
END_DATE = "2026-09-01"

all_data = []

for region, (lat, lon) in REGIONS.items():
    print(f"Fetching {region}...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": "precipitation_sum,temperature_2m_max",
        "timezone": "auto",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()["daily"]

    df = pd.DataFrame({
        "date": data["time"],
        "region": region,
        "precipitation_sum": data["precipitation_sum"],
        "temperature_2m_max": data["temperature_2m_max"],
    })
    all_data.append(df)
    print(f"  got {len(df)} days")

weather_df = pd.concat(all_data, ignore_index=True)
weather_df.to_csv(OUT_DIR / "weather_data.csv", index=False)
import sqlite3

conn = sqlite3.connect("db/food_prices.db")
conn.executescript(Path("db/schema.sql").read_text(encoding="utf-8-sig"))
weather_df.to_sql("weather", conn, if_exists="replace", index=False)
conn.commit()

# Aggregate daily weather into monthly averages/totals — this is what we'll
# actually join against the monthly price data later
query = """
SELECT
    region,
    substr(date, 1, 7) AS month,
    SUM(precipitation_sum) AS total_precipitation,
    AVG(temperature_2m_max) AS avg_max_temp
FROM weather
GROUP BY region, month
ORDER BY region, month DESC
LIMIT 6;
"""
result = pd.read_sql(query, conn)
print("\nMonthly aggregation check:")
print(result)

conn.close()
print(f"\nSaved {len(weather_df)} rows to weather table in db/food_prices.db")