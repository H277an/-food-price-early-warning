import requests
import pandas as pd
import sqlite3
import time
from pathlib import Path

NEW_REGIONS = {
    "indonesia_sumatra": (0.5, 102.0),
    "malaysia_sabah": (5.0, 117.0),
    "brazil_sao_paulo": (-21.5, -47.8),
    "india_up_sugar_belt": (27.0, 80.0),
    "newzealand_waikato": (-37.8, 175.3),
    "ireland": (53.0, -8.0),
    "brazil_mato_grosso": (-15.0, -55.0),
}

START_DATE = "1990-01-01"
END_DATE = "2026-09-01"

conn = sqlite3.connect("db/food_prices.db")

# Skip regions already saved from the previous run
existing = pd.read_sql("SELECT DISTINCT region FROM weather", conn)["region"].tolist()
pending = {k: v for k, v in NEW_REGIONS.items() if k not in existing}
print(f"Already saved: {existing}")
print(f"Still need: {list(pending.keys())}")

for region, (lat, lon) in pending.items():
    print(f"\nFetching {region}...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": START_DATE, "end_date": END_DATE,
        "daily": "precipitation_sum,temperature_2m_max",
        "timezone": "auto",
    }

    success = False
    for attempt in range(5):
        response = requests.get(url, params=params)
        if response.status_code == 200:
            success = True
            break
        wait = 20 * (attempt + 1)
        print(f"  got status {response.status_code}, waiting {wait}s...")
        time.sleep(wait)

    if not success:
        print(f"  FAILED after 5 attempts, skipping {region} for now")
        continue

    data = response.json()["daily"]
    df = pd.DataFrame({
        "date": data["time"],
        "region": region,
        "precipitation_sum": data["precipitation_sum"],
        "temperature_2m_max": data["temperature_2m_max"],
    })

    df.to_sql("weather", conn, if_exists="append", index=False)
    conn.commit()
    print(f"  got {len(df)} days, saved to database")

    time.sleep(15)  # longer gap to respect the free tier's rate limit

check = pd.read_sql("SELECT DISTINCT region FROM weather ORDER BY region", conn)
conn.close()
print(f"\nAll regions now in weather table ({len(check)} total):")
print(check)