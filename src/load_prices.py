import sqlite3
import pandas as pd
from pathlib import Path

CSV_PATH = Path("data/raw/food_price_indices_data.csv")
DB_PATH = Path("db/food_prices.db")
SCHEMA_PATH = Path("db/schema.sql")

# The FAO file has 2 title rows before the real header, so skip them
df = pd.read_csv(CSV_PATH, skiprows=2)

# Drop the blank spacer row and any unnamed trailing columns from stray commas
df = df.dropna(subset=["Date"])
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Reshape from wide (one column per commodity) to long (one row per date+commodity)
# which matches our SQL schema and makes querying/plotting per-commodity easy
long_df = df.melt(id_vars=["Date"], var_name="commodity", value_name="index_value")
long_df = long_df.rename(columns={"Date": "date"})
long_df = long_df.dropna(subset=["index_value"])

print(f"Loaded {len(long_df)} rows across {long_df['commodity'].nunique()} commodities")
print(long_df.head())

# Load into SQLite
conn = sqlite3.connect(DB_PATH)
conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8-sig"))
long_df.to_sql("prices", conn, if_exists="replace", index=False)
conn.commit()

# Sanity-check query: month-over-month change for the main Food Price Index,
# using a window function (LAG) to compare each row to the previous month
query = """
SELECT
    date,
    index_value,
    index_value - LAG(index_value) OVER (ORDER BY date) AS mom_change
FROM prices
WHERE commodity = 'Food Price Index'
ORDER BY date DESC
LIMIT 10;
"""
result = pd.read_sql(query, conn)
print(result)

conn.close()

print(f"Saved to {DB_PATH}")