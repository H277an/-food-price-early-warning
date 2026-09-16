import pandas as pd

df = pd.read_csv("data/processed/final_merged_features.csv")
df = df.sort_values("date").reset_index(drop=True)

# % change month-over-month, based on the actual next-month target
df["pct_change"] = (df["target_next_price"] - df["cereals_price"]) / df["cereals_price"] * 100

print("Distribution of month-over-month % change:")
print(df["pct_change"].describe())

print("\nPercentiles:")
for p in [90, 95, 97.5, 99]:
    print(f"  {p}th percentile: {df['pct_change'].quantile(p/100):.2f}%")

print("\nHow many months exceed various thresholds:")
for threshold in [5, 7, 10, 15]:
    count = (df["pct_change"] > threshold).sum()
    pct = count / len(df) * 100
    print(f"  >{threshold}%: {count} months ({pct:.1f}% of data)")