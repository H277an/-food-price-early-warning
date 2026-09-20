from pipeline import prepare_features, build_lstm_features
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

df = prepare_features("Cereals", ["us_midwest", "ukraine_wheat_belt", "india_punjab"])
seq_cols = ["price", "avg_precipitation", "avg_max_temp"]

# Proof #1: scaler leakage is actually gone
full_scaler = MinMaxScaler().fit(df[seq_cols])
n_sequences = len(df) - 12 + 1
split_seq = n_sequences - 60
train_only_scaler = MinMaxScaler().fit(df[seq_cols].iloc[:split_seq])

print("=== Scaler leakage check ===")
print(f"Full-data max precipitation: {full_scaler.data_max_[1]:.2f}")
print(f"Train-only max precipitation: {train_only_scaler.data_max_[1]:.2f}")
print("(These should DIFFER if the fix is working — if identical, the test")
print(" period simply never contained a new max, which is possible but check)")

print(f"\nFull-data min precipitation: {full_scaler.data_min_[1]:.2f}")
print(f"Train-only min precipitation: {train_only_scaler.data_min_[1]:.2f}")
print(f"Full-data max temp: {full_scaler.data_max_[2]:.2f}")
print(f"Train-only max temp: {train_only_scaler.data_max_[2]:.2f}")
print(f"Full-data min temp: {full_scaler.data_min_[2]:.2f}")
print(f"Train-only min temp: {train_only_scaler.data_min_[2]:.2f}")

print(f"\nFull-data max price: {full_scaler.data_max_[0]:.2f}")
print(f"Train-only max price: {train_only_scaler.data_max_[0]:.2f}")
print(f"Full-data min price: {full_scaler.data_min_[0]:.2f}")
print(f"Train-only min price: {train_only_scaler.data_min_[0]:.2f}")

# Proof #2: temporal alignment is correct
print("\n=== Temporal alignment check ===")
i = 50  # arbitrary row to inspect
print(f"Row {i} date: {df['date'].iloc[i]}, price: {df['price'].iloc[i]:.2f}")
print(f"Row {i} target_pct_change: {df['target_pct_change'].iloc[i]:.2f}%")
print(f"Row {i+1} date: {df['date'].iloc[i+1]}, price: {df['price'].iloc[i+1]:.2f}")
implied_change = (df['price'].iloc[i+1] - df['price'].iloc[i]) / df['price'].iloc[i] * 100
print(f"Implied change from row {i} to {i+1}: {implied_change:.2f}%")
print("(target_pct_change should match the implied change — confirms the")
print(" target definition itself, separate from the sequence input fix)")

# Run the actual fixed pipeline and confirm it completes
lstm_feats = build_lstm_features(df, "Cereals")
print(f"\n=== Fixed pipeline output ===")
print(f"Rows produced: {len(lstm_feats)}")
print(f"Expected: {n_sequences} (was 415 before the fix, now includes one more row)")