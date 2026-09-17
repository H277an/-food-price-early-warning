from pipeline import prepare_features, build_lstm_features, train_random_forest
import pandas as pd
import json

COMMODITY_REGIONS = {
    "Cereals": ["us_midwest", "ukraine_wheat_belt", "india_punjab"],
    "Oils": ["indonesia_sumatra", "malaysia_sabah"],
    "Sugar": ["brazil_sao_paulo", "india_up_sugar_belt"],
    "Dairy": ["newzealand_waikato", "ireland"],
    "Meat": ["brazil_mato_grosso", "us_midwest"],
}

results = []

for commodity, regions in COMMODITY_REGIONS.items():
    print(f"\n=== {commodity} ===")
    try:
        df = prepare_features(commodity, regions)
        print(f"  {len(df)} rows")

        if len(df) < 100:
            print(f"  SKIPPING {commodity} — insufficient data ({len(df)} rows)")
            continue

        lstm_feats = build_lstm_features(df, commodity)
        merged = pd.merge(df, lstm_feats, on="date")
        result = train_random_forest(merged, commodity)

        results.append({
            "commodity": commodity,
            "rows": len(df),
            "mae": round(result["mae"], 3),
            "r2": round(result["r2"], 3),
        })

        merged.to_csv(f"data/processed/final_merged_{commodity.lower()}.csv", index=False)
        result["model"].save = None  # avoid trying to pickle the model in this loop
    except Exception as e:
        print(f"  FAILED for {commodity}: {e}")
        results.append({"commodity": commodity, "error": str(e)})

print("\n\n=== SUMMARY ACROSS ALL COMMODITIES ===")
results_df = pd.DataFrame(results)
print(results_df)

results_df.to_csv("data/processed/all_commodities_results.csv", index=False)
print("\nSaved to data/processed/all_commodities_results.csv")