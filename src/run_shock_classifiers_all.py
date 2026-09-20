from pipeline import prepare_features, build_lstm_features, train_shock_classifier
import pandas as pd

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
        lstm_feats = build_lstm_features(df, commodity)
        merged = pd.merge(df, lstm_feats, on="date")

        result = train_shock_classifier(merged, commodity)
        results.append(result)
    except Exception as e:
        print(f"  FAILED for {commodity}: {e}")
        results.append({"commodity": commodity, "error": str(e)})

print("\n\n=== SHOCK CLASSIFIER SUMMARY ACROSS ALL COMMODITIES ===")
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("data/processed/shock_classifier_results_all.csv", index=False)
print("\nSaved to data/processed/shock_classifier_results_all.csv")