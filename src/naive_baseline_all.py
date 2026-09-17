from pipeline import prepare_features
from sklearn.metrics import mean_absolute_error, r2_score
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
    df = prepare_features(commodity, regions)
    split = len(df) - 60
    test = df.iloc[split:]

    naive_preds = test["price"]
    mae = mean_absolute_error(test["target_next_price"], naive_preds)
    r2 = r2_score(test["target_next_price"], naive_preds)
    results.append({"commodity": commodity, "naive_mae": round(mae, 3), "naive_r2": round(r2, 3)})

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("data/processed/naive_baselines_all.csv", index=False)