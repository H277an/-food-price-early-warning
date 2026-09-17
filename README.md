# Global Food Price Risk Intelligence System

**Status: In progress** — Cereals (reference commodity) pipeline built and validated. Extending to Meat, Dairy, Oils, and Sugar next.

## What this is

A hybrid deep learning + gradient boosting system that forecasts next-month commodity prices and flags elevated price-shock risk, using region-specific weather signals alongside price history. Built as a general-purpose pipeline — one architecture, applied consistently across commodities — rather than a one-off model.

**Core objective:** For each of 5 major food commodities (Cereals, Meat, Dairy, Oils, Sugar), predict next-month price direction and flag shock risk, tied to the actual production geography behind each commodity, with an honest account of what worked and what didn't.

## What makes this different from a typical portfolio project

- **Hybrid architecture, not a single model.** A sequence encoder (LSTM, and a Transformer variant tested as a comparison) extracts temporal features from price + weather sequences; those features feed into a gradient-boosted/ensemble model (XGBoost or Random Forest) alongside hand-engineered lag features.
- **Region-specific data, not generic "weather."** Weather is pulled per commodity from the actual regions that produce it (e.g. US Midwest/Ukraine/Punjab for cereals; São Paulo/Uttar Pradesh for sugar) — not a single abstract column.
- **A rigorous, controlled model comparison.** Three architectures were benchmarked with the downstream model held constant where possible, to isolate what actually drives performance:
  - Model 1: LSTM → XGBoost
  - Model 2: LSTM → Random Forest
  - Model 3: Transformer → Random Forest (paired with the winner of 1 vs 2)
- **Documented negative results, not hidden ones.** Two things were tried and *didn't* help — they're reported honestly below instead of omitted.

## Data sources

| Source | What | Access |
|---|---|---|
| FAO Food Price Index | Monthly commodity sub-indices, 1990–present | Public Excel/CSV download |
| Open-Meteo Historical API | Daily precipitation & max temperature by region | Free, no API key, 1990–present |
| GDELT 2.0 | News event/theme signals (drought, export bans, conflict) | Free, no API key — *planned, not yet integrated* |

## Architecture

```
Price history + Weather (per region)
        │
        ▼
Sequence encoder (LSTM or Transformer) — 12-month lookback
        │
        ▼
Extracted temporal features (16-dim)
        │
        ▼
   + hand-engineered lag features (1/2/3/6/12-month)
        │
        ▼
Downstream model (XGBoost or Random Forest)
        │
        ├──► Next-month % price change (regression)
        └──► Shock risk classification (>5% MoM move)
```

## Key finding #1: Beating a naive baseline is hard — and that's reported honestly

Commodity prices are highly autocorrelated month-to-month. A naive "predict no change" baseline is a genuinely strong benchmark, consistent with known difficulty in commodity/financial forecasting (related to the efficient market hypothesis — easily predictable moves tend to already be priced in).

| Model | Price-level Test MAE | R² |
|---|---|---|
| **Naive baseline** (predict no change) | **3.085** | **0.927** |
| Model 2: LSTM + Random Forest | 3.857 | 0.899 |
| Model 3: Transformer + Random Forest | 3.859 | 0.904 |
| Model 1: LSTM + XGBoost | 4.328 | 0.886 |

Random Forest won the Model 1 vs. Model 2 comparison — its feature importance shows a balanced reliance on weather, LSTM-extracted features, and lag features, rather than XGBoost's narrower dependence on the LSTM output alone. Random Forest was therefore selected as the downstream pairing for the Transformer comparison (Model 3).

**Model 2 and Model 3 are statistically tied** (MAE differs by 0.002; R² marginally favors the Transformer). This is a genuine and informative result: self-attention did not meaningfully outperform recurrent encoding at this data scale (355 training sequences) — consistent with the well-documented fact that Transformer architectures are more data-hungry than LSTMs and typically need far larger datasets to show their advantage. Given the tie, **LSTM + Random Forest was selected as the production model** for the remaining commodities — the simpler, faster-to-train architecture, chosen on engineering grounds rather than a forced preference for the more novel one.

None of the three hybrid models beat the naive baseline on raw point-forecast accuracy. This is reported as a genuine, expected-in-the-literature result rather than adjusted away.

## Key finding #2: More features made the shock classifier worse

The shock classifier (predicting >5% month-over-month price moves, using `TimeSeriesSplit` cross-validation to get a reliable evaluation sample) started at **ROC AUC 0.635**. Adding weather anomaly and price volatility/momentum features — a reasonable hypothesis — actually *reduced* performance to AUC 0.584, most likely due to overfitting a ~30-feature model against only 36 positive (shock) examples in the dataset. The simpler feature set was kept as the final model.

## Key finding #3: Threshold tuning is where the real, usable value is

Rather than reporting a single accuracy number, the shock classifier is evaluated as a business tradeoff between two realistic use cases:

| Threshold | Recall (shocks caught) | Precision |
|---|---|---|
| 0.2 (early-warning mode) | 58% | 13% |
| 0.8 (high-confidence mode) | 17% | 67% |

At the low threshold, the model catches over half of actual price shocks — useful where missing a shock is costlier than a false alarm. At the high threshold, two-thirds of its alerts are correct — useful for a system that should only speak up when confident.

## Tech stack

- **Data:** SQLite (with window-function SQL queries), pandas
- **Modeling:** TensorFlow/Keras (LSTM, Transformer), XGBoost, scikit-learn (Random Forest)
- **Weather:** Open-Meteo API
- **News (planned):** GDELT 2.0
- **Dashboard (planned):** Streamlit, deployed on Streamlit Community Cloud

## Repo structure

```
food-price-early-warning/
├── data/
│   ├── raw/              # FAO price data, weather pulls
│   └── processed/        # merged feature tables per model stage
├── db/
│   ├── schema.sql
│   └── food_prices.db    # SQLite database (gitignored)
├── src/
│   ├── load_prices.py
│   ├── fetch_weather.py
│   ├── prepare_features.py
│   ├── build_lstm.py
│   ├── build_transformer.py
│   ├── train_xgboost.py
│   ├── train_random_forest.py
│   ├── train_shock_classifier.py
│   └── naive_baseline.py
└── README.md
```

## Scope

**In scope (current build):** Cereals as the reference commodity for full model development and comparison; then the same pipeline applied to Meat, Dairy, Oils, and Sugar, each with commodity-specific growing-region weather data.

**Planned:** GDELT news-signal integration, tested against a specific hypothesis (does drought/export-ban/conflict news improve shock recall beyond price + weather alone) rather than added generically. A Streamlit dashboard showing a composite risk index across all 5 commodities.

**Explicitly out of scope (for now):** Commodities beyond these 5; regions beyond the 1-3 per commodity already defined; pretrained foundation-model (e.g. Chronos, TimesFM) comparison, reserved as a final polish step once the core system is complete.

## Setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
python src\load_prices.py
python src\fetch_weather.py
python src\prepare_features.py
python src\build_lstm.py
python src\train_xgboost.py
python src\train_random_forest.py
python src\train_shock_classifier.py
```
