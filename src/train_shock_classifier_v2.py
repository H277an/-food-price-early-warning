import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

np.random.seed(42)

df = pd.read_csv("data/processed/shock_features_v2.csv")
df = df.sort_values("date").reset_index(drop=True)

df["pct_change"] = (df["target_next_price"] - df["cereals_price"]) / df["cereals_price"] * 100
df["is_shock"] = (df["pct_change"] > 5).astype(int)

exclude_cols = ["date", "target_next_price", "pct_change", "is_shock", "month_num", "pct_change_hist"]
feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols]
y = df["is_shock"]

print(f"Total shocks: {y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")

tscv = TimeSeriesSplit(n_splits=5, test_size=40)
all_y_test, all_preds, all_probs = [], [], []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    if pos == 0:
        print(f"Fold {fold+1}: no shocks in training data, skipping")
        continue
    scale_pos_weight = neg / pos

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42, eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"Fold {fold+1}: {y_test.sum()} shocks in test set, model caught {((preds==1) & (y_test==1)).sum()}")

    all_y_test.extend(y_test)
    all_preds.extend(preds)
    all_probs.extend(probs)

print(f"\n=== Combined results across all folds ({len(all_y_test)} test months, {sum(all_y_test)} total shocks) ===")
print(classification_report(all_y_test, all_preds, target_names=["normal", "shock"]))
print("Confusion matrix:")
print(confusion_matrix(all_y_test, all_preds))
if sum(all_y_test) > 0:
    print(f"ROC AUC: {roc_auc_score(all_y_test, all_probs):.3f}")

final_scale_pos_weight = (y == 0).sum() / (y == 1).sum()
final_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=final_scale_pos_weight,
    random_state=42, eval_metric="logloss",
)
final_model.fit(X, y)
importance = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 features (final model):")
print(importance.head(10))

final_model.save_model("db/shock_classifier_v2.json")
print("\nSaved to db/shock_classifier_v2.json")
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(all_y_test, all_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = f1_scores.argmax()
print(f"\nBest threshold by F1: {thresholds[best_idx]:.3f}")
print(f"At that threshold — Precision: {precisions[best_idx]:.3f}, Recall: {recalls[best_idx]:.3f}, F1: {f1_scores[best_idx]:.3f}")

# Show performance at a few fixed thresholds too, for the README
for t in [0.3, 0.4, 0.5]:
    preds_t = [1 if p >= t else 0 for p in all_probs]
    tp = sum((np.array(preds_t) == 1) & (np.array(all_y_test) == 1))
    fp = sum((np.array(preds_t) == 1) & (np.array(all_y_test) == 0))
    fn = sum((np.array(preds_t) == 0) & (np.array(all_y_test) == 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"Threshold {t}: caught {tp}/{tp+fn} shocks, precision {prec:.2f}, recall {rec:.2f}")