import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt


PROCESSED_DIR = "C:/Users/sudee/projects/Final Year Project/data/processed"

df = pd.read_parquet(os.path.join(PROCESSED_DIR, "all_stocks_labeled.parquet"))
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

with open(os.path.join(PROCESSED_DIR, "fold_config.json")) as fp:
    config = json.load(fp)

FOLDS        = config["folds"]
FEATURE_COLS = config["feature_cols"]
LABEL_COL    = config["label_col"]

print(f"Loaded: {df.shape[0]:,} rows | {len(FEATURE_COLS)} features | label: {LABEL_COL}")


pos_count = (df[LABEL_COL] == 1).sum()
neg_count = (df[LABEL_COL] == 0).sum()
spw = round(neg_count / pos_count, 2)
print(f"scale_pos_weight: {spw}  (neg/pos ratio)")

XGB_PARAMS = {
    "n_estimators":      300,    # Number of trees. More = slower but potentially better.
                                 # 300 is a good starting point for this dataset size.

    "max_depth":         4,      # How deep each tree can grow.
                                 # Shallow trees (3-5) prevent overfitting.
                                 # Deep trees memorise noise.

    "learning_rate":     0.05,   # How much each tree contributes to the final answer.
                                 # Small values (0.01-0.1) require more trees but generalise better.
                                 # Think of it as "step size" when learning.

    "subsample":         0.8,    # Each tree only sees 80% of training rows (randomly selected).
                                 # This adds randomness and prevents overfitting.

    "colsample_bytree":  0.8,    # Each tree only sees 80% of features.
                                 # Prevents the model from over-relying on any single feature.

    "min_child_weight":  10,     # A leaf node must have at least 10 samples.
                                 # Higher values = more conservative splits = less overfitting.
                                 # Very important for financial data where patterns are subtle.

    "reg_alpha":         0.1,    # L1 regularisation: pushes less-useful feature weights to zero.
                                 # Effectively performs feature selection automatically.

    "reg_lambda":        1.0,    # L2 regularisation: keeps all weights small.
                                 # Reduces sensitivity to individual noisy samples.

    "scale_pos_weight":  spw,    # Handles class imbalance (calculated above)

    "objective":        "binary:logistic",  # We're predicting probability of label=1
    "eval_metric":      "auc",              # Optimise for AUC during training
    "use_label_encoder": False,
    "random_state":      42,
    "n_jobs":           -1,      # Use all CPU cores for speed
}



all_predictions = []   # Will hold out-of-sample predictions from all folds
fold_metrics    = []   # Will hold AUC scores per fold

print("\n" + "="*65)
print("WALK-FORWARD TRAINING")
print("="*65)

for f in FOLDS:
    fold_num = f["fold"]

    train_df = df[df["Date"] <= f["train_end"]].copy()
    test_df  = df[(df["Date"] >= f["test_start"]) &
                  (df["Date"] <= f["test_end"])].copy()

    train_df = train_df.dropna(subset=FEATURE_COLS + [LABEL_COL])
    test_df  = test_df.dropna(subset=FEATURE_COLS + [LABEL_COL])

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[LABEL_COL].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df[LABEL_COL].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # fit + transform train
    X_test  = scaler.transform(X_test)        # transform test only

    model = XGBClassifier(**XGB_PARAMS, verbosity=0)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba)

    print(f"  Fold {fold_num}: train={len(train_df):>7,} rows | "
          f"test={len(test_df):>6,} rows | AUC={auc:.4f}")

    test_df = test_df.copy()
    test_df["Pred_proba"]    = proba
    test_df["Fold"]          = fold_num
    test_df["Pred_label"]    = (proba >= 0.5).astype(int)

    all_predictions.append(test_df)
    fold_metrics.append({
        "fold":       fold_num,
        "train_rows": len(train_df),
        "test_rows":  len(test_df),
        "auc":        auc,
        "test_period": f"{f['test_start']} → {f['test_end']}"
    })

    import pickle
    model_dir = os.path.join(PROCESSED_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, f"model_fold{fold_num}.pkl"), "wb") as fp:
        pickle.dump({"model": model, "scaler": scaler,
                     "features": FEATURE_COLS}, fp)


combined_preds = pd.concat(all_predictions, ignore_index=True)

print("\n" + "="*65)
print("OVERALL PERFORMANCE SUMMARY")
print("="*65)

overall_auc = roc_auc_score(
    combined_preds[LABEL_COL],
    combined_preds["Pred_proba"]
)
print(f"\nOverall out-of-sample AUC: {overall_auc:.4f}")
print(f"Total out-of-sample predictions: {len(combined_preds):,}")

print(f"\n{'Fold':<6} {'Period':<35} {'AUC':>8}")
print("-"*50)
for fm in fold_metrics:
    bar_len = int((fm["auc"] - 0.45) * 100)
    bar = "█" * max(0, bar_len)
    print(f"  {fm['fold']:<4} {fm['test_period']:<35} {fm['auc']:.4f}  {bar}")

mean_auc = np.mean([fm["auc"] for fm in fold_metrics])
std_auc  = np.std([fm["auc"] for fm in fold_metrics])
print(f"\n  Mean AUC across folds: {mean_auc:.4f} ± {std_auc:.4f}")
print(f"  (Std deviation measures consistency across time periods)")


print("\n" + "="*65)
print("CLASSIFICATION REPORT (threshold = 0.50)")
print("="*65)
print(classification_report(
    combined_preds[LABEL_COL],
    combined_preds["Pred_label"],
    target_names=["Label=0 (fail)", "Label=1 (success)"]
))

last_model = pickle.load(
    open(os.path.join(PROCESSED_DIR, "models", f"model_fold7.pkl"), "rb")
)["model"]

importances = pd.Series(
    last_model.feature_importances_,
    index=FEATURE_COLS
).sort_values(ascending=True)

plt.figure(figsize=(9, 7))
importances.plot(kind="barh", color="steelblue", edgecolor="none")
plt.axvline(1/len(FEATURE_COLS), color="red", linestyle="--",
            label=f"Uniform baseline ({1/len(FEATURE_COLS):.3f})")
plt.title("Feature importance — Fold 7 model\n"
          "(features above red line contribute more than average)")
plt.xlabel("Importance score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "feature_importance.png"), dpi=150)
plt.show()

combined_preds.to_parquet(
    os.path.join(PROCESSED_DIR, "oos_predictions.parquet"), index=False
)

metrics_df = pd.DataFrame(fold_metrics)
metrics_df.to_csv(os.path.join(PROCESSED_DIR, "fold_metrics.csv"), index=False)

print(f"\nSaved out-of-sample predictions → oos_predictions.parquet")
print(f"Saved fold metrics              → fold_metrics.csv")
print(f"\nModel training complete! Next: backtesting with transaction costs.")