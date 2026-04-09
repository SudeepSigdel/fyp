import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROCESSED_DIR = "C:/Users/sudee/projects/Final Year Project/data/processed"

PRIMARY_LABEL = "Label_10d"

FEATURE_COLS = [
    "RSI_dist_50", "RSI_slope_3", "MACD_hist", "MACD_hist_slope_3",
    "EMA_cross", "Price_vs_SMA20",
    "BB_pctB", "BB_width", "ATR_ratio", "Vol_10d",
    "Volume_ratio", "Volume_spike", "OBV_slope_norm",
    "Ret_1d", "Ret_3d", "Ret_5d", "Ret_10d", "Ret_20d", "Ret_momentum",
    "In_uptrend", "RSI_oversold", "RSI_overbought", "HL_range_pct", "Gap_pct"
]
df = pd.read_parquet(os.path.join(PROCESSED_DIR, "all_stocks_labeled.parquet"))
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print(f"Loaded: {df.shape[0]:,} rows")
print(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Stocks: {df['Symbol'].nunique()}")
print(f"Features: {len(FEATURE_COLS)}")

EMBARGO_DAYS = 20

folds = [
    {"fold": 1, "train_end": "2017-12-31", "test_start": "2018-02-01", "test_end": "2018-12-31"},
    {"fold": 2, "train_end": "2018-12-31", "test_start": "2019-02-01", "test_end": "2019-12-31"},
    {"fold": 3, "train_end": "2019-12-31", "test_start": "2020-02-01", "test_end": "2020-12-31"},
    {"fold": 4, "train_end": "2020-12-31", "test_start": "2021-02-01", "test_end": "2021-12-31"},
    {"fold": 5, "train_end": "2021-12-31", "test_start": "2022-02-01", "test_end": "2022-12-31"},
    {"fold": 6, "train_end": "2022-12-31", "test_start": "2023-02-01", "test_end": "2023-12-31"},
    {"fold": 7, "train_end": "2023-12-31", "test_start": "2024-02-01", "test_end": "2025-09-21"},
]


print("\n" + "="*70)
print("WALK-FORWARD FOLD SUMMARY")
print("="*70)
print(f"{'Fold':<6} {'Train period':<25} {'Test period':<22} "
      f"{'Train rows':>11} {'Test rows':>10} {'Train+':>8} {'Test+':>8}")
print("-"*70)

fold_stats = []

for f in folds:
    train_mask = df["Date"] <= f["train_end"]
    test_mask  = (df["Date"] >= f["test_start"]) & (df["Date"] <= f["test_end"])

    train_df = df[train_mask].dropna(subset=FEATURE_COLS + [PRIMARY_LABEL])
    test_df  = df[test_mask].dropna(subset=FEATURE_COLS + [PRIMARY_LABEL])

    train_pos = train_df[PRIMARY_LABEL].sum()
    test_pos  = test_df[PRIMARY_LABEL].sum()

    train_period = f"2012-01-01 → {f['train_end']}"
    test_period  = f"{f['test_start']} → {f['test_end']}"

    print(f"  {f['fold']:<4} {train_period:<25} {test_period:<22} "
          f"{len(train_df):>11,} {len(test_df):>10,} "
          f"{train_pos:>8,} {test_pos:>8,}")

    fold_stats.append({
        "fold":       f["fold"],
        "train_rows": len(train_df),
        "test_rows":  len(test_df),
        "train_pos":  train_pos,
        "test_pos":   test_pos,
        "train_pct":  train_pos / len(train_df) * 100 if len(train_df) > 0 else 0,
        "test_pct":   test_pos  / len(test_df)  * 100 if len(test_df)  > 0 else 0,
    })

print("\n(Train+ and Test+ = number of Label=1 rows in each split)")

print("\n" + "="*50)
print("LEAKAGE CHECK")
print("="*50)

for f in folds:
    train_dates = df[df["Date"] <= f["train_end"]]["Date"]
    test_dates  = df[(df["Date"] >= f["test_start"]) &
                     (df["Date"] <= f["test_end"])]["Date"]

    overlap = (train_dates.max() >= test_dates.min()) if len(test_dates) > 0 else False
    status  = "OVERLAP DETECTED" if overlap else "Clean"
    print(f"  Fold {f['fold']}: {status}")

fig, ax = plt.subplots(figsize=(14, 5))

colors_train = "#2196F3"
colors_test  = "#FF9800"

for i, f in enumerate(folds):
    y = i * 1.2

    # Training bar
    train_start_dt = pd.Timestamp("2012-01-01")
    train_end_dt   = pd.Timestamp(f["train_end"])
    test_start_dt  = pd.Timestamp(f["test_start"])
    test_end_dt    = pd.Timestamp(f["test_end"])

    ax.barh(y, (train_end_dt - train_start_dt).days,
            left=train_start_dt.toordinal() - pd.Timestamp("2012-01-01").toordinal(),
            height=0.8, color=colors_train, alpha=0.7)

    ax.barh(y, (test_end_dt - test_start_dt).days,
            left=(test_start_dt - pd.Timestamp("2012-01-01")).days,
            height=0.8, color=colors_test, alpha=0.9)

    ax.text(-30, y, f"Fold {f['fold']}", va="center", ha="right", fontsize=9)

ax.set_xlabel("Days from 2012-01-01")
ax.set_title("Walk-forward validation folds\n(blue = train, orange = test)")
ax.set_yticks([])

train_patch = mpatches.Patch(color=colors_train, alpha=0.7, label="Training period")
test_patch  = mpatches.Patch(color=colors_test,  alpha=0.9, label="Test period")
ax.legend(handles=[train_patch, test_patch])

plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "walk_forward_folds.png"), dpi=150)
plt.show()
print("\n Fold visualisation saved")

import json

fold_config = {
    "folds":        folds,
    "feature_cols": FEATURE_COLS,
    "label_col":    PRIMARY_LABEL,
    "embargo_days": EMBARGO_DAYS,
}

config_path = os.path.join(PROCESSED_DIR, "fold_config.json")
with open(config_path, "w") as fp:
    json.dump(fold_config, fp, indent=2)

print(f"Fold config saved → {config_path}")
print("\nWalk-forward setup complete! Next: XGBoost model training.")