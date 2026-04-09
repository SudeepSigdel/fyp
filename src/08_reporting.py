import pandas as pd
import numpy as np
import os
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = "C:/Users/sudee/projects/Final Year Project/data/processed"
REPORT_DIR    = os.path.join(PROCESSED_DIR, "report")
os.makedirs(REPORT_DIR, exist_ok=True)


preds   = pd.read_parquet(os.path.join(PROCESSED_DIR, "oos_predictions.parquet"))
preds["Date"] = pd.to_datetime(preds["Date"])

ml_trades = pd.read_parquet(os.path.join(PROCESSED_DIR, "ml_trades.parquet"))

with open(os.path.join(PROCESSED_DIR, "fold_config.json")) as fp:
    config = json.load(fp)

FEATURE_COLS  = config["feature_cols"]
LABEL_COL     = config["label_col"]
TRANS_COST    = 0.01

fold7_bundle  = pickle.load(
    open(os.path.join(PROCESSED_DIR, "models", "model_fold7.pkl"), "rb")
)
model   = fold7_bundle["model"]
scaler  = fold7_bundle["scaler"]

print("All files loaded.")

fig, ax = plt.subplots(figsize=(8, 7))

fold_aucs = []
colors = plt.cm.Blues(np.linspace(0.4, 0.95, 7)) #type: ignore

for fold_num, color in zip(sorted(preds["Fold"].unique()), colors):
    fold_data = preds[preds["Fold"] == fold_num].dropna(
        subset=[LABEL_COL, "Pred_proba"])
    if len(fold_data) < 100:
        continue
    fpr, tpr, _ = roc_curve(fold_data[LABEL_COL], fold_data["Pred_proba"])
    fold_auc     = auc(fpr, tpr)
    fold_aucs.append(fold_auc)
    year_label   = {1:"2018",2:"2019",3:"2020",4:"2021",
                    5:"2022",6:"2023",7:"2024–25"}[fold_num]
    ax.plot(fpr, tpr, color=color, linewidth=1.8,
            label=f"Fold {fold_num} ({year_label})  AUC={fold_auc:.3f}")

ax.plot([0,1],[0,1], "k--", linewidth=1, label="Random (AUC=0.500)")
ax.fill_between([0,1],[0,1], alpha=0.05, color="gray")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Walk-Forward Folds\n"
             "Each curve is fully out-of-sample", fontsize=13)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim([0,1]); ax.set_ylim([0,1]) #type: ignore

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "fig1_roc_curves.png"), dpi=150)
plt.close()
print(" Figure 1: ROC curves saved")

thresholds    = np.arange(0.50, 0.70, 0.01)
thresh_results = []

for t in thresholds:
    mask   = preds["Pred_proba"] >= t
    trades = preds[mask].copy()
    trades["Net_return"] = trades["Fwd_ret_10d"] - TRANS_COST

    if len(trades) < 30:
        break

    net   = trades["Net_return"].dropna()
    wins  = (net > 0).mean() * 100
    gp    = net[net > 0].sum()
    gl    = abs(net[net < 0].sum())
    pf    = gp / gl if gl > 0 else np.nan
    sr    = (net.mean() / net.std()) * np.sqrt(252/10) if net.std() > 0 else 0

    thresh_results.append({
        "threshold": t,
        "trades":    len(trades),
        "win_rate":  wins,
        "pf":        pf,
        "sharpe":    sr,
    })

tdf = pd.DataFrame(thresh_results)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].plot(tdf["threshold"], tdf["win_rate"], "steelblue", linewidth=2)
axes[0,0].axhline(50, color="red", linestyle="--", linewidth=1)
axes[0,0].set_title("Win Rate vs Threshold"); axes[0,0].set_ylabel("Win Rate (%)")
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(tdf["threshold"], tdf["pf"], "green", linewidth=2)
axes[0,1].axhline(1.0, color="red", linestyle="--", linewidth=1, label="Break-even")
axes[0,1].set_title("Profit Factor vs Threshold"); axes[0,1].set_ylabel("Profit Factor")
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(tdf["threshold"], tdf["sharpe"], "purple", linewidth=2)
axes[1,0].axhline(0, color="red", linestyle="--", linewidth=1)
axes[1,0].set_title("Sharpe Ratio vs Threshold"); axes[1,0].set_ylabel("Sharpe Ratio")
axes[1,0].set_xlabel("Probability Threshold"); axes[1,0].grid(True, alpha=0.3)

axes[1,1].bar(tdf["threshold"], tdf["trades"], width=0.008,
              color="steelblue", alpha=0.7)
axes[1,1].set_title("Number of Trades vs Threshold")
axes[1,1].set_ylabel("Trade Count"); axes[1,1].set_xlabel("Probability Threshold")
axes[1,1].grid(True, alpha=0.3)

plt.suptitle("Threshold Sensitivity Analysis\n"
             "Higher threshold = fewer but better-quality trades", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "fig2_threshold_analysis.png"), dpi=150)
plt.close()
print(" Figure 2: Threshold analysis saved")



importances = pd.Series(
    model.feature_importances_,
    index=FEATURE_COLS
).sort_values(ascending=True)

category_colors = {
    "RSI":    "#4472C4",
    "MACD":   "#ED7D31",
    "EMA":    "#ED7D31",
    "Price":  "#ED7D31",
    "BB":     "#A9D18E",
    "ATR":    "#A9D18E",
    "Vol":    "#A9D18E",
    "Volume": "#FFD966",
    "OBV":    "#FFD966",
    "Ret":    "#9DC3E6",
    "In_":    "#C5A5CF",
    "HL":     "#C5A5CF",
    "Gap":    "#C5A5CF",
}

bar_colors = []
for feat in importances.index:
    color = "#AAAAAA"
    for prefix, c in category_colors.items():
        if feat.startswith(prefix):
            color = c
            break
    bar_colors.append(color)

fig, ax = plt.subplots(figsize=(9, 8))
bars = ax.barh(importances.index, importances.values, #type: ignore
               color=bar_colors, edgecolor="none", height=0.7)
ax.axvline(1/len(FEATURE_COLS), color="red", linestyle="--",
           linewidth=1.2, label=f"Uniform baseline")
ax.set_title("Feature Importance — Fold 7 Model\n"
             "(trained on 2012–2023, most representative)", fontsize=13)
ax.set_xlabel("Importance Score")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis="x")

for bar, val in zip(bars, importances.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8)

from matplotlib.patches import Patch
legend_items = [
    Patch(color="#4472C4", label="Momentum (RSI)"),
    Patch(color="#ED7D31", label="Momentum (MACD/EMA/Price)"),
    Patch(color="#A9D18E", label="Volatility (BB/ATR)"),
    Patch(color="#FFD966", label="Volume (OBV)"),
    Patch(color="#9DC3E6", label="Returns"),
    Patch(color="#C5A5CF", label="Context"),
]
ax.legend(handles=legend_items, fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "fig3_feature_importance.png"), dpi=150)
plt.close()
print(" Figure 3: Feature importance saved")

fold_summary = []
PROB_THRESHOLD = 0.55

for fold_num in sorted(preds["Fold"].unique()):
    fd = preds[preds["Fold"] == fold_num].dropna(
        subset=[LABEL_COL, "Pred_proba", "Fwd_ret_10d"])

    fold_auc = auc(*roc_curve(fd[LABEL_COL], fd["Pred_proba"])[:2])

    trades = fd[fd["Pred_proba"] >= PROB_THRESHOLD].copy()
    trades["Net_return"] = trades["Fwd_ret_10d"] - TRANS_COST
    net = trades["Net_return"].dropna()
    sharpe = (net.mean()/net.std()*np.sqrt(252/10)) if (len(net)>10 and net.std()>0) else 0

    fold_summary.append({
        "fold":   fold_num,
        "year":   {1:"2018",2:"2019",3:"2020",4:"2021",
                   5:"2022",6:"2023",7:"2024-25"}[fold_num],
        "auc":    fold_auc,
        "sharpe": sharpe,
    })

fsdf = pd.DataFrame(fold_summary)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

bar_colors_auc = ["#c0392b" if v < 0.50 else
                  "#f39c12" if v < 0.54 else
                  "#27ae60" for v in fsdf["auc"]]
ax1.bar(fsdf["year"], fsdf["auc"], color=bar_colors_auc, edgecolor="none")
ax1.axhline(0.50, color="black", linestyle="--", linewidth=1.2, label="Random (0.50)")
ax1.axhline(fsdf["auc"].mean(), color="steelblue", linestyle=":",
            linewidth=1.2, label=f"Mean ({fsdf['auc'].mean():.3f})")
ax1.set_title("AUC per Fold"); ax1.set_ylabel("ROC-AUC")
ax1.set_ylim([0.44, 0.62]); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

bar_colors_sr = ["#c0392b" if v < 0 else
                 "#f39c12" if v < 0.5 else
                 "#27ae60" for v in fsdf["sharpe"]]
ax2.bar(fsdf["year"], fsdf["sharpe"], color=bar_colors_sr, edgecolor="none")
ax2.axhline(0, color="black", linestyle="--", linewidth=1.2)
ax2.axhline(0.5, color="steelblue", linestyle=":", linewidth=1.2, label="0.5 target")
ax2.set_title(f"Sharpe Ratio per Fold (threshold={PROB_THRESHOLD})")
ax2.set_ylabel("Annualised Sharpe Ratio")
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.suptitle("Walk-Forward Performance by Year\n"
             "Red = poor, Orange = marginal, Green = good", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "fig4_fold_performance.png"), dpi=150)
plt.close()
print(" Figure 4: Fold performance saved")


print("\n" + "="*65)
print("FINAL SUMMARY TABLE — for your project report")
print("="*65)

summary_rows = []
for thresh in [0.55, 0.60, 0.65]:
    mask   = preds["Pred_proba"] >= thresh
    trades = preds[mask].copy()
    trades["Net_return"] = trades["Fwd_ret_10d"] - TRANS_COST
    net = trades["Net_return"].dropna()
    gp  = net[net>0].sum(); gl = abs(net[net<0].sum())
    summary_rows.append({
        "Threshold":      thresh,
        "Trades":         len(net),
        "Win Rate %":     round((net>0).mean()*100, 1),
        "Profit Factor":  round(gp/gl, 3) if gl>0 else "∞",
        "Mean Ret/Trade %": round(net.mean()*100, 4),
        "Sharpe (ann.)":  round((net.mean()/net.std())*np.sqrt(252/10), 3)
                          if net.std()>0 else 0,
        "vs Signal-only PF": "0.795",
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

sig_mask  = ((preds["Signal_RSI_oversold"]==1) |
             (preds["Signal_MACD_cross"]==1)   |
             (preds["Signal_BB_lower"]==1))
sig_net   = (preds[sig_mask]["Fwd_ret_10d"] - TRANS_COST).dropna()
sig_gp    = sig_net[sig_net>0].sum()
sig_gl    = abs(sig_net[sig_net<0].sum())

print(f"\nBaseline (signal-only, no ML filter):")
print(f"  Trades: {len(sig_net):,} | Win Rate: {(sig_net>0).mean()*100:.1f}% | "
      f"Profit Factor: {sig_gp/sig_gl:.3f} | "
      f"Sharpe: {(sig_net.mean()/sig_net.std())*np.sqrt(252/10):.3f}")

print("\n" + "="*65)
print("SIGNAL INTERPRETER — most recent signals across all stocks")
print("="*65)

full_df = pd.read_parquet(
    os.path.join(PROCESSED_DIR, "all_stocks_features.parquet"))
full_df["Date"] = pd.to_datetime(full_df["Date"])

latest = (full_df.sort_values("Date")
                 .groupby("Symbol")
                 .tail(1)
                 .copy())

latest_clean = latest.dropna(subset=FEATURE_COLS).copy()

if latest_clean.empty:
    latest_clean = (full_df.dropna(subset=FEATURE_COLS)
                           .sort_values("Date")
                           .groupby("Symbol", group_keys=False)
                           .tail(1)
                           .copy())
    print("No symbols had complete features on the latest date; using each symbol's most recent complete-feature row.")

if latest_clean.empty:
    print("No complete feature rows found for signal interpretation.")
    latest_clean = latest.head(0).copy()
    latest_clean["ML_confidence"] = np.nan
else:
    X_latest = scaler.transform(latest_clean[FEATURE_COLS].values)
    latest_clean["ML_confidence"] = model.predict_proba(X_latest)[:, 1]

latest_clean["Active_signals"] = (
    latest_clean[["Signal_RSI_oversold",
                  "Signal_MACD_cross",
                  "Signal_BB_lower"]]
    .apply(lambda row: ", ".join([
        ("RSI-oversold"  if row["Signal_RSI_oversold"] else ""),
        ("MACD-cross"    if row["Signal_MACD_cross"]   else ""),
        ("BB-lower"      if row["Signal_BB_lower"]     else ""),
    ]).strip(", ").replace(", ,", ",").replace(",,", ",")
    if row.sum() > 0 else "none", axis=1)
)

high_conf = (latest_clean[latest_clean["ML_confidence"] >= 0.60]
             [["Symbol", "Date", "Close", "RSI_14",
               "Active_signals", "ML_confidence"]]
             .sort_values("ML_confidence", ascending=False)
             .head(20))

if len(high_conf) > 0:
    print(f"\nStocks with ML confidence ≥ 0.60 as of latest data:")
    print(f"{'Symbol':<8} {'Date':<12} {'Close':>8} {'RSI':>6} "
          f"{'Signals':<20} {'Confidence':>10}")
    print("-"*68)
    for _, row in high_conf.iterrows():
        print(f"  {row['Symbol']:<6} {str(row['Date'].date()):<12} "
              f"{row['Close']:>8.2f} {row['RSI_14']:>6.1f} "
              f"{row['Active_signals']:<20} {row['ML_confidence']:>10.3f}")
else:
    print("No stocks currently above 0.60 confidence threshold.")

print(f"\nConfidence distribution across all {len(latest_clean)} stocks:")
for low, high, label in [(0.0,0.45,"Low (<0.45)"),
                          (0.45,0.55,"Neutral (0.45–0.55)"),
                          (0.55,0.65,"Moderate (0.55–0.65)"),
                          (0.65,1.01,"High (>0.65)")]:
    count = ((latest_clean["ML_confidence"] >= low) &
             (latest_clean["ML_confidence"] < high)).sum()
    print(f"  {label:<25}: {count:>3} stocks")


summary_df.to_csv(os.path.join(REPORT_DIR, "summary_table.csv"), index=False)
latest_clean[["Symbol","Date","Close","RSI_14","MACD",
              "BB_pctB","Volume_ratio","ML_confidence",
              "Active_signals"]].to_csv(
    os.path.join(REPORT_DIR, "latest_signals.csv"), index=False)

print(f"\n Report figures saved to: {REPORT_DIR}")
print(f"   fig1_roc_curves.png")
print(f"   fig2_threshold_analysis.png")
print(f"   fig3_feature_importance.png")
print(f"   fig4_fold_performance.png")
print(f"   summary_table.csv")
print(f"   latest_signals.csv")
print(f"\n All steps complete. Your project pipeline is fully built.")