import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR  = "C:/Users/sudee/projects/Final Year Project/data/processed"
OUTPUTS_DIR    = "C:/Users/sudee/projects/Final Year Project/outputs"
TRANS_COST     = 0.01    # 1% round-trip NEPSE transaction cost
PROB_THRESHOLD = 0.55    # Only trade when model is at least 55% confident
                         # This filters out low-confidence predictions
HOLD_DAYS      = 10      # Days we hold each position

os.makedirs(OUTPUTS_DIR, exist_ok=True)

preds = pd.read_parquet(os.path.join(PROCESSED_DIR, "oos_predictions.parquet"))
preds["Date"] = pd.to_datetime(preds["Date"])
preds = preds.sort_values(["Symbol", "Date"]).reset_index(drop=True)

print(f"Loaded {len(preds):,} out-of-sample predictions")
print(f"Date range: {preds['Date'].min().date()} → {preds['Date'].max().date()}")
print(f"Probability threshold: {PROB_THRESHOLD}")


def simulate_strategy(df, entry_mask, label="strategy"):
    """
    Simulate a strategy given a boolean mask of entry points.
    Returns a DataFrame of individual trades with their outcomes.

    Parameters:
        df         : the predictions DataFrame
        entry_mask : boolean Series — True on rows where we enter a trade
        label      : name for this strategy
    """
    trades = df[entry_mask].copy()

    trades["Gross_return"] = trades["Fwd_ret_10d"]

    trades["Net_return"] = trades["Gross_return"] - TRANS_COST

    trades["Win"] = (trades["Net_return"] > 0).astype(int)

    trades["Strategy"] = label
    return trades


ml_mask     = preds["Pred_proba"] >= PROB_THRESHOLD
ml_trades   = simulate_strategy(preds, ml_mask, "ML-validated")

signal_mask  = (
    (preds["Signal_RSI_oversold"] == 1) |
    (preds["Signal_MACD_cross"]   == 1) |
    (preds["Signal_BB_lower"]     == 1)
)
sig_trades   = simulate_strategy(preds, signal_mask, "Signal-only")

always_mask  = pd.Series(True, index=preds.index)
always_trades = simulate_strategy(preds, always_mask, "Always-in")


def calc_metrics(trades, strategy_name):
    """
    Calculate all performance metrics for a set of trades.
    Returns a dict of metrics and prints a summary.
    """
    if len(trades) == 0:
        print(f"{strategy_name}: No trades generated")
        return {}

    net_rets = trades["Net_return"].dropna()
    wins     = trades["Win"].dropna()

    win_rate = wins.mean() * 100

    gross_profit = net_rets[net_rets > 0].sum()
    gross_loss   = abs(net_rets[net_rets < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    total_return = net_rets.sum() * 100

    mean_return = net_rets.mean() * 100
    if net_rets.std() > 0:
        annualise    = np.sqrt(252 / HOLD_DAYS)
        sharpe       = (net_rets.mean() / net_rets.std()) * annualise
    else:
        sharpe = 0.0

    trades_sorted = trades.sort_values("Date")
    cum_ret       = (1 + trades_sorted["Net_return"].fillna(0)).cumprod()
    rolling_max   = cum_ret.expanding().max()
    drawdown      = (cum_ret - rolling_max) / rolling_max
    max_drawdown  = drawdown.min() * 100

    return {
        "Strategy":      strategy_name,
        "Trades":        len(net_rets),
        "Win Rate %":    round(win_rate, 2),
        "Profit Factor": round(profit_factor, 3),
        "Mean Ret %":    round(mean_return, 4),
        "Total Ret %":   round(total_return, 2),
        "Sharpe":        round(sharpe, 3),
        "Max DD %":      round(max_drawdown, 2),
    }


print("\n" + "="*70)
print("STRATEGY PERFORMANCE COMPARISON")
print("="*70)

results = []
for trades, name in [
    (ml_trades,     "ML-validated"),
    (sig_trades,    "Signal-only"),
    (always_trades, "Always-in"),
]:
    m = calc_metrics(trades, name)
    results.append(m)

metrics_df = pd.DataFrame(results).set_index("Strategy")
print(metrics_df.to_string())

print("\nMetric guide:")
print("  Win Rate  : % trades profitable after costs (>50% is good)")
print("  Prof Factor: gross profit / gross loss (>1.0 = overall profitable)")
print("  Mean Ret  : average return per trade after costs")
print("  Total Ret : sum of all trade returns (not compounded)")
print("  Sharpe    : risk-adjusted return, annualised (>0.5 acceptable)")
print("  Max DD    : worst peak-to-trough loss (smaller magnitude = better)")


print("\n" + "="*70)
print("ML-VALIDATED STRATEGY: PERFORMANCE BY FOLD")
print("="*70)
print(f"{'Fold':<6} {'Trades':>7} {'Win%':>7} {'ProfFact':>9} "
      f"{'MeanRet%':>9} {'Sharpe':>8}")
print("-"*50)

for fold_num in sorted(ml_trades["Fold"].unique()):
    fold_t = ml_trades[ml_trades["Fold"] == fold_num]
    m      = calc_metrics(fold_t, f"Fold {fold_num}")
    if m:
        print(f"  {fold_num:<4} {m['Trades']:>7,} {m['Win Rate %']:>7.1f} "
              f"{m['Profit Factor']:>9.3f} {m['Mean Ret %']:>9.4f} "
              f"{m['Sharpe']:>8.3f}")

print("\n" + "="*70)
print("THRESHOLD SENSITIVITY (ML-validated strategy)")
print("="*70)
print(f"{'Threshold':>10} {'Trades':>8} {'Win%':>7} {'ProfFact':>10} {'Sharpe':>8}")
print("-"*50)

for thresh in [0.50, 0.52, 0.55, 0.58, 0.60, 0.63, 0.65]:
    mask   = preds["Pred_proba"] >= thresh
    trades = simulate_strategy(preds, mask, f"thresh_{thresh}")
    if len(trades) > 50:
        m = calc_metrics(trades, str(thresh))
        print(f"  {thresh:>9.2f} {m['Trades']:>8,} {m['Win Rate %']:>7.1f} "
              f"{m['Profit Factor']:>10.3f} {m['Sharpe']:>8.3f}")
    else:
        print(f"  {thresh:>9.2f}  < 50 trades — threshold too high")


fig, axes = plt.subplots(2, 1, figsize=(13, 9))

ax1 = axes[0]
for trades, name, color in [
    (ml_trades,     "ML-validated", "steelblue"),
    (sig_trades,    "Signal-only",  "orange"),
    (always_trades, "Always-in",    "gray"),
]:
    if len(trades) == 0:
        continue
    t = trades.sort_values("Date").copy()
    t["Cum_return"] = (1 + t["Net_return"].fillna(0)).cumprod() - 1
    ax1.plot(t["Date"], t["Cum_return"] * 100,
             label=name, color=color, linewidth=1.5, alpha=0.85)

ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax1.set_title("Cumulative net return by strategy")
ax1.set_ylabel("Cumulative return (%)")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.hist(preds["Pred_proba"], bins=60, color="steelblue",
         alpha=0.7, edgecolor="none", label="All predictions")
ax2.axvline(PROB_THRESHOLD, color="red", linestyle="--",
            linewidth=2, label=f"Threshold = {PROB_THRESHOLD}")
ax2.set_title("Distribution of predicted probabilities")
ax2.set_xlabel("Predicted probability (P = signal succeeds)")
ax2.set_ylabel("Count")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "backtest_results.png"), dpi=150)
plt.show()

metrics_df.to_csv(os.path.join(OUTPUTS_DIR, "strategy_metrics.csv"))
ml_trades.to_parquet(os.path.join(PROCESSED_DIR, "ml_trades.parquet"), index=False)

print(f"\n Saved strategy metrics → strategy_metrics.csv")
print(f" Saved ML trades        → ml_trades.parquet")
print(f"\n Backtesting complete! Next: reporting layer.")