# ============================================================
# 04_label_construction.py
# PURPOSE: Create the target labels (Y) that the ML model learns to predict
#
# KEY CONCEPT: For each row at time T, we define:
#   - Label_5d:  Did this trade earn >1% net over the next 5 days?
#   - Label_10d: Did this trade earn >1% net over the next 10 days?
#
# The ~1% threshold represents NEPSE transaction costs (round trip).
# A trade must clear this hurdle to be genuinely profitable.
# ============================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

PROCESSED_DIR = "C:/Users/sudee/projects/Final Year Project/data/processed"

TRANSACTION_COST = 0.01   

df = pd.read_parquet(os.path.join(PROCESSED_DIR, "all_stocks_features.parquet"))
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")


def per_stock(func, df):
    """Apply func per symbol and keep Symbol available as a normal column."""
    if "Symbol" not in df.columns:
        if isinstance(df.index, pd.MultiIndex) and "Symbol" in df.index.names:
            df = df.reset_index(level="Symbol")
        elif df.index.name == "Symbol":
            df = df.reset_index()
        else:
            raise KeyError("'Symbol' not found as column or index level")

    out = df.groupby("Symbol", group_keys=False).apply(
        lambda g: func(g).assign(Symbol=g.name)
    )

    if "Symbol" not in out.columns:
        if isinstance(out.index, pd.MultiIndex) and "Symbol" in out.index.names:
            out = out.reset_index(level="Symbol")
        elif out.index.name == "Symbol":
            out = out.reset_index()

    return out


def compute_forward_returns(g, horizons=[5, 10]):
    g = g.sort_values("Date").copy()
    for h in horizons:
        future_price = g["Close"].shift(-h)
        g[f"Fwd_ret_{h}d"] = (future_price - g["Close"]) / g["Close"]
    return g

df = per_stock(lambda g: compute_forward_returns(g, horizons=[5, 10]), df)
print(" Forward returns computed")


df["Label_5d"]  = (df["Fwd_ret_5d"]  > TRANSACTION_COST).astype(int)
df["Label_10d"] = (df["Fwd_ret_10d"] > TRANSACTION_COST).astype(int)

print(" Binary labels created")



print("\n" + "="*55)
print("LABEL DISTRIBUTION")
print("="*55)

for label in ["Label_5d", "Label_10d"]:
    count_1 = df[label].sum()
    count_0 = (df[label] == 0).sum()
    total   = count_1 + count_0
    pct_1   = count_1 / total * 100
    pct_0   = count_0 / total * 100
    print(f"\n{label}:")
    print(f"  Label=1 (trade succeeded): {count_1:>7,}  ({pct_1:.1f}%)")
    print(f"  Label=0 (trade failed):    {count_0:>7,}  ({pct_0:.1f}%)")
    print(f"  Imbalance ratio: {max(pct_0,pct_1)/min(pct_0,pct_1):.2f}:1")


def add_signals(g):
    g = g.sort_values("Date").copy()

    rsi_prev = g["RSI_14"].shift(1)
    g["Signal_RSI_oversold"] = (
        (rsi_prev < 30) & (g["RSI_14"] >= 30)
    ).astype(int)

    macd_prev   = g["MACD"].shift(1)
    signal_prev = g["MACD_Signal"].shift(1)
    g["Signal_MACD_cross"] = (
        (macd_prev < signal_prev) & (g["MACD"] >= g["MACD_Signal"])
    ).astype(int)

    g["Signal_BB_lower"] = (
        g["Close"] <= g["BB_Lower"]
    ).astype(int)

    return g

df = per_stock(add_signals, df)
print("\n Candidate signals defined")


print("\n" + "="*55)
print("SIGNAL FREQUENCY (across all stocks, all years)")
print("="*55)

signals = ["Signal_RSI_oversold", "Signal_MACD_cross", "Signal_BB_lower"]

for sig in signals:
    fires      = df[sig].sum()
    total_rows = len(df)
    pct        = fires / total_rows * 100
    print(f"\n{sig}:")
    print(f"  Total fires: {fires:,}  ({pct:.2f}% of all trading days)")

    signal_rows = df[df[sig] == 1]
    for label in ["Label_5d", "Label_10d"]:
        if label in df.columns:
            valid = signal_rows[label].dropna()
            if len(valid) > 0:
                win_rate = valid.mean() * 100
                print(f"  Win rate ({label}): {win_rate:.1f}%  "
                      f"(out of {len(valid):,} signal instances)")


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for i, (ret_col, label_col, horizon) in enumerate([
    ("Fwd_ret_5d", "Label_5d", "5-day"),
    ("Fwd_ret_10d", "Label_10d", "10-day")
]):
    ax = axes[i]
    data = df[ret_col].dropna()

    ax.hist(data.clip(-0.3, 0.3), bins=80, color="steelblue",
            alpha=0.7, edgecolor="none")
    ax.axvline(TRANSACTION_COST, color="red", linestyle="--",
               linewidth=1.5, label=f"Cost threshold ({TRANSACTION_COST*100:.0f}%)")
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)

    pct_above = (data > TRANSACTION_COST).mean() * 100
    ax.set_title(f"{horizon} forward return distribution\n"
                 f"Label=1: {pct_above:.1f}% of rows")
    ax.set_xlabel("Forward return (clipped at ±30%)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "label_distribution.png"), dpi=150)
plt.show()
print(" Distribution chart saved")

rows_before = len(df)
df_labeled = df.dropna(subset=["Label_5d", "Label_10d"]).copy()
rows_dropped = rows_before - len(df_labeled)

print(f"\n Dropped {rows_dropped:,} rows with no forward label "
      f"(last 10 trading days of each stock)")
print(f"   Labeled dataset: {len(df_labeled):,} rows")

df.to_parquet(os.path.join(PROCESSED_DIR, "all_stocks_features.parquet"), index=False)

df_labeled.to_parquet(
    os.path.join(PROCESSED_DIR, "all_stocks_labeled.parquet"), index=False
)

print(f"\n Saved full dataset  → all_stocks_features.parquet")
print(f" Saved labeled dataset → all_stocks_labeled.parquet")
print(f"\nLabel construction complete! Next: walk-forward validation setup.")