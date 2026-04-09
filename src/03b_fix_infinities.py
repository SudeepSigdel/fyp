import pandas as pd
import numpy as np
import os

PROCESSED_DIR = "C:/Users/sudee/projects/Final Year Project/data/processed"

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



price_cols = ["Open", "High", "Low", "Close"]

zero_mask = (df[price_cols] <= 0).any(axis=1)
print(f"\n Rows with zero or negative prices: {zero_mask.sum()}")

if zero_mask.sum() > 0:
    affected = df[zero_mask].groupby("Symbol").size()
    print("   Affected stocks:")
    print(affected.to_string())

df = df[~zero_mask].copy()
print(f" Removed {zero_mask.sum()} bad price rows")
print(f" Remaining rows: {df.shape[0]:,}")


def recalc_log_return(g):
    g = g.sort_values("Date").copy()
    g["Log_Return"] = np.log(g["Close"] / g["Close"].shift(1))
    g["Daily_Return"] = g["Close"].pct_change()
    return g

df = per_stock(recalc_log_return, df)

inf_count = np.isinf(df["Log_Return"]).sum()
print(f"\n Log_Return infinities after fix: {inf_count}")



def recalc_features(g):
    g = g.sort_values("Date").copy()

    close_safe = g["Close"].replace(0, np.nan)
    g["ATR_ratio"] = g["ATR_14"] / close_safe * 100

    prev_close = g["Close"].shift(1).replace(0, np.nan)
    g["Gap_pct"] = (g["Open"] - prev_close) / prev_close * 100

    g["Ret_1d"]  = g["Log_Return"]
    g["Ret_3d"]  = g["Log_Return"].rolling(3).sum()
    g["Ret_5d"]  = g["Log_Return"].rolling(5).sum()
    g["Ret_10d"] = g["Log_Return"].rolling(10).sum()
    g["Ret_20d"] = g["Log_Return"].rolling(20).sum()
    g["Ret_momentum"] = g["Ret_3d"] - g["Ret_10d"] / 2

    return g

df = per_stock(recalc_features, df)
print(" Recalculated ATR_ratio, Gap_pct, and return features")

features_to_clip = [
    "MACD_hist", "MACD_hist_slope_3", "EMA_cross",
    "OBV_slope_5", "OBV_slope_norm",
    "ATR_ratio", "Gap_pct", "Volume_ratio",
    "HL_range_pct", "Price_vs_SMA20", "BB_width",
    "Ret_3d", "Ret_5d", "Ret_10d", "Ret_20d"
]

print("\n Clipping outliers at 1st/99th percentile:")
for feat in features_to_clip:
    if feat not in df.columns:
        continue
    p01 = df[feat].quantile(0.01)
    p99 = df[feat].quantile(0.99)
    before_inf = np.isinf(df[feat]).sum()
    df[feat] = df[feat].clip(lower=p01, upper=p99)
    after_inf = np.isinf(df[feat]).sum()
    print(f"   {feat:<25} clipped to [{p01:.3f}, {p99:.3f}]  "
          f"(inf before: {before_inf}, after: {after_inf})")



print("\n" + "="*60)
print("FINAL CHECK — inf and NaN counts")
print("="*60)

feature_cols = [
    "RSI_dist_50", "RSI_slope_3", "MACD_hist", "MACD_hist_slope_3",
    "EMA_cross", "Price_vs_SMA20", "BB_pctB", "BB_width",
    "ATR_ratio", "Vol_10d", "Volume_ratio", "Volume_spike",
    "OBV_slope_5", "OBV_slope_norm",
    "Ret_1d", "Ret_3d", "Ret_5d", "Ret_10d", "Ret_20d", "Ret_momentum",
    "In_uptrend", "RSI_oversold", "RSI_overbought",
    "HL_range_pct", "Gap_pct"
]

print(f"{'Feature':<25} {'NaN%':>6}  {'Inf count':>10}  {'Min':>10}  {'Max':>10}")
print("-"*65)

all_clean = True
for feat in feature_cols:
    nan_pct   = df[feat].isna().mean() * 100
    inf_count = np.isinf(df[feat]).sum()
    fmin      = df[feat].replace([np.inf, -np.inf], np.nan).min()
    fmax      = df[feat].replace([np.inf, -np.inf], np.nan).max()

    flag = " !!!" if inf_count > 0 or nan_pct > 5 else ""
    print(f"{feat:<25} {nan_pct:>5.1f}%  {inf_count:>10}  "
          f"{fmin:>10.3f}  {fmax:>10.3f}{flag}")
    if inf_count > 0:
        all_clean = False

if all_clean:
    print("\n No infinities remaining. Data is clean.")
else:
    print("\n  Some infinities still present — share output for investigation.")


output_path = os.path.join(PROCESSED_DIR, "all_stocks_features.parquet")
df.to_parquet(output_path, index=False)

print(f"\n Saved clean features → {output_path}")
print(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("\nReady for label construction!")