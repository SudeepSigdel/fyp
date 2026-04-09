import pandas as pd
import numpy as np
import os

PROCESSED_DIR = "C:/Users/sudee/projects/Final Year Project/data/processed"

df = pd.read_parquet(os.path.join(PROCESSED_DIR, "all_stocks_clean.parquet"))
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

def per_stock(func, df):
    """Apply func to each stock's data separately, then recombine."""
    if "Symbol" not in df.columns:
        if isinstance(df.index, pd.MultiIndex) and "Symbol" in df.index.names:
            df = df.reset_index(level="Symbol")
        elif df.index.name == "Symbol":
            df = df.reset_index()
        else:
            raise KeyError("'Symbol' not found as column or index level")

    result = df.groupby("Symbol", group_keys=False).apply(
        lambda g: func(g).assign(Symbol=g.name)
    )

    if "Symbol" not in result.columns:
        if isinstance(result.index, pd.MultiIndex) and "Symbol" in result.index.names:
            result = result.reset_index(level="Symbol")
        elif result.index.name == "Symbol":
            result = result.reset_index()

    return result

def add_momentum_features(g):
    g["RSI_dist_50"]   = g["RSI_14"] - 50
    g["RSI_slope_3"]   = g["RSI_14"].diff(3)
    g["MACD_hist"]     = g["MACD"] - g["MACD_Signal"]
    g["MACD_hist_slope_3"] = g["MACD_hist"].diff(3)
    g["EMA_cross"]     = g["EMA_12"] - g["EMA_26"]
    g["Price_vs_SMA20"] = (g["Close"] - g["SMA_20"]) / g["SMA_20"] * 100
    return g

df = per_stock(add_momentum_features, df)
print("Momentum features added")

def add_volatility_features(g):
    band_range = g["BB_Upper"] - g["BB_Lower"]
    g["BB_pctB"] = (g["Close"] - g["BB_Lower"]) / band_range
    g["BB_width"] = band_range / g["BB_Middle"] * 100
    g["ATR_ratio"] = g["ATR_14"] / g["Close"] * 100
    g["Vol_10d"] = g["Log_Return"].rolling(10).std()
    return g

df = per_stock(add_volatility_features, df)
print("Volatility features added")

def add_volume_features(g):
    vol_mean_20 = g["Volume"].rolling(20).mean()
    g["Volume_ratio"] = g["Volume"] / vol_mean_20
    g["Volume_spike"] = (g["Volume_ratio"] > 2.0).astype(int)
    g["OBV_slope_5"] = g["OBV"].diff(5)
    obv_std = g["OBV"].rolling(20).std()
    g["OBV_slope_norm"] = g["OBV_slope_5"] / (obv_std + 1e-9)
    return g

df = per_stock(add_volume_features, df)
print("Volume features added")

def add_return_features(g):
    g["Ret_1d"]  = g["Log_Return"]
    g["Ret_3d"]  = g["Log_Return"].rolling(3).sum()
    g["Ret_5d"]  = g["Log_Return"].rolling(5).sum()
    g["Ret_10d"] = g["Log_Return"].rolling(10).sum()
    g["Ret_20d"] = g["Log_Return"].rolling(20).sum()
    g["Ret_momentum"] = g["Ret_3d"] - g["Ret_10d"] / 2
    return g

df = per_stock(add_return_features, df)
print("Return features added")

def add_context_features(g):
    g["In_uptrend"] = (g["EMA_12"] > g["EMA_26"]).astype(int)
    g["RSI_oversold"]  = (g["RSI_14"] < 30).astype(int)
    g["RSI_overbought"] = (g["RSI_14"] > 70).astype(int)
    g["HL_range_pct"] = (g["High"] - g["Low"]) / g["Close"] * 100
    g["Gap_pct"] = (g["Open"] - g["Close"].shift(1)) / g["Close"].shift(1) * 100

    return g

df = per_stock(add_context_features, df)
print("Context features added")

new_features = [
    "RSI_dist_50", "RSI_slope_3", "MACD_hist", "MACD_hist_slope_3",
    "EMA_cross", "Price_vs_SMA20",
    "BB_pctB", "BB_width", "ATR_ratio", "Vol_10d",
    "Volume_ratio", "Volume_spike", "OBV_slope_5", "OBV_slope_norm",
    "Ret_1d", "Ret_3d", "Ret_5d", "Ret_10d", "Ret_20d", "Ret_momentum",
    "In_uptrend", "RSI_oversold", "RSI_overbought", "HL_range_pct", "Gap_pct"
]

print(f"\nNew features created: {len(new_features)}")
print(f"   Total columns now: {df.shape[1]}")
print(f"   Total rows: {df.shape[0]:,}")

print("\n" + "="*60)
print("FEATURE QUALITY CHECK")
print("="*60)
print(f"{'Feature':<22} {'NaN%':>6}  {'Min':>10}  {'Mean':>10}  {'Max':>10}")
print("-"*60)

for feat in new_features:
    nan_pct = df[feat].isna().mean() * 100
    if df[feat].dtype in [float, int] or np.issubdtype(df[feat].dtype, np.number):
        fmin  = df[feat].min()
        fmean = df[feat].mean()
        fmax  = df[feat].max()
        print(f"{feat:<22} {nan_pct:>5.1f}%  {fmin:>10.3f}  {fmean:>10.3f}  {fmax:>10.3f}")

output_path = os.path.join(PROCESSED_DIR, "all_stocks_features.parquet")
df.to_parquet(output_path, index=False)
print(f"\nSaved → {output_path}")
print("Feature engineering complete! Next: label construction.")