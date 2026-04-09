import pandas as pd
import numpy as np
import os

PROCESSED_DIR = "C:/Users/sudee/projects/Final Year Project/data/processed"

df = pd.read_parquet(os.path.join(PROCESSED_DIR, "all_stocks_combined.parquet"))
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

df["Volume"] = df.groupby("Symbol")["Volume"].ffill()

df["Daily_Return"] = df["Daily_Return"].fillna(0)
df["Log_Return"] = df["Log_Return"].fillna(0)

print("Fixed missing Volume, Daily_Return, Log_Return")

before = len(df)
df = df[df["Symbol"] != "SANVI"].copy()
after = len(df)
print(f"Removed SANVI: {before - after} rows dropped")
print(f"Remaining stocks: {df['Symbol'].nunique()}")

bnl_info = df[df["Symbol"] == "BNL"]
print(f"\n BNL note: {len(bnl_info)} rows from")
f"{bnl_info['Date'].min().date()} to {bnl_info['Date'].max().date()}"
print(f"Average trading days per year: "
      f"{len(bnl_info)/13:.0f} (very low- normal NEPSE stocks trade ~250/year)")


print("\n" + "=" * 50)
print("CHECKING FOR MID-SERIES NANS (should all be 0)")
print("="*50)

problem_cols = ["Close", "Open", "High", "Low", "Volume",
                "RSI_14", "MACD", "BB_Upper"]

for symbol, group in df.groupby("Symbol"):
    group_sorted = group.sort_values("Date")
    for col in problem_cols:
        series = group_sorted[col].values
        first_valid = pd.Series(series).first_valid_index()
        if first_valid is None:
            continue
        mid_nans = pd.Series(series[first_valid:]).isnull().sum()
        if mid_nans > 0:
            print(f"  {symbol} | {col}: {mid_nans} NaN(s) after first valid row")

print("Check complete")

output_path = os.path.join(PROCESSED_DIR, "all_stocks_clean.parquet")
df.to_parquet(output_path, index=False)

print(f"\nSaved clean data → {output_path}")
print(f"   Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

print("\n" + "="*50)
print("REMAINING MISSING VALUES (should only be warm-up NaNs)")
print("="*50)
missing = df.isnull().sum()
print(missing[missing > 0])
print("\nData is ready for feature engineering!")