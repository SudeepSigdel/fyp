import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

RAW_DATA_DIR = "C:/Users/sudee/projects/Final Year Project/data/raw"
OUTPUT_DIR= "C:/Users/sudee/projects/Final Year Project/data/processed"
os.makedirs(OUTPUT_DIR, exist_ok= True)

all_dataframes = []

for filename in sorted(os.listdir(RAW_DATA_DIR)):
    if not filename.endswith(".csv"):
        continue
    
    filepath = os.path.join(RAW_DATA_DIR, filename)

    df = pd.read_csv(filepath)

    all_dataframes.append(df)

combined = pd.concat(all_dataframes, ignore_index= True)

print(f"Loaded {len(all_dataframes)} files")

print(f"Total rows: {len(combined):,}")

print(f"Columns:{list(combined.columns)}\n")


combined["Date"] = pd.to_datetime(combined["Date"])

combined = combined.sort_values(["Symbol", "Date"]).reset_index(drop= True)

print(f"Date range: {combined['Date'].min().date()} -> {combined['Date'].max().date()}\n")

print("=" * 50)
print("MISSING VALUES PER COLUMN")
print("=" * 50)

missing = combined.isnull().sum()

missing = combined.isnull().sum()
missing_pct= (missing/len(combined) * 100).round(2)

missing_report = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_pct
})

print(missing_report[missing_report["Missing Count"] > 0])

print()

duplicates = combined.duplicated(subset=["Symbol", "Date"]).sum()
print(f"Duplicate (Symbol, Date) pairs: {duplicates}")

if duplicates > 0:
    print("Duplicates found! Keeping the first occurence.")
    combined = combined.drop_duplicates(subset=["Symbol", "Date"], keep="first")

print()

print("=" * 50)
print("ROW COUNT AND DATE RANGE PER STOCK")
print("=" * 50)

stock_summary = combined.groupby("Symbol").agg(
    Rows=("Date", "count"),
    Start=("Date", "min"),
    End=("Date", "max")
).reset_index()

stock_summary["Start"] = stock_summary["Start"].dt.date
stock_summary["End"] = stock_summary["End"].dt.date

print(stock_summary.to_string(index=False))
print()

median_rows = stock_summary["Rows"].median()
thin_stocks = stock_summary[stock_summary["Rows"] < median_rows * 0.5]

if len(thin_stocks) > 0:
    print(f"These stocks have less than half the median row count ({int(median_rows)}):")
    print(thin_stocks.to_string(index=False))
else:
    print(f"All stocks have reasonable row counts (median: {int(median_rows)} rows)")
print()

output_path = os.path.join(OUTPUT_DIR, "all_stocks_combined.parquet")
combined.to_parquet(output_path, index=False)
print(f"Saved combined data to: {output_path}")
print(f"   Shape: {combined.shape[0]:,} rows × {combined.shape[1]} columns")

plt.figure(figsize=(18, 5))
plt.bar(stock_summary["Symbol"], stock_summary["Rows"], color="steelblue")
plt.axhline(median_rows, color="red", linestyle="--", label=f"Median ({int(median_rows)})")
plt.xticks(rotation=90, fontsize=7)
plt.ylabel("Number of Trading Days")
plt.title("Data Coverage per Stock")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join("C:/Users/sudee/projects/Final Year Project/outputs", "data_coverage.png"), dpi=150)
plt.show()
print("Chart saved to outputs/data_coverage.png")