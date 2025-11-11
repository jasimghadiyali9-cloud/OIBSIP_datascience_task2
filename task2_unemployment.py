import os, glob
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROLL_WINDOW = 5
CLIP_Q = 0.95
path = kagglehub.dataset_download("gokulrajkmv/unemployment-in-india")
print("Downloaded to:", path)
csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
if not csvs:
    raise FileNotFoundError("No CSV found")
csv_path = csvs[0]
print("Using CSV:", csv_path)
df = pd.read_csv(csv_path)
df.columns = [c.strip().lower() for c in df.columns]
def pick(names):
    for n in names:
        if n in df.columns:
            return n
    return None
date_col = pick(["date","month","period","time"])
region_col = pick(["region","state","area","district","place"])
rate_col = None
for c in df.columns:
    if "unemployment" in c and "%" in c:
        rate_col = c
        break
if rate_col is None:
    for c in df.columns:
        if "unemployment" in c or "rate" in c:
            rate_col = c
            break
if not date_col or not rate_col:
    raise ValueError("Columns not detected")
df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce")
df = df.dropna(subset=[date_col, rate_col]).sort_values(date_col)
df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
monthly = df.groupby("month", as_index=False)[rate_col].mean()
monthly.rename(columns={rate_col: "rate"}, inplace=True)
upper = monthly["rate"].quantile(CLIP_Q)
monthly["rate"] = monthly["rate"].clip(0, upper)
monthly["rate_roll"] = monthly["rate"].rolling(ROLL_WINDOW, min_periods=1).mean()
plt.figure(figsize=(12,6))
plt.plot(monthly["month"], monthly["rate"], linewidth=2, marker="o", alpha=0.7)
plt.plot(monthly["month"], monthly["rate_roll"], linewidth=3)
plt.title("Unemployment Rate Over Time (India)")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
if region_col:
    latest_month = monthly["month"].max()
    snap = (df[df["month"] == latest_month]
            .groupby(region_col)[rate_col]
            .mean()
            .sort_values(ascending=False)
            .head(10))
    if not snap.empty:
        print(f"\nTop Regions — {latest_month.date()}\n", snap)
        plt.figure(figsize=(10,6))
        plt.barh(snap.index, snap.values)
        plt.gca().invert_yaxis()
        plt.title(f"Top Regions — {latest_month.date()}")
        plt.xlabel("Unemployment Rate (%)")
        plt.tight_layout()
        plt.show()
peak = monthly.loc[monthly["rate"].idxmax()]
low  = monthly.loc[monthly["rate"].idxmin()]
avg  = monthly["rate"].mean()
print(f"\nHighest: {peak['rate']:.2f}% on {peak['month'].date()}")
print(f"Lowest: {low['rate']:.2f}% on {low['month'].date()}")
print(f"Average: {avg:.2f}%")
