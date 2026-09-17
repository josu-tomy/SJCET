"""
MachineGuard — Household Data Inspection Script
Inspects the raw UCI Appliances Energy Prediction dataset (UCI ID: 374).
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "household", "raw", "energydata_complete.csv")


def inspect():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw household dataset not found at {RAW_DATA_PATH}")

    print("=" * 70)
    print("MachineGuard — UCI Household Appliances Energy Dataset Inspection")
    print("=" * 70)

    df = pd.read_csv(RAW_DATA_PATH)

    # 1. Basic Shape and Memory
    n_rows, n_cols = df.shape
    print(f"\n1. DATASET DIMENSIONS")
    print(f"   - Rows:    {n_rows:,}")
    print(f"   - Columns: {n_cols}")

    # 2. Date/Time Coverage & Sampling Interval
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    time_min = df["date"].min()
    time_max = df["date"].max()
    duration = time_max - time_min
    time_diffs = df["date"].diff().dropna()
    median_interval = time_diffs.median()

    print(f"\n2. DATE / TIME COVERAGE & RESOLUTION")
    print(f"   - Start timestamp:  {time_min}")
    print(f"   - End timestamp:    {time_max}")
    print(f"   - Total duration:   {duration.days} days ({duration})")
    print(f"   - Sampling interval: Median = {median_interval} ({median_interval.total_seconds() / 60:.0f} minutes)")

    # 3. Missing Values & Duplicate Rows
    missing = df.isnull().sum()
    total_missing = missing.sum()
    duplicate_rows = df.duplicated().sum()
    print(f"\n3. INTEGRITY & DATA QUALITY")
    print(f"   - Total missing values: {total_missing}")
    print(f"   - Duplicate rows:       {duplicate_rows}")

    # 4. Column Types & Categorization
    print(f"\n4. COLUMNS & VARIABLE CATEGORIES")
    zone_temps = [c for c in df.columns if c.startswith("T") and c not in ["T_out", "Tdewpoint"]]
    zone_rhs = [c for c in df.columns if c.startswith("RH_") and c != "RH_out"]
    weather_cols = ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]
    energy_cols = ["Appliances", "lights"]
    random_cols = ["rv1", "rv2"]

    print(f"   - Energy Target/Usage : {energy_cols}")
    print(f"   - Indoor Temperatures : {zone_temps} (9 sensors, °C)")
    print(f"   - Indoor Humidity     : {zone_rhs} (9 sensors, %)")
    print(f"   - Weather Variables   : {weather_cols}")
    print(f"   - Synthetic/Random    : {random_cols} (Random variables from original benchmark study)")

    # 5. Target Variable Distribution: 'Appliances' (Wh)
    app = df["Appliances"]
    print(f"\n5. TARGET DISTRIBUTION: 'Appliances' [Wh]")
    print(f"   - Min:     {app.min():.1f} Wh")
    print(f"   - 25%:     {app.quantile(0.25):.1f} Wh")
    print(f"   - Median:  {app.median():.1f} Wh")
    print(f"   - Mean:    {app.mean():.2f} Wh")
    print(f"   - 75%:     {app.quantile(0.75):.1f} Wh")
    print(f"   - 95%:     {app.quantile(0.95):.1f} Wh")
    print(f"   - Max:     {app.max():.1f} Wh")
    print(f"   - Std Dev: {app.std():.2f} Wh")

    # 6. Target Correlations
    numeric_df = df.select_dtypes(include=[np.number])
    corr_with_target = numeric_df.corr()["Appliances"].sort_values(ascending=False)

    print(f"\n6. CORRELATION WITH TARGET ('Appliances'):")
    print("   Top positive correlations:")
    for col, val in corr_with_target.head(6).items():
        if col != "Appliances":
            print(f"     - {col:15s}: {val:+.4f}")
    print("   Top negative correlations:")
    for col, val in corr_with_target.tail(5).items():
        print(f"     - {col:15s}: {val:+.4f}")

    print("\n   Random Variables check (should be near 0 correlation):")
    for r_col in ["rv1", "rv2"]:
        if r_col in corr_with_target:
            print(f"     - {r_col:15s}: {corr_with_target[r_col]:+.5f} (Confirm non-predictive; exclude from features)")

    # 7. Summary & Recommendations
    print("\n7. PREPROCESSING RECOMMENDATIONS:")
    print("   1) Parse 'date' to datetime and sort chronologically.")
    print("   2) Extract temporal features: hour, day of week, month, is_weekend, sin_hour, cos_hour.")
    print("   3) Target is 'Appliances' (energy consumption in Wh).")
    print("   4) Exclude non-predictive columns 'rv1', 'rv2', and 'date' from regression feature matrix.")
    print("   5) Keep indoor zone conditions (T1-T9, RH_1-RH_9) and outdoor weather features.")
    print("=" * 70)


if __name__ == "__main__":
    inspect()

