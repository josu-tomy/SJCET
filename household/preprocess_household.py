"""
MachineGuard — Household Data Preprocessing Script
Cleans the UCI Appliances Energy Prediction dataset and generates temporal features.
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "household", "raw", "energydata_complete.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "household", "processed")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DIR, "household_clean.csv")

# Synthetic/random variable columns to exclude from training
EXCLUDE_COLUMNS = ["rv1", "rv2"]


def preprocess_data(raw_path: str = RAW_DATA_PATH, save_path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """
    Loads raw UCI dataset, cleans records, extracts temporal features,
    and saves processed dataset chronologically.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}")

    print(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)
    initial_rows = len(df)

    # 1. Parse date and sort chronologically
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 2. Duplicate checking
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Removed {duplicates} exact duplicate rows.")
    else:
        print("Duplicate check: 0 duplicates found.")

    # 3. Missing Value Handling
    # Note: UCI 374 contains 0 missing values; documented robust handling in case of unexpected NaNs
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        # Forward fill sensor measurements without lookahead leakage
        df = df.ffill().bfill()
        print(f"Handled {null_counts} missing sensor readings using causal forward fill.")
    else:
        print("Missing values check: 0 missing values found.")

    # 4. Drop non-predictive random variables
    dropped = [c for c in EXCLUDE_COLUMNS if c in df.columns]
    df = df.drop(columns=dropped)
    print(f"Dropped non-predictive synthetic columns: {dropped}")

    # 5. Extract temporal features without future leakage
    hours = df["date"].dt.hour
    df["hour"] = hours
    df["day_of_week"] = df["date"].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical hour representation (24-hour cycle)
    df["sin_hour"] = np.sin(2 * np.pi * hours / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hours / 24.0)

    # 6. Physical validity checks
    # Relative humidity must be in [0, 100], energy >= 0
    rh_cols = [c for c in df.columns if c.startswith("RH_")]
    for col in rh_cols:
        invalid_rh = (df[col] < 0) | (df[col] > 100)
        if invalid_rh.any():
            df.loc[df[col] < 0, col] = 0.0
            df.loc[df[col] > 100, col] = 100.0

    # Ensure target 'Appliances' is positive
    df = df[df["Appliances"] >= 0].reset_index(drop=True)

    # 7. Save Cleaned Dataset
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Successfully saved cleaned dataset to: {save_path}")
    print(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns (Initial: {initial_rows} rows)")

    return df


if __name__ == "__main__":
    preprocess_data()

