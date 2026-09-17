"""
Data Acquisition & Preprocessing Pipeline for MachineGuard
UCI AI4I 2020 Predictive Maintenance Dataset

Person 1: Data & Preprocessing Engineer
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "ai4i2020.csv")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")

# Feature selection specification
# EXCLUDED: 'UDI', 'Product ID', 'Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
SELECTED_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET_COLUMN = "Machine failure"


def load_dataset(csv_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw dataset from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    return pd.read_csv(csv_path)


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Inspect dataset properties and return summary statistics."""
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "target_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
        "target_distribution_pct": df[TARGET_COLUMN].value_counts(normalize=True).to_dict(),
    }
    return info


def preprocess_and_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Select the 5 mandatory features, separate X and y, and perform
    an 80/20 stratified train/test split.
    """
    X = df[SELECTED_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Combine back into train and test DataFrames for clean persistence
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    return X_train, X_test, y_train, y_test, train_df, test_df


def run():
    print("=" * 60)
    print("MachineGuard — Data Acquisition & Preprocessing Pipeline")
    print("=" * 60)

    # 1. Load data
    df = load_dataset(RAW_DATA_PATH)
    print(f"\n1. Loaded raw dataset: {RAW_DATA_PATH}")
    print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Inspect data
    inspection = inspect_dataset(df)
    print("\n2. Dataset Columns & Data Types:")
    for col, dtype in inspection["dtypes"].items():
        print(f"   - {col:30s}: {dtype}")

    total_missing = sum(inspection["missing_values"].values())
    print(f"\n3. Missing values check: {total_missing} total missing values across all columns.")

    print(f"\n4. Target ({TARGET_COLUMN}) Distribution:")
    for cls, count in inspection["target_distribution"].items():
        pct = inspection["target_distribution_pct"][cls] * 100
        print(f"   Class {cls}: {count} ({pct:.2f}%)")

    # 3. Preprocess and split
    print(f"\n5. Selecting 5 specified features:")
    for feat in SELECTED_FEATURES:
        print(f"   - {feat}")

    X_train, X_test, y_train, y_test, train_df, test_df = preprocess_and_split(df)

    print(f"\n6. Stratified Train/Test Split (80/20, random_state=42):")
    print(f"   Training samples: {len(train_df)}")
    print(f"     Class 0: {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.2f}%)")
    print(f"     Class 1: {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.2f}%)")
    print(f"   Testing samples:  {len(test_df)}")
    print(f"     Class 0: {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.2f}%)")
    print(f"     Class 1: {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.2f}%)")

    # 4. Save processed splits
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"\n7. Saved split datasets:")
    print(f"   - Train: {TRAIN_DATA_PATH} ({train_df.shape})")
    print(f"   - Test:  {TEST_DATA_PATH} ({test_df.shape})")
    print("=" * 60)


if __name__ == "__main__":
    run()

