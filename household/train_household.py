"""
MachineGuard — Household Model Training Pipeline
Trains and compares candidate regression models on the UCI Appliances Energy Prediction dataset
using chronological time-series splitting (70% train / 15% validation / 15% test).
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "household", "processed", "household_clean.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "household_model.joblib")


def load_and_split_data(data_path: str = PROCESSED_DATA_PATH):
    """Loads cleaned data and performs chronological 70/15/15 time-series split."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {data_path}. Run preprocess_household.py first.")

    df = pd.read_csv(data_path)

    # Feature selection: all numeric sensor/weather/time columns, excluding timestamp and target
    feature_cols = [c for c in df.columns if c not in ["date", "Appliances"]]
    target_col = "Appliances"

    X = df[feature_cols]
    y = df[target_col]

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def train_and_evaluate():
    print("=" * 70)
    print("MachineGuard — Household Energy AI Model Training")
    print("=" * 70)

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_and_split_data()
    print(f"Data partitioning (Chronological, No Shuffling):")
    print(f"  - Training Set:   {len(X_train):,} samples (70%)")
    print(f"  - Validation Set: {len(X_val):,} samples (15%)")
    print(f"  - Test Set:       {len(X_test):,} samples (15%)")
    print(f"  - Features:       {len(feature_cols)} input dimensions")

    # Candidate Models Comparison
    candidates = {
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=100, max_depth=14, min_samples_leaf=10, random_state=42, n_jobs=-1
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=120, max_leaf_nodes=31, min_samples_leaf=20, random_state=42
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    print("\nEvaluating Candidate Models on Validation Set:")
    best_name = None
    best_val_mae = float("inf")
    best_model = None

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)

        mae = mean_absolute_error(y_val, val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        r2 = r2_score(y_val, val_pred)

        results[name] = {"model": model, "mae": mae, "rmse": rmse, "r2": r2}
        print(f"  [{name:28s}] Val MAE: {mae:6.2f} Wh | RMSE: {rmse:6.2f} Wh | R²: {r2:+.4f}")

        if mae < best_val_mae:
            best_val_mae = mae
            best_name = name
            best_model = model

    print(f"\n--> Selected Best Model: {best_name} (Lowest Validation MAE: {best_val_mae:.2f} Wh)")

    # Compute Anomaly Thresholds using strictly Validation Absolute Residuals
    val_pred_best = best_model.predict(X_val)
    val_abs_residual = np.abs(y_val - val_pred_best)

    warning_threshold = float(np.percentile(val_abs_residual, 80))  # 80th percentile
    anomaly_threshold = float(np.percentile(val_abs_residual, 95))  # 95th percentile

    print(f"\nCalibrated Residual Anomaly Thresholds (from Validation Set):")
    print(f"  - Warning Threshold (80th percentile): {warning_threshold:.2f} Wh")
    print(f"  - Anomaly Threshold (95th percentile): {anomaly_threshold:.2f} Wh")

    # Final Single Evaluation on Unseen Test Set
    test_pred = best_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)

    test_abs_residual = np.abs(y_test - test_pred)
    pct_normal = (test_abs_residual < warning_threshold).mean() * 100.0
    pct_warning = ((test_abs_residual >= warning_threshold) & (test_abs_residual < anomaly_threshold)).mean() * 100.0
    pct_anomaly = (test_abs_residual >= anomaly_threshold).mean() * 100.0

    print(f"\nFinal Test Set Performance (Evaluated Once on Held-out 15%):")
    print(f"  - Test MAE:  {test_mae:.2f} Wh")
    print(f"  - Test RMSE: {test_rmse:.2f} Wh")
    print(f"  - Test R²:   {test_r2:.4f}")
    print(f"  - Test Categorization Breakdown:")
    print(f"      * Normal Pattern:    {pct_normal:.2f}%")
    print(f"      * Unusual / Warning: {pct_warning:.2f}%")
    print(f"      * Potential Anomaly: {pct_anomaly:.2f}%")

    # Save Pipeline Artifact
    os.makedirs(MODEL_DIR, exist_ok=True)
    artifact = {
        "model": best_model,
        "model_name": best_name,
        "feature_names": feature_cols,
        "target_name": "Appliances",
        "unit": "Wh",
        "thresholds": {
            "warning": warning_threshold,
            "anomaly": anomaly_threshold,
        },
        "metrics": {
            "val_mae": float(results[best_name]["mae"]),
            "val_rmse": float(results[best_name]["rmse"]),
            "val_r2": float(results[best_name]["r2"]),
            "test_mae": float(test_mae),
            "test_rmse": float(test_rmse),
            "test_r2": float(test_r2),
        },
    }

    joblib.dump(artifact, MODEL_SAVE_PATH)
    print(f"\nSuccessfully saved household model artifact to: {MODEL_SAVE_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    train_and_evaluate()

