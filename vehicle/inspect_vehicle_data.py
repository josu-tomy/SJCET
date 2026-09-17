"""
Inspect OBD-II Vehicle Sensor/Telemetry Dataset.

Provides detailed statistical and structural diagnostics:
- Shape, columns, datatypes
- Missing values and cardinality
- Distribution of key telemetry parameters (RPM, Speed, Coolant Temp, Engine Load, etc.)
- Investigation of Diagnostic Trouble Codes (DTC_NUMBER, TROUBLE_CODES)
- Assessment of supervised vs unsupervised modeling suitability
"""

import os
import pandas as pd
import numpy as np

RAW_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "vehicle", "raw", "exp1_14drivers_14cars_dailyRoutes.csv"
)

def inspect_obd_data():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Dataset not found at {RAW_DATA_PATH}")
        return

    print("=" * 70)
    print("OBD-II VEHICLE TELEMETRY DATASET INSPECTION")
    print("=" * 70)
    print(f"Source file: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")

    print("\n--- Column List & Non-Null Counts ---")
    for col in df.columns:
        non_null = df[col].notna().sum()
        pct = (non_null / len(df)) * 100
        dtype = df[col].dtype
        sample_val = df[col].dropna().iloc[0] if non_null > 0 else "N/A"
        print(f"  {col:<30} | {dtype} | {non_null:,} non-null ({pct:.1f}%) | sample: {sample_val}")

    print("\n--- Target / Diagnostic Fields Analysis ---")
    if "DTC_NUMBER" in df.columns:
        print("\nDTC_NUMBER unique values and frequencies:")
        print(df["DTC_NUMBER"].value_counts(dropna=False).head(10))

    if "TROUBLE_CODES" in df.columns:
        print("\nTROUBLE_CODES non-null count:", df["TROUBLE_CODES"].notna().sum())
        print("TROUBLE_CODES unique values:")
        print(df["TROUBLE_CODES"].value_counts(dropna=False).head(10))

    if "VEHICLE_ID" in df.columns:
        print("\nVehicle breakdown (VEHICLE_ID):")
        print(df["VEHICLE_ID"].value_counts())

    print("\n--- Numerical Telemetry Candidates ---")
    # Clean percentages and comma decimals for inspection
    sample_cols = [
        "ENGINE_RPM", "SPEED", "ENGINE_COOLANT_TEMP", 
        "ENGINE_LOAD", "THROTTLE_POS", "AIR_INTAKE_TEMP", 
        "MAF", "INTAKE_MANIFOLD_PRESSURE", "BAROMETRIC_PRESSURE(KPA)"
    ]
    
    clean_dict = {}
    for col in sample_cols:
        if col in df.columns:
            s = df[col].astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False).str.strip()
            num_s = pd.to_numeric(s, errors="coerce")
            clean_dict[col] = num_s
            print(f"  {col:<28}: non-null numeric = {num_s.notna().sum():,} | min={num_s.min():.1f} | median={num_s.median():.1f} | mean={num_s.mean():.1f} | max={num_s.max():.1f}")

    num_df = pd.DataFrame(clean_dict)
    print("\n--- Summary Statistics of Key Sensor Metrics ---")
    print(num_df.describe().T[["count", "mean", "std", "min", "50%", "max"]])

    # Modeling paradigm assessment
    print("\n--- Machine Learning Formulation Assessment ---")
    has_mil_on = False
    if "DTC_NUMBER" in df.columns:
        mil_on_count = df["DTC_NUMBER"].astype(str).str.contains("MIL is ON", na=False).sum()
        codes_count = df["TROUBLE_CODES"].notna().sum()
        print(f"Records with 'MIL is ON': {mil_on_count:,} ({mil_on_count/len(df)*100:.2f}%)")
        print(f"Records with Diagnostic Trouble Codes: {codes_count:,} ({codes_count/len(df)*100:.2f}%)")
        if mil_on_count > 0 or codes_count > 0:
            has_mil_on = True

    print("\nConclusion:")
    if has_mil_on:
        print("Dataset contains real Diagnostic Trouble Code / MIL indicator events.")
        print("Can be used for supervised fault classification or dual-mode (unsupervised anomaly detection + DTC fault categorization).")
    else:
        print("Dataset lacks explicit fault labels. Unsupervised Anomaly Detection (e.g. IsolationForest) is the methodologically sound choice.")

if __name__ == "__main__":
    inspect_obd_data()

