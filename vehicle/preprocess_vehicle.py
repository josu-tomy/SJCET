"""
Data Preprocessing Pipeline for Vehicle Maintenance AI (OBD-II Telemetry).

Features:
- Cleans raw OBD telemetry strings (percentages, European comma decimals, whitespaces)
- Validates physical plausible boundaries for automotive engine sensors
- Computes automotive domain derived features:
    - SPEED_TO_RPM: Gear/transmission engagement proxy
    - LOAD_TO_THROTTLE: Mechanical load vs throttle input ratio
    - TEMP_DIFF: Thermal gradient between coolant and intake air
    - IS_IDLE: Vehicle stationary with running engine indicator
- Extracts ground truth DTC indicators (FAULT_PRESENT, FAULT_TYPE)
- Exports cleaned data to data/vehicle/processed/vehicle_clean.csv
"""

import os
import re
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "vehicle", "raw", "exp1_14drivers_14cars_dailyRoutes.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "vehicle", "processed", "vehicle_clean.csv")

CORE_SENSORS = [
    "ENGINE_RPM",
    "SPEED",
    "ENGINE_COOLANT_TEMP",
    "ENGINE_LOAD",
    "THROTTLE_POS",
    "AIR_INTAKE_TEMP"
]

DERIVED_FEATURES = [
    "SPEED_TO_RPM",
    "LOAD_TO_THROTTLE",
    "TEMP_DIFF",
    "IS_IDLE"
]

ALL_MODEL_FEATURES = CORE_SENSORS + DERIVED_FEATURES

# Physical plausibility boundaries for automotive sensor sanity checks
PHYSICAL_LIMITS = {
    "ENGINE_RPM": (400.0, 7500.0),
    "SPEED": (0.0, 220.0),
    "ENGINE_COOLANT_TEMP": (-20.0, 130.0),
    "ENGINE_LOAD": (0.0, 100.0),
    "THROTTLE_POS": (0.0, 100.0),
    "AIR_INTAKE_TEMP": (-30.0, 95.0),
}


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """Cleans numeric series that may contain % signs, comma decimals, or whitespace."""
    s = series.astype(str).str.replace("%", "", regex=False)
    s = s.str.replace(",", ".", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def parse_fault_type(trouble_code: str) -> str:
    """Categorizes raw trouble code strings into standardized fault classifications."""
    if pd.isna(trouble_code) or str(trouble_code).strip() in ("", "nan", "None"):
        return "NORMAL"
    code = str(trouble_code).strip().upper()
    if "P0133" in code:
        return "O2_SENSOR_SLOW_RESPONSE"
    elif "C0300" in code:
        return "CHASSIS_SPEED_SENSOR_FAULT"
    elif any(k in code for k in ["P0078", "P0079", "P007E", "P007F"]):
        return "VALVE_CONTROL_SOLENOID_FAULT"
    elif "P2004" in code:
        return "INTAKE_MANIFOLD_RUNNER_FAULT"
    elif "P3000" in code:
        return "CONTROL_MODULE_FAULT"
    else:
        return "GENERIC_DTC_FAULT"


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Computes automotive physics-based derived indicators from core sensor readings."""
    df_out = df.copy()
    
    # Gear/Transmission ratio proxy: speed / (rpm + 1)
    df_out["SPEED_TO_RPM"] = df_out["SPEED"] / (df_out["ENGINE_RPM"] + 1.0)
    
    # Load to throttle opening ratio: load / (throttle + 1)
    df_out["LOAD_TO_THROTTLE"] = df_out["ENGINE_LOAD"] / (df_out["THROTTLE_POS"] + 1.0)
    
    # Thermal gradient between engine coolant and intake air
    df_out["TEMP_DIFF"] = df_out["ENGINE_COOLANT_TEMP"] - df_out["AIR_INTAKE_TEMP"]
    
    # Idle engine condition (engine running at idle, vehicle at rest)
    df_out["IS_IDLE"] = ((df_out["ENGINE_RPM"] > 400) & (df_out["SPEED"] < 3.0)).astype(int)
    
    return df_out


def preprocess_vehicle_data(raw_path: str = RAW_DATA_PATH, save_path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """
    Main preprocessing pipeline for raw OBD-II vehicle telemetry.
    Reads, cleans, filters physical bounds, extracts derived metrics and fault labels,
    and saves to processed CSV.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw vehicle dataset not found at: {raw_path}")

    print(f"[Vehicle Preprocessing] Loading raw dataset: {raw_path}")
    raw_df = pd.read_csv(raw_path, low_memory=False)
    
    # Filter rows with valid timestamp
    valid_df = raw_df[raw_df["TIMESTAMP"].notna()].copy()
    print(f"[Vehicle Preprocessing] Valid timestamped records: {len(valid_df):,}")

    # Clean numeric representations
    for col in CORE_SENSORS:
        if col in valid_df.columns:
            valid_df[col] = clean_numeric_series(valid_df[col])
        else:
            raise KeyError(f"Required OBD core sensor missing: {col}")

    # Drop records missing any core sensor
    clean_df = valid_df.dropna(subset=CORE_SENSORS).copy()
    print(f"[Vehicle Preprocessing] Records with complete core sensors: {len(clean_df):,}")

    # Apply automotive physical plausibility bounds
    mask = pd.Series(True, index=clean_df.index)
    for col, (min_val, max_val) in PHYSICAL_LIMITS.items():
        mask &= (clean_df[col] >= min_val) & (clean_df[col] <= max_val)
    
    filtered_df = clean_df[mask].copy()
    print(f"[Vehicle Preprocessing] Plausible physical records: {len(filtered_df):,}")

    # Compute derived automotive engineering features
    processed_df = compute_derived_metrics(filtered_df)

    # Label extraction from ground-truth DTCs
    trouble_codes = processed_df["TROUBLE_CODES"] if "TROUBLE_CODES" in processed_df.columns else pd.Series(np.nan, index=processed_df.index)
    processed_df["FAULT_PRESENT"] = trouble_codes.notna().astype(int)
    processed_df["FAULT_TYPE"] = trouble_codes.apply(parse_fault_type)

    # Reorder columns: identifiers, sensors, derived, targets
    id_cols = [c for c in ["TIMESTAMP", "VEHICLE_ID", "MARK", "MODEL", "CAR_YEAR"] if c in processed_df.columns]
    target_cols = ["FAULT_PRESENT", "FAULT_TYPE"]
    final_cols = id_cols + ALL_MODEL_FEATURES + target_cols

    # Keep any existing extra columns at the end if present
    remaining_cols = [c for c in processed_df.columns if c not in final_cols]
    final_df = processed_df[final_cols + remaining_cols]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    final_df.to_csv(save_path, index=False)
    print(f"[Vehicle Preprocessing] Saved cleaned dataset to: {save_path} ({len(final_df):,} rows, {len(final_df.columns)} columns)")

    fault_counts = final_df["FAULT_PRESENT"].value_counts().to_dict()
    print(f"[Vehicle Preprocessing] Class distribution: Normal={fault_counts.get(0, 0):,}, Fault={fault_counts.get(1, 0):,}")
    print(f"[Vehicle Preprocessing] Fault types breakdown:")
    for ft, cnt in final_df["FAULT_TYPE"].value_counts().items():
        print(f"   {ft:<30}: {cnt:,}")

    return final_df


def load_clean_vehicle_data(processed_path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Loads the preprocessed clean vehicle dataset."""
    if not os.path.exists(processed_path):
        return preprocess_vehicle_data(save_path=processed_path)
    return pd.read_csv(processed_path, low_memory=False)


def get_feature_names() -> list:
    """Returns the list of feature column names used for vehicle models."""
    return list(ALL_MODEL_FEATURES)


def prepare_single_reading(reading_dict: dict) -> pd.DataFrame:
    """
    Validates and transforms a single dictionary of sensor readings into model input format.
    Accepts keys: ENGINE_RPM, SPEED, ENGINE_COOLANT_TEMP, ENGINE_LOAD, THROTTLE_POS, AIR_INTAKE_TEMP.
    """
    row = {}
    for sensor in CORE_SENSORS:
        val = reading_dict.get(sensor, None)
        if val is None:
            raise ValueError(f"Missing required sensor reading: {sensor}")
        try:
            row[sensor] = float(val)
        except (ValueError, TypeError):
            raise ValueError(f"Sensor reading '{sensor}' must be numeric, got: {val}")

    df_single = pd.DataFrame([row])
    df_derived = compute_derived_metrics(df_single)
    return df_derived[ALL_MODEL_FEATURES]


if __name__ == "__main__":
    preprocess_vehicle_data()

