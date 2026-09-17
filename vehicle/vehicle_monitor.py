"""
Vehicle Maintenance AI — Telemetry Monitoring & Batch Analysis Service.

Provides high-level monitoring utilities:
1. Preset scenario definitions for interactive demos and testing.
2. Batch CSV telemetry analyzer (processes uploaded trip logs, calculates fleet/trip health metrics).
3. Flexible column mapping for various OBD scanner export formats.
"""

import os
import sys
import io
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from vehicle.predict_vehicle import analyze_vehicle, inspect_sensor_anomalies, get_model_bundle, parse_dtc_details

PRESET_VEHICLE_SCENARIOS = {
    "Normal Highway Cruising": {
        "description": "Smooth highway cruising at constant speed, optimal operating temperature and low engine stress.",
        "readings": {
            "ENGINE_RPM": 1850.0,
            "SPEED": 85.0,
            "ENGINE_COOLANT_TEMP": 88.0,
            "ENGINE_LOAD": 38.0,
            "THROTTLE_POS": 22.0,
            "AIR_INTAKE_TEMP": 32.0,
            "TROUBLE_CODES": None
        },
        "expected_status": "HEALTHY"
    },
    "City Stop-and-Go (Idle / Traffic)": {
        "description": "Urban driving with frequent idling and low vehicle speeds.",
        "readings": {
            "ENGINE_RPM": 850.0,
            "SPEED": 0.0,
            "ENGINE_COOLANT_TEMP": 93.0,
            "ENGINE_LOAD": 28.0,
            "THROTTLE_POS": 12.0,
            "AIR_INTAKE_TEMP": 42.0,
            "TROUBLE_CODES": None
        },
        "expected_status": "HEALTHY / ADVISORY"
    },
    "Engine Overheating Warning": {
        "description": "Thermostat failure or low coolant leading to dangerous coolant temperature rise.",
        "readings": {
            "ENGINE_RPM": 2600.0,
            "SPEED": 45.0,
            "ENGINE_COOLANT_TEMP": 112.0,
            "ENGINE_LOAD": 65.0,
            "THROTTLE_POS": 35.0,
            "AIR_INTAKE_TEMP": 55.0,
            "TROUBLE_CODES": None
        },
        "expected_status": "CRITICAL"
    },
    "O2 Sensor Circuit Slow Response (P0133)": {
        "description": "Upstream lambda oxygen sensor degradation causing rich stumbling and delayed closed-loop feedback.",
        "readings": {
            "ENGINE_RPM": 1400.0,
            "SPEED": 25.0,
            "ENGINE_COOLANT_TEMP": 74.0,
            "ENGINE_LOAD": 42.0,
            "THROTTLE_POS": 12.0,
            "AIR_INTAKE_TEMP": 38.0,
            "TROUBLE_CODES": "P0133"
        },
        "expected_status": "ADVISORY / CRITICAL"
    },
    "ABS / Wheel Speed Sensor Malfunction (C0300)": {
        "description": "Faulty wheel speed sensor wiring or reluctor ring debris causing chassis communication fault.",
        "readings": {
            "ENGINE_RPM": 2100.0,
            "SPEED": 60.0,
            "ENGINE_COOLANT_TEMP": 86.0,
            "ENGINE_LOAD": 45.0,
            "THROTTLE_POS": 24.0,
            "AIR_INTAKE_TEMP": 35.0,
            "TROUBLE_CODES": "C0300"
        },
        "expected_status": "CRITICAL / ADVISORY"
    }
}

# Column normalization mapping for third-party OBD CSV export files (Torque Pro, Car Scanner, OBD Auto Doctor)
COLUMN_ALIASES = {
    "ENGINE_RPM": ["engine_rpm", "rpm", "engine speed", "revolutions", "engine_rpm(rpm)"],
    "SPEED": ["speed", "vehicle_speed", "vehicle speed", "spd", "speed(km/h)", "speed(kph)"],
    "ENGINE_COOLANT_TEMP": ["engine_coolant_temp", "coolant_temp", "coolant", "ect", "engine coolant temperature(°c)", "engine_coolant_temperature"],
    "ENGINE_LOAD": ["engine_load", "calculated_engine_load", "load", "engine load(%)", "calculated load"],
    "THROTTLE_POS": ["throttle_pos", "throttle_position", "throttle", "tpos", "throttle position(%)"],
    "AIR_INTAKE_TEMP": ["air_intake_temp", "intake_air_temp", "iat", "air intake temperature(°c)", "intake_temp"],
    "TROUBLE_CODES": ["trouble_codes", "dtc", "dtc_number", "fault_codes", "trouble codes"]
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes heterogeneous OBD export column headers to standard uppercase feature names."""
    df_clean = df.copy()
    col_map = {}
    lower_cols = {str(c).strip().lower(): c for c in df_clean.columns}

    for target_col, aliases in COLUMN_ALIASES.items():
        if target_col in df_clean.columns:
            continue
        for alias in aliases:
            if alias in lower_cols:
                col_map[lower_cols[alias]] = target_col
                break

    if col_map:
        df_clean = df_clean.rename(columns=col_map)
    return df_clean


def analyze_batch_telemetry(data_input: Union[pd.DataFrame, str, bytes]) -> Dict[str, Any]:
    """
    Analyzes a batch of vehicle telemetry records (e.g. from an uploaded trip log CSV).
    Computes summary health stats, risk progression, DTC frequencies, and anomalous events.
    """
    if isinstance(data_input, pd.DataFrame):
        df = data_input.copy()
    elif isinstance(data_input, bytes):
        df = pd.read_csv(io.BytesIO(data_input), low_memory=False)
    elif isinstance(data_input, str):
        if os.path.exists(data_input):
            df = pd.read_csv(data_input, low_memory=False)
        else:
            df = pd.read_csv(io.StringIO(data_input), low_memory=False)
    else:
        raise ValueError("Unsupported data input type. Pass DataFrame, file path, or bytes/string.")

    total_rows = len(df)
    if total_rows == 0:
        return {"status": "error", "message": "Uploaded telemetry dataset is empty."}

    df = normalize_column_names(df)

    core_cols = ["ENGINE_RPM", "SPEED", "ENGINE_COOLANT_TEMP", "ENGINE_LOAD", "THROTTLE_POS", "AIR_INTAKE_TEMP"]
    missing_cols = [c for c in core_cols if c not in df.columns]
    if missing_cols:
        return {
            "status": "error",
            "message": f"Missing required OBD sensor columns: {missing_cols}. Available columns: {list(df.columns)}"
        }

    # Clean numeric columns
    for col in core_cols:
        s = df[col].astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False).str.strip()
        df[col] = pd.to_numeric(s, errors="coerce")

    clean_df = df.dropna(subset=core_cols).copy()
    valid_rows = len(clean_df)
    if valid_rows == 0:
        return {"status": "error", "message": "No valid numeric rows found after cleaning required OBD sensors."}

    # Vectorized derived feature computation
    clean_df["SPEED_TO_RPM"] = clean_df["SPEED"] / (clean_df["ENGINE_RPM"] + 1.0)
    clean_df["LOAD_TO_THROTTLE"] = clean_df["ENGINE_LOAD"] / (clean_df["THROTTLE_POS"] + 1.0)
    clean_df["TEMP_DIFF"] = clean_df["ENGINE_COOLANT_TEMP"] - clean_df["AIR_INTAKE_TEMP"]
    clean_df["IS_IDLE"] = ((clean_df["ENGINE_RPM"] > 400.0) & (clean_df["SPEED"] < 3.0)).astype(int)

    # Batch model scoring
    bundle = get_model_bundle()
    iso_forest = bundle["isolation_forest"]
    rf_clf = bundle["supervised_classifier"]
    scaler = bundle["scaler"]
    feature_names = bundle["features"]
    score_min = bundle["score_min"]
    score_max = bundle["score_max"]

    X_scaled = scaler.transform(clean_df[feature_names])
    raw_scores = -iso_forest.decision_function(X_scaled)
    risk_scores = np.clip((raw_scores - score_min) / (score_max - score_min) * 100.0, 0.0, 100.0)
    supervised_probs = rf_clf.predict_proba(X_scaled)[:, 1] * 100.0

    clean_df["ANOMALY_SCORE"] = raw_scores
    clean_df["RISK_SCORE_PCT"] = risk_scores
    clean_df["FAULT_PROB_PCT"] = supervised_probs

    # Categorize risk buckets
    clean_df["CONDITION"] = "HEALTHY"
    clean_df.loc[clean_df["RISK_SCORE_PCT"] >= 38.0, "CONDITION"] = "ADVISORY"
    clean_df.loc[(clean_df["RISK_SCORE_PCT"] >= 65.0) | (clean_df["ENGINE_COOLANT_TEMP"] > 105.0), "CONDITION"] = "CRITICAL"

    healthy_count = int((clean_df["CONDITION"] == "HEALTHY").sum())
    advisory_count = int((clean_df["CONDITION"] == "ADVISORY").sum())
    critical_count = int((clean_df["CONDITION"] == "CRITICAL").sum())

    # DTC summary
    dtc_summary = {}
    if "TROUBLE_CODES" in clean_df.columns:
        dtc_non_null = clean_df["TROUBLE_CODES"].dropna()
        for code, count in dtc_non_null.value_counts().items():
            parsed = parse_dtc_details(str(code))
            dtc_summary[str(code)] = {
                "count": int(count),
                "title": parsed["title"] if parsed else "Diagnostic Trouble Code",
                "severity": parsed["severity"] if parsed else "MODERATE"
            }

    # Summary metrics
    mean_risk = float(risk_scores.mean())
    max_risk = float(risk_scores.max())
    overheat_events = int((clean_df["ENGINE_COOLANT_TEMP"] > 105.0).sum())

    if critical_count > 0 or max_risk > 85.0 or overheat_events > 0:
        trip_status = "CRITICAL RISK DETECTED"
        trip_category = "CRITICAL"
    elif advisory_count > (valid_rows * 0.20) or mean_risk > 35.0 or len(dtc_summary) > 0:
        trip_status = "MODERATE ANOMALY DETECTED"
        trip_category = "ADVISORY"
    else:
        trip_status = "TRIP TELEMETRY HEALTHY"
        trip_category = "NORMAL"

    # Top anomaly records
    top_anomalies = clean_df.sort_values(by="RISK_SCORE_PCT", ascending=False).head(5)
    anomalous_samples = []
    for _, row in top_anomalies.iterrows():
        anomalous_samples.append({
            "rpm": round(float(row["ENGINE_RPM"]), 1),
            "speed": round(float(row["SPEED"]), 1),
            "coolant": round(float(row["ENGINE_COOLANT_TEMP"]), 1),
            "load": round(float(row["ENGINE_LOAD"]), 1),
            "throttle": round(float(row["THROTTLE_POS"]), 1),
            "risk_pct": round(float(row["RISK_SCORE_PCT"]), 1),
            "fault_prob_pct": round(float(row["FAULT_PROB_PCT"]), 1),
            "condition": str(row["CONDITION"])
        })

    return {
        "status": "success",
        "trip_status": trip_status,
        "trip_category": trip_category,
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "mean_risk_score_pct": round(mean_risk, 1),
        "max_risk_score_pct": round(max_risk, 1),
        "condition_counts": {
            "healthy": healthy_count,
            "advisory": advisory_count,
            "critical": critical_count
        },
        "condition_percentages": {
            "healthy_pct": round(healthy_count / valid_rows * 100.0, 1),
            "advisory_pct": round(advisory_count / valid_rows * 100.0, 1),
            "critical_pct": round(critical_count / valid_rows * 100.0, 1)
        },
        "overheat_events_count": overheat_events,
        "dtc_summary": dtc_summary,
        "top_anomalous_events": anomalous_samples
    }


def get_preset_scenario(scenario_name: str) -> Dict[str, Any]:
    """Retrieves preset scenario readings and description."""
    if scenario_name not in PRESET_VEHICLE_SCENARIOS:
        raise KeyError(f"Unknown preset scenario '{scenario_name}'. Available: {list(PRESET_VEHICLE_SCENARIOS.keys())}")
    return PRESET_VEHICLE_SCENARIOS[scenario_name]


def get_available_presets() -> List[str]:
    """Returns list of available preset vehicle scenario names."""
    return list(PRESET_VEHICLE_SCENARIOS.keys())


if __name__ == "__main__":
    print("Testing Vehicle Monitor Presets:")
    for name in get_available_presets():
        scenario = get_preset_scenario(name)
        res = analyze_vehicle(scenario["readings"])
        print(f"[{name}] -> Status: {res['health_status']} (Risk: {res['risk_score_pct']}%)")

    print("\nTesting Batch CSV Analysis on sample subset:")
    sample_csv = os.path.join(BASE_DIR, "data", "vehicle", "processed", "vehicle_clean.csv")
    if os.path.exists(sample_csv):
        batch_res = analyze_batch_telemetry(sample_csv)
        print(f"Batch Analysis Results: Valid Rows = {batch_res['valid_rows']:,}")
        print(f"Mean Risk: {batch_res['mean_risk_score_pct']}% | Overall: {batch_res['trip_status']}")
        print(f"Breakdown: Healthy={batch_res['condition_counts']['healthy']:,} | Advisory={batch_res['condition_counts']['advisory']:,} | Critical={batch_res['condition_counts']['critical']:,}")
