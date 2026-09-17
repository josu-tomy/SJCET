"""
MachineGuard — Household Energy Inference & Anomaly Detection Pipeline
Provides reusable prediction and anomaly scoring based on expected vs. actual energy use.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "household_model.joblib")

_CACHED_ARTIFACT = None

# Baseline Median Default Readings from UCI AI4I Dataset for non-entered variables
FEATURE_DEFAULTS = {
    "lights": 0.0,
    "T1": 21.6, "RH_1": 39.65,
    "T2": 20.0, "RH_2": 37.60,
    "T3": 22.1, "RH_3": 38.53,
    "T4": 20.85, "RH_4": 38.40,
    "T5": 19.39, "RH_5": 49.09,
    "T6": 7.3, "RH_6": 53.77,
    "T7": 20.03, "RH_7": 34.86,
    "T8": 22.1, "RH_8": 42.37,
    "T9": 19.39, "RH_9": 40.90,
    "T_out": 6.9, "Press_mm_hg": 756.1,
    "RH_out": 82.0, "Windspeed": 4.0,
    "Visibility": 40.0, "Tdewpoint": 3.4,
    "hour": 12, "day_of_week": 2, "month": 3,
    "is_weekend": 0, "sin_hour": 0.0, "cos_hour": -1.0,
}


def load_household_model(model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """Loads and caches the trained household model bundle."""
    global _CACHED_ARTIFACT
    if _CACHED_ARTIFACT is not None:
        return _CACHED_ARTIFACT
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Household model bundle not found at: {model_path}. Run train_household.py first.")
    _CACHED_ARTIFACT = joblib.load(model_path)
    return _CACHED_ARTIFACT


def predict_household(
    readings: Optional[Dict[str, float]] = None,
    actual_energy: Optional[float] = None,
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Predicts expected household energy consumption in Wh and computes
    an anomaly score if actual consumption measurement is provided.

    Parameters:
        readings (dict, optional): Dictionary containing sensor/environmental readings
        actual_energy (float, optional): Actual measured energy consumption in Wh
        model_path (str, optional): Custom path to model artifact

    Returns:
        dict: {
            "expected_energy": float (Wh),
            "actual_energy": float or None (Wh),
            "residual": float or None (Wh),
            "abs_residual": float or None (Wh),
            "anomaly_score": float or None (0.0 to 1.0+ normalized score),
            "status": "NORMAL" | "WARNING" | "ANOMALY",
            "warning_threshold": float (Wh),
            "anomaly_threshold": float (Wh),
            "explanation": str
        }
    """
    bundle = load_household_model(model_path or DEFAULT_MODEL_PATH)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    thresholds = bundle["thresholds"]
    warning_thresh = thresholds["warning"]
    anomaly_thresh = thresholds["anomaly"]

    # Merge user-supplied readings with baseline defaults
    full_readings = dict(FEATURE_DEFAULTS)
    if readings:
        full_readings.update(readings)

    # If hour is provided in readings, calculate sin_hour and cos_hour automatically
    if "hour" in full_readings:
        hr = float(full_readings["hour"])
        full_readings["sin_hour"] = np.sin(2 * np.pi * hr / 24.0)
        full_readings["cos_hour"] = np.cos(2 * np.pi * hr / 24.0)

    # Build input DataFrame matching exact feature column order
    row_data = [[full_readings.get(col, 0.0) for col in feature_names]]
    X_input = pd.DataFrame(row_data, columns=feature_names)

    expected_energy = float(model.predict(X_input)[0])
    expected_energy = max(0.0, expected_energy)  # Physical constraint

    result = {
        "expected_energy": round(expected_energy, 2),
        "actual_energy": None,
        "residual": None,
        "abs_residual": None,
        "anomaly_score": None,
        "status": "NORMAL",
        "warning_threshold": round(warning_thresh, 2),
        "anomaly_threshold": round(anomaly_thresh, 2),
        "explanation": "Expected energy consumption calculated under given environmental baseline.",
    }

    if actual_energy is not None:
        act = float(actual_energy)
        residual = act - expected_energy
        abs_res = abs(residual)
        # Normalized anomaly score relative to the 95th percentile threshold
        # score = 0.0 to 1.0 indicates normal-to-warning; > 1.0 indicates anomaly
        score = round(abs_res / anomaly_thresh, 3)

        if abs_res < warning_thresh:
            status = "NORMAL"
            badge = "🟢 NORMAL"
            explanation = (
                f"Observed energy ({act:.1f} Wh) closely matches model-expected demand ({expected_energy:.1f} Wh). "
                f"Operating pattern aligns with historical training baseline."
            )
        elif abs_res < anomaly_thresh:
            status = "WARNING"
            badge = "🟡 WARNING"
            explanation = (
                f"Observed energy ({act:.1f} Wh) deviates moderately from expected baseline ({expected_energy:.1f} Wh) "
                f"by {abs_res:.1f} Wh (exceeds 80th percentile threshold of {warning_thresh:.1f} Wh)."
            )
        else:
            status = "ANOMALY"
            badge = "🔴 ANOMALY"
            explanation = (
                f"Significant unusual energy spike/drop detected! Actual consumption ({act:.1f} Wh) differs from "
                f"expected baseline ({expected_energy:.1f} Wh) by {abs_res:.1f} Wh (exceeds 95th percentile threshold of {anomaly_thresh:.1f} Wh). "
                f"Suggests potential equipment malfunction, standby leakage, or abnormal operating load."
            )

        result.update({
            "actual_energy": round(act, 2),
            "residual": round(residual, 2),
            "abs_residual": round(abs_res, 2),
            "anomaly_score": score,
            "status": status,
            "badge": badge,
            "explanation": explanation,
        })

    return result


if __name__ == "__main__":
    print("Testing household energy prediction pipeline:")
    res_normal = predict_household({"hour": 14, "T_out": 12.0}, actual_energy=95.0)
    print("Normal test:", res_normal)

    res_anomaly = predict_household({"hour": 14, "T_out": 12.0}, actual_energy=450.0)
    print("Anomaly test:", res_anomaly)

