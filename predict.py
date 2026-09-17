"""
MachineGuard — Inference Pipeline
Provides reusable prediction functions for machine failure risk assessment.

Person 2: Machine Learning Engineer
"""

import os
import joblib
import pandas as pd
from typing import Tuple, Optional

# Paths & Feature Specification
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Cached model instance for efficient repeated inference (e.g. in Streamlit)
_CACHED_MODEL = None


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load model from disk or return cached instance."""
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. Please run train.py first."
        )
    _CACHED_MODEL = joblib.load(model_path)
    return _CACHED_MODEL


def predict_risk(
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
    model_path: Optional[str] = None,
) -> Tuple[int, float]:
    """
    Generate failure risk prediction for a single set of operating parameters.

    Important Interpretation Note:
    - This model detects operating patterns statistically associated with machine failure.
    - It does NOT claim deterministic certainty ('The machine will definitely fail').
    - The returned probability represents a model-generated failure-associated probability.

    Parameters:
        air_temperature (float): Air temperature in Kelvin [K]
        process_temperature (float): Process temperature in Kelvin [K]
        rotational_speed (float): Rotational speed in revolutions per minute [rpm]
        torque (float): Torque in Newton-meters [Nm]
        tool_wear (float): Tool wear in minutes [min]
        model_path (str, optional): Custom path to joblib model file

    Returns:
        tuple (predicted_class, failure_associated_probability):
            predicted_class (int): 0 (normal operating pattern) or 1 (failure-associated pattern)
            failure_associated_probability (float): Probability score in [0.0, 1.0] for class 1
    """
    model = load_model(model_path or DEFAULT_MODEL_PATH)

    # Prepare single-row DataFrame with exact feature names
    input_data = pd.DataFrame(
        [
            [
                float(air_temperature),
                float(process_temperature),
                float(rotational_speed),
                float(torque),
                float(tool_wear),
            ]
        ],
        columns=FEATURES,
    )

    pred_class = int(model.predict(input_data)[0])
    failure_probability = float(model.predict_proba(input_data)[0, 1])

    return pred_class, failure_probability


if __name__ == "__main__":
    print("=" * 60)
    print("MachineGuard — Inference Test Demonstration")
    print("=" * 60)

    # Sample test case 1: Typical normal operating conditions
    sample_normal = {
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1500.0,
        "torque": 40.0,
        "tool_wear": 10.0,
    }

    # Sample test case 2: Stressed operating conditions (high torque, high tool wear, low speed)
    sample_stress = {
        "air_temperature": 304.5,
        "process_temperature": 313.2,
        "rotational_speed": 1200.0,
        "torque": 68.0,
        "tool_wear": 220.0,
    }

    cls_norm, prob_norm = predict_risk(**sample_normal)
    print("\nSample 1 (Nominal Conditions):")
    for k, v in sample_normal.items():
        print(f"  {k:22s}: {v}")
    print(f"  -> Predicted Pattern:           {cls_norm} ({'Failure-Associated Pattern' if cls_norm == 1 else 'Normal Operating Pattern'})")
    print(f"  -> Failure-Associated Probability: {prob_norm:.4f} ({prob_norm * 100:.2f}%)")

    cls_stress, prob_stress = predict_risk(**sample_stress)
    print("\nSample 2 (High-Stress Operating Conditions):")
    for k, v in sample_stress.items():
        print(f"  {k:22s}: {v}")
    print(f"  -> Predicted Pattern:           {cls_stress} ({'Failure-Associated Pattern' if cls_stress == 1 else 'Normal Operating Pattern'})")
    print(f"  -> Failure-Associated Probability: {prob_stress:.4f} ({prob_stress * 100:.2f}%)")
    print("=" * 60)

