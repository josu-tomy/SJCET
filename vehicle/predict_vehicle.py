"""
Vehicle Maintenance AI — Prediction and Diagnostic Inference Engine.

Provides unified inference for OBD-II vehicle telemetry:
- Unsupervised Anomaly Scoring (Isolation Forest)
- Health / Risk Index calculation (0 - 100%)
- Supervised Fault Probability & Classification (Random Forest)
- Automotive Domain Sensor Boundary Checks (Coolant Overheating, Load/Throttle Discordance, etc.)
- SAE J2012 Diagnostic Trouble Code (DTC) Decoder & Actionable Maintenance Recommendations
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "vehicle_model.joblib")

# Nominal automotive operating ranges for healthy passenger vehicles
SENSOR_NOMINAL_RANGES = {
    "ENGINE_RPM": {"min": 650.0, "max": 4000.0, "unit": "RPM", "label": "Engine RPM"},
    "SPEED": {"min": 0.0, "max": 130.0, "unit": "km/h", "label": "Vehicle Speed"},
    "ENGINE_COOLANT_TEMP": {"min": 75.0, "max": 100.0, "unit": "°C", "label": "Coolant Temperature"},
    "ENGINE_LOAD": {"min": 10.0, "max": 80.0, "unit": "%", "label": "Calculated Engine Load"},
    "THROTTLE_POS": {"min": 5.0, "max": 65.0, "unit": "%", "label": "Throttle Position"},
    "AIR_INTAKE_TEMP": {"min": 10.0, "max": 55.0, "unit": "°C", "label": "Intake Air Temperature"},
}

# Standard SAE J2012 OBD-II Diagnostic Trouble Code Database
DTC_KNOWLEDGE_BASE = {
    "P0133": {
        "title": "O2 Sensor Circuit Slow Response (Bank 1, Sensor 1)",
        "subsystem": "Exhaust & Emissions / Air-Fuel Metering",
        "severity": "MODERATE",
        "symptoms": "Sluggish engine response, degraded fuel economy, rough idle, higher tailpipe emissions.",
        "causes": "Contaminated oxygen sensor element, exhaust leak upstream of sensor, sensor heater failure, wiring harness resistance.",
        "action": "Inspect upstream O2 sensor wiring and connector. Test heater circuit resistance (standard 5-15 ohms). If slow transition time persists (>100ms switch time), replace upstream O2 sensor."
    },
    "C0300": {
        "title": "Rear Speed Sensor / Chassis Communication Malfunction",
        "subsystem": "Chassis & ABS / Wheel Speed Monitoring",
        "severity": "HIGH",
        "symptoms": "ABS warning lamp illuminated, traction control disabled, speedometer needle jitter.",
        "causes": "Debris on wheel speed reluctor ring, damaged sensor wiring harness near suspension arm, faulty Hall-effect wheel speed sensor.",
        "action": "Inspect wheel hub sensor wiring for chafing. Clean tone wheel/reluctor ring. Measure wheel sensor output signal with oscilloscope or scanner during wheel spin."
    },
    "P0078": {
        "title": "Exhaust Valve Control Solenoid Circuit (Bank 1)",
        "subsystem": "Engine Valvetrain / Variable Valve Timing (VVT)",
        "severity": "MODERATE",
        "symptoms": "Reduced low-end torque, check engine light, slight engine misfire at idle.",
        "causes": "Clogged VVT solenoid oil screen, low or dirty engine oil, open/short in solenoid coil.",
        "action": "Check engine oil level and viscosity. Remove VVT solenoid, clean oil mesh filter with solvent, test coil resistance (~7-12 ohms), and apply 12V test pulse to verify mechanical plunger travel."
    },
    "P0079": {
        "title": "Exhaust Valve Control Solenoid Circuit Low (Bank 1)",
        "subsystem": "Engine Valvetrain / Variable Valve Timing (VVT)",
        "severity": "MODERATE",
        "symptoms": "VVT timing lockup, power loss at higher RPM.",
        "causes": "Short to ground in solenoid control wire, failed driver in Powertrain Control Module (PCM).",
        "action": "Disconnect solenoid harness. Test for short to ground on control circuit. Replace solenoid if coil is shorted."
    },
    "P0300": {
        "title": "Random / Multiple Cylinder Misfire Detected",
        "subsystem": "Ignition & Combustion System",
        "severity": "CRITICAL",
        "symptoms": "Noticeable engine stumbling, flashing MIL/Check Engine light, catalytic converter overheating risk.",
        "causes": "Worn spark plugs, failing ignition coil packs, low fuel rail pressure, vacuum leak.",
        "action": "CAUTION: Flashing MIL requires immediate stoppage to prevent catalytic converter destruction. Inspect spark plug wear, test fuel pressure, inspect vacuum hoses."
    },
    "P0420": {
        "title": "Catalyst System Efficiency Below Threshold (Bank 1)",
        "subsystem": "Exhaust & Aftertreatment",
        "severity": "MODERATE",
        "symptoms": "Exhaust odor, failed emissions inspection, steady MIL.",
        "causes": "Aged catalytic converter substrate, exhaust leak near cat, downstream O2 sensor drift.",
        "action": "Compare upstream vs downstream O2 sensor waveforms. If downstream sensor mirrors upstream switching, replace catalytic converter."
    },
    "P0171": {
        "title": "System Too Lean (Bank 1)",
        "subsystem": "Fuel & Air Metering",
        "severity": "HIGH",
        "symptoms": "Hesitation under acceleration, pinging/knock, hard cold start.",
        "causes": "Vacuum leak downstream of throttle body, weak fuel pump, dirty Mass Airflow (MAF) sensor.",
        "action": "Perform smoke test for intake vacuum leaks. Clean MAF sensor with approved cleaner. Verify fuel rail pressure under load."
    }
}

# Cache for loaded model bundle
_MODEL_BUNDLE: Optional[Dict[str, Any]] = None


def get_model_bundle() -> Dict[str, Any]:
    """Loads and caches the vehicle model bundle."""
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Vehicle model bundle not found at {MODEL_PATH}. Please run vehicle/train_vehicle.py first.")
        _MODEL_BUNDLE = joblib.load(MODEL_PATH)
    return _MODEL_BUNDLE


def parse_dtc_details(dtc_code: Optional[str]) -> Optional[Dict[str, str]]:
    """Decodes standard OBD-II DTC codes into human-readable automotive diagnostic records."""
    if not dtc_code or str(dtc_code).strip() in ("", "nan", "None", "0"):
        return None
    
    clean_code = str(dtc_code).strip().upper()
    # Extract first standard 5-character DTC (e.g. P0133, C0300)
    import re
    match = re.search(r"([PCBU]\d{4})", clean_code)
    code_key = match.group(1) if match else clean_code[:5]

    if code_key in DTC_KNOWLEDGE_BASE:
        info = DTC_KNOWLEDGE_BASE[code_key].copy()
        info["code"] = code_key
        return info
    
    # Generic SAE J2012 rule-based decoder
    prefix = code_key[0] if len(code_key) > 0 else "P"
    prefix_map = {
        "P": "Powertrain (Engine, Transmission, Fuel, Emissions)",
        "C": "Chassis (ABS, ESP, Suspension, Steering)",
        "B": "Body (Airbags, Climate, Lighting, Instruments)",
        "U": "Network (CAN bus, Module Communication)"
    }
    return {
        "code": code_key,
        "title": f"Diagnostic Trouble Code {code_key}",
        "subsystem": prefix_map.get(prefix, "Electronic Control Subsystem"),
        "severity": "MODERATE",
        "symptoms": "Diagnostic trouble code logged in ECU memory; Malfunction Indicator Lamp may illuminate.",
        "causes": "Subsystem operating variance detected by onboard OBD-II diagnostics.",
        "action": f"Connect diagnostic scanner to query freeze frame data for {code_key}. Verify sensor wiring and ECU ground points."
    }


def inspect_sensor_anomalies(readings: Dict[str, float]) -> list:
    """Checks individual sensor values against automotive mechanical and safety limits."""
    anomalies = []
    
    rpm = readings.get("ENGINE_RPM", 0.0)
    speed = readings.get("SPEED", 0.0)
    coolant = readings.get("ENGINE_COOLANT_TEMP", 85.0)
    load = readings.get("ENGINE_LOAD", 30.0)
    throttle = readings.get("THROTTLE_POS", 15.0)
    iat = readings.get("AIR_INTAKE_TEMP", 35.0)

    # 1. Coolant thermal checks
    if coolant > 105.0:
        anomalies.append({
            "sensor": "ENGINE_COOLANT_TEMP",
            "value": coolant,
            "status": "CRITICAL_OVERHEAT",
            "message": f"Engine coolant temperature ({coolant:.1f}°C) exceeds safe ceiling (105°C). Severe risk of head gasket failure or engine seizure."
        })
    elif coolant > 98.0:
        anomalies.append({
            "sensor": "ENGINE_COOLANT_TEMP",
            "value": coolant,
            "status": "WARNING_HIGH_TEMP",
            "message": f"Engine coolant elevated ({coolant:.1f}°C). Cooling fan or thermostat restriction may be present."
        })
    elif coolant < 65.0 and speed > 10.0:
        anomalies.append({
            "sensor": "ENGINE_COOLANT_TEMP",
            "value": coolant,
            "status": "WARNING_RUNNING_COLD",
            "message": f"Engine running below nominal operating temperature ({coolant:.1f}°C). Thermostat stuck open or cold-enrichment stuck."
        })

    # 2. Load vs Throttle concordancy
    if load > 85.0 and throttle < 25.0:
        anomalies.append({
            "sensor": "ENGINE_LOAD",
            "value": load,
            "status": "HIGH_LOAD_DISCORDANCE",
            "message": f"Engine load is very high ({load:.1f}%) despite low throttle opening ({throttle:.1f}%). Suggests mechanical drag, vacuum loss, or uphill straining."
        })

    # 3. RPM vs Speed concordancy (transmission / clutch slippage)
    if rpm > 3200.0 and speed < 5.0:
        anomalies.append({
            "sensor": "ENGINE_RPM",
            "value": rpm,
            "status": "HIGH_RPM_AT_REST",
            "message": f"Engine RPM is high ({rpm:.0f} RPM) while vehicle is stationary ({speed:.1f} km/h). Severe clutch slipping or neutral revving."
        })

    # 4. Intake Air Temperature
    if iat > 65.0:
        anomalies.append({
            "sensor": "AIR_INTAKE_TEMP",
            "value": iat,
            "status": "HIGH_INTAKE_TEMP",
            "message": f"Intake air temperature ({iat:.1f}°C) indicates severe engine bay heat soak. Air density and power will be restricted."
        })

    return anomalies


def analyze_vehicle(readings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Primary Vehicle AI Inference function.
    
    Parameters:
    - readings: dictionary containing:
        - ENGINE_RPM (float)
        - SPEED (float)
        - ENGINE_COOLANT_TEMP (float)
        - ENGINE_LOAD (float)
        - THROTTLE_POS (float)
        - AIR_INTAKE_TEMP (float)
        - TROUBLE_CODES or DTC_CODE (optional str, e.g. "P0133")
        - VEHICLE_ID (optional str)
        - MODEL (optional str)
    
    Returns:
    - Comprehensive vehicle maintenance risk assessment dictionary.
    """
    bundle = get_model_bundle()
    iso_forest = bundle["isolation_forest"]
    rf_clf = bundle["supervised_classifier"]
    scaler = bundle["scaler"]
    feature_names = bundle["features"]
    score_min = bundle["score_min"]
    score_max = bundle["score_max"]

    # Extract and cast core features
    core_inputs = {}
    for col in ["ENGINE_RPM", "SPEED", "ENGINE_COOLANT_TEMP", "ENGINE_LOAD", "THROTTLE_POS", "AIR_INTAKE_TEMP"]:
        val = readings.get(col, None)
        if val is None:
            raise ValueError(f"Missing required vehicle sensor: {col}")
        try:
            core_inputs[col] = float(val)
        except (ValueError, TypeError):
            raise ValueError(f"Sensor value for {col} must be numeric. Got: {val}")

    # Compute derived features
    rpm = core_inputs["ENGINE_RPM"]
    speed = core_inputs["SPEED"]
    coolant = core_inputs["ENGINE_COOLANT_TEMP"]
    load = core_inputs["ENGINE_LOAD"]
    throttle = core_inputs["THROTTLE_POS"]
    iat = core_inputs["AIR_INTAKE_TEMP"]

    speed_to_rpm = speed / (rpm + 1.0)
    load_to_throttle = load / (throttle + 1.0)
    temp_diff = coolant - iat
    is_idle = 1 if (rpm > 400.0 and speed < 3.0) else 0

    feature_dict = {
        "ENGINE_RPM": rpm,
        "SPEED": speed,
        "ENGINE_COOLANT_TEMP": coolant,
        "ENGINE_LOAD": load,
        "THROTTLE_POS": throttle,
        "AIR_INTAKE_TEMP": iat,
        "SPEED_TO_RPM": speed_to_rpm,
        "LOAD_TO_THROTTLE": load_to_throttle,
        "TEMP_DIFF": temp_diff,
        "IS_IDLE": is_idle
    }

    # Prepare vector for model inference
    X_input = pd.DataFrame([feature_dict])[feature_names]
    X_scaled = scaler.transform(X_input)

    # 1. Unsupervised Anomaly Scoring
    raw_anomaly_score = float(-iso_forest.decision_function(X_scaled)[0])
    # Calibrate risk score to 0 - 100%
    norm_risk = (raw_anomaly_score - score_min) / (score_max - score_min) * 100.0
    risk_score = float(np.clip(norm_risk, 0.0, 100.0))

    # 2. Supervised Diagnostic Prediction
    supervised_prob = float(rf_clf.predict_proba(X_scaled)[0, 1])
    supervised_pred = int(rf_clf.predict(X_scaled)[0])

    # 3. Physical sensor boundary analysis
    sensor_anomalies = inspect_sensor_anomalies(core_inputs)
    has_critical_sensor = any(a["status"] == "CRITICAL_OVERHEAT" for a in sensor_anomalies)

    # 4. DTC code analysis
    dtc_raw = readings.get("TROUBLE_CODES") or readings.get("DTC_CODE") or readings.get("DTC_NUMBER")
    dtc_info = parse_dtc_details(dtc_raw)

    # 5. Determine Overall Health Status & Maintenance Urgency
    if has_critical_sensor or (dtc_info and dtc_info.get("severity") == "CRITICAL") or risk_score >= 65.0 or supervised_prob >= 0.70:
        health_status = "CRITICAL RISK / FAULT DETECTED"
        health_category = "CRITICAL"
        action_urgency = "Immediate Service Required"
    elif len(sensor_anomalies) > 0 or dtc_info is not None or risk_score >= 38.0 or supervised_prob >= 0.40:
        health_status = "MODERATE RISK / ADVISORY"
        health_category = "ADVISORY"
        action_urgency = "Schedule Diagnostic Inspection"
    else:
        health_status = "HEALTHY OPERATING STATE"
        health_category = "NORMAL"
        action_urgency = "Normal Operation — Routine Maintenance Only"

    # 6. Synthesize Recommended Actions
    recommendations = []
    if dtc_info:
        recommendations.append(f"DTC [{dtc_info['code']}]: {dtc_info['action']}")
    for anom in sensor_anomalies:
        recommendations.append(f"Sensor Warning: {anom['message']}")
    if risk_score > 50.0 and not recommendations:
        recommendations.append("Multi-parameter operating discordance detected. Inspect air intake hoses, throttle body cleanliness, and spark plug wear.")
    if not recommendations:
        recommendations.append("All powertrain telemetry parameters within normal operating envelopes. Continue standard service intervals.")

    return {
        "status": "success",
        "health_status": health_status,
        "health_category": health_category,
        "action_urgency": action_urgency,
        "risk_score_pct": round(risk_score, 1),
        "fault_probability_pct": round(supervised_prob * 100.0, 1),
        "raw_anomaly_score": round(raw_anomaly_score, 4),
        "inputs": core_inputs,
        "derived_metrics": {
            "speed_to_rpm_ratio": round(speed_to_rpm, 4),
            "load_to_throttle_ratio": round(load_to_throttle, 2),
            "coolant_to_air_delta_c": round(temp_diff, 1),
            "is_engine_idling": bool(is_idle)
        },
        "sensor_anomalies": sensor_anomalies,
        "diagnostic_trouble_code": dtc_info,
        "recommendations": recommendations,
        "model_confidence": {
            "unsupervised_engine": "IsolationForest (150 trees)",
            "supervised_engine": "RandomForestClassifier (100 trees, balanced)"
        }
    }


if __name__ == "__main__":
    # Test nominal reading
    normal_case = {
        "ENGINE_RPM": 1600.0,
        "SPEED": 55.0,
        "ENGINE_COOLANT_TEMP": 88.0,
        "ENGINE_LOAD": 35.0,
        "THROTTLE_POS": 18.0,
        "AIR_INTAKE_TEMP": 32.0,
        "TROUBLE_CODES": None
    }
    print("Normal Highway Case:")
    res_normal = analyze_vehicle(normal_case)
    print(f"Status: {res_normal['health_status']} | Risk: {res_normal['risk_score_pct']}% | Supervised Fault Prob: {res_normal['fault_probability_pct']}%")

    # Test overheating and high load fault case
    fault_case = {
        "ENGINE_RPM": 3200.0,
        "SPEED": 15.0,
        "ENGINE_COOLANT_TEMP": 109.0,
        "ENGINE_LOAD": 92.0,
        "THROTTLE_POS": 22.0,
        "AIR_INTAKE_TEMP": 58.0,
        "TROUBLE_CODES": "P0133"
    }
    print("\nOverheating + O2 Sensor Fault Case:")
    res_fault = analyze_vehicle(fault_case)
    print(f"Status: {res_fault['health_status']} | Risk: {res_fault['risk_score_pct']}% | Supervised Fault Prob: {res_fault['fault_probability_pct']}%")
    print("Recommendations:", res_fault["recommendations"])

