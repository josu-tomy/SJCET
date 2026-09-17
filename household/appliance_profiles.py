"""
MachineGuard — Appliance Manual Monitoring Profiles & Reference Rules
Contains profile definitions, typical operating ranges, and reference heuristic rules
for the 6 supported household equipment types.

IMPORTANT NOTICE:
These reference thresholds represent prototype monitoring heuristics based on general engineering
rules of thumb, NOT manufacturer-certified safety limits or physical breakdown guarantees.
"""

from typing import Dict, Any, Tuple, List

APPLIANCE_PROFILES = {
    "Air Conditioner": {
        "icon": "❄️",
        "description": "Split / Window HVAC cooling equipment",
        "fields": [
            {"key": "room_temp", "label": "Room Temperature", "unit": "°C", "min": 15.0, "max": 45.0, "default": 26.0, "step": 0.5, "normal": (22.0, 30.0)},
            {"key": "set_temp", "label": "Set Temperature", "unit": "°C", "min": 16.0, "max": 30.0, "default": 24.0, "step": 0.5, "normal": (20.0, 26.0)},
            {"key": "power", "label": "Power Consumption", "unit": "W", "min": 200.0, "max": 4000.0, "default": 1400.0, "step": 50.0, "normal": (800.0, 2200.0)},
            {"key": "current", "label": "Operating Current", "unit": "A", "min": 1.0, "max": 20.0, "default": 6.5, "step": 0.1, "normal": (4.0, 10.0)},
            {"key": "vibration", "label": "Compressor Vibration", "unit": "mm/s RMS", "min": 0.0, "max": 15.0, "default": 2.0, "step": 0.1, "normal": (0.5, 4.5)},
        ],
    },
    "Refrigerator": {
        "icon": "🧊",
        "description": "Domestic food preservation and refrigeration unit",
        "fields": [
            {"key": "internal_temp", "label": "Internal Temperature", "unit": "°C", "min": -5.0, "max": 20.0, "default": 4.0, "step": 0.5, "normal": (1.0, 6.0)},
            {"key": "ambient_temp", "label": "Ambient Room Temp", "unit": "°C", "min": 10.0, "max": 45.0, "default": 25.0, "step": 0.5, "normal": (18.0, 32.0)},
            {"key": "power", "label": "Active Power Consumption", "unit": "W", "min": 0.0, "max": 800.0, "default": 120.0, "step": 10.0, "normal": (80.0, 250.0)},
            {"key": "compressor_state", "label": "Compressor State", "unit": "State", "type": "select", "options": ["Running", "Idle"], "default": "Running"},
            {"key": "door_open_sec", "label": "Door Open Duration", "unit": "sec", "min": 0.0, "max": 3600.0, "default": 15.0, "step": 5.0, "normal": (0.0, 60.0)},
        ],
    },
    "Washing Machine": {
        "icon": "🧺",
        "description": "Automatic washing and high-speed spin drying machine",
        "fields": [
            {"key": "motor_speed", "label": "Motor Drum Speed", "unit": "rpm", "min": 0.0, "max": 1600.0, "default": 800.0, "step": 50.0, "normal": (400.0, 1200.0)},
            {"key": "water_temp", "label": "Water Temperature", "unit": "°C", "min": 10.0, "max": 90.0, "default": 40.0, "step": 1.0, "normal": (20.0, 60.0)},
            {"key": "current", "label": "Motor Current", "unit": "A", "min": 0.5, "max": 16.0, "default": 4.0, "step": 0.2, "normal": (2.0, 8.0)},
            {"key": "vibration", "label": "Chassis Vibration", "unit": "mm/s RMS", "min": 0.0, "max": 25.0, "default": 3.5, "step": 0.5, "normal": (1.0, 8.0)},
            {"key": "cycle_duration", "label": "Cycle Duration", "unit": "min", "min": 5.0, "max": 180.0, "default": 45.0, "step": 5.0, "normal": (20.0, 90.0)},
        ],
    },
    "Television": {
        "icon": "📺",
        "description": "Smart LED/OLED display equipment",
        "fields": [
            {"key": "power", "label": "Operating Power", "unit": "W", "min": 0.5, "max": 500.0, "default": 85.0, "step": 5.0, "normal": (40.0, 180.0)},
            {"key": "operating_temp", "label": "Chassis/Panel Temperature", "unit": "°C", "min": 15.0, "max": 80.0, "default": 38.0, "step": 1.0, "normal": (25.0, 50.0)},
            {"key": "operating_hours", "label": "Daily Operating Hours", "unit": "hrs/day", "min": 0.5, "max": 24.0, "default": 5.0, "step": 0.5, "normal": (1.0, 10.0)},
            {"key": "state", "label": "Operating State", "unit": "State", "type": "select", "options": ["Active Display", "Standby"], "default": "Active Display"},
        ],
    },
    "Water Pump": {
        "icon": "💧",
        "description": "Domestic centrifugal water pumping system",
        "fields": [
            {"key": "current", "label": "Motor Current", "unit": "A", "min": 1.0, "max": 25.0, "default": 5.5, "step": 0.1, "normal": (3.5, 8.0)},
            {"key": "pressure", "label": "Discharge Pressure", "unit": "bar", "min": 0.5, "max": 10.0, "default": 3.2, "step": 0.1, "normal": (2.0, 5.0)},
            {"key": "flow_rate", "label": "Flow Rate", "unit": "L/min", "min": 0.0, "max": 120.0, "default": 45.0, "step": 1.0, "normal": (25.0, 75.0)},
            {"key": "vibration", "label": "Pump Head Vibration", "unit": "mm/s RMS", "min": 0.0, "max": 20.0, "default": 2.8, "step": 0.2, "normal": (1.0, 5.5)},
            {"key": "temperature", "label": "Motor Casing Temperature", "unit": "°C", "min": 20.0, "max": 110.0, "default": 55.0, "step": 1.0, "normal": (35.0, 75.0)},
        ],
    },
    "Ceiling Fan": {
        "icon": "🌀",
        "description": "Brushless DC / Induction ceiling ventilation fan",
        "fields": [
            {"key": "speed", "label": "Fan Speed", "unit": "rpm", "min": 50.0, "max": 450.0, "default": 280.0, "step": 10.0, "normal": (150.0, 380.0)},
            {"key": "power", "label": "Power Consumption", "unit": "W", "min": 10.0, "max": 150.0, "default": 55.0, "step": 2.0, "normal": (25.0, 75.0)},
            {"key": "temperature", "label": "Motor Housing Temperature", "unit": "°C", "min": 20.0, "max": 95.0, "default": 42.0, "step": 1.0, "normal": (30.0, 65.0)},
            {"key": "vibration", "label": "Wobble / Vibration", "unit": "mm/s RMS", "min": 0.0, "max": 15.0, "default": 1.5, "step": 0.2, "normal": (0.5, 3.5)},
            {"key": "operating_hours", "label": "Operating Hours", "unit": "hrs/day", "min": 0.5, "max": 24.0, "default": 8.0, "step": 0.5, "normal": (2.0, 16.0)},
        ],
    },
}


def evaluate_manual_appliance(appliance_name: str, readings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates manually entered sensor parameters against prototype reference ranges.
    Returns status (NORMAL, WARNING, ANOMALY), flagged abnormalities, and inspection recommendations.
    """
    if appliance_name not in APPLIANCE_PROFILES:
        raise KeyError(f"Unknown appliance profile: {appliance_name}")

    profile = APPLIANCE_PROFILES[appliance_name]
    abnormalities = []
    warnings = []

    for field in profile["fields"]:
        k = field["key"]
        val = readings.get(k)
        if val is None or "normal" not in field:
            continue

        low, high = field["normal"]
        label = field["label"]
        unit = field["unit"]

        # Evaluation thresholds: >1.5x normal or <0.5x normal indicates anomaly; 1.0-1.5x indicates warning
        if val > high:
            severity = (val - high) / (high - low)
            if severity > 0.4:
                abnormalities.append(f"{label} ({val} {unit}) severely exceeds reference ceiling ({high} {unit})")
            else:
                warnings.append(f"{label} ({val} {unit}) exceeds expected nominal range ({low}–{high} {unit})")
        elif val < low:
            severity = (low - val) / (high - low)
            if severity > 0.4:
                abnormalities.append(f"{label} ({val} {unit}) falls significantly below reference baseline ({low} {unit})")
            else:
                warnings.append(f"{label} ({val} {unit}) is below expected nominal range ({low}–{high} {unit})")

    # Appliance-Specific Logic Combinations
    if appliance_name == "Air Conditioner":
        # Check delta T: if room temp is much higher than set temp despite high power, cooling failure suspected
        room_t = readings.get("room_temp", 26.0)
        set_t = readings.get("set_temp", 24.0)
        pwr = readings.get("power", 1400.0)
        if (room_t - set_t) >= 5.0 and pwr > 1800.0:
            abnormalities.append("Ineffective Cooling: High power draw with inability to achieve set temperature (refrigerant leak or compressor inefficiency).")

    elif appliance_name == "Refrigerator":
        door = readings.get("door_open_sec", 0.0)
        t_int = readings.get("internal_temp", 4.0)
        if door > 120.0:
            warnings.append(f"Prolonged door ajar ({door:.0f}s) causing thermal dissipation.")
        if t_int > 10.0:
            abnormalities.append(f"Internal food zone temperature critical ({t_int}°C). Thermostat or sealed system fault.")

    elif appliance_name == "Washing Machine":
        vib = readings.get("vibration", 3.0)
        speed = readings.get("motor_speed", 800.0)
        if vib > 10.0:
            abnormalities.append(f"Severe mechanical vibration ({vib} mm/s). Unbalanced drum load or defective suspension shock absorbers.")

    elif appliance_name == "Water Pump":
        p = readings.get("pressure", 3.0)
        fl = readings.get("flow_rate", 45.0)
        cur = readings.get("current", 5.0)
        if cur > 9.0 and fl < 10.0:
            abnormalities.append("Deadhead / Impeller lock: High current draw with near-zero flow.")
        elif cur < 3.0 and fl < 5.0:
            abnormalities.append("Dry Run: Extremely low current and flow indicates suction cavitation or dry well.")

    elif appliance_name == "Ceiling Fan":
        vib = readings.get("vibration", 1.5)
        if vib > 5.0:
            abnormalities.append("Excessive blade wobble/vibration: Blade aerodynamic unbalance or loose downrod bearing.")

    # Status Determination
    if abnormalities:
        status = "ANOMALY"
        badge = "🔴 ANOMALY"
        explanation = "One or more operating parameters significantly violate prototype reference operating boundaries."
        recommendation = "Schedule visual and electrical inspection. Check electrical contacts, thermal clearances, and mechanical mountings."
    elif warnings:
        status = "WARNING"
        badge = "🟡 WARNING"
        explanation = "Operating parameters show moderate deviations from typical nominal baseline."
        recommendation = "Monitor equipment trends over next operating cycle; verify load balance and clean ventilation filters."
    else:
        status = "NORMAL"
        badge = "🟢 NORMAL"
        explanation = "All entered measurements fall squarely within standard reference operating bounds."
        recommendation = "No immediate maintenance action required. Continue standard operational schedule."

    return {
        "status": status,
        "badge": badge,
        "abnormalities": abnormalities,
        "warnings": warnings,
        "explanation": explanation,
        "recommendation": recommendation,
        "disclaimer": "Prototype reference heuristics only — not manufacturer certified specifications.",
    }

