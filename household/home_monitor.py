"""
MachineGuard — Home Equipment Monitor & Transparent NLP Symptom Parser
Coordinates appliance tracking and transparent text symptom extraction.

IMPORTANT:
Does NOT claim an AI/NLP model has learned natural language.
Uses transparent rule-based keyword extraction and explicitly reports:
'Numeric measurement not provided' rather than fabricating fake numbers.
"""

import re
from typing import Dict, Any, List, Optional
import datetime


# Keyword mappings for transparent rule-based extraction
APPLIANCE_KEYWORDS = {
    "Air Conditioner": ["ac", "air conditioner", "air conditioning", "cooling", "chiller", "hvac", "thermostat"],
    "Refrigerator": ["fridge", "refrigerator", "freezer", "compressor", "cooling box", "chilling"],
    "Washing Machine": ["washing machine", "washer", "spin cycle", "drum", "laundry", "wash cycle"],
    "Television": ["tv", "television", "screen", "display", "monitor", "smart tv"],
    "Water Pump": ["water pump", "pump", "motor pump", "submersible", "impeller", "pressure pump"],
    "Ceiling Fan": ["ceiling fan", "fan", "regulator", "blade", "wobble"],
}

SYMPTOM_PATTERNS = [
    {
        "pattern": r"(vibrat\w+|shak\w+|wobbl\w+|rattl\w+)",
        "signal": "Vibration",
        "description": "Mechanical vibration / oscillation observed",
        "concern": "Unbalanced rotating component, loose mounting, or bearing wear",
    },
    {
        "pattern": r"(hot\w*|overheat\w*|warm\w*|burning|heat\w*)",
        "signal": "Temperature (High)",
        "description": "Elevated thermal condition reported",
        "concern": "Inadequate heat dissipation, overload, or cooling system failure",
    },
    {
        "pattern": r"(not cooling|warm room|no cool\w*|insufficient cool\w*)",
        "signal": "Cooling Deficiency",
        "description": "Inability to maintain target temperature gradient",
        "concern": "Refrigerant loss, compressor valve defect, or thermal leak",
    },
    {
        "pattern": r"(power|electric\w*|bill|current|tripping|short\w*|spark\w*|draw\w*|consum\w+)",
        "signal": "Power / Electrical",
        "description": "High electrical draw or power fluctuation noted",
        "concern": "Electrical inefficiency, internal short, or winding degradation",
    },
    {
        "pattern": r"(continuous\w*|constantly|non-stop|always running|never stops|won't stop)",
        "signal": "Continuous Duty Cycle",
        "description": "Appliance running continuously without cycling off",
        "concern": "Thermostat failure, sensor fault, or inability to satisfy setpoint",
    },
    {
        "pattern": r"(noise|sound|loud|squeak\w*|grind\w*|humming|buzz\w*)",
        "signal": "Acoustic Noise",
        "description": "Abnormal acoustic emission reported",
        "concern": "Mechanical friction, cavitation, or loose housing",
    },
    {
        "pattern": r"(normal\w*|fine|working well|good condition|no problem\w*|ok|okay)",
        "signal": "Nominal Condition",
        "description": "No adverse operational symptoms reported",
        "concern": "Normal operational observation",
    },
]


def parse_natural_language_symptom(text: str) -> Dict[str, Any]:
    """
    Transparently parses user text descriptions into detected equipment,
    symptoms, and concerns without fabricating any numeric values.
    """
    if not text or not text.strip():
        return {
            "has_input": False,
            "detected_appliance": None,
            "detected_symptoms": [],
            "detected_concerns": [],
            "missing_measurements": ["Numeric measurement not provided."],
            "is_normal": False,
        }

    clean_text = text.lower().strip()

    # 1. Detect Appliance
    detected_appliance = None
    for app_name, keywords in APPLIANCE_KEYWORDS.items():
        for kw in keywords:
            # Word boundary matching
            if re.search(rf"\b{re.escape(kw)}\b", clean_text):
                detected_appliance = app_name
                break
        if detected_appliance:
            break

    # 2. Detect Symptoms & Concerns
    detected_symptoms = []
    detected_concerns = []
    is_normal = False

    for item in SYMPTOM_PATTERNS:
        match = re.search(item["pattern"], clean_text)
        if match:
            if item["signal"] == "Nominal Condition":
                is_normal = True
            else:
                detected_symptoms.append(f"{item['signal']}: {item['description']}")
                detected_concerns.append(item["concern"])

    # Fallback if no specific symptom was detected
    if not detected_symptoms and not is_normal:
        detected_symptoms.append("Unspecified user-reported observation")
        detected_concerns.append("Requires quantitative parameter check to verify condition")

    # Explicit statement on missing measurements
    missing_measurements = [
        "Numeric measurement not provided in natural-language description.",
        "Please confirm or enter sensor values below for quantitative evaluation.",
    ]

    return {
        "has_input": True,
        "raw_text": text,
        "detected_appliance": detected_appliance,
        "detected_symptoms": detected_symptoms,
        "detected_concerns": list(set(detected_concerns)),
        "missing_measurements": missing_measurements,
        "is_normal": is_normal and len(detected_symptoms) == 0,
    }


# Centralized Session State key for household equipment
SESSION_KEY = "household_equipment_status"


def format_last_updated(timestamp: Optional[datetime.datetime]) -> str:
    """Formats a datetime object to 'DD Mon YYYY • HH:MM AM/PM', or returns '—' if None."""
    if timestamp is None:
        return "—"
    return timestamp.strftime("%d %b %Y • %I:%M %p")


def create_initial_equipment_state() -> Dict[str, Any]:
    """Generates a clean un-analyzed state for equipment before manual readings are entered."""
    return {
        "status": "NOT ANALYZED",
        "badge": "⚪ NOT ANALYZED",
        "last_reading": "No reading yet",
        "last_updated": "—",
        "last_updated_dt": None,
        "summary": "No reading yet",
        "anomaly": "No inspection recorded",
        "explanation": "No manual telemetry readings entered in this session.",
        "recommendation": "Select this equipment in 'Check Equipment' to enter manual readings.",
        "readings": {},
        "abnormalities": [],
        "warnings": [],
    }


def initialize_household_state(appliance_list: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Initializes st.session_state[SESSION_KEY] as the single source of truth.
    Preserves all existing states across Streamlit reruns.
    """
    import streamlit as st

    if SESSION_KEY not in st.session_state:
        default_apps = appliance_list or [
            "Air Conditioner",
            "Refrigerator",
            "Washing Machine",
            "Television",
            "Water Pump",
            "Ceiling Fan",
        ]
        st.session_state[SESSION_KEY] = {app: create_initial_equipment_state() for app in default_apps}

    return st.session_state[SESSION_KEY]


def save_equipment_analysis(
    appliance: str,
    eval_result: Dict[str, Any],
    readings: Dict[str, Any],
    summary_str: str,
    timestamp: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """
    Updates session state for a specific appliance at the exact moment of analysis.
    Leaves all other appliances completely untouched.
    """
    state_map = initialize_household_state()

    now_dt = timestamp or datetime.datetime.now()
    formatted_ts = format_last_updated(now_dt)

    anomaly_str = eval_result["abnormalities"][0] if eval_result.get("abnormalities") else (
        "Minor deviation" if eval_result.get("warnings") else "None detected"
    )

    app_record = {
        "status": eval_result["status"],
        "badge": eval_result["badge"],
        "last_reading": summary_str,
        "last_updated": formatted_ts,
        "last_updated_dt": now_dt,
        "summary": summary_str,
        "anomaly": anomaly_str,
        "explanation": eval_result.get("explanation", ""),
        "recommendation": eval_result.get("recommendation", ""),
        "readings": dict(readings),
        "abnormalities": list(eval_result.get("abnormalities", [])),
        "warnings": list(eval_result.get("warnings", [])),
    }

    state_map[appliance] = app_record
    return app_record


def get_equipment_status(appliance: str) -> Dict[str, Any]:
    """Retrieves current session state for an appliance."""
    state_map = initialize_household_state()
    return state_map.get(appliance, create_initial_equipment_state())


