"""
MachineGuard — AI-Powered Equipment Monitoring
Unified Multi-Domain Diagnostic & Predictive Maintenance Dashboard

College Hackathon Prototype — Professional Light Theme Edition
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import io
import json
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Core MachineGuard Pipeline Imports
from predict import predict_risk, FEATURES as INDUSTRIAL_FEATURES
from household.predict_household import predict_household, load_household_model
from household.appliance_profiles import APPLIANCE_PROFILES, evaluate_manual_appliance
from household.home_monitor import (
    parse_natural_language_symptom,
    initialize_household_state,
    save_equipment_analysis,
    get_equipment_status,
    format_last_updated,
    SESSION_KEY,
)
from household.sensor_interface import sensor_gateway

# Vehicle AI Pipeline Imports
from vehicle.predict_vehicle import (
    analyze_vehicle,
    parse_dtc_details,
    SENSOR_NOMINAL_RANGES,
)
from vehicle.vehicle_monitor import (
    analyze_batch_telemetry,
    PRESET_VEHICLE_SCENARIOS,
    get_available_presets,
    get_preset_scenario,
    normalize_column_names,
)

# Base File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDUSTRIAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
HOUSEHOLD_MODEL_PATH = os.path.join(BASE_DIR, "models", "household_model.joblib")
VEHICLE_MODEL_PATH = os.path.join(BASE_DIR, "models", "vehicle_model.joblib")

CM_IMAGE_PATH = os.path.join(BASE_DIR, "outputs", "confusion_matrix.png")
FI_IMAGE_PATH = os.path.join(BASE_DIR, "outputs", "feature_importance.png")

VEHICLE_METRICS_PATH = os.path.join(BASE_DIR, "outputs", "vehicle", "evaluation_metrics.json")
VEHICLE_FI_PATH = os.path.join(BASE_DIR, "outputs", "vehicle", "feature_importance.png")
VEHICLE_ANOM_PATH = os.path.join(BASE_DIR, "outputs", "vehicle", "anomaly_score_distribution.png")
VEHICLE_CM_PATH = os.path.join(BASE_DIR, "outputs", "vehicle", "supervised_confusion_matrix.png")
VEHICLE_CLEAN_CSV_PATH = os.path.join(BASE_DIR, "data", "vehicle", "processed", "vehicle_clean.csv")
VEHICLE_TEST_CSV_PATH = os.path.join(BASE_DIR, "data", "vehicle", "test_vehicle_obd.csv")

# Benchmark Training Data Ranges for Industrial Model (UCI AI4I 2020 Dataset)
INDUSTRIAL_RANGES = {
    "Air temperature [K]": {"min": 295.3, "max": 304.5, "unit": "K", "label": "Air Temperature"},
    "Process temperature [K]": {"min": 305.7, "max": 313.8, "unit": "K", "label": "Process Temperature"},
    "Rotational speed [rpm]": {"min": 1168.0, "max": 2886.0, "unit": "rpm", "label": "Rotational Speed"},
    "Torque [Nm]": {"min": 3.8, "max": 76.6, "unit": "Nm", "label": "Torque"},
    "Tool wear [min]": {"min": 0.0, "max": 253.0, "unit": "min", "label": "Tool Wear"},
}

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & INJECTED PROFESSIONAL LIGHT STYLESHEET
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MachineGuard — Equipment Monitoring",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Injected Light Theme CSS
st.markdown(
    """
    <style>
    /* Global Background & Typography */
    .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }
    
    .block-container {
        padding-top: 2.0rem !important;
        padding-bottom: 3.0rem !important;
        max-width: 1120px !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif !important;
        color: #0F172A !important;
        letter-spacing: -0.015em !important;
    }

    p, span, label, div {
        color: #0F172A;
    }

    .stCaption, small, [data-testid="stCaptionContainer"] p {
        color: #475569 !important;
        font-size: 0.88rem !important;
    }

    button, button * {
        text-shadow: none !important;
    }
    
    /* ------------------------------------------------------------------------- */
    /* NAVIGATION BAR (LIGHT BUTTON STYLE)                                       */
    /* ------------------------------------------------------------------------- */
    div[data-testid="stSegmentedControl"] {
        display: flex !important;
        justify-content: center !important;
        background: #FFFFFF !important;
        padding: 6px !important;
        border-radius: 10px !important;
        border: 1px solid #CBD5E1 !important;
        margin-top: 0.5rem !important;
        margin-bottom: 2.0rem !important;
        width: 100% !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    }
    
    div[data-testid="stSegmentedControl"] button {
        border-radius: 7px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 8px 16px !important;
        border: 1px solid transparent !important;
        transition: all 0.15s ease-in-out !important;
        flex-grow: 1 !important;
        margin: 0 3px !important;
    }
    
    /* Inactive Navigation Button */
    div[data-testid="stSegmentedControl"] button[aria-checked="false"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #E2E8F0 !important;
    }
    
    div[data-testid="stSegmentedControl"] button[aria-checked="false"] * {
        color: #1E293B !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stSegmentedControl"] button[aria-checked="false"]:hover {
        background-color: #F1F5F9 !important;
        background: #F1F5F9 !important;
        color: #0F172A !important;
        border-color: #CBD5E1 !important;
    }
    
    div[data-testid="stSegmentedControl"] button[aria-checked="false"]:hover * {
        color: #0F172A !important;
    }
    
    /* Active Navigation Button (Dark Blue + Pure White Text) */
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background-color: #1D4ED8 !important;
        background: #1D4ED8 !important;
        color: #FFFFFF !important;
        border: 1px solid #1E40AF !important;
        box-shadow: 0 2px 4px rgba(29, 78, 216, 0.25) !important;
    }
    
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] * {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* ------------------------------------------------------------------------- */
    /* PRIMARY ACTION BUTTONS (DARK BLUE + PURE WHITE TEXT)                      */
    /* ------------------------------------------------------------------------- */
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="baseButton-primary"] {
        background-color: #1D4ED8 !important;
        background: #1D4ED8 !important;
        color: #FFFFFF !important;
        border: 1.5px solid #1E40AF !important;
        border-radius: 8px !important;
        padding: 9px 24px !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(29, 78, 216, 0.20) !important;
        transition: all 0.15s ease-in-out !important;
        cursor: pointer !important;
    }
    
    div.stButton > button[kind="primary"] *,
    div.stButton > button[data-testid="baseButton-primary"] * {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="baseButton-primary"]:hover {
        background-color: #1E3A8A !important;
        background: #1E3A8A !important;
        color: #FFFFFF !important;
        border-color: #172554 !important;
        box-shadow: 0 3px 6px rgba(29, 78, 216, 0.30) !important;
    }
    
    div.stButton > button[kind="primary"]:hover *,
    div.stButton > button[data-testid="baseButton-primary"]:hover * {
        color: #FFFFFF !important;
    }
    
    div.stButton > button[kind="primary"]:focus,
    div.stButton > button[kind="primary"]:active,
    div.stButton > button[data-testid="baseButton-primary"]:focus,
    div.stButton > button[data-testid="baseButton-primary"]:active {
        background-color: #172554 !important;
        background: #172554 !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 0 2px rgba(29, 78, 216, 0.40) !important;
    }
    
    div.stButton > button[kind="primary"]:focus *,
    div.stButton > button[kind="primary"]:active *,
    div.stButton > button[data-testid="baseButton-primary"]:focus *,
    div.stButton > button[data-testid="baseButton-primary"]:active * {
        color: #FFFFFF !important;
    }

    /* ------------------------------------------------------------------------- */
    /* SECONDARY ACTION BUTTONS (WHITE + DARK CHARCOAL TEXT + VISIBLE BORDER)    */
    /* ------------------------------------------------------------------------- */
    div.stButton > button[kind="secondary"],
    div.stButton > button[data-testid="baseButton-secondary"],
    div.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]) {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        padding: 7px 16px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.15s ease-in-out !important;
        cursor: pointer !important;
    }
    
    div.stButton > button[kind="secondary"] *,
    div.stButton > button[data-testid="baseButton-secondary"] *,
    div.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]) * {
        color: #0F172A !important;
        font-weight: 600 !important;
    }
    
    div.stButton > button[kind="secondary"]:hover,
    div.stButton > button[data-testid="baseButton-secondary"]:hover,
    div.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]):hover {
        background-color: #F1F5F9 !important;
        background: #F1F5F9 !important;
        color: #0F172A !important;
        border-color: #94A3B8 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06) !important;
    }
    
    div.stButton > button[kind="secondary"]:hover *,
    div.stButton > button[data-testid="baseButton-secondary"]:hover *,
    div.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]):hover * {
        color: #0F172A !important;
    }
    
    div.stButton > button[kind="secondary"]:focus,
    div.stButton > button[kind="secondary"]:active,
    div.stButton > button[data-testid="baseButton-secondary"]:focus,
    div.stButton > button[data-testid="baseButton-secondary"]:active {
        background-color: #E2E8F0 !important;
        background: #E2E8F0 !important;
        color: #0F172A !important;
        border-color: #64748B !important;
    }
    
    div.stButton > button[kind="secondary"]:focus *,
    div.stButton > button[kind="secondary"]:active *,
    div.stButton > button[data-testid="baseButton-secondary"]:focus *,
    div.stButton > button[data-testid="baseButton-secondary"]:active * {
        color: #0F172A !important;
    }

    /* ------------------------------------------------------------------------- */
    /* WHITE CONTENT CARDS & FRAMING                                             */
    /* ------------------------------------------------------------------------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 10px !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        padding: 10px !important;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #CBD5E1 !important;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.06) !important;
    }
    
    /* ------------------------------------------------------------------------- */
    /* EXPANDERS (WHITE CARD + CRISP BORDER)                                     */
    /* ------------------------------------------------------------------------- */
    div[data-testid="stExpander"] {
        border-radius: 8px !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #FFFFFF !important;
        margin-top: 12px !important;
        box-shadow: none !important;
    }
    
    div[data-testid="stExpander"] details summary {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
    }
    
    div[data-testid="stExpander"] details summary:hover {
        color: #1D4ED8 !important;
    }

    /* ------------------------------------------------------------------------- */
    /* METRIC CARDS                                                              */
    /* ------------------------------------------------------------------------- */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        padding: 12px 16px !important;
        border-radius: 8px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }

    /* ------------------------------------------------------------------------- */
    /* INPUT FIELDS, SELECTBOXES & NUMBER INPUTS                                 */
    /* ------------------------------------------------------------------------- */
    input, textarea {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border-radius: 6px !important;
        border: 1px solid #CBD5E1 !important;
    }

    input:focus, textarea:focus {
        border-color: #1D4ED8 !important;
        box-shadow: 0 0 0 1px #1D4ED8 !important;
    }

    div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
    }

    div[data-baseweb="input"] input {
        color: #0F172A !important;
        background-color: #FFFFFF !important;
    }

    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border-radius: 6px !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
    }

    div[data-baseweb="select"] span {
        color: #0F172A !important;
    }

    /* File Uploader */
    div[data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
    }

    div[data-testid="stFileUploader"] section {
        background-color: #F8FAFC !important;
        border: 1.5px dashed #CBD5E1 !important;
        border-radius: 8px !important;
    }

    div[data-testid="stFileUploader"] section:hover {
        border-color: #1D4ED8 !important;
        background-color: #F1F5F9 !important;
    }

    div[data-testid="stFileUploader"] button {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 1.5px solid #94A3B8 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }

    div[data-testid="stFileUploader"] button * {
        color: #0F172A !important;
        font-weight: 600 !important;
    }

    div[data-testid="stFileUploader"] button:hover {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        border-color: #64748B !important;
    }

    div[data-testid="stFileUploader"] button:hover * {
        color: #0F172A !important;
    }

    /* Architecture Code Box */
    pre, code {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 8px !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS: TEXT STATUS BADGES & TIMESTAMPS (NO EMOJIS / NO DOTS)
# -----------------------------------------------------------------------------
def render_badge(status_text: str, category: str = "NORMAL") -> str:
    """Generates a clean text status badge with semantic coloring and visible border (no dots or emojis)."""
    cat = (category or "").upper()
    clean_txt = str(status_text or "")
    for sym in ["●", "⚪", "🟢", "🟡", "🔴", "▸", "⚠️", "🚨", "🛡️", "•", "✔"]:
        clean_txt = clean_txt.replace(sym, "")
    display_txt = clean_txt.strip()

    if "NORMAL" in cat or "HEALTHY" in cat:
        bg = "#ECFDF5"
        border = "#059669"
        color = "#065F46"
        if not display_txt:
            display_txt = "NORMAL"
    elif "WARN" in cat or "ADVISORY" in cat:
        bg = "#FFFBEB"
        border = "#D97706"
        color = "#92400E"
        if not display_txt:
            display_txt = "WARNING"
    elif "CRITICAL" in cat or "ANOMALY" in cat or "HIGH" in cat or "FAIL" in cat:
        bg = "#FEF2F2"
        border = "#DC2626"
        color = "#991B1B"
        if not display_txt:
            display_txt = "ANOMALY" if "ANOMALY" in cat else "HIGH RISK"
    else:
        bg = "#F8FAFC"
        border = "#CBD5E1"
        color = "#475569"
        if not display_txt:
            display_txt = "NOT ANALYZED"

    return (
        f'<div style="display: inline-block; padding: 4px 12px; '
        f'border-radius: 6px; background: {bg}; border: 1.5px solid {border}; color: {color}; '
        f'font-size: 0.80rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;">'
        f'{display_txt}</div>'
    )


def format_card_timestamp(ts) -> str:
    """Formats timestamps into human-readable strings like 'Updated 17 Sep 2026 • 03:08 PM' or '—'."""
    if not ts or str(ts).strip() in ("—", "None", "", "No inspection recorded"):
        return "—"
    ts_str = str(ts).strip()
    if ts_str.startswith("Updated "):
        return ts_str
    if isinstance(ts, (datetime.datetime, datetime.date)):
        return f"Updated {ts.strftime('%d %b %Y • %I:%M %p')}"
    return f"Updated {ts_str}"


# -----------------------------------------------------------------------------
# CENTRALIZED SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
# 1. Household Equipment State (Preserves existing analysis across reruns)
household_status_map = initialize_household_state(list(APPLIANCE_PROFILES.keys()))

# 2. Vehicle Telemetry State
if "vehicle_status" not in st.session_state:
    st.session_state["vehicle_status"] = {
        "status": "NOT ANALYZED",
        "badge_text": "NOT ANALYZED",
        "category": "NOT ANALYZED",
        "last_updated": "—",
        "summary": "No inspection recorded",
        "latest_reading": "No reading yet",
        "risk_score": None,
        "fault_probability": None,
        "raw_anomaly_score": None,
        "recommendations": [],
        "explanation": "No vehicle telemetry analyzed in this session.",
        "readings": {},
        "result": None,
    }

# 3. Industrial Equipment State
if "industrial_status" not in st.session_state:
    st.session_state["industrial_status"] = {
        "status": "NOT ANALYZED",
        "badge_text": "NOT ANALYZED",
        "category": "NOT ANALYZED",
        "last_updated": "—",
        "prob": None,
        "pred": None,
        "summary": "No inspection recorded",
        "explanation": "No machine telemetry analyzed in this session.",
        "readings": {},
    }

# 4. Clean Navigation State & Programmatic Navigation Handler
NAV_OPTIONS = [
    "Home Overview",
    "Check Equipment",
    "Energy Behavior",
    "Vehicle AI",
    "Industrial AI",
    "Sensor Integration",
]

# Consume any pending programmatic navigation BEFORE the navigation widget is instantiated
if "pending_nav" in st.session_state:
    st.session_state["nav_bar"] = st.session_state.pop("pending_nav")

if "nav_bar" not in st.session_state or st.session_state["nav_bar"] not in NAV_OPTIONS:
    st.session_state["nav_bar"] = "Home Overview"


# -----------------------------------------------------------------------------
# APPLICATION HEADER & NAVIGATION BAR
# -----------------------------------------------------------------------------
header_col1, header_col2 = st.columns([0.7, 0.3])
with header_col1:
    st.title("MachineGuard")
    st.caption("AI-Powered Equipment Monitoring & Predictive Maintenance Dashboard")

# Navigation Segmented Control Bar (High-Contrast Button System)
selected_nav = st.segmented_control(
    "Navigation Menu",
    NAV_OPTIONS,
    label_visibility="collapsed",
    key="nav_bar",
)

current_page = selected_nav if selected_nav else st.session_state.get("nav_bar", "Home Overview")


# =============================================================================
# HELPER: HOME OVERVIEW APPLIANCE CARD RENDERER
# =============================================================================
def render_appliance_card(app_name: str, state: dict, profile: dict):
    """Renders a clean Home Overview card for an appliance with actual readings when analyzed."""
    status_cat = state.get("status", "NOT ANALYZED")
    badge_text = state.get("badge", "NOT ANALYZED")
    last_updated = state.get("last_updated", "—")
    readings = state.get("readings", {})
    explanation = state.get("explanation", "")

    with st.container(border=True):
        st.markdown(f"#### {app_name}")
        st.markdown(render_badge(badge_text, status_cat), unsafe_allow_html=True)
        st.caption(format_card_timestamp(last_updated))

        # Only display readings and analysis summary if the appliance has actually been analyzed
        if status_cat != "NOT ANALYZED" and readings:
            fields = profile.get("fields", [])
            lines = []
            for field in fields:
                k = field["key"]
                if k in readings:
                    val = readings[k]
                    unit = field.get("unit", "")
                    if isinstance(val, float):
                        val_str = f"{int(val)}" if val.is_integer() else f"{val:.1f}"
                    else:
                        val_str = str(val)
                    if unit and unit != "State":
                        val_str = f"{val_str} {unit}"
                    lines.append(
                        f'<div style="display: flex; justify-content: space-between; padding: 2px 0; border-bottom: 1px solid #F1F5F9; font-size: 0.83rem;">'
                        f'<span style="color: #475569;">{field["label"]}</span>'
                        f'<span style="font-weight: 600; color: #0F172A;">{val_str}</span></div>'
                    )

            if lines:
                st.markdown(
                    f'<div style="margin: 10px 0; border-top: 1px solid #E2E8F0; padding-top: 4px;">'
                    f'{"".join(lines)}</div>',
                    unsafe_allow_html=True,
                )

            if explanation and explanation != "No manual telemetry readings entered in this session.":
                st.markdown(
                    f'<div style="margin-top: 8px; margin-bottom: 10px; font-size: 0.82rem; line-height: 1.4;">'
                    f'<strong style="color: #0F172A;">Analysis:</strong><br>'
                    f'<span style="color: #475569;">{explanation}</span></div>',
                    unsafe_allow_html=True,
                )

        if st.button("Analyze", key=f"btn_jump_{app_name}", use_container_width=True):
            st.session_state["pending_nav"] = "Check Equipment"
            st.session_state["selected_appliance_override"] = app_name
            st.rerun()


# =============================================================================
# SECTION 1: HOME OVERVIEW (EXECUTIVE LANDING PAGE)
# =============================================================================
if current_page == "Home Overview":
    st.markdown("### Equipment Status Overview")
    st.caption("Operational status and diagnostic screening across all monitored assets.")

    # 1. Household Equipment (6 Clean Cards in 3 Columns)
    st.markdown("##### Household Appliances")
    cols_h = st.columns(3)
    app_names = list(APPLIANCE_PROFILES.keys())

    for idx, name in enumerate(app_names):
        state = household_status_map.get(name, {})
        profile = APPLIANCE_PROFILES.get(name, {})
        col_idx = idx % 3

        with cols_h[col_idx]:
            render_appliance_card(name, state, profile)

    # 2. Fleet Telemetry & Industrial Machinery (2 Wide Cards)
    st.markdown("##### Fleet & Industrial Assets")
    col_v, col_ind = st.columns(2)

    # Vehicle Card
    v_state = st.session_state.get("vehicle_status", {})
    v_cat = v_state.get("category", v_state.get("status", "NOT ANALYZED"))
    v_badge = v_state.get("badge_text", v_state.get("badge", "NOT ANALYZED"))
    v_updated = v_state.get("last_updated", "—")

    with col_v:
        with st.container(border=True):
            st.markdown("#### Vehicle AI (OBD-II)")
            st.markdown(render_badge(v_badge, v_cat), unsafe_allow_html=True)
            st.caption(format_card_timestamp(v_updated))

            if v_cat != "NOT ANALYZED":
                v_readings = v_state.get("readings", {})
                if v_readings:
                    v_labels = [
                        ("Engine RPM", v_readings.get("ENGINE_RPM"), "RPM"),
                        ("Vehicle Speed", v_readings.get("SPEED"), "km/h"),
                        ("Coolant Temperature", v_readings.get("ENGINE_COOLANT_TEMP"), "°C"),
                        ("Engine Load", v_readings.get("ENGINE_LOAD"), "%"),
                        ("Throttle Position", v_readings.get("THROTTLE_POS"), "%"),
                        ("Intake Air Temperature", v_readings.get("AIR_INTAKE_TEMP"), "°C"),
                    ]
                    v_lines = []
                    for lbl, val, unit in v_labels:
                        if val is not None:
                            val_str = f"{int(val)}" if isinstance(val, float) and val.is_integer() else (f"{val:.1f}" if isinstance(val, float) else str(val))
                            v_lines.append(
                                f'<div style="display: flex; justify-content: space-between; padding: 2px 0; border-bottom: 1px solid #F1F5F9; font-size: 0.83rem;">'
                                f'<span style="color: #475569;">{lbl}</span>'
                                f'<span style="font-weight: 600; color: #0F172A;">{val_str} {unit}</span></div>'
                            )
                    if v_lines:
                        st.markdown(
                            f'<div style="margin: 10px 0; border-top: 1px solid #E2E8F0; padding-top: 4px;">'
                            f'{"".join(v_lines)}</div>',
                            unsafe_allow_html=True,
                        )
                elif v_state.get("latest_reading") and v_state.get("latest_reading") != "No reading yet":
                    st.caption(v_state.get("latest_reading"))

                v_expl = v_state.get("summary") or v_state.get("explanation")
                if v_expl and v_expl != "No vehicle telemetry analyzed in this session.":
                    st.markdown(
                        f'<div style="margin-top: 8px; margin-bottom: 10px; font-size: 0.82rem; line-height: 1.4;">'
                        f'<strong style="color: #0F172A;">Analysis:</strong><br>'
                        f'<span style="color: #475569;">{v_expl}</span></div>',
                        unsafe_allow_html=True,
                    )

            if st.button("Analyze Vehicle", key="btn_jump_v", use_container_width=True):
                st.session_state["pending_nav"] = "Vehicle AI"
                st.rerun()

    # Industrial Machine Card
    ind_state = st.session_state.get("industrial_status", {})
    ind_cat = ind_state.get("category", "NOT ANALYZED")
    ind_badge = ind_state.get("badge_text", "NOT ANALYZED")
    ind_updated = ind_state.get("last_updated", "—")

    with col_ind:
        with st.container(border=True):
            st.markdown("#### Industrial Machine (AI4I)")
            st.markdown(render_badge(ind_badge, ind_cat), unsafe_allow_html=True)
            st.caption(format_card_timestamp(ind_updated))

            if ind_cat != "NOT ANALYZED":
                ind_readings = ind_state.get("readings", {})
                if ind_readings:
                    ind_labels = [
                        ("Air Temperature", ind_readings.get("air_temp"), "K"),
                        ("Process Temperature", ind_readings.get("proc_temp"), "K"),
                        ("Rotational Speed", ind_readings.get("speed"), "rpm"),
                        ("Torque", ind_readings.get("torque"), "Nm"),
                        ("Tool Wear", ind_readings.get("tool_wear"), "min"),
                    ]
                    ind_lines = []
                    for lbl, val, unit in ind_labels:
                        if val is not None:
                            val_str = f"{int(val)}" if isinstance(val, float) and val.is_integer() else (f"{val:.1f}" if isinstance(val, float) else str(val))
                            ind_lines.append(
                                f'<div style="display: flex; justify-content: space-between; padding: 2px 0; border-bottom: 1px solid #F1F5F9; font-size: 0.83rem;">'
                                f'<span style="color: #475569;">{lbl}</span>'
                                f'<span style="font-weight: 600; color: #0F172A;">{val_str} {unit}</span></div>'
                            )
                    if ind_lines:
                        st.markdown(
                            f'<div style="margin: 10px 0; border-top: 1px solid #E2E8F0; padding-top: 4px;">'
                            f'{"".join(ind_lines)}</div>',
                            unsafe_allow_html=True,
                        )
                elif ind_state.get("prob") is not None:
                    st.caption(f"Failure Probability: {ind_state['prob']:.1f}%")

                ind_expl = ind_state.get("explanation")
                if ind_expl and ind_expl != "No machine telemetry analyzed in this session.":
                    st.markdown(
                        f'<div style="margin-top: 8px; margin-bottom: 10px; font-size: 0.82rem; line-height: 1.4;">'
                        f'<strong style="color: #0F172A;">Analysis:</strong><br>'
                        f'<span style="color: #475569;">{ind_expl}</span></div>',
                        unsafe_allow_html=True,
                    )

            if st.button("Analyze Machine", key="btn_jump_ind", use_container_width=True):
                st.session_state["pending_nav"] = "Industrial AI"
                st.rerun()


# =============================================================================
# SECTION 2: CHECK EQUIPMENT (HOUSEHOLD WORKFLOW)
# =============================================================================
elif current_page == "Check Equipment":
    st.markdown("### Check Equipment")
    st.caption("Inspect physical operating parameters for household equipment to detect abnormal stress.")

    # Optional Symptom Parser Expander
    with st.expander("Describe Symptoms in Natural Language (Optional)", expanded=False):
        st.caption("Keyword pattern extractor identifies mentioned equipment and operational concerns without fabricating numeric readings.")
        user_symptom_text = st.text_area(
            "Observation:",
            placeholder="e.g. 'Washing machine motor feels hot and vibrates heavily during spin cycle.'",
            label_visibility="collapsed",
        )
        detected_appliance = None
        if user_symptom_text.strip():
            nlp_res = parse_natural_language_symptom(user_symptom_text)
            detected_appliance = nlp_res["detected_appliance"]
            if nlp_res["detected_symptoms"]:
                st.markdown(f"**Identified Symptoms:** {', '.join(nlp_res['detected_symptoms'])}")
            if nlp_res["detected_concerns"]:
                st.caption(f"Concerns: {', '.join(nlp_res['detected_concerns'])}")

    # 1. Equipment Selection
    equipment_list = list(APPLIANCE_PROFILES.keys())

    # Handle programmatic appliance selection from Home Overview jump buttons
    if "selected_appliance_override" in st.session_state:
        override = st.session_state.pop("selected_appliance_override")
        if override in equipment_list:
            st.session_state["selected_equipment"] = override
    elif detected_appliance and detected_appliance in equipment_list:
        st.session_state["selected_equipment"] = detected_appliance

    if "selected_equipment" not in st.session_state or st.session_state["selected_equipment"] not in equipment_list:
        st.session_state["selected_equipment"] = equipment_list[0]

    selected_appliance = st.selectbox(
        "Select Equipment to Analyze:",
        equipment_list,
        key="selected_equipment",
    )

    profile = APPLIANCE_PROFILES[selected_appliance]

    # 2. Dynamic Input Fields Card
    with st.container(border=True):
        st.markdown(f"##### {selected_appliance} Operating Readings")
        readings_input = {}
        fields = profile["fields"]
        num_fields = len(fields)
        mid_idx = (num_fields + 1) // 2

        c_left, c_right = st.columns(2)
        for i, field in enumerate(fields):
            col = c_left if i < mid_idx else c_right
            with col:
                k = field["key"]
                lbl = f"{field['label']} [{field['unit']}]"
                if field.get("type") == "select":
                    readings_input[k] = st.selectbox(
                        lbl,
                        field["options"],
                        index=field["options"].index(field["default"]),
                        key=f"manual_{selected_appliance}_{k}",
                    )
                else:
                    readings_input[k] = st.number_input(
                        lbl,
                        min_value=float(field["min"]),
                        max_value=float(field["max"]),
                        value=float(field["default"]),
                        step=float(field["step"]),
                        help=f"Normal: {field['normal'][0]} – {field['normal'][1]} {field['unit']}",
                        key=f"manual_{selected_appliance}_{k}",
                    )

        analyze_equipment_btn = st.button(
            "Analyze Equipment",
            type="primary",
            use_container_width=True,
        )

    # 3. Assessment Result Card
    if analyze_equipment_btn:
        eval_res = evaluate_manual_appliance(selected_appliance, readings_input)
        analysis_timestamp = datetime.datetime.now()
        summary_str = " | ".join([
            f"{readings_input[f['key']]} {f.get('unit', '')}".strip()
            for f in fields[:3]
        ])

        # Persist to Centralized State (Only this appliance is updated)
        save_equipment_analysis(
            appliance=selected_appliance,
            eval_result=eval_res,
            readings=readings_input,
            summary_str=summary_str,
            timestamp=analysis_timestamp,
        )
        st.session_state["last_appliance_eval"] = {
            "appliance": selected_appliance,
            "res": eval_res,
            "readings": readings_input,
            "timestamp": analysis_timestamp,
        }

    if "last_appliance_eval" in st.session_state:
        eval_data = st.session_state["last_appliance_eval"]
        res = eval_data["res"]
        app_name = eval_data["appliance"]

        with st.container(border=True):
            st.markdown(f"#### {app_name.upper()} STATUS")
            st.markdown(render_badge(res["badge"], res["status"]), unsafe_allow_html=True)
            st.markdown(f"**Assessment:** {res['explanation']}")
            if res.get("recommendation"):
                st.markdown(f"**Recommended Inspection:** {res['recommendation']}")

            # Secondary Technical Details
            with st.expander("View Details", expanded=False):
                if res["abnormalities"]:
                    st.error("**Identified Out-of-Range Parameters:**")
                    for abn in res["abnormalities"]:
                        st.markdown(f"- {abn}")
                if res["warnings"]:
                    st.warning("**Elevated Parameters:**")
                    for w in res["warnings"]:
                        st.markdown(f"- {w}")
                st.markdown("**Submitted Telemetry:**")
                st.json(eval_data["readings"])
                st.caption(f"Analyzed at: {eval_data['timestamp'].strftime('%Y-%m-%d %I:%M %p')}")
                st.caption(res['disclaimer'])


# =============================================================================
# SECTION 3: ENERGY BEHAVIOR (CONSUMPTION ANOMALY MONITOR)
# =============================================================================
elif current_page == "Energy Behavior":
    st.markdown("### Energy Behavior")
    st.caption("Compare observed household electricity consumption against expected baseline demand.")

    # 1. Inputs Card
    with st.container(border=True):
        st.markdown("##### Operating Conditions & Measured Consumption")
        e_col1, e_col2 = st.columns(2)

        with e_col1:
            h_hour = st.slider("Hour of Day (0–23)", min_value=0, max_value=23, value=14, step=1)
            h_t_out = st.number_input("Outdoor Temperature (°C)", min_value=-10.0, max_value=40.0, value=12.5, step=0.5)
            h_rh_out = st.number_input("Outdoor Humidity (%)", min_value=10.0, max_value=100.0, value=75.0, step=1.0)
            h_t_indoor = st.number_input("Average Indoor Temperature (°C)", min_value=10.0, max_value=35.0, value=21.5, step=0.5)

        with e_col2:
            st.caption("Quick Presets:")
            p_e1, p_e2 = st.columns(2)
            if p_e1.button("Preset: Normal", use_container_width=True):
                st.session_state["actual_energy_val"] = 90.0
            if p_e2.button("Preset: Spike Anomaly", use_container_width=True):
                st.session_state["actual_energy_val"] = 380.0

            h_actual = st.number_input(
                "Actual Appliance Energy [Wh]",
                min_value=0.0,
                max_value=1500.0,
                value=st.session_state.get("actual_energy_val", 95.0),
                step=10.0,
                help="Measured total appliance energy over a 10-minute window.",
            )

        check_energy_btn = st.button("Check Energy Behavior", type="primary", use_container_width=True)

    # 2. Result Card
    if check_energy_btn or "household_pred" in st.session_state:
        readings_dict = {
            "hour": h_hour,
            "T_out": h_t_out,
            "RH_out": h_rh_out,
            "T1": h_t_indoor,
            "T2": h_t_indoor,
            "T3": h_t_indoor,
        }
        res_h = predict_household(readings_dict, actual_energy=h_actual, model_path=HOUSEHOLD_MODEL_PATH)
        st.session_state["household_pred"] = res_h

        with st.container(border=True):
            st.markdown("#### ENERGY BEHAVIOR STATUS")
            st.markdown(render_badge(res_h["badge"], res_h["status"]), unsafe_allow_html=True)
            st.markdown(f"**Assessment:** {res_h['explanation']}")

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Expected Baseline", f"{res_h['expected_energy']} Wh")
            m_col2.metric("Actual Measured", f"{res_h['actual_energy']} Wh")
            m_col3.metric("Deviation (Residual)", f"{res_h['residual']:+0.1f} Wh")

            # Clean Comparison Bar Chart (Light Theme Styled)
            fig, ax = plt.subplots(figsize=(7, 2.6), dpi=140)
            fig.patch.set_facecolor('#FFFFFF')
            ax.set_facecolor('#FFFFFF')
            categories = ["Expected Baseline", "Actual Measured"]
            values = [res_h["expected_energy"], res_h["actual_energy"]]
            bar_color = "#10B981" if res_h["status"] == "NORMAL" else ("#F59E0B" if res_h["status"] == "WARNING" else "#DC2626")
            colors = ["#2563EB", bar_color]

            bars = ax.bar(categories, values, color=colors, width=0.42, edgecolor="#CBD5E1", alpha=0.95)
            ax.set_ylabel("Energy (Wh)", fontsize=8.5, fontweight="bold", color="#0F172A")
            ax.set_ylim(0, max(values) * 1.3)
            ax.grid(axis="y", linestyle=":", color="#E2E8F0", alpha=0.9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CBD5E1')
            ax.spines['bottom'].set_color('#CBD5E1')
            ax.tick_params(colors='#475569')

            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + (max(values) * 0.03), f"{h:.1f} Wh",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0F172A")

            exp = res_h["expected_energy"]
            ax.axhline(exp + res_h["warning_threshold"], color="#D97706", linestyle="--", linewidth=1.1, label=f"Warning Ceiling (+{res_h['warning_threshold']:.0f} Wh)")
            ax.axhline(exp + res_h["anomaly_threshold"], color="#DC2626", linestyle="--", linewidth=1.1, label=f"Anomaly Ceiling (+{res_h['anomaly_threshold']:.0f} Wh)")
            ax.legend(loc="upper right", fontsize=7.5, framealpha=0.95, facecolor="#FFFFFF", edgecolor="#CBD5E1", labelcolor="#0F172A")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Secondary Technical Explanation
            with st.expander("View Details", expanded=False):
                st.markdown(
                    "An ExtraTrees regression model estimates nominal baseline demand from environmental variables. "
                    "Residual deviations are tested against empirical test-set percentiles:\n"
                    f"- **Warning Ceiling (80th %ile):** +{res_h['warning_threshold']} Wh\n"
                    f"- **Anomaly Ceiling (95th %ile):** +{res_h['anomaly_threshold']} Wh\n\n"
                    "Source: UCI Appliances Energy Prediction Dataset (UCI ID: 374)."
                )


# =============================================================================
# SECTION 4: VEHICLE AI (OBD-II MONITORING & SCREENING)
# =============================================================================
elif current_page == "Vehicle AI":
    st.markdown("### Vehicle AI")
    st.caption("Vehicle operating-pattern and maintenance-risk screening.")
    st.caption("Notice: This is a prototype screening system. It does not provide a definitive mechanical diagnosis.")

    # Ingestion Mode High-Contrast Segmented Selector
    v_mode = st.segmented_control(
        "Vehicle Ingestion Mode",
        ["Manual OBD", "Upload CSV"],
        default="Manual OBD",
        label_visibility="collapsed",
    )

    # -------------------------------------------------------------------------
    # MODE 1: MANUAL OBD READING
    # -------------------------------------------------------------------------
    if v_mode == "Manual OBD":
        # Scenario Presets
        def set_v_preset(rpm, spd, cool, load, throt, iat, dtc=""):
            st.session_state["v_rpm"] = float(rpm)
            st.session_state["v_spd"] = float(spd)
            st.session_state["v_cool"] = float(cool)
            st.session_state["v_load"] = float(load)
            st.session_state["v_throt"] = float(throt)
            st.session_state["v_iat"] = float(iat)
            st.session_state["v_dtc"] = str(dtc) if dtc else ""

        with st.container(border=True):
            st.caption("Quick Scenario Presets:")
            vp1, vp2, vp3, vp4, vp5 = st.columns(5)
            vp1.button("Highway", on_click=set_v_preset, args=(1850.0, 85.0, 88.0, 38.0, 22.0, 32.0, ""), use_container_width=True)
            vp2.button("City Idle", on_click=set_v_preset, args=(850.0, 0.0, 93.0, 28.0, 12.0, 42.0, ""), use_container_width=True)
            vp3.button("Overheat", on_click=set_v_preset, args=(2600.0, 45.0, 112.0, 65.0, 35.0, 55.0, ""), use_container_width=True)
            vp4.button("O2 Fault", on_click=set_v_preset, args=(1400.0, 25.0, 74.0, 42.0, 12.0, 38.0, "P0133"), use_container_width=True)
            vp5.button("ABS Fault", on_click=set_v_preset, args=(2100.0, 60.0, 86.0, 45.0, 24.0, 35.0, "C0300"), use_container_width=True)

            st.markdown("##### Core Telemetry Sensors")
            vc_l, vc_r = st.columns(2)
            with vc_l:
                v_rpm = st.number_input("Engine RPM [RPM]", 400.0, 7500.0, st.session_state.get("v_rpm", 1850.0), 50.0)
                v_coolant = st.number_input("Coolant Temperature [°C]", -20.0, 130.0, st.session_state.get("v_cool", 88.0), 0.5)
                v_throttle = st.number_input("Throttle Position [%]", 0.0, 100.0, st.session_state.get("v_throt", 22.0), 1.0)
            with vc_r:
                v_speed = st.number_input("Vehicle Speed [km/h]", 0.0, 220.0, st.session_state.get("v_spd", 85.0), 1.0)
                v_load = st.number_input("Engine Load [%]", 0.0, 100.0, st.session_state.get("v_load", 38.0), 1.0)
                v_iat = st.number_input("Intake Air Temperature [°C]", -30.0, 95.0, st.session_state.get("v_iat", 32.0), 0.5)

            v_dtc = st.text_input(
                "Active Diagnostic Trouble Code (Optional):",
                value=st.session_state.get("v_dtc", ""),
                placeholder="e.g. P0133, C0300, P0078 (or leave blank)",
                help="Standard 5-character SAE J2012 fault code.",
            )

            analyze_v_btn = st.button("Analyze Vehicle", type="primary", use_container_width=True)

        if analyze_v_btn:
            v_payload = {
                "ENGINE_RPM": float(v_rpm),
                "SPEED": float(v_speed),
                "ENGINE_COOLANT_TEMP": float(v_coolant),
                "ENGINE_LOAD": float(v_load),
                "THROTTLE_POS": float(v_throttle),
                "AIR_INTAKE_TEMP": float(v_iat),
                "TROUBLE_CODES": v_dtc.strip() if v_dtc.strip() else None,
            }
            try:
                res_v = analyze_vehicle(v_payload)
                st.session_state["v_last_eval"] = res_v
                now_str = datetime.datetime.now().strftime("%d %b %Y • %I:%M %p")

                status_label = "NORMAL" if res_v["health_category"] == "NORMAL" else ("WARNING" if res_v["health_category"] == "ADVISORY" else "ANOMALY")
                st.session_state["vehicle_status"] = {
                    "status": res_v["health_category"],
                    "badge_text": status_label,
                    "category": res_v["health_category"],
                    "last_updated": now_str,
                    "summary": f"{res_v['action_urgency']} (Risk: {res_v['risk_score_pct']}%)",
                    "latest_reading": f"RPM: {v_rpm:.0f} | Speed: {v_speed:.0f} km/h | Coolant: {v_coolant:.1f}°C",
                    "risk_score": res_v["risk_score_pct"],
                    "fault_probability": res_v["fault_probability_pct"],
                    "raw_anomaly_score": res_v["raw_anomaly_score"],
                    "recommendations": res_v["recommendations"],
                    "explanation": " | ".join(res_v["recommendations"]),
                    "readings": v_payload,
                    "result": res_v,
                }
            except Exception as e:
                st.error(f"Inference notice: {e}")

        # Result Card
        if "v_last_eval" in st.session_state:
            res_v = st.session_state["v_last_eval"]
            status_label = "NORMAL" if res_v["health_category"] == "NORMAL" else ("WARNING" if res_v["health_category"] == "ADVISORY" else "ANOMALY")

            with st.container(border=True):
                st.markdown("#### VEHICLE STATUS")
                st.markdown(render_badge(status_label, res_v["health_category"]), unsafe_allow_html=True)
                st.markdown(f"**Assessment:** {res_v['action_urgency']}")

                with st.expander("View Details", expanded=False):
                    vm1, vm2, vm3 = st.columns(3)
                    vm1.metric("Anomaly Risk Index", f"{res_v['risk_score_pct']}%")
                    vm2.metric("Isolation Forest Score", f"{res_v['raw_anomaly_score']:.4f}")
                    vm3.metric("Supervised Fault Prob", f"{res_v['fault_probability_pct']}%")

                    if res_v["sensor_anomalies"]:
                        st.markdown("##### Physical Sensor Boundary Alerts:")
                        for anom in res_v["sensor_anomalies"]:
                            st.warning(f"- {anom['message']}")

                    if res_v["diagnostic_trouble_code"]:
                        dtc = res_v["diagnostic_trouble_code"]
                        st.markdown(f"##### DTC: `{dtc.get('code')}` — {dtc.get('title')}")
                        st.markdown(f"**Subsystem:** {dtc.get('subsystem')} | **Severity:** `{dtc.get('severity')}`")
                        st.markdown(f"**Action:** {dtc.get('action')}")

                    st.markdown("##### Recommendations:")
                    for rec in res_v["recommendations"]:
                        st.markdown(f"- {rec}")

    # -------------------------------------------------------------------------
    # MODE 2: UPLOAD OBD CSV (POLISHED WORKFLOW)
    # -------------------------------------------------------------------------
    else:
        with st.container(border=True):
            st.markdown("##### Upload OBD-II Telemetry Log")
            st.caption(
                "Upload a trip log CSV (e.g. Torque Pro, Car Scanner, or standard OBD log). "
                "Required sensor PIDs: ENGINE_RPM, SPEED, ENGINE_COOLANT_TEMP, ENGINE_LOAD, THROTTLE_POS, AIR_INTAKE_TEMP."
            )

            col_up1, col_up2 = st.columns([0.72, 0.28])
            with col_up1:
                uploaded_csv = st.file_uploader(
                    "Choose CSV File",
                    type=["csv"],
                    label_visibility="collapsed",
                    key="vehicle_csv_uploader",
                )
            with col_up2:
                load_test_btn = st.button("Load Sample Trip", use_container_width=True)

            # Ingest Data Safely
            if uploaded_csv is not None:
                try:
                    df_in = pd.read_csv(uploaded_csv, low_memory=False)
                    st.session_state["vehicle_df_raw"] = df_in
                    st.session_state["vehicle_df_filename"] = uploaded_csv.name
                except Exception as read_err:
                    st.error(f"Unable to parse CSV file: {read_err}")
            elif load_test_btn:
                sample_path = VEHICLE_TEST_CSV_PATH if os.path.exists(VEHICLE_TEST_CSV_PATH) else VEHICLE_CLEAN_CSV_PATH
                if os.path.exists(sample_path):
                    df_in = pd.read_csv(sample_path, low_memory=False)
                    st.session_state["vehicle_df_raw"] = df_in
                    st.session_state["vehicle_df_filename"] = os.path.basename(sample_path)

        # Validation, 5-Row Preview, and Analysis CTA
        if "vehicle_df_raw" in st.session_state and st.session_state["vehicle_df_raw"] is not None:
            raw_df = st.session_state["vehicle_df_raw"]
            file_name = st.session_state.get("vehicle_df_filename", "Uploaded File")

            if len(raw_df) == 0:
                st.error("Uploaded telemetry dataset is empty.")
            else:
                norm_df = normalize_column_names(raw_df)
                core_pids = ["ENGINE_RPM", "SPEED", "ENGINE_COOLANT_TEMP", "ENGINE_LOAD", "THROTTLE_POS", "AIR_INTAKE_TEMP"]
                missing_pids = [p for p in core_pids if p not in norm_df.columns]

                if missing_pids:
                    st.error(
                        f"Validation Notice: Missing required OBD sensor columns: {', '.join(missing_pids)}. "
                        f"Found columns: {', '.join(list(raw_df.columns)[:8])}"
                    )
                else:
                    # Clean Validation Message Box
                    st.success(f"CSV validated successfully ({file_name} — Rows: {len(raw_df)})")

                    # Preview table
                    st.caption("Telemetry Preview (First 5 Rows):")
                    display_cols = [c for c in core_pids if c in norm_df.columns]
                    if "TROUBLE_CODES" in norm_df.columns:
                        display_cols.append("TROUBLE_CODES")
                    st.dataframe(norm_df[display_cols].head(5), use_container_width=True)

                    # Prominent CTA Button
                    if st.button("Analyze Vehicle Data", type="primary", key="btn_exec_vehicle_csv", use_container_width=True):
                        try:
                            batch_res = analyze_batch_telemetry(norm_df)
                            if batch_res.get("status") == "error":
                                st.error(f"Analysis Notice: {batch_res.get('message')}")
                            else:
                                st.session_state["vehicle_csv_last_result"] = batch_res
                                t_label = "NORMAL" if batch_res["trip_category"] == "NORMAL" else ("WARNING" if batch_res["trip_category"] == "ADVISORY" else "ANOMALY")
                                now_str = datetime.datetime.now().strftime("%d %b %Y • %I:%M %p")

                                # Synchronize with Home Overview state
                                st.session_state["vehicle_status"] = {
                                    "status": batch_res["trip_category"],
                                    "badge_text": t_label,
                                    "category": batch_res["trip_category"],
                                    "last_updated": now_str,
                                    "summary": f"Trip Log ({batch_res['valid_rows']} frames) — Mean Risk: {batch_res['mean_risk_score_pct']}%",
                                    "latest_reading": f"Analyzed {batch_res['valid_rows']} trip frames",
                                    "risk_score": batch_res["mean_risk_score_pct"],
                                    "fault_probability": round(batch_res["condition_percentages"]["critical_pct"], 1),
                                    "raw_anomaly_score": None,
                                    "recommendations": [f"Trip Assessment: {t_label}"],
                                    "explanation": f"Evaluated {batch_res['valid_rows']} OBD snapshots.",
                                    "readings": {},
                                    "result": batch_res,
                                }
                        except Exception as eval_err:
                            st.error(f"Unable to process telemetry log: {eval_err}")

        # Batch Analysis Result Card
        if "vehicle_csv_last_result" in st.session_state and st.session_state["vehicle_csv_last_result"] is not None:
            b_res = st.session_state["vehicle_csv_last_result"]
            t_label = "NORMAL" if b_res["trip_category"] == "NORMAL" else ("WARNING" if b_res["trip_category"] == "ADVISORY" else "ANOMALY")

            with st.container(border=True):
                st.markdown("#### TRIP STATUS")
                st.markdown(render_badge(t_label, b_res["trip_category"]), unsafe_allow_html=True)
                st.markdown(
                    f"**Assessment:** Analyzed {b_res['valid_rows']} telemetry frames — "
                    f"Mean Risk: {b_res['mean_risk_score_pct']}%"
                )

                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric("Healthy Frames", f"{b_res['condition_percentages']['healthy_pct']}%")
                tc2.metric("Advisory Frames", f"{b_res['condition_percentages']['advisory_pct']}%")
                tc3.metric("Critical Frames", f"{b_res['condition_percentages']['critical_pct']}%")
                tc4.metric("Mean Risk Index", f"{b_res['mean_risk_score_pct']}%")

                with st.expander("View Details", expanded=False):
                    if b_res.get("dtc_summary"):
                        st.markdown("##### Detected Trouble Codes:")
                        for code, details in b_res["dtc_summary"].items():
                            st.markdown(f"- **{code}** ({details['count']} occurrences): {details['title']} [Severity: {details['severity']}]")

                    if b_res.get("top_anomalous_events"):
                        st.markdown("##### High-Risk Telemetry Snapshots:")
                        st.dataframe(pd.DataFrame(b_res["top_anomalous_events"]), use_container_width=True)

    # Expandable Technical Information at Bottom
    with st.expander("Model Performance & Diagnostics", expanded=False):
        st.markdown("##### Unsupervised Isolation Forest Anomaly Separation")
        if os.path.exists(VEHICLE_ANOM_PATH):
            st.image(VEHICLE_ANOM_PATH, use_container_width=True)
        st.markdown("##### Supervised Random Forest Confusion Matrix")
        if os.path.exists(VEHICLE_CM_PATH):
            st.image(VEHICLE_CM_PATH, use_container_width=True)

    with st.expander("Vehicle Sensor Feature Importance", expanded=False):
        if os.path.exists(VEHICLE_FI_PATH):
            st.image(VEHICLE_FI_PATH, use_container_width=True)

    with st.expander("Technical Architecture & Future OBD-II", expanded=False):
        st.markdown(
            "- **Dataset:** Real-World Fleet OBD-II Telemetry (14 passenger vehicles, 29,100 clean records)\n"
            "- **Dual ML Engine:** Unsupervised Isolation Forest (150 trees) + Supervised Random Forest (100 trees)\n"
            "- **Hardware Abstraction:** `vehicle/obd_interface.py` supports ELM327 Bluetooth/USB/WiFi adapters."
        )


# =============================================================================
# SECTION 5: INDUSTRIAL AI (PREDICTIVE MAINTENANCE)
# =============================================================================
elif current_page == "Industrial AI":
    st.markdown("### Industrial AI")
    st.caption("Milling machine thermodynamic and rotational telemetry predictive maintenance screening.")

    # Preset handler
    def set_ind_preset(air, proc, speed, torq, wear):
        st.session_state["ind_air"] = air
        st.session_state["ind_proc"] = proc
        st.session_state["ind_spd"] = speed
        st.session_state["ind_trq"] = torq
        st.session_state["ind_wr"] = wear

    # Inputs Card
    with st.container(border=True):
        st.caption("Quick Machine Presets:")
        ip1, ip2, ip3 = st.columns(3)
        ip1.button("Nominal Baseline", on_click=set_ind_preset, args=(300.0, 310.0, 1500.0, 40.0, 50.0), use_container_width=True)
        ip2.button("Elevated Stress", on_click=set_ind_preset, args=(302.0, 311.0, 1350.0, 52.0, 210.0), use_container_width=True)
        ip3.button("High Risk", on_click=set_ind_preset, args=(304.5, 313.2, 1200.0, 68.0, 220.0), use_container_width=True)

        st.markdown("##### Machine Operating Parameters")
        ic_l, ic_r = st.columns(2)
        with ic_l:
            ind_air = st.number_input("Air Temperature [K]", 280.0, 320.0, st.session_state.get("ind_air", 300.0), 0.1)
            ind_speed = st.number_input("Rotational Speed [rpm]", 800.0, 3200.0, st.session_state.get("ind_spd", 1500.0), 10.0)
            ind_wear = st.number_input("Tool Wear [min]", 0.0, 350.0, st.session_state.get("ind_wr", 100.0), 1.0)
        with ic_r:
            ind_proc = st.number_input("Process Temperature [K]", 290.0, 330.0, st.session_state.get("ind_proc", 310.0), 0.1)
            ind_torque = st.number_input("Torque [Nm]", 0.0, 120.0, st.session_state.get("ind_trq", 40.0), 0.5)

        analyze_ind_btn = st.button("Analyze Machine", type="primary", use_container_width=True)

    # Result Card
    if analyze_ind_btn or "ind_last_eval" in st.session_state:
        pred_class, failure_prob = predict_risk(
            air_temperature=ind_air,
            process_temperature=ind_proc,
            rotational_speed=ind_speed,
            torque=ind_torque,
            tool_wear=ind_wear,
            model_path=INDUSTRIAL_MODEL_PATH,
        )
        st.session_state["ind_last_eval"] = (pred_class, failure_prob)

        prob_pct = failure_prob * 100.0
        if failure_prob < 0.40:
            ind_risk = "NORMAL"
            ind_text = "Operating parameters align with normal conditions."
        elif failure_prob < 0.70:
            ind_risk = "WARNING"
            ind_text = "Operating parameters show similarity to failure-associated conditions."
        else:
            ind_risk = "HIGH RISK"
            ind_text = "Operating parameters strongly resemble failure-associated conditions."

        # Sync to Home Overview State
        ind_now = datetime.datetime.now().strftime("%d %b %Y • %I:%M %p")
        st.session_state["industrial_status"] = {
            "status": ind_risk,
            "badge_text": ind_risk,
            "category": ind_risk,
            "last_updated": ind_now,
            "prob": prob_pct,
            "pred": pred_class,
            "summary": f"{ind_risk} ({prob_pct:.1f}% risk)",
            "explanation": ind_text,
            "readings": {
                "air_temp": ind_air,
                "proc_temp": ind_proc,
                "speed": ind_speed,
                "torque": ind_torque,
                "tool_wear": ind_wear,
            },
        }

        with st.container(border=True):
            st.markdown("#### MACHINE STATUS")
            st.markdown(render_badge(ind_risk, ind_risk), unsafe_allow_html=True)
            st.markdown(f"**Assessment:** {ind_text}")

            p1, p2 = st.columns([0.4, 0.6])
            with p1:
                st.metric("Failure-Associated Probability", f"{prob_pct:.1f}%")
            with p2:
                st.progress(float(failure_prob), text=f"Probability: {prob_pct:.1f}%")
                st.caption("Prototype visualization threshold — not an industrial safety shutoff limit.")

        # Expandable Technical Sections (View Details, Model Performance, Feature Importance)
        with st.expander("View Details", expanded=False):
            features_data = [
                ("Air Temperature", ind_air, INDUSTRIAL_RANGES["Air temperature [K]"]),
                ("Process Temperature", ind_proc, INDUSTRIAL_RANGES["Process temperature [K]"]),
                ("Rotational Speed", ind_speed, INDUSTRIAL_RANGES["Rotational speed [rpm]"]),
                ("Torque", ind_torque, INDUSTRIAL_RANGES["Torque [Nm]"]),
                ("Tool Wear", ind_wear, INDUSTRIAL_RANGES["Tool wear [min]"]),
            ]
            plot_names = [f[0] for f in features_data]
            plot_norms = [((f[1] - f[2]["min"]) / (f[2]["max"] - f[2]["min"])) * 100.0 for f in features_data]
            plot_labels = [f"{f[1]:.1f} {f[2]['unit']} ({norm:.1f}%)" for f, norm in zip(features_data, plot_norms)]

            fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=140)
            fig.patch.set_facecolor('#FFFFFF')
            ax.set_facecolor('#FFFFFF')
            y_pos = np.arange(len(plot_names))
            b_color = "#DC2626" if ind_risk == "HIGH RISK" else ("#F59E0B" if ind_risk == "WARNING" else "#2563EB")
            widths = [max(0.0, min(110.0, n)) for n in plot_norms]
            bars = ax.barh(y_pos, widths, color=b_color, edgecolor="#CBD5E1", height=0.52, alpha=0.95)
            ax.axvline(50, color="#64748B", linestyle="--", linewidth=1.1, label="Midpoint (50%)")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(plot_names, fontsize=8.5, fontweight="bold", color="#0F172A")
            ax.set_xlabel("Relative Position in Benchmark Range (%)", fontsize=8, color="#0F172A")
            ax.set_xlim(0, max(105, max(widths) + 15))
            ax.grid(axis="x", linestyle=":", color="#E2E8F0", alpha=0.9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CBD5E1')
            ax.spines['bottom'].set_color('#CBD5E1')
            ax.tick_params(colors='#475569')

            for bar, lbl in zip(bars, plot_labels):
                x_val = bar.get_width()
                ha = "right" if x_val > 80 else "left"
                x_pos = (x_val - 2.0) if x_val > 80 else (x_val + 1.5)
                c = "#FFFFFF" if x_val > 80 else "#0F172A"
                ax.text(x_pos, bar.get_y() + bar.get_height() / 2, lbl, va="center", ha=ha, fontsize=7.5, fontweight="bold", color=c)

            ax.legend(loc="lower right", fontsize=7.5, facecolor="#FFFFFF", edgecolor="#CBD5E1", labelcolor="#0F172A")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with st.expander("Model Performance", expanded=False):
            st.caption("Benchmark Metrics: Precision: 71.21% | Recall: 69.12% | F1-Score: 70.15% | Accuracy: 98.00%")
            if os.path.exists(CM_IMAGE_PATH):
                st.image(CM_IMAGE_PATH, use_container_width=True, caption="Confusion Matrix")

        with st.expander("Feature Importance", expanded=False):
            if os.path.exists(FI_IMAGE_PATH):
                st.image(FI_IMAGE_PATH, use_container_width=True, caption="Feature Importance")


# =============================================================================
# SECTION 6: SENSOR INTEGRATION & LIMITATIONS
# =============================================================================
elif current_page == "Sensor Integration":
    st.markdown("### Sensor Integration")
    st.caption("Telemetry ingestion architecture bridging edge IoT hardware with MachineGuard AI models.")

    # Visual Architecture Diagram
    with st.container(border=True):
        st.markdown("##### System Architecture Flow")
        st.code(
            """
    ┌─────────────────────────────────────────────────────────────┐
    │                 Physical Telemetry Sources                  │
    │        Household Sensors  |  OBD-II Vehicle  |  Industrial   │
    └──────────────────────────────┬──────────────────────────────┘
                                   │ MQTT / Serial / CAN / REST
                                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              MachineGuard Protocol Gateway                  │
    │       Standardizes TelemetryPackets & Protocol Adapters     │
    └──────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                     Multi-Domain AI Engine                  │
    │      Residual Energy | Isolation Forest | Random Forest     │
    └──────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              Real-Time Alert Dashboard & Screening          │
    └─────────────────────────────────────────────────────────────┘
            """,
            language="text",
        )

    # Current vs Planned Comparison
    col_cur, col_fut = st.columns(2)

    with col_cur:
        with st.container(border=True):
            st.markdown("#### CURRENT PROTOTYPE")
            st.markdown(
                "- **Manual Telemetry:** Physical bounds verification\n"
                "- **Batch Log Ingestion:** OBD-II CSV processing\n"
                "- **Keyword Symptom Parser:** Transparent keyword mapping\n"
                "- **Virtual Endpoints:** Mocked IoT sensors (ESP32-HVAC, OBD-BT01, CT-PANEL)"
            )

    with col_fut:
        with st.container(border=True):
            st.markdown("#### PLANNED HARDWARE")
            st.markdown(
                "- **Live Household IoT:** ESP32 current clamps (CT)\n"
                "- **Vehicle Telematics:** ELM327 Bluetooth/WiFi adapter\n"
                "- **Industrial Telemetry:** Modbus TCP & PLC bridge\n"
                "- **Automated Alerts:** Push notifications on anomaly"
            )

    # Disclosures & Limitations Expander
    with st.expander("Prototype Limitations & AI Tool Disclosure", expanded=False):
        st.markdown(
            "1. **Public Benchmark Datasets:** Industrial milling machine model uses UCI AI4I 2020; household model uses UCI Appliances Energy (UCI ID: 374); vehicle model uses open OBD-II fleet data.\n"
            "2. **Screening vs. Diagnosis:** Predictions represent statistical pattern similarity to training distributions; they do **not** provide definitive mechanical failure guarantees or certified repairs.\n"
            "3. **Prototype Thresholds:** Risk levels (NORMAL, WARNING, ANOMALY, HIGH RISK) are screening heuristics, **not industrial safety shutoff limits**.\n"
            "4. **AI Tool Disclosure:** Generative AI tools were utilized for code scaffolding, layout styling, and test scripts. All ML logic, thresholds, and validation were engineered and verified."
        )


# -----------------------------------------------------------------------------
# GLOBAL FOOTER
# -----------------------------------------------------------------------------
st.divider()
st.caption(
    "MachineGuard — AI-Powered Equipment Monitoring | SJCET Hackathon Project | Powered by Scikit-Learn & Streamlit"
)
