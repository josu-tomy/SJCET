"""
MachineGuard — AI Predictive Maintenance Risk Detector
Streamlit Dashboard

Person 3: Streamlit Application Engineer
"""

import os
import streamlit as st
from predict import predict_risk

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
CM_IMAGE_PATH = os.path.join(BASE_DIR, "outputs", "confusion_matrix.png")
FI_IMAGE_PATH = os.path.join(BASE_DIR, "outputs", "feature_importance.png")

# Page Configuration
st.set_page_config(
    page_title="MachineGuard — AI Predictive Maintenance Risk Detector",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header Section
st.title("MachineGuard")
st.subheader("AI Predictive Maintenance Risk Detector")
st.markdown(
    """
    Operating pattern risk detection dashboard trained on the 
    **UCI AI4I 2020 Predictive Maintenance Dataset**. 
    Evaluates real-time sensor measurements to detect equipment condition risks.
    """
)
st.divider()

# Sidebar: Inputs & Controls
st.sidebar.header("Operating Sensor Telemetry")
st.sidebar.markdown("Configure operational parameters or load a preset:")

# Preset handler callbacks
def set_nominal_preset():
    st.session_state["air_temp"] = 300.0
    st.session_state["proc_temp"] = 310.0
    st.session_state["rot_speed"] = 1500.0
    st.session_state["torque"] = 40.0
    st.session_state["tool_wear"] = 50.0

def set_warning_preset():
    st.session_state["air_temp"] = 302.0
    st.session_state["proc_temp"] = 311.0
    st.session_state["rot_speed"] = 1350.0
    st.session_state["torque"] = 52.0
    st.session_state["tool_wear"] = 210.0

def set_stress_preset():
    st.session_state["air_temp"] = 304.5
    st.session_state["proc_temp"] = 313.2
    st.session_state["rot_speed"] = 1200.0
    st.session_state["torque"] = 68.0
    st.session_state["tool_wear"] = 220.0

col_preset1, col_preset2, col_preset3 = st.sidebar.columns(3)
col_preset1.button("Nominal", on_click=set_nominal_preset, width="stretch", help="Loads nominal parameters (Normal risk)")
col_preset2.button("Elevated", on_click=set_warning_preset, width="stretch", help="Loads elevated stress parameters (Warning risk)")
col_preset3.button("High Stress", on_click=set_stress_preset, width="stretch", help="Loads high stress parameters (High Risk)")

# Ensure defaults in session_state
if "air_temp" not in st.session_state:
    st.session_state["air_temp"] = 300.0
if "proc_temp" not in st.session_state:
    st.session_state["proc_temp"] = 310.0
if "rot_speed" not in st.session_state:
    st.session_state["rot_speed"] = 1500.0
if "torque" not in st.session_state:
    st.session_state["torque"] = 40.0
if "tool_wear" not in st.session_state:
    st.session_state["tool_wear"] = 100.0

# Numerical Sensor Inputs (Sensible defaults based on UCI dataset)
air_temp = st.sidebar.number_input(
    "Air Temperature [K]",
    min_value=280.0,
    max_value=320.0,
    step=0.1,
    key="air_temp",
    help="Ambient factory temperature in Kelvin (Dataset typical: ~295K – 305K)",
)

proc_temp = st.sidebar.number_input(
    "Process Temperature [K]",
    min_value=290.0,
    max_value=330.0,
    step=0.1,
    key="proc_temp",
    help="Internal equipment process temperature in Kelvin (Dataset typical: ~305K – 314K)",
)

rot_speed = st.sidebar.number_input(
    "Rotational Speed [rpm]",
    min_value=800.0,
    max_value=3200.0,
    step=10.0,
    key="rot_speed",
    help="Spindle rotational speed in revolutions per minute (Dataset typical: ~1168 – 2886 rpm)",
)

torque = st.sidebar.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=120.0,
    step=0.5,
    key="torque",
    help="Applied mechanical torque in Newton-meters (Dataset typical: ~3.8 – 76.6 Nm)",
)

tool_wear = st.sidebar.number_input(
    "Tool Wear [min]",
    min_value=0.0,
    max_value=350.0,
    step=1.0,
    key="tool_wear",
    help="Cumulative active cutting tool usage time in minutes (Dataset range: 0 – 253 min)",
)

st.sidebar.markdown("---")
analyze_btn = st.sidebar.button("ANALYZE MACHINE CONDITION", type="primary", width="stretch")

# Main Dashboard Layout
main_col1, main_col2 = st.columns([1.1, 0.9])

with main_col1:
    st.subheader("Condition Analysis & Risk Assessment")

    # Run inference upon clicking button or default initial state
    if analyze_btn or "has_run" in st.session_state:
        try:
            pred_class, failure_prob = predict_risk(
                air_temperature=float(air_temp),
                process_temperature=float(proc_temp),
                rotational_speed=float(rot_speed),
                torque=float(torque),
                tool_wear=float(tool_wear),
                model_path=MODEL_PATH,
            )
            st.session_state["has_run"] = True
            st.session_state["last_pred"] = pred_class
            st.session_state["last_prob"] = failure_prob
        except Exception as e:
            st.error(f"Error loading model or generating prediction: {e}")
            pred_class, failure_prob = None, None
    else:
        # Prompt user before initial analysis
        st.info("👈 Set sensor values in the sidebar and click **ANALYZE MACHINE CONDITION**.")
        pred_class, failure_prob = None, None

    if failure_prob is not None:
        # Prototype Visualization Risk Thresholds:
        # 0.00–0.39 -> NORMAL
        # 0.40–0.69 -> WARNING
        # 0.70–1.00 -> HIGH RISK
        if failure_prob < 0.40:
            risk_level = "NORMAL"
            badge_icon = "🟢"
            status_text = "The current operating condition resembles normal operating patterns according to the trained model."
        elif failure_prob < 0.70:
            risk_level = "WARNING"
            badge_icon = "🟡"
            status_text = "The current operating condition shows elevated stress patterns approaching failure-associated operating regions."
        else:
            risk_level = "HIGH RISK"
            badge_icon = "🔴"
            status_text = "The current operating condition resembles failure-associated patterns according to the trained model."

        # Metrics Display Cards
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric(
                label="Predicted Class",
                value=f"Class {pred_class} ({'Failure Risk' if pred_class == 1 else 'Normal'})",
            )
        with m_col2:
            st.metric(
                label="Failure-Associated Probability",
                value=f"{failure_prob * 100:.2f}%",
            )
        with m_col3:
            st.metric(
                label="Risk Level",
                value=f"{badge_icon} {risk_level}",
            )

        # Visual progress bar representing failure probability
        st.progress(float(failure_prob), text=f"Failure-Associated Probability: {failure_prob * 100:.1f}%")

        # Interpretation & Mandatory Disclaimers
        if risk_level == "HIGH RISK":
            st.error(f"**Assessment:** {status_text}")
        elif risk_level == "WARNING":
            st.warning(f"**Assessment:** {status_text}")
        else:
            st.success(f"**Assessment:** {status_text}")

        st.caption(
            "⚠️ **Prototype visualization threshold — not an industrial safety limit.** "
            "The system detects operating patterns associated with machine failure. "
            "It does not claim that the machine is guaranteed to fail or guaranteed to remain operable."
        )

    # Model Performance on Held-Out Test Set
    st.markdown("### Model Performance (Held-Out Test Set)")
    st.markdown(
        "Performance calculated on the 2,000 held-out test samples (20% stratified test set):"
    )
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    p_col1.metric(label="Precision", value="71.21%", help="TP / (TP + FP) on held-out test set")
    p_col2.metric(label="Recall", value="69.12%", help="TP / (TP + FN) on held-out test set")
    p_col3.metric(label="F1-Score", value="70.15%", help="Harmonic mean of precision and recall")
    p_col4.metric(label="Accuracy", value="98.00%", help="Overall correct predictions")

with main_col2:
    st.subheader("Model Diagnostic & Evaluation Visuals")

    tab1, tab2 = st.tabs(["Feature Importance", "Confusion Matrix"])

    with tab1:
        st.markdown("#### Random Forest Feature Importance")
        st.info(
            "**Feature importance indicates which input variables were most influential to the trained model overall. "
            "It does not establish causality.**"
        )
        if os.path.exists(FI_IMAGE_PATH):
            st.image(FI_IMAGE_PATH, width="stretch", caption="Model feature importance (Gini impurity reduction)")
        else:
            st.warning(f"Feature importance image not found at {FI_IMAGE_PATH}. Run evaluate.py to generate it.")

    with tab2:
        st.markdown("#### Test Set Confusion Matrix")
        st.markdown(
            "Confusion Matrix evaluated on 2,000 held-out test samples: "
            "**1,913** True Negatives, **47** True Positives, **19** False Positives, **21** False Negatives."
        )
        if os.path.exists(CM_IMAGE_PATH):
            st.image(CM_IMAGE_PATH, width="stretch", caption="Confusion Matrix on held-out test set (n=2000)")
        else:
            st.warning(f"Confusion matrix image not found at {CM_IMAGE_PATH}. Run evaluate.py to generate it.")

st.divider()
st.caption(
    "MachineGuard — Built with Python, Scikit-Learn, and Streamlit | Dataset: UCI Machine Learning Repository AI4I 2020"
)

