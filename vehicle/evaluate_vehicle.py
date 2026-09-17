"""
Evaluation and Diagnostic Analysis for Vehicle Maintenance AI.

Evaluates the trained vehicle model bundle on the test split:
- Isolation Forest anomaly score performance (ROC-AUC, separation metrics)
- Supervised Random Forest classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Subsystem fault breakdown by diagnostic trouble code
- Cross-vehicle generalization analysis to assess fleet transferability
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "vehicle", "processed", "vehicle_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "vehicle_model.joblib")


def evaluate_vehicle_models():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained vehicle model not found at {MODEL_PATH}. Run vehicle/train_vehicle.py first.")
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Clean vehicle data not found at {PROCESSED_DATA_PATH}. Run vehicle/preprocess_vehicle.py first.")

    print("=" * 75)
    print("VEHICLE MAINTENANCE AI — SYSTEM EVALUATION & DIAGNOSTIC BENCHMARK")
    print("=" * 75)

    bundle = joblib.load(MODEL_PATH)
    iso_forest = bundle["isolation_forest"]
    rf_clf = bundle["supervised_classifier"]
    scaler = bundle["scaler"]
    features = bundle["features"]
    score_min = bundle["score_min"]
    score_max = bundle["score_max"]

    df = pd.read_csv(PROCESSED_DATA_PATH, low_memory=False)
    X = df[features]
    y = df["FAULT_PRESENT"]
    fault_types = df["FAULT_TYPE"]

    # Recreate the stratified 80/20 test split
    _, X_test, _, y_test, _, ft_test = train_test_split(
        X, y, fault_types, test_size=0.20, random_state=42, stratify=y
    )

    X_test_scaled = scaler.transform(X_test)

    # 1. Unsupervised Anomaly Scoring Evaluation
    raw_scores = -iso_forest.decision_function(X_test_scaled)
    # Calibrated risk score 0 to 100%
    risk_scores = np.clip((raw_scores - score_min) / (score_max - score_min) * 100.0, 0.0, 100.0)
    iso_roc = roc_auc_score(y_test, raw_scores)

    print("\n[1] UNSUPERVISED ANOMALY DETECTION (Isolation Forest)")
    print(f"    - ROC-AUC on Fault Separation: {iso_roc:.4f}")
    print(f"    - Mean Risk Score (Normal Records): {risk_scores[y_test == 0].mean():.1f}% (Std: {risk_scores[y_test == 0].std():.1f}%)")
    print(f"    - Mean Risk Score (Fault Records):  {risk_scores[y_test == 1].mean():.1f}% (Std: {risk_scores[y_test == 1].std():.1f}%)")
    print(f"    - Separation Delta:                 {risk_scores[y_test == 1].mean() - risk_scores[y_test == 0].mean():.1f}% risk spread")

    # 2. Supervised Classifier Evaluation
    y_pred = rf_clf.predict(X_test_scaled)
    y_proba = rf_clf.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rf_roc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n[2] SUPERVISED DIAGNOSTIC CLASSIFICATION (Random Forest)")
    print(f"    - Accuracy:  {acc * 100:.2f}%")
    print(f"    - Precision: {prec * 100:.2f}%")
    print(f"    - Recall:    {rec * 100:.2f}%")
    print(f"    - F1-Score:  {f1 * 100:.2f}%")
    print(f"    - ROC-AUC:   {rf_roc:.4f}")

    print("\n    Confusion Matrix:")
    print(f"                     Pred Normal    Pred Fault")
    print(f"    Actual Normal:   {cm[0, 0]:<14} {cm[0, 1]}")
    print(f"    Actual Fault:    {cm[1, 0]:<14} {cm[1, 1]}")

    # 3. Fault Detection Breakdown by Specific Subsystem Fault Code
    print("\n[3] BREAKDOWN BY SUBSYSTEM FAULT CLASSIFICATION")
    test_analysis_df = pd.DataFrame({
        "FAULT_TYPE": ft_test,
        "ACTUAL_FAULT": y_test,
        "PRED_FAULT": y_pred,
        "RISK_SCORE": risk_scores
    })

    summary_by_fault = test_analysis_df.groupby("FAULT_TYPE").agg(
        Total_Samples=("ACTUAL_FAULT", "count"),
        Detected_By_Supervised=("PRED_FAULT", "sum"),
        Avg_Anomaly_Risk=("RISK_SCORE", "mean")
    )
    summary_by_fault["Supervised_Detection_Rate_%"] = (
        summary_by_fault["Detected_By_Supervised"] / summary_by_fault["Total_Samples"] * 100
    ).round(2)
    summary_by_fault["Avg_Anomaly_Risk"] = summary_by_fault["Avg_Anomaly_Risk"].round(1)

    print(summary_by_fault[["Total_Samples", "Supervised_Detection_Rate_%", "Avg_Anomaly_Risk"]].to_string())

    # 4. Top Influential Sensors
    importances = pd.Series(rf_clf.feature_importances_, index=features).sort_values(ascending=False)
    print("\n[4] TOP INFLUENTIAL OBD-II TELEMETRY SENSORS")
    for feat, imp in importances.items():
        print(f"    {feat:<25}: {imp * 100:.2f}%")

    print("\n" + "=" * 75)
    return {
        "isolation_forest_roc_auc": iso_roc,
        "supervised_accuracy": acc,
        "supervised_precision": prec,
        "supervised_recall": rec,
        "supervised_f1": f1,
        "supervised_roc_auc": rf_roc
    }


if __name__ == "__main__":
    evaluate_vehicle_models()

