"""
Model Training Pipeline for Vehicle Maintenance AI.

Trains dual predictive maintenance models on OBD-II telemetry:
1. Unsupervised Anomaly Detector (Isolation Forest):
   Learns multi-dimensional normal vehicle operating manifolds (RPM, Speed,
   Coolant Temp, Engine Load, Throttle Position, Intake Air Temp, Derived Ratios).
   Yields continuous anomaly scores and calibrated Risk Indices [0-100%].
2. Supervised Diagnostic Classifier (Random Forest):
   Trained on ground-truth Diagnostic Trouble Code (DTC) occurrences to classify
   and quantify specific automotive subsystem failure risks.

Outputs:
- models/vehicle_model.joblib: Complete serialized model bundle
- outputs/vehicle/anomaly_score_distribution.png
- outputs/vehicle/supervised_confusion_matrix.png
- outputs/vehicle/feature_importance.png
- outputs/vehicle/evaluation_metrics.json
"""

import os
import json
import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "vehicle", "processed", "vehicle_clean.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "vehicle_model.joblib")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "vehicle")

FEATURE_COLUMNS = [
    "ENGINE_RPM",
    "SPEED",
    "ENGINE_COOLANT_TEMP",
    "ENGINE_LOAD",
    "THROTTLE_POS",
    "AIR_INTAKE_TEMP",
    "SPEED_TO_RPM",
    "LOAD_TO_THROTTLE",
    "TEMP_DIFF",
    "IS_IDLE"
]


def train_vehicle_models():
    print("=" * 70)
    print("VEHICLE MAINTENANCE AI — MODEL TRAINING PIPELINE")
    print("=" * 70)

    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Clean vehicle data not found at {PROCESSED_DATA_PATH}. Run preprocess_vehicle.py first.")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[1/5] Loading preprocessed telemetry from {PROCESSED_DATA_PATH}...")
    df = pd.read_csv(PROCESSED_DATA_PATH, low_memory=False)
    print(f"      Total records: {len(df):,} | Features: {len(FEATURE_COLUMNS)}")

    X = df[FEATURE_COLUMNS].copy()
    y = df["FAULT_PRESENT"].copy()

    # Stratified 80/20 train/test split with deterministic random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Train set: {len(X_train):,} (Normal: {(y_train == 0).sum():,}, Fault: {(y_train == 1).sum():,})")
    print(f"      Test set:  {len(X_test):,} (Normal: {(y_test == 0).sum():,}, Fault: {(y_test == 1).sum():,})")

    # Fit feature standardizer
    print("[2/5] Standardizing telemetry features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Unsupervised Anomaly Detection: Isolation Forest trained on healthy baselines
    print("[3/5] Training Unsupervised Isolation Forest on normal baseline telemetry...")
    normal_train_mask = (y_train == 0)
    X_normal_train = X_train_scaled[normal_train_mask]

    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=0.15,
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_normal_train)

    # Anomaly decision function: raw scores (negative of decision_function so higher = more anomalous)
    raw_train_scores = -iso_forest.decision_function(X_normal_train)
    raw_test_scores = -iso_forest.decision_function(X_test_scaled)

    # Calibration parameters for risk scoring [0-100%]
    score_min = float(np.percentile(raw_train_scores, 1))
    score_max = float(np.percentile(raw_test_scores, 99))
    if score_max <= score_min:
        score_max = score_min + 1.0

    iso_roc_auc = float(roc_auc_score(y_test, raw_test_scores))
    iso_pr_auc = float(average_precision_score(y_test, raw_test_scores))
    print(f"      Isolation Forest ROC-AUC on fault detection: {iso_roc_auc:.4f}")
    print(f"      Isolation Forest PR-AUC on fault detection:  {iso_pr_auc:.4f}")

    # 2. Supervised Classifier: Random Forest
    print("[4/5] Training Supervised Random Forest Classifier on labeled fault conditions...")
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(X_train_scaled, y_train)

    y_test_pred = rf_clf.predict(X_test_scaled)
    y_test_proba = rf_clf.predict_proba(X_test_scaled)[:, 1]

    rf_acc = float(accuracy_score(y_test, y_test_pred))
    rf_prec = float(precision_score(y_test, y_test_pred))
    rf_rec = float(recall_score(y_test, y_test_pred))
    rf_f1 = float(f1_score(y_test, y_test_pred))
    rf_roc_auc = float(roc_auc_score(y_test, y_test_proba))
    rf_pr_auc = float(average_precision_score(y_test, y_test_proba))

    print(f"      Supervised RF Accuracy:  {rf_acc * 100:.2f}%")
    print(f"      Supervised RF Precision: {rf_prec * 100:.2f}%")
    print(f"      Supervised RF Recall:    {rf_rec * 100:.2f}%")
    print(f"      Supervised RF F1-Score:  {rf_f1 * 100:.2f}%")
    print(f"      Supervised RF ROC-AUC:   {rf_roc_auc:.4f}")

    # 3. Generate Visual Artifacts and Evaluation Outputs
    print("[5/5] Generating evaluation reports and diagnostic plots in outputs/vehicle/...")
    
    # A. Anomaly Score Distribution Plot
    plt.figure(figsize=(9, 5))
    sns.set_theme(style="whitegrid")
    sns.kdeplot(raw_test_scores[y_test == 0], label="Normal Telemetry (No DTC)", fill=True, color="#2ecc71", alpha=0.4, linewidth=2)
    sns.kdeplot(raw_test_scores[y_test == 1], label="Fault Telemetry (DTC Logged)", fill=True, color="#e74c3c", alpha=0.4, linewidth=2)
    plt.axvline(np.percentile(raw_train_scores, 85), color="#f39c12", linestyle="--", linewidth=1.5, label="Advisory Threshold (85th %ile)")
    plt.title("Vehicle AI — Isolation Forest Anomaly Score Distribution", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Anomaly Score (Higher indicates greater deviation from normal operating envelope)")
    plt.ylabel("Density")
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    anomaly_plot_path = os.path.join(OUTPUT_DIR, "anomaly_score_distribution.png")
    plt.savefig(anomaly_plot_path, dpi=180)
    plt.close()
    print(f"      Saved anomaly plot: {anomaly_plot_path}")

    # B. Supervised Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal (0)", "Fault Present (1)"],
        yticklabels=["Normal (0)", "Fault Present (1)"],
        cbar=True
    )
    plt.title("Vehicle AI — Random Forest Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Condition")
    plt.ylabel("Ground Truth Condition")
    plt.tight_layout()
    cm_plot_path = os.path.join(OUTPUT_DIR, "supervised_confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=180)
    plt.close()
    print(f"      Saved confusion matrix: {cm_plot_path}")

    # C. Feature Importance Plot
    feat_imp = pd.Series(rf_clf.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=True)
    plt.figure(figsize=(9, 6))
    feat_imp.plot(kind="barh", color="#3498db", edgecolor="#2980b9", alpha=0.85)
    plt.title("Vehicle AI — Feature Importance (Random Forest)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Gini Feature Importance")
    plt.ylabel("OBD-II Sensor Feature")
    plt.tight_layout()
    feat_plot_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(feat_plot_path, dpi=180)
    plt.close()
    print(f"      Saved feature importance plot: {feat_plot_path}")

    # D. Save JSON metrics
    metrics_summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": FEATURE_COLUMNS,
        "isolation_forest": {
            "n_estimators": 150,
            "contamination": 0.15,
            "roc_auc": round(iso_roc_auc, 4),
            "pr_auc": round(iso_pr_auc, 4),
            "score_min": round(score_min, 4),
            "score_max": round(score_max, 4)
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 12,
            "accuracy": round(rf_acc, 4),
            "precision": round(rf_prec, 4),
            "recall": round(rf_rec, 4),
            "f1_score": round(rf_f1, 4),
            "roc_auc": round(rf_roc_auc, 4),
            "pr_auc": round(rf_pr_auc, 4),
            "confusion_matrix": cm.tolist()
        },
        "feature_importances": {k: round(float(v), 4) for k, v in feat_imp.items()}
    }

    metrics_json_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"      Saved metrics JSON: {metrics_json_path}")

    # Package and save complete model bundle
    bundle = {
        "isolation_forest": iso_forest,
        "supervised_classifier": rf_clf,
        "scaler": scaler,
        "features": FEATURE_COLUMNS,
        "score_min": score_min,
        "score_max": score_max,
        "metrics": metrics_summary,
        "saved_at": datetime.datetime.now().isoformat()
    }

    joblib.dump(bundle, MODEL_SAVE_PATH, compress=3)
    print(f"\n[Vehicle Training Complete] Model bundle successfully saved to {MODEL_SAVE_PATH}")
    print(f"File size: {os.path.getsize(MODEL_SAVE_PATH) / 1024 / 1024:.2f} MB")
    print("=" * 70)


if __name__ == "__main__":
    train_vehicle_models()

