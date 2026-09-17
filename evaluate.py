"""
MachineGuard — Model Evaluation Pipeline
Evaluates the trained RandomForestClassifier on the held-out test set.

Person 2: Machine Learning Engineer
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "ai4i2020.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
CM_PLOT_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
FI_PLOT_PATH = os.path.join(OUTPUTS_DIR, "feature_importance.png")

# Required 5 features & target
FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"


def load_test_data(csv_path: str = DATA_PATH):
    """Load and return the held-out test split (20% stratified, random_state=42)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    X = df[FEATURES]
    y = df[TARGET]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_test, y_test


def evaluate():
    print("=" * 60)
    print("MachineGuard — Model Evaluation on Held-Out Test Set")
    print("=" * 60)

    # 1. Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run train.py first.")
    
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")

    # 2. Load held-out test set
    X_test, y_test = load_test_data()
    print(f"Held-out test set: {len(X_test)} samples (Class 0: {(y_test == 0).sum()}, Class 1: {(y_test == 1).sum()})")

    # 3. Generate predictions & probability predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Failure-associated probability

    # 4. Calculate metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Evaluation Metrics (Held-Out Test Set) ---")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1 * 100:.2f}%)")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Failure Risk (1)"]))

    # 5. Generate and save Confusion Matrix plot
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal (0)", "Failure (1)"])
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title("MachineGuard — Confusion Matrix (Held-Out Test Set)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Pattern", fontsize=10)
    ax.set_ylabel("True Condition", fontsize=10)
    plt.tight_layout()
    plt.savefig(CM_PLOT_PATH, dpi=300)
    plt.close()
    print(f"\nSaved Confusion Matrix plot: {CM_PLOT_PATH}")

    # 6. Generate and save Feature Importance plot
    importances = model.feature_importances_
    indices = np.argsort(importances)  # Ascending for horizontal bar plot

    sorted_features = [FEATURES[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(sorted_features, sorted_importances, color="#1f77b4", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Model Importance (Gini Impurity Reduction)", fontsize=10, fontweight="bold")
    ax.set_title("Random Forest Feature Importance\n(Statistical Model Association, Not Causality)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlim(0, max(sorted_importances) * 1.15)

    # Annotate bar values
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.008, bar.get_y() + bar.get_height() / 2, f"{width:.4f}", ha="left", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(FI_PLOT_PATH, dpi=300)
    plt.close()
    print(f"Saved Feature Importance plot: {FI_PLOT_PATH}")

    print("\nFeature Model Importances:")
    for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True):
        print(f"  - {feat:30s}: {imp:.4f} ({imp * 100:.2f}%)")

    print("=" * 60)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "feature_importances": dict(zip(FEATURES, importances)),
    }


if __name__ == "__main__":
    evaluate()

