"""
MachineGuard — Model Training Pipeline
Trains a RandomForestClassifier on the UCI AI4I 2020 Predictive Maintenance Dataset.

Person 2: Machine Learning Engineer
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "ai4i2020.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")

# Required 5 features & target
FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"


def load_and_split_data(csv_path: str = DATA_PATH):
    """Load dataset, select the 5 features, and perform 80/20 stratified split."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train RandomForestClassifier with balanced class weighting."""
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    return rf


def save_model(model: RandomForestClassifier, output_path: str = MODEL_PATH):
    """Save trained model using joblib."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model successfully saved to: {output_path}")


def main():
    print("=" * 60)
    print("MachineGuard — Training Pipeline")
    print("=" * 60)

    print(f"Loading data from: {DATA_PATH}")
    X_train, X_test, y_train, y_test = load_and_split_data()
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set:     {X_test.shape[0]} samples")

    print("\nTraining RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')...")
    model = train_model(X_train, y_train)
    print("Model training complete.")

    save_model(model, MODEL_PATH)
    print("=" * 60)


if __name__ == "__main__":
    main()

