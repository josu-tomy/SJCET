# MachineGuard — AI Predictive Maintenance Risk Detector

An AI-driven predictive maintenance risk detection application that analyzes multivariate industrial sensor telemetry to detect equipment operating conditions resembling failure-associated patterns.

- **Live Public Demo:** [https://stories-cement-remember-shift.trycloudflare.com](https://stories-cement-remember-shift.trycloudflare.com)
- **GitHub Repository:** [https://github.com/josu-tomy/sjcet](https://github.com/josu-tomy/sjcet)

---

## 1. Problem Statement

Modern manufacturing machinery operates under continuous thermal, rotational, and mechanical stresses. Single-parameter threshold alarms often fail to detect impending malfunctions because machine failures frequently originate from complex, non-linear interactions across multiple operating parameters (e.g., moderate torque combined with high tool wear and elevated thermal gradient). These multi-parameter failure-associated patterns are difficult to detect manually or through static threshold rules.

---

## 2. Solution

**MachineGuard** implements a machine learning classification and risk estimation pipeline trained on empirical industrial milling machine telemetry. Using an ensemble **Random Forest Classifier** with class-weighted balancing, MachineGuard:
1. Ingests 5 primary physical operating measurements.
2. Identifies multivariate patterns statistically associated with machine failures.
3. Computes a continuous, calibrated **failure-associated probability score** ($0.00$ to $1.00$).
4. Categorizes the operational state into prototype risk tiers (`NORMAL`, `WARNING`, `HIGH RISK`) to assist maintenance engineers in prioritizing inspection.

---

## 3. Dataset & Attribution

- **Dataset:** [UCI Machine Learning Repository: AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset)
- **Official Source URL:** `https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset`
- **Citation & License:**
  > Stephan Matzka, "AI4I 2020 Predictive Maintenance Dataset", UCI Machine Learning Repository, 2020. DOI: [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C). Distributed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **Data Scope:** 10,000 synthetic records reflecting real-world industrial milling machine operations with zero missing values.
- **Model Features (5 Selected Inputs):**
  1. `Air temperature [K]` (ambient factory temperature)
  2. `Process temperature [K]` (internal equipment temperature)
  3. `Rotational speed [rpm]` (spindle angular velocity)
  4. `Torque [Nm]` (applied mechanical torque)
  5. `Tool wear [min]` (cumulative tool cut duration)
- **Target Variable:** `Machine failure` (binary: `0` = normal operation, `1` = equipment failure).
- **Excluded Attributes:** Identification columns (`UDI`, `Product ID`), machine type, and explicit failure-mode labels (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) to prevent target leakage.

---

## 4. AI / Machine Learning Architecture

- **Learning Paradigm:** Supervised Learning (Binary Classification).
- **Algorithm:** `RandomForestClassifier` from Scikit-Learn:
  - `n_estimators = 100`
  - `random_state = 42`
  - `class_weight = "balanced"` (corrects for the ~96.6% / 3.4% class imbalance without synthetic data distortion).
- **Data Partitioning:** 80% training (8,000 samples) and 20% held-out test split (2,000 samples), strictly stratified by the target label (`random_state=42`).
- **Probability Estimation:** Evaluated via ensemble vote fraction (`predict_proba()[:, 1]`), quantifying the model-generated failure-associated probability.
- **Feature Importance:** Derived using Gini impurity reduction across all decision trees.

---

## 5. Evaluation & Performance Metrics

All performance metrics were calculated on the **held-out test set** (2,000 samples, containing 68 failure cases and 1,932 normal cases). Training set metrics are not reported as test performance.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Precision** | **71.21%** (0.7121) | When the model flags a failure risk, it is correct in ~71.2% of cases. |
| **Recall** | **69.12%** (0.6912) | The model successfully captures ~69.1% of actual failure-associated conditions. |
| **F1-Score** | **70.15%** (0.7015) | Balanced harmonic mean optimizing both precision and sensitivity under high class imbalance. |
| **Overall Accuracy** | **98.00%** (0.9800) | Total percentage of correct predictions across all test instances. |

### Confusion Matrix (Held-Out Test Set, n=2,000)

| | Predicted Normal (0) | Predicted Failure Risk (1) |
| :--- | :---: | :---: |
| **Actual Normal (0)** | **1,913** (True Negatives) | **19** (False Positives) |
| **Actual Failure (1)** | **21** (False Negatives) | **47** (True Positives) |

---

## 6. Visualizations

The pipeline generates and displays two diagnostic visual artifacts stored in `outputs/`:

1. **Confusion Matrix (`outputs/confusion_matrix.png`):**
   Visualizes the breakdown of True Negatives (1,913), True Positives (47), False Positives (19), and False Negatives (21) on the held-out test set.
2. **Feature Importance Bar Chart (`outputs/feature_importance.png`):**
   Displays the relative influence of each input variable to the Random Forest model:
   - **Torque [Nm]:** 33.30%
   - **Rotational speed [rpm]:** 30.09%
   - **Tool wear [min]:** 22.04%
   - **Air temperature [K]:** 8.60%
   - **Process temperature [K]:** 5.96%

---

## 7. Prototype Risk Level Thresholds

| Probability Range | Prototype Status | Visual Indicator | Operational Interpretation |
| :--- | :--- | :---: | :--- |
| **0.00 – 0.39** | **NORMAL** | 🟢 | Operating condition resembles nominal baseline patterns. |
| **0.40 – 0.69** | **WARNING** | 🟡 | Operating condition exhibits elevated stress patterns approaching failure regions. |
| **0.70 – 1.00** | **HIGH RISK** | 🔴 | Operating condition strongly resembles failure-associated patterns according to the trained model. |

> **Disclaimer:** These thresholds represent a **prototype visualization threshold — not an industrial safety limit**.

---

## 8. Technologies Used

- **Language:** Python 3.14 / 3.10+
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, Joblib
- **Visualization:** Matplotlib, Seaborn
- **Web Application:** Streamlit

---

## 9. AI Tool Disclosure

Generative AI and coding assistants (Antigravity / Gemini) were utilized during project development for code scaffolding, boilerplate generation, automated testing scripts, and documentation drafting. The engineering team reviewed, tested, validated, and verified all code, mathematical formulations, models, and outputs.

---

## 10. Limitations & Responsible AI Notice

1. **Benchmark Data:** Trained on the synthetic public benchmark UCI AI4I 2020 dataset; not calibrated to any specific physical industrial facility or individual manufacturing machine.
2. **Probabilistic Association vs. Certainty:** The failure-associated probability score reflects statistical pattern correlation; it **does not** guarantee that a physical breakdown will or will not occur.
3. **Safety Criticality:** The prototype thresholds are intended for operational prioritization and visualization, **not** as certified industrial safety limits or emergency shutdown parameters.
4. **Causality:** Feature importance indicates statistical model influence across decision tree splits, **not** direct physical or mechanical causality.

---

## 11. Quickstart & Local Execution

### Prerequisites
Python 3.10+ installed.

### Setup and Execution
```bash
# 1. Clone repository
git clone https://github.com/josu-tomy/sjcet.git
cd sjcet

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Run preprocessing and evaluation
python preprocess.py
python train.py
python evaluate.py

# 4. Launch Streamlit Dashboard
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
