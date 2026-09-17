# MachineGuard — AI-Powered Equipment & Energy Anomaly Monitoring

An AI-driven predictive maintenance and equipment anomaly detection platform that bridges **industrial machinery risk detection** and **household equipment energy/operating behavior monitoring**.

- **GitHub Repository:** [https://github.com/josu-tomy/sjcet](https://github.com/josu-tomy/sjcet)
- **Primary Framework:** Python, Scikit-Learn, Streamlit, Pandas, Matplotlib

---

## 1. Problem Statement

Both industrial machinery and household appliances operate under continuous thermal, electrical, and mechanical stresses. Equipment frequently exhibits subtle, non-linear operating pattern deviations—such as unusual energy consumption spikes, elevated vibration, thermal buildup, or abnormal duty cycles—well before physical breakdown or catastrophic failure occurs. These multivariate warning patterns are difficult to detect manually or through static single-parameter trip thresholds.

---

## 2. Solution Overview

**MachineGuard** implements a modular, two-layer diagnostic architecture:

1. **Industrial Predictive Maintenance Layer:**
   - Pre-trained on the **UCI AI4I 2020 Predictive Maintenance Dataset**.
   - Evaluates 5 mechanical/thermal measurements using a class-balanced **Random Forest Classifier**.
   - Outputs failure-associated probabilities and classifies operating states into prototype risk tiers (`NORMAL`, `WARNING`, `HIGH RISK`).
2. **Household Equipment & Energy Monitoring Layer:**
   - Trained on the **UCI Appliances Energy Prediction Dataset (UCI ID: 374)**.
   - An **ExtraTreesRegressor** model predicts expected household appliance energy demand in Watt-hours [Wh] given ambient environmental and temporal conditions.
   - Computes prediction residuals ($|\text{actual} - \text{predicted}|$) to score and flag anomalous energy spikes or standby power leakage.
3. **Appliance-Specific Manual Monitoring:**
   - Dedicated profiles for 6 household equipment types: **Air Conditioner, Refrigerator, Washing Machine, Television, Water Pump, Ceiling Fan**.
   - Evaluates manually entered telemetry against documented engineering reference heuristics.
4. **Transparent Natural-Language Symptom Parser:**
   - Allows users to describe symptoms in plain text.
   - Transparent keyword/rule extraction detects equipment type, symptoms, and potential concerns without fabricating numeric sensor data.
5. **Future Sensor Integration Architecture:**
   - Hardware-independent gateway abstraction (`household/sensor_interface.py`) designed to interface with physical IoT edge microcontrollers (ESP32, current transducers, accelerometers) over MQTT or REST in future deployments.

---

## 3. Dataset Attribution & Integrity

### A. Industrial Dataset: UCI AI4I 2020 Predictive Maintenance Dataset

- **UCI Dataset ID:** 601
- **Official Source URL:** https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset
- **Citation:** Stephan Matzka, "AI4I 2020 Predictive Maintenance Dataset", UCI Machine Learning Repository, 2020. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Scope:** 10,000 synthetic records reflecting real-world industrial milling machine operations.
- **Model Features (5 Inputs):** Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min].
- **Target:** `Machine failure` (binary classification).

### B. Household Dataset: UCI Appliances Energy Prediction Dataset

- **UCI Dataset ID:** 374
- **Official Source URL:** https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
- **Citation:** Luis M. Candanedo, Véronique Feldheim, Dominique Deramaix, "Data driven prediction models of energy use of appliances in a low-energy house", Energy and Buildings, Volume 140, 2017, Pages 81-97. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Scope:** 19,735 records sampled every 10 minutes over 4.5 months across 9 indoor household climate zones and local weather telemetry.
- **Target:** `Appliances` (Energy use in Watt-hours [Wh]).
- **Features Used:** Indoor zone temperatures (`T1`–`T9`), relative humidity (`RH_1`–`RH_9`), outdoor weather (`T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint`), and engineered time features (`hour`, `day_of_week`, `month`, `is_weekend`, `sin_hour`, `cos_hour`). Non-predictive synthetic columns (`rv1`, `rv2`) were removed.

> **Data Integrity Notice:** The UCI household dataset measures aggregate household appliance energy consumption; it does **not** provide separate labeled failure classifications for individual appliances. MachineGuard keeps data-driven energy regression strictly separated from appliance-specific heuristic monitoring.

---

## 4. AI / Machine Learning Architecture

### Model 1: Industrial Random Forest Classifier

- **Algorithm:** `RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')`
- **Data Partitioning:** 80% training (8,000 samples) and 20% held-out test split (2,000 samples), strictly stratified (`random_state=42`).
- **Performance (Held-Out Test Set):**
  - **Precision:** `71.21%`
  - **Recall:** `69.12%`
  - **F1-Score:** `70.15%`
  - **Overall Accuracy:** `98.00%`

### Model 2: Household Energy Regression & Anomaly Detector

- **Learning Paradigm:** Supervised Regression with Residual-Based Anomaly Scoring.
- **Algorithm:** `ExtraTreesRegressor(n_estimators=100, max_depth=14, min_samples_leaf=10, random_state=42)`
- **Data Partitioning:** Chronological time-series split (no random shuffling):
  - **Training Set (Earliest 70%):** 13,814 samples
  - **Validation Set (Next 15%):** 2,960 samples
  - **Final Test Set (Latest 15%):** 2,961 samples
- **Model Selection (Validation Set):**
  - `ExtraTreesRegressor`: **Val MAE: 46.32 Wh** | RMSE: 85.21 Wh | R²: +0.1466 _(Selected Best)_
  - `HistGradientBoostingRegressor`: Val MAE: 48.97 Wh | RMSE: 88.02 Wh | R²: +0.0894
  - `RandomForestRegressor`: Val MAE: 52.96 Wh | RMSE: 90.06 Wh | R²: +0.0466
- **Test Set Evaluation (Evaluated Once on Held-out 15%):**
  - **Test MAE:** `83.34 Wh`
  - **Test RMSE:** `112.27 Wh`
  - **Test R²:** `-0.5264` _(Reflects significant seasonal climate drift between winter training data and late spring test data)_
- **Calibrated Residual Anomaly Thresholds (from Validation Errors):**
  - **Warning Threshold (80th percentile):** `65.68 Wh` deviation
  - **Anomaly Threshold (95th percentile):** `151.35 Wh` deviation

---

## 5. Prototype Risk & Anomaly Level Thresholds

| System Layer            | Level         | Indicator | Operational Meaning                                                                       |
| :---------------------- | :------------ | :-------: | :---------------------------------------------------------------------------------------- |
| **Industrial Model**    | **NORMAL**    |    🟢     | Probability $< 40.0\%$. Matches nominal historical baseline.                              |
| **Industrial Model**    | **WARNING**   |    🟡     | Probability $40.0\% \le p < 70.0\%$. Elevated multi-parameter stress.                     |
| **Industrial Model**    | **HIGH RISK** |    🔴     | Probability $\ge 70.0\%$. Strongly resembles failure-associated patterns.                 |
| **Household Energy AI** | **NORMAL**    |    🟢     | Residual $< 65.68\text{ Wh}$. Observed energy matches model expectations.                 |
| **Household Energy AI** | **WARNING**   |    🟡     | $65.68\text{ Wh} \le \text{Residual} < 151.35\text{ Wh}$. Moderate consumption variance.  |
| **Household Energy AI** | **ANOMALY**   |    🔴     | $\text{Residual} \ge 151.35\text{ Wh}$. Significant spike/leakage; potential malfunction. |

---

## 6. Transparent Natural-Language Symptom Parser

The application includes an optional text observation interface for user convenience. It employs a transparent rule-based keyword pattern extraction pipeline:

- **Detected Equipment:** Identifies appliance type (e.g., Washing Machine, Air Conditioner).
- **Detected Symptoms:** Identifies qualitative physical signals (e.g., High mechanical vibration, cooling deficiency, thermal elevation).
- **Detected Concerns:** Suggests potential underlying mechanisms for inspection.
- **Honest Data Handling:** Explicitly displays `"Numeric measurement not provided"` rather than fabricating artificial numbers. Prompts user to confirm physical readings for quantitative evaluation.

---

## 7. Future Physical Sensor Integration Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                 Physical Sensors / IoT Edge                 │
│  (ESP32, Current CT Transducer, Accelerometer, Thermocouple)│
└──────────────────────────────┬──────────────────────────────┘
                               │ MQTT / Serial / REST
                               ▼
┌─────────────────────────────────────────────────────────────┐
│          household/sensor_interface.py (Gateway)            │
│      (Standardizes TelemetryPacket & Protocol Adapters)     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      MachineGuard Core                      │
│   - Industrial Classifier (RandomForest 5-Feature Risk)     │
│   - Household Energy Regressor (ExtraTrees Residual Anomaly)│
│   - Appliance Reference Rules & NLP Keyword Symptom Parser  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  Real-Time Alert Dashboard                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Limitations & Responsible AI Notice

1. **Benchmark Scope:** The industrial model is trained on synthetic milling machine telemetry; the household model is trained on a single low-energy smart house testbed. Neither is certified for arbitrary physical equipment.
2. **Probability vs. Guarantee:** Statistical pattern similarity **does not guarantee** that a machine will or will not fail in the future.
3. **Threshold Heuristics:** Prototype thresholds (`NORMAL`, `WARNING`, `ANOMALY`, `HIGH RISK`) are visualization and prioritization tools, **not certified industrial safety trip points**.
4. **Appliance-Specific Failure Data:** The UCI household dataset does not contain individual appliance failure labels. Manual appliance evaluation uses heuristic reference boundaries.
5. **Causality:** Feature importance describes statistical tree split contribution, **not physical causality**.
6. **Natural Language Input:** A convenience extraction tool; it does not replace calibrated physical sensor instrumentation.

---

## 9. Future Work

- [ ] Microcontroller firmware integration (ESP32 / Arduino) sending real-time packets via MQTT.
- [ ] Ingestion of appliance-specific labeled vibration and acoustic run-to-failure datasets.
- [ ] Dynamic self-calibrating household baselines adapted to individual occupancy schedules.
- [ ] Automated SMS / webhook notification dispatch upon sustained anomaly detection.

---

## 10. Quickstart & Local Execution

```bash
# 1. Clone repository
git clone https://github.com/josu-tomy/sjcet.git
cd sjcet

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Run pipeline scripts
python preprocess.py
python train.py
python evaluate.py
python household/inspect_household_data.py
python household/preprocess_household.py
python household/train_household.py

# 4. Launch Unified Streamlit Dashboard
streamlit run app.py
```

Open your browser at `http://localhost:8501`.
