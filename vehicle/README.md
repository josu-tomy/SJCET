# Vehicle Maintenance AI — OBD-II Telemetry Architecture

## 1. Overview

The **Vehicle Maintenance AI** layer provides intelligent diagnostics, anomaly detection, and predictive maintenance risk scoring for passenger and fleet automotive vehicles using standard **On-Board Diagnostics (OBD-II)** sensor streams.

It is part of the three-domain **MachineGuard** architecture:

```text
MachineGuard
│
├── 🏭 Industrial AI
│   └── AI4I 2020 Predictive Maintenance (Milling Machine Tool Wear & Failure)
│
├── 🏠 Household AI
│   ├── Household Energy Anomaly Detection (Appliances Energy Consumption)
│   └── Manual Appliance Monitoring (Refrigerator, AC, Washing Machine, etc.)
│
└── 🚗 Vehicle AI
    ├── OBD-II Telemetry Sensor Analysis
    ├── Dual Machine Learning Engine (Unsupervised Anomaly Scoring + Supervised DTC Diagnostics)
    ├── Batch Trip Log Analysis (CSV Upload)
    └── Future ELM327 Hardware Abstraction Layer
```

---

## 2. Dataset Attribution & Integrity

- **Source**: Real-world fleet OBD-II dataset collected across 14 passenger vehicles and 14 drivers during daily driving commutes in Natal-RN, Brazil.
- **Format**: Raw 60,439 rows (47,514 valid timestamped OBD frames), 33 columns (`data/vehicle/raw/exp1_14drivers_14cars_dailyRoutes.csv`).
- **Cleaned Subset**: 29,100 high-quality records with complete core sensor readings and automotive sanity checks (`data/vehicle/processed/vehicle_clean.csv`).
- **Target / Diagnostics**: Real Diagnostic Trouble Codes (DTCs) logged in vehicle ECU memory:
  - `P0133`: O2 Sensor Circuit Slow Response (Bank 1, Sensor 1) — 5,104 clean records
  - `C0300`: Rear Speed Sensor / Chassis Communication Malfunction — 5,611 clean records
  - `P0078` / `P0079` / `P007E` / `P007F`: Exhaust Valve Control Solenoid Circuit — 127 clean records
  - `NORMAL` (No DTC): 18,258 clean records

---

## 3. Telemetry Features & Engineering Metrics

### Core SAE J1979 OBD-II Mode 01 PIDs:

1. `ENGINE_RPM` (`0x0C`): Engine crankshaft rotational speed (RPM).
2. `SPEED` (`0x0D`): Vehicle ground speed from wheel speed sensors (km/h).
3. `ENGINE_COOLANT_TEMP` (`0x05`): Engine cooling circuit temperature (°C).
4. `ENGINE_LOAD` (`0x04`): Calculated engine load percentage (%).
5. `THROTTLE_POS` (`0x11`): Absolute throttle valve opening angle (%).
6. `AIR_INTAKE_TEMP` (`0x0F`): Temperature of ambient air entering the intake manifold (°C).

### Derived Automotive Domain Features:

1. `SPEED_TO_RPM`: Ratio of vehicle speed to engine RPM (effective gear ratio and clutch/torque-converter engagement proxy).
2. `LOAD_TO_THROTTLE`: Engine load relative to throttle opening (indicates manifold vacuum loss, mechanical drag, or engine straining).
3. `TEMP_DIFF`: Coolant temperature minus intake air temperature (monitors thermal gradient and heat rejection efficiency).
4. `IS_IDLE`: Binary indicator for vehicle stationary with running engine.

---

## 4. Dual AI Modeling Architecture

Automotive telemetry datasets often exhibit **vehicle-clustered faults** (e.g. particular vehicles have specific persistent DTCs while others remain healthy). To prevent overfitting or artificial leakage across vehicles, MachineGuard implements a **Dual-Engine Architecture**:

### Engine 1: Unsupervised Anomaly Detection (`IsolationForest`)

- **Training**: Fitted exclusively on normal, healthy baseline telemetry (18,258 records) across all vehicles.
- **Objective**: Maps the high-dimensional normal operating manifold of speed, RPM, load, and temperatures.
- **Output**: Continuous anomaly decision score calibrated to a **[0–100%] Vehicle Risk Index**.
- **ROC-AUC on Fault Separation**: **0.7409** without having seen any fault labels during training.

### Engine 2: Supervised Diagnostic Classifier (`RandomForestClassifier`)

- **Training**: 100 balanced decision trees trained on ground-truth DTC events.
- **Objective**: Predicts the likelihood of specific automotive subsystem failures.
- **Accuracy**: **96.82%**
- **Precision**: **95.88%**
- **Recall**: **95.57%**
- **F1-Score**: **95.73%**
- **ROC-AUC**: **0.9946**

### Top Influential OBD-II Telemetry Sensors:

1. `ENGINE_COOLANT_TEMP`: 35.89%
2. `THROTTLE_POS`: 26.50%
3. `TEMP_DIFF`: 11.83%
4. `LOAD_TO_THROTTLE`: 8.81%
5. `ENGINE_LOAD`: 3.81%
6. `SPEED_TO_RPM`: 3.59%

---

## 5. File Structure & Component Reference

```text
sjcet/
├── data/
│   └── vehicle/
│       ├── raw/
│       │   └── exp1_14drivers_14cars_dailyRoutes.csv   # Raw OBD-II telemetry
│       ├── processed/
│       │   └── vehicle_clean.csv                       # Cleaned, standardized dataset
│       └── README.md                                   # Dataset attribution & metadata
├── vehicle/
│   ├── inspect_vehicle_data.py                         # Statistical & structural inspection
│   ├── preprocess_vehicle.py                           # Preprocessing & feature extraction
│   ├── train_vehicle.py                                # Dual-model training pipeline
│   ├── evaluate_vehicle.py                             # Evaluation benchmark & metrics
│   ├── predict_vehicle.py                              # Real-time inference API
│   ├── vehicle_monitor.py                              # Batch CSV analysis & scenario presets
│   ├── obd_interface.py                                # Hardware abstraction (ELM327 & Simulation)
│   └── README.md                                       # Architecture documentation
├── models/
│   ├── model.joblib                                    # Industrial AI model
│   ├── household_model.joblib                          # Household AI model
│   └── vehicle_model.joblib                            # Vehicle AI model bundle
└── outputs/
    └── vehicle/
        ├── anomaly_score_distribution.png              # Anomaly score density plot
        ├── supervised_confusion_matrix.png             # Confusion matrix heatmap
        ├── feature_importance.png                      # Telemetry feature importances
        └── evaluation_metrics.json                     # Serialized evaluation metrics
```

---

## 6. How to Run & Verify

### Inspect Dataset:

```bash
.venv/bin/python vehicle/inspect_vehicle_data.py
```

### Run Preprocessing:

```bash
.venv/bin/python vehicle/preprocess_vehicle.py
```

### Train Vehicle Models:

```bash
.venv/bin/python vehicle/train_vehicle.py
```

### Run Diagnostic Evaluation:

```bash
.venv/bin/python vehicle/evaluate_vehicle.py
```

### Test Inference:

```bash
.venv/bin/python vehicle/predict_vehicle.py
```

### Test Telemetry Monitor & Batch CSV:

```bash
.venv/bin/python vehicle/vehicle_monitor.py
```

### Test Hardware Abstraction:

```bash
.venv/bin/python vehicle/obd_interface.py
```

---

## 7. Future Physical Hardware Integration (ELM327)

`vehicle/obd_interface.py` contains production-ready AT command drivers for physical ELM327 Bluetooth/USB/WiFi adapters:

1. **Bluetooth Setup (Linux)**:
   ```bash
   # Pair scanner and bind to virtual serial port
   sudo rfcomm bind 0 <BLUETOOTH_MAC_ADDRESS>
   ```
2. **Initialize Connection**:

   ```python
   from vehicle.obd_interface import create_obd_connection

   # Connects via /dev/rfcomm0, falls back gracefully to simulation if unplugged
   adapter = create_obd_connection(mode="hardware", port="/dev/rfcomm0")
   for frame in adapter.stream_telemetry(interval_sec=1.0):
       print(frame.to_dict())
   ```
