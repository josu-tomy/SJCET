# Vehicle Maintenance AI Dataset — OBD-II Telemetry

## 1. Dataset Overview & Source

- **Dataset Name**: OBD-II Fleet Telemetry & Trouble Codes Dataset (`exp1_14drivers_14cars_dailyRoutes.csv`)
- **Original Research / Source**: Collected via ELM327 OBD-II Bluetooth adapters and Android OBD Reader application during daily commutes across 14 drivers and 14 distinct vehicles in Natal-RN, Brazil.
- **Public Repository / Reference**: Cephas Barreto / Yudong Fan, hosted on Kaggle and GitHub (`cephasax/obdii-ds3`, `Yudong-Fan/OBD-II-Eco-driving-assistant`).
- **Licensing**: Public Academic Research Dataset (Open Access for research, education, and development).

---

## 2. Dataset Dimensions & Structure

- **Raw Total Rows**: 60,439 rows (47,514 valid timestamped OBD readings)
- **Raw Columns**: 33 features
- **Monitored Vehicles**: 14 passenger vehicles across different manufacturers (Chevrolet Agile, Toyota Corolla, Nissan Versa, Citroen C3, VW Voyage, Ford EcoSport, Honda Fit, Renault Duster, VW Polo, Ford Focus, Fiat Palio, Peugeot 208 Allure, etc.) spanning model years 2003–2016.

---

## 3. Key Telemetry Parameters

The dataset records standard SAE J1979 / ISO 15031-5 OBD-II Mode 01 Parameter IDs (PIDs):

| OBD-II Parameter           | PID    | Description                                         | Units        |
| :------------------------- | :----- | :-------------------------------------------------- | :----------- |
| `ENGINE_RPM`               | `0x0C` | Rotational speed of the crankshaft                  | RPM          |
| `SPEED`                    | `0x0D` | Vehicle speed reported by wheel speed sensors       | km/h         |
| `ENGINE_COOLANT_TEMP`      | `0x05` | Internal combustion engine coolant temperature      | °C           |
| `ENGINE_LOAD`              | `0x04` | Calculated engine load percentage                   | %            |
| `THROTTLE_POS`             | `0x11` | Absolute throttle plate position                    | %            |
| `AIR_INTAKE_TEMP`          | `0x0F` | Temperature of air entering intake manifold         | °C           |
| `INTAKE_MANIFOLD_PRESSURE` | `0x0B` | Manifold Absolute Pressure (MAP)                    | kPa          |
| `TIMING_ADVANCE`           | `0x0E` | Ignition spark timing advance before TDC            | Degrees      |
| `DTC_NUMBER`               | `0x01` | Malfunction Indicator Lamp (MIL) status & DTC count | Text / Count |
| `TROUBLE_CODES`            | `0x03` | Diagnostic Trouble Codes (SAE J2012 fault codes)    | Alphanumeric |

---

## 4. Ground Truth & Diagnostic Trouble Codes (DTCs)

The dataset contains real diagnostic trouble codes recorded during vehicle operation:

- **`P0133`** (6,070 records): O2 Sensor Circuit Slow Response (Bank 1, Sensor 1) — indicates oxygen sensor degradation or air-fuel ratio lag.
- **`C0300`** (5,673 records): Rear Speed Sensor / Chassis Communication Malfunction — indicates ABS wheel speed sensor anomaly.
- **`P0078` / `P0079` / `P007E` / `P007F`** (182 records): Exhaust Valve Control Solenoid Circuit (Bank 1/2) — indicates variable valve timing (VVT) actuator circuit fault.
- **Healthy / No DTC (`NaN`)**: 35,589 records (74.9%)

---

## 5. Machine Learning Formulation & Methodological Integrity

A critical finding in automotive fleet datasets is **vehicle-specific fault clustering**:
In this dataset, specific faults are associated with particular vehicles (e.g., `car6` logged `P0133`, `car9` logged `C0300`, `car13` logged `P0078`, while other vehicles logged no DTCs). A naive random 80/20 train/test split leaks trip and vehicle dynamics into the test set, leading to artificially high accuracy without true generalizability.

To maintain methodological rigor without fabricating labels:

1. **Primary Model — Unsupervised Anomaly Detection (`IsolationForest`)**:
   Trained on standard multi-dimensional vehicle operating envelopes across normal telemetry (RPM, Speed, Coolant Temp, Engine Load, Throttle Position, Intake Air Temp). Generates an unbiased continuous **Anomaly Score** and **Health / Risk Index** that detects engine overheating, vacuum leaks, sensor discrepancies, and abnormal operating stresses.
2. **Diagnostic Trouble Code (DTC) Expert System**:
   Integrates standard SAE J2012 diagnostic code parsing with physical safety thresholds to deliver actionable diagnostic advisories.
3. **Supervised Fault Benchmark**:
   Evaluates a supervised Random Forest classifier for DTC presence with explicit Group-based vehicle cross-validation to illustrate the real-world challenge of fleet cross-vehicle generalization.
