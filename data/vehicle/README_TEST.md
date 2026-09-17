# Vehicle AI — OBD-II Telemetry Test Dataset

This directory contains `test_vehicle_obd.csv`, a realistic benchmark vehicle telemetry sample designed for testing the MachineGuard Vehicle AI inference engine, batch processing pipeline, and Streamlit user interface without needing a live vehicle or physical OBD-II hardware adapter.

---

## 1. File Overview

- **Filename:** `test_vehicle_obd.csv`
- **Format:** Standard Comma-Separated Values (CSV)
- **Record Count:** 35 telemetry frames
- **Data Source:** Synthesized from real-world driving distributions matching the project's processed fleet dataset (`exp1_14drivers_14cars_dailyRoutes.csv`).

---

## 2. Telemetry Schema & Features

The dataset conforms strictly to the standard SAE OBD-II Parameter IDs (PIDs) expected by `vehicle/predict_vehicle.py` and `vehicle/vehicle_monitor.py`:

| Column Header         | Standard PID | Units  | Typical Range    | Description                                 |
| :-------------------- | :----------- | :----- | :--------------- | :------------------------------------------ |
| `ENGINE_RPM`          | 0x0C         | RPM    | 800 – 3,000      | Crankshaft rotational speed                 |
| `SPEED`               | 0x0D         | km/h   | 0 – 105          | Vehicle ground speed                        |
| `ENGINE_COOLANT_TEMP` | 0x05         | °C     | 78 – 96          | Engine cooling circuit temperature          |
| `ENGINE_LOAD`         | 0x04         | %      | 22 – 68          | Calculated instantaneous engine load        |
| `THROTTLE_POS`        | 0x11         | %      | 11 – 42          | Throttle valve position angle               |
| `AIR_INTAKE_TEMP`     | 0x0F         | °C     | 28 – 38          | Intake manifold air charge temperature      |
| `TROUBLE_CODES`       | 0x03 / 0x07  | String | DTC code / blank | Active Diagnostic Trouble Codes (SAE J2012) |

---

## 3. Trip Scenario Progression

The 35 sequential frames model a complete micro-trip cycle:

1. **Frames 1–6 (Idle / Warm-up):** Vehicle stationary (`SPEED = 0`), engine at idle (`ENGINE_RPM ~ 840`), coolant warming from 78.5°C to 85.2°C.
2. **Frames 7–15 (Urban Acceleration):** Stop-and-go driving with progressive throttle input, accelerating from 15 km/h to 46 km/h.
3. **Frames 16–27 (Steady Highway Cruising):** Stable cruising speed (80–92 km/h) at optimal engine load (33–38%) and operating temperature (88–91°C).
4. **Frames 28–31 (High Load / Grade Climb):** Overtaking acceleration (up to 102 km/h) with elevated engine load (55–66%) and engine speed (2600–2850 RPM).
5. **Frames 32–35 (Deceleration & Stop):** Deceleration to exit ramp and final stop (`SPEED = 0`).

---

## 4. How to Test in the MachineGuard Streamlit Dashboard

1. Launch the application:
   ```bash
   streamlit run app.py
   ```
2. In the top navigation bar, select **Vehicle AI**.
3. Toggle the ingestion mode to **Upload CSV**.
4. Drag and drop or browse to `data/vehicle/test_vehicle_obd.csv`.
5. The interface will display a validation confirmation box:
   - Status: `CSV validated successfully`
   - Total rows: `35`
   - Clean 5-row table preview.
6. Click the primary button: **Analyze Vehicle Data**.
7. The application executes both the Isolation Forest anomaly detector and the Supervised Random Forest risk scorer, displaying:
   - Overall Trip Status badge (`NORMAL`, `WARNING`, or `ANOMALY`)
   - Breakdown metrics (Healthy Frames %, Advisory Frames %, Critical Frames %)
   - Mean risk index and high-risk snapshot table.

---

## 5. Technical Notes & Disclosures

- **Offline Telemetry Testing:** This file is intended for offline demonstration and testing of MachineGuard's ML ingestion logic.
- **Screening Heuristic:** Predictions generated from telemetry logs reflect statistical similarity to baseline distributions and do not constitute a certified automotive diagnostic assessment.
