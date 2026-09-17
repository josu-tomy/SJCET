# MachineGuard

## AI-Powered Equipment Monitoring and Predictive Maintenance Platform

MachineGuard is an AI-powered equipment monitoring platform designed to identify abnormal operating patterns that may be difficult for humans to recognize manually.

The system brings together three specialized monitoring layers:

- Industrial machine predictive maintenance
- Household equipment and energy behavior monitoring
- Vehicle OBD-II based maintenance-risk screening

Instead of using one model for every type of equipment, MachineGuard uses **domain-specific datasets and machine-learning pipelines** for each monitoring layer. This makes the system more appropriate for the different types of sensor and operational data produced by industrial machines, household equipment, and vehicles.

---

# 1. Problem Statement

Equipment failures and abnormal operating conditions are often difficult to detect before they become visible problems.

Traditional monitoring commonly depends on:

- Manual inspection
- Fixed threshold rules
- Human observation
- Periodic maintenance
- Reactive repair after failure

These approaches can miss subtle patterns in sensor readings.

For example:

- An industrial machine may operate at a combination of temperature, rotational speed, torque, and tool-wear values associated with failure.
- A household may consume significantly more or less energy than expected under similar environmental conditions.
- A vehicle may exhibit an unusual combination of engine RPM, speed, temperature, load, throttle position, or other OBD-II parameters.

The core problem addressed by MachineGuard is:

> **How can AI identify abnormal equipment operating patterns that may be difficult for humans to recognize through manual monitoring alone?**

---

# 2. Proposed Solution

MachineGuard provides a unified monitoring dashboard containing specialized AI modules.

## Industrial AI

Uses the UCI AI4I 2020 Predictive Maintenance Dataset to estimate whether current industrial-machine operating conditions resemble failure-associated patterns.

## Household Energy AI

Uses the UCI Appliances Energy Prediction Dataset to predict expected appliance energy consumption based on environmental and temporal conditions.

The system then compares:

```text
Actual Energy Consumption
            vs
AI Expected Energy Consumption
````

Large deviations are flagged as unusual energy behavior.

## Vehicle AI

Uses vehicle/OBD telemetry to analyze operating patterns and identify anomalous behavior that may indicate the need for maintenance inspection.

The vehicle layer is designed around parameters available from OBD-style telemetry and provides a path toward future physical OBD-II adapter integration.

---

# 3. Key Features

## 3.1 Unified Equipment Dashboard

MachineGuard provides a single interface for monitoring multiple equipment domains.

The dashboard includes:

- Home Overview
- Check Equipment
- Energy Behavior
- Vehicle AI
- Industrial AI
- Future Sensor Integration

---

## 3.2 Home Overview

The Home Overview provides a high-level status view of monitored equipment.

Supported household equipment includes:

- Refrigerator
- Air Conditioner
- Washing Machine
- Television
- Water Pump
- Ceiling Fan

The dashboard can display:

- Equipment name
- Current analysis status
- Last analysis time
- Latest readings
- Analysis summary

Equipment that has not yet been analyzed is displayed as:

```text
NOT ANALYZED
```

rather than using fabricated status values.

The current prototype stores the latest analysis results in the Streamlit session state.

---

# 4. Industrial Predictive Maintenance

## Dataset

MachineGuard uses the:

**UCI AI4I 2020 Predictive Maintenance Dataset**

Dataset source:

[https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenanc](https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenanc)

The dataset contains approximately 10,000 machine operating records.

Important variables include:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

Target:

```text
Machine failure
```

---

## Why This Dataset Was Selected

The AI4I dataset was selected because it directly represents the predictive-maintenance problem targeted by the industrial monitoring module.

It provides:

- Machine sensor measurements
- Operating-condition variables
- A machine-failure target
- A manageable dataset size
- A public and reproducible source
- A well-defined classification problem

This allows the project to demonstrate an actual supervised machine-learning workflow instead of relying only on manually created threshold rules.

The dataset is also appropriate for a hackathon prototype because it allows model training, evaluation, visualization, and real-time manual inference within a limited development time.

---

# 5. Industrial Machine Learning Approach

The industrial module uses:

```text
Random Forest Classifier
```

Configuration:

```text
n_estimators = 100
random_state = 42
class_weight = balanced
```

The model receives machine operating conditions and predicts whether the conditions are associated with machine failure.

The system also produces a probability-like model output that is used to create a prototype risk visualization.

---

## Industrial Data Pipeline

```text
UCI AI4I Dataset
       |
       v
Data Inspection
       |
       v
Data Cleaning
       |
       v
Feature Selection
       |
       v
Stratified Train/Test Split
       |
       v
Random Forest
       |
       v
Prediction + Probability
       |
       v
Risk Visualization
       |
       v
Streamlit Dashboard
```

---

# 6. Industrial Features Used

The industrial model uses:

| Feature | Description |
| --- | --- |
| Air Temperature     | Temperature around the machine |
| Process Temperature | Process operating temperature  |
| Rotational Speed    | Machine rotational speed       |
| Torque              | Applied machine torque         |
| Tool Wear           | Accumulated tool wear          |

The identifiers such as UDI and Product ID are not used as predictive features.

---

# 7. Industrial Evaluation

The industrial model is evaluated using:

- Precision
- Recall
- F1-score
- Confusion Matrix
- Feature Importance

The project emphasizes multiple classification metrics because predictive-maintenance datasets can contain class imbalance.

A confusion matrix is used to inspect:

```text
True Positives
True Negatives
False Positives
False Negatives
```

Feature importance is used to understand which input variables contributed most strongly to the Random Forest's predictions.

Feature importance is treated as a model-level association and **not as proof of causality**.

---

# 8. Industrial Risk Visualization

For the prototype interface, the model output is visually grouped into:

```text
0.00 - 0.39   NORMAL
0.40 - 0.69   WARNING
0.70 - 1.00   HIGH RISK
```

These ranges are used only for visualization and demonstration.

They are **not industrial safety limits** and should not be used as engineering or safety thresholds without validation using real machine data.

MachineGuard does not claim to guarantee that a machine will fail in the future.

Instead, the result represents how closely the current operating conditions resemble failure-associated patterns learned by the model.

---

# 9. Household Equipment Monitoring

MachineGuard includes a household monitoring layer designed to demonstrate how AI-based equipment monitoring can extend beyond industrial environments.

Supported equipment includes:

```text
Refrigerator
Air Conditioner
Washing Machine
Television
Water Pump
Ceiling Fan
```

The prototype currently supports manual readings.

The architecture is designed so that future sensor hardware can provide these readings automatically.

---

# 10. Household Energy AI

## Dataset

MachineGuard uses the:

**UCI Appliances Energy Prediction Dataset**

Dataset source:

[https://archive.ics.uci.edu/dataset/374/appliances%2Benergy%2Bprediction](https://archive.ics.uci.edu/dataset/374/appliances%2Benergy%2Bprediction)

The dataset contains approximately 19,735 observations recorded at 10-minute intervals.

It contains household environmental and energy measurements such as:

- Indoor temperature
- Indoor humidity
- Outdoor temperature
- Outdoor humidity
- Weather information
- Time-related information
- Appliance energy consumption

Target:

```text
Appliances
```

which represents appliance energy consumption.

---

# 11. Why the Household Dataset Was Selected

The dataset was selected because it provides real household energy measurements together with environmental and temporal variables.

This allows MachineGuard to demonstrate a more meaningful AI task than simply using fixed energy thresholds.

The model learns:

```text
Environmental Conditions
+
Time Information
        |
        v
Expected Energy Consumption
```

The measured energy can then be compared with the AI's expectation.

This makes the household module useful for demonstrating **energy behavior anomaly detection**.

---

# 12. Household Machine Learning Approach

The household module treats energy consumption as a regression problem.

The model predicts:

```text
Expected Appliance Energy Consumption
```

The prototype then calculates:

```text
Deviation = Actual Consumption - Expected Consumption
```

Large deviations can indicate unusual energy behavior.

The household system therefore performs:

```text
Environmental + Temporal Inputs
             |
             v
       Regression Model
             |
             v
     Expected Energy
             |
             |
Actual Energy --------+
             |
             v
      Residual / Deviation
             |
             v
      Anomaly Evaluation
```

---

# 13. Household Model Evaluation

The household regression model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score

The data is split chronologically rather than randomly to reduce the possibility of future-data leakage.

The general split is:

```text
70% Training
15% Validation
15% Testing
```

The anomaly threshold is calibrated using training/validation information rather than using arbitrary values.

---

# 14. Important Household Limitation

The Household Energy AI module is an **energy behavior anomaly detector**.

It does NOT directly diagnose:

```text
Refrigerator failure
Air-conditioner failure
Washing-machine failure
Television failure
Water-pump failure
Ceiling-fan failure
```

The UCI Appliances Energy dataset does not provide appliance-specific failure labels for all these devices.

Therefore, the current system should be understood as:

> AI-based household energy behavior monitoring.

Future appliance-specific sensor datasets can be added to create dedicated predictive-maintenance models for individual appliances.

---

# 15. Manual Household Equipment Analysis

The current prototype supports manual equipment readings.

Example refrigerator inputs may include:

- Internal temperature
- Ambient temperature
- Power consumption
- Compressor status
- Door-open duration

Example air-conditioner inputs may include:

- Room temperature
- Set temperature
- Power consumption
- Current
- Vibration

Example washing-machine inputs may include:

- Motor speed
- Water temperature
- Motor current
- Vibration
- Cycle duration

The exact fields are selected according to the equipment profile implemented in the application.

---

# 16. Vehicle AI

MachineGuard also contains a vehicle monitoring layer designed around OBD-style telemetry.

The objective is:

> Identify unusual vehicle operating patterns from sensor telemetry that may justify maintenance inspection.

The system is designed for future integration with a physical OBD-II adapter.

---

# 17. Vehicle Dataset

The vehicle module uses public vehicle telemetry data containing OBD-style measurements.

The selected data includes vehicle operating parameters such as available:

- Engine RPM
- Vehicle speed
- Engine load
- Coolant temperature
- Intake air temperature
- Throttle position
- MAP
- MAF
- Battery voltage
- Timing advance
- Diagnostic information
- Timestamp
- Vehicle/trip information

The exact features used by the trained model depend on the available columns and preprocessing pipeline.

The raw and processed vehicle datasets are stored under:

```text
data/vehicle/
```

---

# 18. Why Vehicle Telemetry Was Selected

Vehicle maintenance increasingly relies on electronic sensor information available through vehicle diagnostic systems.

OBD-II provides access to operating information that can include:

- Engine conditions
- Temperature
- Speed
- Load
- Throttle behavior
- Air measurements
- Diagnostic codes

This makes OBD-style telemetry a natural data source for a vehicle monitoring system.

The dataset was selected to allow MachineGuard to demonstrate an AI pipeline based on real vehicle telemetry rather than manually invented sensor values.

---

# 19. Vehicle Machine Learning Approach

The vehicle module analyzes operating patterns using the available vehicle dataset.

Depending on the availability and quality of maintenance/fault labels, the system can use:

```text
Supervised Classification
```

when trustworthy labels are available, or:

```text
Anomaly Detection
```

when reliable maintenance labels are not available.

The implemented model and prediction pipeline are stored under:

```text
vehicle/
models/vehicle_model.joblib
```

---

# 20. Vehicle Evaluation

The vehicle model evaluation includes appropriate metrics depending on the selected learning task.

For supervised classification:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC / PR-AUC where appropriate

For anomaly detection:

- Anomaly score distribution
- Percentage of observations flagged
- Temporal anomaly distribution
- Per-vehicle anomaly counts
- Per-trip anomaly counts

The project avoids inventing labels or performance metrics when the original dataset does not provide them.

---

# 21. Vehicle OBD-II Integration

The project includes an abstraction layer:

```text
vehicle/obd_interface.py
```

This is intended to provide a future interface between the application and physical OBD-II hardware.

Conceptually:

```text
Vehicle
   |
   v
OBD-II Adapter
   |
   v
OBD Interface
   |
   v
Sensor Data
   |
   v
Vehicle AI Model
   |
   v
Maintenance-Risk Analysis
   |
   v
MachineGuard Dashboard
```

The current prototype does not pretend that a physical OBD device is connected when one is not available.

---

# 22. Vehicle CSV Upload

The Vehicle AI module can also work with vehicle telemetry supplied through CSV data.

This allows the system to be demonstrated without physical OBD hardware.

The project includes a test telemetry file for prototype inference:

```text
data/vehicle/test_vehicle_obd.csv
```

The CSV is intended for testing the existing inference pipeline and does not represent a real-time vehicle connection.

---

# 23. Diagnostic Trouble Codes

OBD diagnostic trouble codes can provide useful additional context.

MachineGuard treats DTC information separately from machine-learning predictions.

A diagnostic code indicates that a vehicle diagnostic system detected a condition associated with that code.

It should not automatically be interpreted as definitive proof of a particular physical component failure.

Future versions can integrate a structured DTC lookup database to provide:

```text
DTC
 |
 v
Code Description
 |
 v
Possible System / Condition
 |
 v
Recommended Inspection Area
```

This would complement, rather than replace, the AI analysis.

---

# 24. Overall System Architecture

```text
                         MachineGuard
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
        Industrial        Household        Vehicle
            AI               AI              AI
              |               |               |
              v               v               v
        Machine Data     Energy Data      OBD Telemetry
              |               |               |
              v               v               v
        Classification    Regression      Classification /
              |            + Residual       Anomaly Detection
              |               |               |
              +---------------+---------------+
                              |
                              v
                     Streamlit Dashboard
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
    Home Overview       Equipment Analysis    AI Results
```

---

# 25. Modular Design

MachineGuard intentionally keeps the domains separate.

The project does **not** merge:

```text
Industrial Dataset
+
Household Dataset
+
Vehicle Dataset
```

into one model.

Instead:

```text
Industrial Model
Household Model
Vehicle Model
```

are maintained independently.

This is because each domain has different:

- Sensor meanings
- Data distributions
- Targets
- Sampling rates
- Failure mechanisms
- Feature semantics
- Validation requirements

The unified dashboard is therefore an interface layer rather than a single universal ML model.

---

# 26. Project Structure

```text
sjcet/
│
├── app.py
│
├── data/
│   ├── ai4i2020.csv
│   │
│   ├── household/
│   │   ├── raw/
│   │   │   └── energydata_complete.csv
│   │   ├── processed/
│   │   │   └── household_clean.csv
│   │   └── ATTRIBUTION.md
│   │
│   └── vehicle/
│       ├── raw/
│       ├── processed/
│       │   └── vehicle_clean.csv
│       ├── test_vehicle_obd.csv
│       └── README.md
│
├── household/
│   ├── appliance_profiles.py
│   ├── home_monitor.py
│   ├── inspect_household_data.py
│   ├── predict_household.py
│   ├── preprocess_household.py
│   ├── sensor_interface.py
│   └── train_household.py
│
├── vehicle/
│   ├── evaluate_vehicle.py
│   ├── inspect_vehicle_data.py
│   ├── obd_interface.py
│   ├── predict_vehicle.py
│   ├── preprocess_vehicle.py
│   ├── train_vehicle.py
│   ├── vehicle_monitor.py
│   └── README.md
│
├── models/
│   ├── model.joblib
│   ├── household_model.joblib
│   └── vehicle_model.joblib
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── vehicle/
│       ├── anomaly_score_distribution.png
│       ├── evaluation_metrics.json
│       ├── feature_importance.png
│       └── supervised_confusion_matrix.png
│
├── train.py
├── predict.py
├── evaluate.py
├── preprocess.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 27. Technologies Used

## Programming Language

- Python

## Machine Learning

- Scikit-learn
- Random Forest
- Regression models
- Anomaly detection
- Model evaluation

## Data Processing

- Pandas
- NumPy

## Visualization

- Matplotlib
- Streamlit visualizations

## Web Application

- Streamlit

## Model Serialization

- Joblib

## Development Tools

- Git
- GitHub
- Jupyter / Anaconda
- Python virtual environment

## Hardware / Future Integration

- OBD-II adapter
- Vehicle diagnostic interface
- Future household sensors
- Future industrial sensors

---

# 28. How the Application Works

The basic workflow is:

```text
User
 |
 v
MachineGuard Dashboard
 |
 +--> Select Equipment
 |
 +--> Enter / Upload Sensor Data
 |
 v
Data Validation
 |
 v
Preprocessing
 |
 v
Domain-Specific ML Model
 |
 v
Prediction / Expected Value / Anomaly Score
 |
 v
Risk or Status Interpretation
 |
 v
Dashboard Visualization
```

---

# 29. Running the Project Locally

## 29.1 Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd sjcet
```

---

## 29.2 Create a Virtual Environment

Using Python:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

---

## 29.3 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 29.4 Run the Streamlit Application

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# 30. Running With the Project Virtual Environment

If the project already contains `.venv`, the application can be started using:

```bash
.venv/bin/streamlit run app.py --server.port 8501 --server.headless true --server.address 0.0.0.0
```

---

# 31. Model Training

The project contains separate training pipelines.

## Industrial

```bash
python train.py
```

## Household

```bash
python household/train_household.py
```

## Vehicle

```bash
python vehicle/train_vehicle.py
```

Training should generally be performed during development rather than every time the Streamlit dashboard starts.

The Streamlit application loads the saved models for inference.

---

# 32. Model Evaluation

Industrial evaluation:

```bash
python evaluate.py
```

Vehicle evaluation:

```bash
python vehicle/evaluate_vehicle.py
```

Household evaluation is performed through its corresponding training/evaluation workflow.

Evaluation outputs are stored under:

```text
outputs/
```

---

# 33. Data Preprocessing

Each domain has its own preprocessing pipeline.

## Industrial

- Select relevant machine sensor features
- Remove identifier columns from model features
- Handle target labels
- Perform stratified train/test splitting

## Household

- Parse timestamps
- Sort chronologically
- Handle missing values
- Remove duplicates where appropriate
- Create temporal features
- Avoid future-data leakage
- Perform chronological train/validation/test splitting

## Vehicle

- Parse timestamps where available
- Sort telemetry chronologically
- Remove duplicate records
- Handle missing sensor values
- Validate sensor ranges
- Encode required categorical information
- Prevent vehicle/trip identifiers from becoming misleading predictive features
- Avoid future-data leakage

---

# 34. Data Leakage Prevention

MachineGuard attempts to prevent information leakage during model development.

For industrial classification:

```text
Stratified Train/Test Split
```

is used.

For household time-series data:

```text
Chronological Train / Validation / Test Split
```

is used.

For vehicle telemetry:

```text
Time-aware or vehicle/trip-aware validation
```

is preferred where the dataset structure allows it.

Future observations should not be used to predict past observations.

---

# 35. AI/ML Techniques

The project demonstrates multiple machine-learning techniques:

### Supervised Classification

Used for industrial predictive-maintenance classification and applicable to vehicle fault classification when trustworthy labels are available.

### Regression

Used in household energy prediction.

### Residual-Based Anomaly Detection

Used by comparing:

```text
Actual Energy
-
Expected Energy
```

### Anomaly Detection

Used in the vehicle layer when reliable maintenance labels are unavailable.

### Feature Importance

Used to provide model-level interpretability for tree-based models.

---

# 36. Why Machine Learning Instead of Only Rules?

A fixed rule might look like:

```text
IF temperature > X
THEN warning
```

However, equipment behavior can depend on multiple variables simultaneously.

For example:

```text
Temperature
+
RPM
+
Torque
+
Tool Wear
```

may collectively represent a pattern that is difficult to capture with one simple threshold.

Machine-learning models can learn relationships between multiple variables from historical observations.

MachineGuard therefore combines:

```text
Machine Learning
+
Visualization
+
Domain-specific monitoring
```

instead of relying only on manually selected thresholds.

---

# 37. Innovation

MachineGuard's main innovation is the combination of multiple equipment-monitoring scenarios into one modular AI platform.

Instead of creating a single generic "equipment failure" model, the platform recognizes that:

```text
Industrial equipment
Household equipment
Vehicles
```

produce fundamentally different data.

Therefore, MachineGuard uses specialized AI pipelines while providing one unified user interface.

The architecture also provides a path from:

```text
Manual Input
       |
       v
CSV Telemetry
       |
       v
Physical Sensors / OBD
       |
       v
Automated Monitoring
```

This allows the prototype to demonstrate both immediate usability and future deployment potential.

---

# 38. Prototype vs Real-World Deployment

The current project is a functional prototype.

Current data acquisition includes:

- Manual readings
- Public datasets
- CSV telemetry

Future deployment can add:

- IoT sensors
- ESP32 / Raspberry Pi
- Industrial sensor systems
- Smart energy meters
- OBD-II Bluetooth/USB adapters
- Real-time data streaming
- Database storage
- Automated alerts

---

# 39. Future Sensor Integration

The project contains sensor-interface abstractions that can later be connected to physical devices.

Conceptual architecture:

```text
Physical Sensor
      |
      v
Sensor Interface
      |
      v
Data Validation
      |
      v
ML Model
      |
      v
Risk / Anomaly Analysis
      |
      v
MachineGuard Dashboard
```

For vehicles:

```text
OBD-II Port
    |
    v
OBD Adapter
    |
    v
obd_interface.py
    |
    v
Vehicle AI
```

For household equipment:

```text
Appliance Sensor
       |
       v
sensor_interface.py
       |
       v
Household AI
```

---

# 40. Limitations

MachineGuard has several important limitations.

## Dataset Limitations

Public datasets may not perfectly represent every real-world machine, appliance, or vehicle.

## Industrial Dataset

The AI4I dataset is synthetic and should not be treated as a substitute for validated industrial sensor data.

## Household Dataset

The household energy dataset is primarily an energy-consumption dataset and does not provide comprehensive failure labels for individual appliances.

Therefore, the household module currently detects unusual energy behavior rather than diagnosing appliance faults.

## Vehicle Dataset

Vehicle telemetry datasets can vary significantly by:

- Vehicle model
- ECU implementation
- Sensor availability
- Driving conditions
- Sampling rate
- Diagnostic interface

Therefore, a model trained on one dataset may not generalize directly to every vehicle.

## Thresholds

Prototype status thresholds are visualization aids and are not certified safety or maintenance limits.

## Predictions

MachineGuard does not guarantee future equipment failure or provide a definitive physical diagnosis.

A flagged result should be treated as a signal for further inspection.

---

# 41. Safety and Responsible Use

MachineGuard is intended as a decision-support prototype.

It should not replace:

- Qualified technicians
- Manufacturer service procedures
- Vehicle diagnostic procedures
- Industrial safety systems
- Certified engineering inspection
- Professional maintenance decisions

AI predictions should be validated against appropriate real-world measurements before deployment in safety-critical environments.

---

# 42. Dataset Attribution

## Industrial Dataset

UCI AI4I 2020 Predictive Maintenance Dataset

UCI Machine Learning Repository:

[https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenanc](https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenanc)

The dataset is provided under its stated license and attribution requirements.

---

## Household Dataset

UCI Appliances Energy Prediction Dataset

UCI Machine Learning Repository:

[https://archive.ics.uci.edu/dataset/374/appliances%2Benergy%2Bprediction](https://archive.ics.uci.edu/dataset/374/appliances%2Benergy%2Bprediction)

The dataset is used for the household energy prediction and anomaly-detection component.

---

## Vehicle Dataset

Vehicle telemetry data is sourced from the public dataset documented in:

```text
data/vehicle/README.md
```

The repository maintains the corresponding dataset attribution and source information.

---

# 43. External Resources

MachineGuard uses publicly available datasets and open-source Python libraries.

External resources include:

- UCI Machine Learning Repository
- Public vehicle telemetry data
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Streamlit
- Joblib

All external datasets and resources should be used according to their respective licenses and attribution requirements.

---

# 44. AI-Assisted Development Disclosure

AI-assisted development tools were used during the development of this hackathon prototype for:

- Code generation assistance
- Debugging assistance
- UI implementation assistance
- Documentation drafting
- Development workflow support

The team remains responsible for:

- Understanding the implementation
- Testing the application
- Selecting datasets
- Training and evaluating models
- Integrating the components
- Demonstrating the working prototype

The AI tools were used as development assistance and not as a replacement for the project's technical implementation and evaluation.

---

# 45. Hackathon Relevance

MachineGuard directly addresses the theme:

> **Hack the Detector – Build AI that can identify something humans might miss.**

The project focuses on identifying subtle equipment operating patterns that may not be obvious through manual observation.

Examples include:

```text
Industrial:
Failure-associated machine operating patterns

Household:
Unexpected energy-consumption behavior

Vehicle:
Unusual OBD telemetry patterns
```

The project demonstrates how AI can act as an additional layer of monitoring rather than relying entirely on manual inspection.

---

# 46. Expected Demonstration Flow

During the project demonstration, the following workflow can be shown.

## Step 1 — Home Overview

Show the unified equipment dashboard.

```text
Refrigerator
Air Conditioner
Washing Machine
Television
Water Pump
Ceiling Fan
Vehicle
```

---

## Step 2 — Check Equipment

Select an appliance and enter manual readings.

Click:

```text
Analyze Equipment
```

The AI analyzes the supplied readings and displays the result.

---

## Step 3 — Energy Behavior

Show:

```text
Actual Energy
Expected Energy
Deviation
Anomaly Status
```

This demonstrates the household regression and anomaly-detection pipeline.

---

## Step 4 — Vehicle AI

Use:

```text
Manual OBD Input
```

or:

```text
Vehicle CSV Upload
```

to demonstrate vehicle telemetry analysis.

---

## Step 5 — Industrial AI

Enter:

```text
Air Temperature
Process Temperature
Rotational Speed
Torque
Tool Wear
```

and run the industrial predictive-maintenance model.

Show:

- Predicted status
- Failure-associated probability
- Current operating profile
- Feature importance
- Model evaluation metrics

---

# 47. Results

The repository contains generated evaluation artifacts under:

```text
outputs/
```

These include model evaluation visualizations such as:

```text
confusion_matrix.png
feature_importance.png
```

and vehicle-specific outputs:

```text
outputs/vehicle/
```

The application displays the corresponding evaluation information where applicable.

Actual numerical results should be taken directly from the generated evaluation files rather than manually entered into the documentation.

---

# 48. Demo

Live Demo:

```text
<YOUR_STREAMLIT_DEMO_LINK>
```

GitHub Repository:

```text
<YOUR_GITHUB_REPOSITORY_LINK>
```

Replace the placeholders above with the final deployed Streamlit and GitHub links before submitting the project.

---

# 49. Team

MachineGuard was developed as a hackathon project.

Team Members:

- \<TEAM MEMBER 1>
- \<TEAM MEMBER 2>
- \<TEAM MEMBER 3>
- \<TEAM MEMBER 4>

---

# 50. Conclusion

MachineGuard demonstrates a practical approach to AI-powered equipment monitoring by combining specialized machine-learning pipelines for industrial machines, household energy behavior, and vehicle telemetry.

The system focuses on a common principle:

```text
Sensor Data
     |
     v
AI Pattern Recognition
     |
     v
Abnormality / Risk Signal
     |
     v
Human Inspection and Decision
```

Rather than attempting to replace human maintenance expertise, MachineGuard provides an additional layer of intelligent pattern detection that can help highlight operating conditions that deserve attention.

The architecture is modular and can be extended in the future with real-time sensors, OBD-II hardware, IoT devices, databases, and automated monitoring.

---

# MachineGuard

### AI-Powered Equipment Monitoring

```text
Industrial AI
+
Household Energy AI
+
Vehicle OBD AI
+
Future Sensor Integration
```

**Built for Hack the Detector**
