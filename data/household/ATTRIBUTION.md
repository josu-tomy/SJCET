# Household Datasets Attribution & Documentation

## Primary Dataset: UCI Appliances Energy Prediction

- **Dataset Name:** Appliances Energy Prediction Dataset
- **UCI Dataset ID:** 374
- **Official Source URL:** https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
- **Citation:**
  > Luis M. Candanedo, Véronique Feldheim, Dominique Deramaix, "Data driven prediction models of energy use of appliances in a low-energy house", Energy and Buildings, Volume 140, 2017, Pages 81-97, ISSN 0378-7788, https://doi.org/10.1016/j.enbuild.2017.01.083.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Time Coverage & Resolution:** January 11, 2016 to May 27, 2016 (4.5 months), logged at 10-minute intervals (19,735 rows).
- **What it Measures:**
  - `Appliances`: Total appliance energy consumption in Watt-hours [Wh] (target variable).
  - `lights`: Energy consumption of light fixtures in Wh.
  - Indoor environmental conditions: Temperature (`T1`–`T9`) in °C and relative humidity (`RH_1`–`RH_9`) in % across 9 household zones (kitchen, living room, laundry, office, bathroom, outside north, ironing room, teen room, parents room).
  - Outdoor weather conditions from nearby weather station: Temperature (`T_out`), pressure (`Press_mm_hg`), humidity (`RH_out`), wind speed (`Windspeed`), visibility (`Visibility`), dew point (`Tdewpoint`).
  - `rv1`, `rv2`: Non-predictive random variables included in original study for feature selection evaluation (excluded from model training).
- **Role in MachineGuard:**
  Serves as the training ground for the Household Energy Prediction & Anomaly Detector. A regression model predicts expected appliance energy consumption based on environmental/weather and temporal conditions. Large residuals ($|\text{actual} - \text{predicted}|$) flag unusual energy operating patterns or potential electrical anomalies.

---

## Inspected Secondary Dataset: UCI Individual Household Electric Power Consumption

- **Dataset Name:** Individual Household Electric Power Consumption
- **UCI Dataset ID:** 235
- **Official Source URL:** https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- **Citation:**
  > Georges Hebrail, Alice Berard, "Individual household electric power consumption", UCI Machine Learning Repository, 2012.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Inspection Findings:** Contains 2,075,259 measurements gathered between December 2006 and November 2010 (1-minute sampling) focusing on aggregate household active/reactive power, voltage, and 3 aggregate sub-metering circuits (kitchen, laundry, water heater/AC).
- **Role in MachineGuard:** Inspected for benchmark architectural context. UCI 374 was selected for core regression modeling due to its rich multivariate zone-level temperature, humidity, and weather features that correlate directly with expected appliance energy demand.
