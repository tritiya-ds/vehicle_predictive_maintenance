# AI/ML-Based Predictive Maintenance and Vehicle Health Monitoring System

## Project Goal
Predict vehicle/component failure before it occurs using vehicle sensor parameters and machine learning.

This implementation uses:
- Random Forest
- XGBoost
- Explainable ML using feature importance
- Vehicle Health Score
- Failure Probability
- Maintenance Alert

## Input Parameters
- Engine temperature
- RPM
- Oil pressure
- Vibration
- Battery voltage
- Coolant temperature
- Fuel consumption
- Vehicle speed
- Operating hours

## Architecture
Vehicle Sensors
→ Data Preprocessing
→ ML Model
→ Failure Probability
→ Vehicle Health Score
→ Maintenance Alert

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

The app generates a realistic synthetic training dataset automatically, so no paid API or external service is required.

## Sample interpretation
- Health Score >= 75: HEALTHY
- Health Score 50–74: WARNING
- Health Score < 50: CRITICAL

The model prediction is based on the sensor pattern, while the explanation panel shows which features contributed most strongly to the prediction.
