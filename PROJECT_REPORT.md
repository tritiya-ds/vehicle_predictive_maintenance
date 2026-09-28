# Project Report
## AI/ML-Based Predictive Maintenance and Vehicle Health Monitoring System

### 1. Introduction
Vehicle failures can cause downtime, safety concerns, and expensive maintenance. This project develops an AI/ML-based predictive maintenance prototype that uses vehicle sensor readings to estimate the probability of failure before it occurs.

### 2. Objectives
- Predict possible vehicle/component failure.
- Generate a vehicle health score.
- Classify the vehicle as Healthy, Warning, or Critical.
- Identify important sensor factors influencing the prediction.
- Provide a simple dashboard for maintenance decisions.

### 3. Input Features
Engine temperature, RPM, oil pressure, vibration, battery voltage, coolant temperature, fuel consumption, vehicle speed, and operating hours.

### 4. Methodology
Synthetic sensor data is generated for the prototype. The data is divided into training and testing sets. Random Forest and XGBoost classifiers are trained to predict the Failure target. The predicted failure probability is converted into a Vehicle Health Score.

### 5. Explainable AI
Feature importance from the trained tree-based model is displayed to identify the most influential sensor parameters.

### 6. System Architecture
Vehicle Sensors → Data Preprocessing → ML Model → Failure Probability → Vehicle Health Score → Maintenance Alert

### 7. Output
The dashboard displays:
- Vehicle Health %
- Failure Probability %
- Status
- Major contributing factors
- Model performance metrics

### 8. Technologies
Python, Pandas, NumPy, Scikit-learn, XGBoost, Plotly, Streamlit.

### 9. Limitations
This is an educational prototype using synthetic data. Real deployment would require validated vehicle sensor data, domain-specific thresholds, safety validation, and extensive testing.

### 10. Future Scope
- Real-time IoT sensor integration.
- LSTM/GRU for time-series prediction.
- SHAP-based local explanations.
- Cloud-based monitoring.
- SMS/WhatsApp/email maintenance alerts.
