# 🚗 AI-Based Predictive Maintenance and Vehicle Health Monitoring

An AI/ML-based predictive maintenance system that analyzes vehicle sensor parameters to estimate **failure probability**, calculate a **Vehicle Health Score**, and generate maintenance alerts.

## 📌 Project Overview

Vehicle failures can lead to unexpected downtime, expensive repairs, and operational issues. This project uses machine learning to analyze vehicle health indicators and identify potential failure conditions before they occur.

The system provides:

* 🔍 Failure probability prediction
* ❤️ Vehicle Health Score
* ⚠️ Maintenance status classification
* 📊 Sensor-based analysis
* 🧠 Explainable ML using feature importance
* 📈 Model performance evaluation
* 🖥️ Interactive Streamlit dashboard

---

## 🎯 Objectives

1. Predict potential vehicle/component failure.
2. Estimate the probability of failure from sensor readings.
3. Generate a Vehicle Health Score.
4. Classify vehicle condition as **Healthy, Warning, or Critical**.
5. Identify the major sensor factors influencing the prediction.
6. Provide an interactive dashboard for monitoring vehicle health.

---

## 📊 Input Parameters

The model analyzes the following vehicle parameters:

| Parameter           | Description                  |
| ------------------- | ---------------------------- |
| Engine Temperature  | Engine operating temperature |
| RPM                 | Engine rotational speed      |
| Oil Pressure        | Engine oil pressure          |
| Vibration           | Mechanical vibration level   |
| Battery Voltage     | Vehicle battery voltage      |
| Coolant Temperature | Cooling system temperature   |
| Fuel Consumption    | Fuel usage                   |
| Vehicle Speed       | Current vehicle speed        |
| Operating Hours     | Total operating hours        |

---

## 🏗️ System Architecture

```text
        Vehicle Sensors
              │
              ▼
     Data Preprocessing
              │
              ▼
       ML Classification
       ┌───────────────┐
       │ Random Forest │
       │   XGBoost     │
       └───────────────┘
              │
              ▼
     Failure Probability
              │
              ▼
     Vehicle Health Score
              │
              ▼
      Maintenance Alert
              │
              ▼
     Explainable Factors
```

---

## 🤖 Machine Learning Models

### Random Forest

An ensemble learning algorithm that combines multiple decision trees to classify whether a vehicle is likely to experience failure.

### XGBoost

A gradient-boosting algorithm used as an alternative model for failure classification.

The application allows the user to select either model directly from the Streamlit interface.

---

## 🧠 Explainable AI

The system uses model feature importance to identify the sensor parameters that contribute most strongly to the model's prediction.

For example:

```text
Major Contributing Factors

1. Vibration
2. Engine Temperature
3. Oil Pressure
```

This makes the prediction easier to interpret rather than presenting only a failure probability.

---

## ❤️ Vehicle Health Score

The predicted failure probability is converted into a health score:

```text
Health Score = 100 × (1 − Failure Probability)
```

The prototype uses the following interpretation:

| Health Score | Status      |
| ------------ | ----------- |
| ≥ 75%        | 🟢 HEALTHY  |
| 50–74%       | 🟡 WARNING  |
| < 50%        | 🔴 CRITICAL |

---

## 🖥️ Application

The Streamlit dashboard allows users to:

1. Select the ML model.
2. Enter vehicle sensor readings.
3. Analyze vehicle health.
4. View failure probability.
5. View the Vehicle Health Score.
6. Check the maintenance status.
7. View important contributing factors.
8. Visualize feature importance.
9. View model performance metrics.

---

## 🛠️ Technology Stack

**Programming Language**

* Python

**Machine Learning**

* Scikit-learn
* Random Forest
* XGBoost

**Data Processing**

* Pandas
* NumPy

**Visualization**

* Plotly

**Application**

* Streamlit

---

## 📁 Project Structure

```text
AI-Vehicle-Predictive-Maintenance/
│
├── data/
│   └── README.txt
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── evaluate.py
│   └── model.py
│
├── app.py
├── requirements.txt
├── PROJECT_REPORT.md
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AI-Vehicle-Predictive-Maintenance.git
```

Move into the project directory:

```bash
cd AI-Vehicle-Predictive-Maintenance
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python -m streamlit run app.py
```

The application will open locally in your browser.

---

## 📈 Model Evaluation

The application displays:

* Accuracy
* Precision
* Recall
* F1 Score

These metrics help evaluate the classification performance of the trained model.

---

## 📌 Dataset

This prototype generates a **synthetic vehicle sensor dataset** automatically using realistic sensor ranges and failure-risk relationships.

No external API or paid service is required.

For real-world deployment, the system would require validated vehicle sensor data collected from actual vehicles.

---

## 🚀 Future Scope

* Real-time IoT sensor integration
* LSTM/GRU-based time-series prediction
* SHAP-based individual prediction explanations
* Cloud-based vehicle monitoring
* Real-time maintenance notifications
* Integration with vehicle diagnostic systems
* Historical health trend analysis

---

## ⚠️ Disclaimer

This project is an educational prototype and uses synthetic data. It should not be used as a real vehicle safety or maintenance decision system without proper validation using real-world automotive data.

---

## 👩‍💻 Author

**Tritiya Dey Sarkar**

B.Tech — Electronics & Communication Engineering
Institute of Engineering & Management, Kolkata
