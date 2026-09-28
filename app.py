import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.model import train_model, predict_vehicle

st.set_page_config(page_title="Vehicle Health Monitor", page_icon="🚗", layout="wide")

st.title("🚗 AI Vehicle Predictive Maintenance System")
st.caption("Predictive maintenance using Random Forest / XGBoost and explainable feature importance.")

@st.cache_resource
def get_model(model_name):
    return train_model(model_name)

with st.sidebar:
    st.header("Model Configuration")
    model_name = st.selectbox("Choose ML Model", ["Random Forest", "XGBoost"])
    st.info("The training data is generated automatically from realistic vehicle sensor ranges.")

model, metrics, feature_names, importances = get_model(model_name)

st.subheader("Enter Vehicle Sensor Readings")

defaults = {
    "Engine temperature": 92.0,
    "RPM": 2200.0,
    "Oil pressure": 42.0,
    "Vibration": 2.0,
    "Battery voltage": 12.6,
    "Coolant temperature": 88.0,
    "Fuel consumption": 7.0,
    "Vehicle speed": 55.0,
    "Operating hours": 2500.0,
}

limits = {
    "Engine temperature": (60.0, 130.0),
    "RPM": (500.0, 5000.0),
    "Oil pressure": (10.0, 80.0),
    "Vibration": (0.1, 10.0),
    "Battery voltage": (9.0, 15.0),
    "Coolant temperature": (50.0, 130.0),
    "Fuel consumption": (2.0, 20.0),
    "Vehicle speed": (0.0, 160.0),
    "Operating hours": (0.0, 20000.0),
}

cols = st.columns(3)
values = {}
for i, name in enumerate(defaults):
    with cols[i % 3]:
        values[name] = st.number_input(
            name,
            min_value=limits[name][0],
            max_value=limits[name][1],
            value=defaults[name],
            step=0.1
        )

if st.button("🔍 Analyze Vehicle Health", type="primary"):
    result = predict_vehicle(model, values, feature_names, importances)

    c1, c2, c3 = st.columns(3)
    c1.metric("Vehicle Health", f"{result['health_score']:.1f}%")
    c2.metric("Failure Probability", f"{result['failure_probability']*100:.1f}%")
    c3.metric("Status", result["status"])

    st.subheader("Maintenance Assessment")
    if result["status"] == "HEALTHY":
        st.success("Vehicle condition appears healthy. Continue regular monitoring.")
    elif result["status"] == "WARNING":
        st.warning("Potential abnormal behavior detected. Schedule an inspection.")
    else:
        st.error("High failure risk detected. Immediate maintenance inspection is recommended.")

    st.subheader("Major Contributing Factors")
    exp_df = pd.DataFrame(result["explanations"], columns=["Feature", "Importance"])
    exp_df["Importance"] = exp_df["Importance"].round(4)
    st.dataframe(exp_df, use_container_width=True, hide_index=True)

    fig = px.bar(
        exp_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Explainable ML: Feature Importance"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Model Performance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
m2.metric("Precision", f"{metrics['precision']:.3f}")
m3.metric("Recall", f"{metrics['recall']:.3f}")
m4.metric("F1 Score", f"{metrics['f1']:.3f}")

st.caption("Note: This educational prototype uses synthetic training data. It should not be used as a real vehicle safety system.")
