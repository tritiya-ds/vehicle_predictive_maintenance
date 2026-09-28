import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

from .data_generator import generate_data, FEATURES

def train_model(model_name="Random Forest"):
    df = generate_data()
    X = df[FEATURES]
    y = df["Failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_name == "XGBoost":
        model = XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42
        )
    else:
        model = RandomForestClassifier(
            n_estimators=180,
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }

    importances = model.feature_importances_
    return model, metrics, FEATURES, importances

def predict_vehicle(model, values, feature_names, importances):
    X = pd.DataFrame([[values[f] for f in feature_names]], columns=feature_names)
    failure_probability = float(model.predict_proba(X)[0][1])

    health_score = max(0.0, min(100.0, 100 * (1 - failure_probability)))

    if health_score >= 75:
        status = "HEALTHY"
    elif health_score >= 50:
        status = "WARNING"
    else:
        status = "CRITICAL"

    pairs = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    return {
        "failure_probability": failure_probability,
        "health_score": health_score,
        "status": status,
        "explanations": pairs,
    }
