import numpy as np
import pandas as pd

FEATURES = [
    "Engine temperature",
    "RPM",
    "Oil pressure",
    "Vibration",
    "Battery voltage",
    "Coolant temperature",
    "Fuel consumption",
    "Vehicle speed",
    "Operating hours",
]

def generate_data(n=8000, seed=42):
    rng = np.random.default_rng(seed)

    engine_temp = rng.normal(92, 12, n).clip(60, 130)
    rpm = rng.normal(2200, 650, n).clip(500, 5000)
    oil_pressure = rng.normal(43, 9, n).clip(10, 80)
    vibration = rng.gamma(2.0, 1.1, n).clip(0.1, 10)
    battery = rng.normal(12.5, 0.7, n).clip(9, 15)
    coolant = (engine_temp - rng.normal(4, 3, n)).clip(50, 130)
    fuel = rng.normal(7.5, 2.0, n).clip(2, 20)
    speed = rng.normal(55, 25, n).clip(0, 160)
    hours = rng.uniform(0, 20000, n)

    risk = (
        0.045 * np.maximum(engine_temp - 100, 0)
        + 0.55 * np.maximum(vibration - 3.0, 0)
        + 0.06 * np.maximum(30 - oil_pressure, 0)
        + 0.7 * np.maximum(11.5 - battery, 0)
        + 0.025 * np.maximum(coolant - 100, 0)
        + 0.00004 * hours
        + 0.00012 * np.maximum(rpm - 3500, 0)
        + 0.06 * np.maximum(fuel - 11, 0)
        + rng.normal(0, 0.55, n)
    )

    probability = 1 / (1 + np.exp(-(risk - 2.3)))
    failure = (rng.random(n) < probability).astype(int)

    df = pd.DataFrame({
        "Engine temperature": engine_temp,
        "RPM": rpm,
        "Oil pressure": oil_pressure,
        "Vibration": vibration,
        "Battery voltage": battery,
        "Coolant temperature": coolant,
        "Fuel consumption": fuel,
        "Vehicle speed": speed,
        "Operating hours": hours,
        "Failure": failure,
    })
    return df
