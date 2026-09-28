from .model import train_model

if __name__ == "__main__":
    for name in ["Random Forest", "XGBoost"]:
        _, metrics, _, _ = train_model(name)
        print(f"\n{name}")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
