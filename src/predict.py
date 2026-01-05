import mlflow
import mlflow.pyfunc
import pandas as pd

FEATURES = [
    "recency_days",
    "frequency",
    "total_revenue",
    "avg_order_value",
    "tenure_days",
    "active_months",
    "purchase_velocity",
    "avg_gap_days",
    "std_gap_days",
]


class CLVPredictor:
    def __init__(self, model_path: str):
        self.model = mlflow.pyfunc.load_model(model_path)

    def predict(self, data: dict) -> float:
        df = pd.DataFrame([data], columns=FEATURES)
        raw_pred = self.model.predict(df)[0]
        final_pred = max(0.0, raw_pred)
        return float(final_pred)
