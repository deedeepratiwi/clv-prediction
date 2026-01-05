import argparse
import shutil
from math import sqrt
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MODELS_DIR = PROJECT_ROOT / "models" / "clv_linear_regression"


def regression_metrics(y_true, y_pred):
    return {
        "rmse": sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def get_production_rmse(model_name: str):
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            return None
        run_id = versions[0].run_id
        run = client.get_run(run_id)
        return run.data.metrics.get("val_rmse")
    except Exception:
        return None


def promote_if_better(model_name: str, new_val_rmse: float):
    client = MlflowClient()

    current_rmse = get_production_rmse(model_name)

    if current_rmse is None:
        print("No Production model found → promoting")
        promote = True
    else:
        promote = new_val_rmse < current_rmse
        print(f"Current prod RMSE: {current_rmse:.4f}")
        print(f"New model RMSE:    {new_val_rmse:.4f}")

    if promote:
        client.transition_model_version_stage(
            name=model_name,
            version=client.get_latest_versions(model_name, stages=["None"])[0].version,
            stage="Production",
            archive_existing_versions=True,
        )
        print("✅ Model promoted to Production")
        return True
    else:
        print("❌ Model NOT promoted")
        return False


def main(data_path: str):
    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_parquet(data_path)

    target = "clv_6m"

    features = [
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

    X = df[features]
    y = df[target]

    # -----------------------------
    # Train / Val / Test split
    # -----------------------------
    # Single-snapshot CLV → random split with fixed seed

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # -----------------------------
    # Model (champion)
    # -----------------------------
    model = Pipeline(
        steps=[("scaler", StandardScaler()), ("model", LinearRegression())]
    )

    # -----------------------------
    # MLflow tracking
    # -----------------------------

    MODEL_NAME = "clv_linear_regression"
    EXPORT_PATH = Path("models/clv_linear_regression")

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("clv_prediction")

    with mlflow.start_run(run_name="linear_regression_final"):
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)

        val_metrics = regression_metrics(y_val, val_preds)
        test_metrics = regression_metrics(y_test, test_preds)

        mlflow.log_params(
            {
                "model_type": "linear_regression",
                "scaling": "standard",
                "features": features,
                "target": target,
                "split_strategy": "random_single_snapshot",
                "random_state": 42,
            }
        )

        mlflow.log_metrics(
            {
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
            }
        )

        mlflow.sklearn.log_model(model, name="model", registered_model_name=MODEL_NAME)

        # -----------------------------
        # Promotion decision
        # -----------------------------
        promoted = promote_if_better(
            model_name=MODEL_NAME, new_val_rmse=val_metrics["rmse"]
        )

        # -----------------------------
        # Export ONLY if promoted
        # -----------------------------
        if promoted:
            if EXPORT_PATH.exists():
                shutil.rmtree(EXPORT_PATH)

            mlflow.sklearn.save_model(sk_model=model, path=EXPORT_PATH)
            print("📦 Champion model exported")
        else:
            print("📦 Export skipped (not champion)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/clv_modeling_table.parquet",
        help="Path to CLV modeling table (parquet)",
    )
    args = parser.parse_args()

    main(args.data_path)
