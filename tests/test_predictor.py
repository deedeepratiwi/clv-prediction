from unittest.mock import patch

import numpy as np
import pytest

from src.predict import CLVPredictor


class DummyModel:
    def predict(self, X):
        # Return positive CLV values
        return np.array([1000.0])


@pytest.fixture
def predictor():
    with patch("mlflow.pyfunc.load_model") as mock_load:
        mock_load.return_value = DummyModel()
        yield CLVPredictor(model_path="dummy-path")


def test_prediction_is_non_negative(predictor):
    sample_input = {
        "recency_days": 10,
        "frequency": 5,
        "total_revenue": 500,
        "avg_order_value": 100,
        "tenure_days": 300,
        "active_months": 6,
        "purchase_velocity": 0.2,
        "avg_gap_days": 30,
        "std_gap_days": 5,
    }

    prediction = predictor.predict(sample_input)

    assert prediction >= 0
