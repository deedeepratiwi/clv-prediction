from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_predict_endpoint():
    response = client.post(
        "/predict",
        json={
            "recency_days": 14,
            "frequency": 10,
            "total_revenue": 5000,
            "avg_order_value": 500,
            "tenure_days": 365,
            "active_months": 12,
            "purchase_velocity": 1.2,
            "avg_gap_days": 20,
            "std_gap_days": 5,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "predicted_clv_6m" in data
    assert data["predicted_clv_6m"] >= 0
