import time

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel

from src.predict import CLVPredictor

REQUEST_COUNT = Counter("clv_requests_total", "Total number of prediction requests")

REQUEST_LATENCY = Histogram(
    "clv_request_latency_seconds", "Latency of prediction requests"
)


app = FastAPI(title="CLV Prediction API")

predictor = CLVPredictor(model_path="models/clv_linear_regression")


class CLVRequest(BaseModel):
    recency_days: float
    frequency: float
    total_revenue: float
    avg_order_value: float
    tenure_days: float
    active_months: float
    purchase_velocity: float
    avg_gap_days: float
    std_gap_days: float


class CLVResponse(BaseModel):
    predicted_clv_6m: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=CLVResponse)
def predict_clv(request: CLVRequest):
    REQUEST_COUNT.inc()

    start = time.time()
    prediction = predictor.predict(request.model_dump())
    REQUEST_LATENCY.observe(time.time() - start)

    return {"predicted_clv_6m": prediction}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
