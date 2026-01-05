from predict import CLVPredictor

predictor = CLVPredictor("models/clv_linear_regression")

sample = {
    "recency_days": 14,
    "frequency": 18,
    "total_revenue": 8500,
    "avg_order_value": 470,
    "tenure_days": 600,
    "active_months": 12,
    "purchase_velocity": 1.5,
    "avg_gap_days": 15,
    "std_gap_days": 6,
}

print(f"Predicted_clv_6m: {predictor.predict(sample)}")
