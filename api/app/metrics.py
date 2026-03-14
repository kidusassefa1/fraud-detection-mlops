from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

PREDICTION_COUNTER = Counter("prediction_requests_total", "Total number of prediction requests")
FRAUD_PREDICTION_COUNTER = Counter("fraud_predictions_total", "Total number of fraud predictions")
API_ERROR_COUNTER = Counter("api_errors_total", "Total number of API errors")
PREDICTION_LATENCY_SECONDS = Histogram("prediction_latency_seconds", "Latency of prediction requests in seconds")

def metrics_response():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)