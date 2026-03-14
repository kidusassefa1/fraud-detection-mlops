from contextlib import asynccontextmanager
import time

from fastapi import FastAPI, HTTPException, Request

from api.app.schemas import TransactionFeatures, PredictionResponse
from api.app.predictor import predict_transaction
from api.app.model_loader import load_model
from api.app.metrics import (
    PREDICTION_REQUESTS_TOTAL, 
    FRAUD_PREDICTION_COUNTER, 
    API_ERRORS_TOTAL, 
    PREDICTION_LATENCY_SECONDS, 
    metrics_response
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield
    app.state.model = None

app = FastAPI(
    title="Fraud Detection Inference API",
    version="0.1.0",
    lifespan=lifespan,
)

    
@app.get("/health")
def health_check(request: Request):
    return {
        "status": "ok",
        "model_loaded": request.app.state.model is not None,
    }


@app.get("/metrics")
def get_metrics():
    return metrics_response()


@app.post("/predict", response_model=PredictionResponse)
def predict(features: TransactionFeatures, request: Request):
    model = request.app.state.model

    if model is None:
        API_ERRORS_TOTAL.inc()
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    PREDICTION_REQUESTS_TOTAL.inc()

    start_time = time.time()

    try:
        result = predict_transaction(model, features.model_dump())

        if result["prediction"] == 1:
            FRAUD_PREDICTION_COUNTER.inc()

        return result
    
    except Exception:
        API_ERRORS_TOTAL.inc()
        raise HTTPException(status_code=500, detail="Error during prediction")
    
    finally:
        latency = time.time() - start_time
        PREDICTION_LATENCY_SECONDS.observe(latency)