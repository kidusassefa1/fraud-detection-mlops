from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from api.app.schemas import TransactionFeatures, PredictionResponse
from api.app.predictor import predict_transaction
from api.app.model_loader import load_model

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

@app.post("/predict", response_model=PredictionResponse)
def predict(features: TransactionFeatures, request: Request):
    model = request.app.state.model

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    result = predict_transaction(model, features.model_dump())
    return result