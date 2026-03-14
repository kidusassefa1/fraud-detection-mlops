from pathlib import Path
import joblib

MODEL_PATH = Path("models/exported/fraud_model.pkl")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Exported model not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    return model