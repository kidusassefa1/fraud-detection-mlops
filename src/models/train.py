import joblib
import time
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

from src.data.load_data import load_train_data, load_test_data
from src.models.evaluate import evaluate_model

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

MODEL_DIR = Path("models/artifacts")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_model():
    print("Loading training data...")

    X_train, y_train = load_train_data()
    X_test, y_test = load_test_data()

    params = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
    }

    with mlflow.start_run():

        start_time = time.time()

        print("Training RandomForest model...")

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            random_state=params["random_state"],
            n_jobs=params["n_jobs"],
        )

        model.fit(X_train, y_train.values.ravel())

        training_time = time.time() - start_time

    print("Evaluating model...")

    amounts = X_test["Amount"]
    metrics = evaluate_model(model, X_test, y_test.values.ravel(), amounts)

    metrics["training_time_seconds"] = training_time
    metrics["train_samples"] = len(X_train)
    metrics["test_samples"] = len(X_test)
    metrics["fraud_ratio"] = float(y_train.mean())

    mlflow.log_params(params)
    print("\nModel Parameters")
    for key, value in params.items():
        print(f"{key}: {value}")

    print("\nModel Metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    mlflow.log_metrics(metrics)
    model_path = MODEL_DIR / "fraud_model.pkl"
    print(f"\nSaving model to {model_path}")
    
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
    )

    joblib.dump(model, model_path)
    mlflow.log_artifact(str(model_path))

    return model, metrics, params


if __name__ == "__main__":
    train_model()