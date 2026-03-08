import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from src.data.load_data import load_train_data, load_test_data

MODEL_DIR = Path("models/artifacts")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_model():

    print("Loading training data...")

    X_train, y_train = load_train_data()
    X_test, y_test = load_test_data()

    print("Training RandomForest model...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train.values.ravel())

    print("Evaluating model...")

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, probabilities)

    print("\nModel Evaluation")
    print(classification_report(y_test, predictions))
    print("ROC AUC:", roc_auc)

    model_path = MODEL_DIR / "fraud_model.pkl"

    print("Saving model to", model_path)

    joblib.dump(model, model_path)

    return model


if __name__ == "__main__":
    train_model()