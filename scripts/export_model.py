from pathlib import Path
import shutil

SOURCE_MODEL = Path("models/artifacts/fraud_model.pkl")
EXPORT_DIR = Path("models/exported")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_MODEL = EXPORT_DIR / "fraud_model.pkl"


def export_model():
    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(f"Source model not found: {SOURCE_MODEL}")

    shutil.copy2(SOURCE_MODEL, TARGET_MODEL)
    print(f"Exported model to {TARGET_MODEL}")


if __name__ == "__main__":
    export_model()