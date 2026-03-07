import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")


def load_train_data():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")

    return X_train, y_train


def load_test_data():
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")

    return X_test, y_test