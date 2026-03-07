import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_PATH = Path("data/raw/creditcard.csv")

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    print("Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("Saving processed datasets...")

    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    print("Done!")


if __name__ == "__main__":
    main()