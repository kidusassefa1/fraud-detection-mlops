import os
import requests
from pathlib import Path

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

DATA_URL = Path("data/raw")
DATA_URL.mkdir(parents=True, exist_ok=True) 

OUTPUT_FILE = DATA_URL / "creditcard.csv"

def download_data():
    if OUTPUT_FILE.exists():
        print(f"Data file already exists at {OUTPUT_FILE}. Skipping download.")
        return

    print(f"Downloading data from {DATA_URL}...")

    response = requests.get(DATA_URL, stream=True)
    response.raise_for_status()

    with open(OUTPUT_FILE, "wb") as f:
        f.write(response.content)

    print(f"Data downloaded and saved to {OUTPUT_FILE}.")


if __name__ == "__main__":
    download_data()