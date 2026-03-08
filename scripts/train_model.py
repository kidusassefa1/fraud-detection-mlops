from src.models.train import train_model

def main():
    print("Starting fraud model training pipeline")

    train_model()

    print("Training Complete!")

if __name__ == "__main__":
    main()