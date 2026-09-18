import argparse
import sys
from src.train import train_model
from src.predict import make_predictions

def main():
    parser = argparse.ArgumentParser(description="Customer Churn Prediction CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    # Train sub-command
    train_parser = subparsers.add_parser("train", help="Train the churn prediction model")
    train_parser.add_argument("--data-path", required=True, help="Path to training CSV dataset")

    # Predict sub-command
    predict_parser = subparsers.add_parser("predict", help="Run batch predictions on new dataset")
    predict_parser.add_argument("--input-path", required=True, help="Path to input CSV dataset")
    predict_parser.add_argument("--output-path", required=True, help="Path to save predictions CSV")

    args = parser.parse_args()

    if args.command == "train":
        train_model(args.data_path)
    elif args.command == "predict":
        make_predictions(args.input_path, args.output_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
