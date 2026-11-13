
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def train_model(config_path="config.yaml"):

    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Load cleaned dataset
    data_path = config["data"]["processed_path"]
    df = pd.read_csv(data_path)

    # Select features and target
    features = config["features"]["numeric"]
    target = config["data"]["target"]

    X = df[features]
    y = df[target]

    # Split data into training/validation and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=y,
    )

    # Split remaining data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=config["split"]["validation_size"],
        random_state=config["split"]["random_state"],
        stratify=y_temp,
    )

    print("Dataset splits:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Initialize Random Forest model
    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        random_state=config["model"]["random_state"],
    )

    # Configure MLflow tracking
    mlflow.set_tracking_uri(
        config["tracking"]["tracking_uri"]
    )

    mlflow.set_experiment(
        config["tracking"]["experiment_name"]
    )

    # Start MLflow experiment
    with mlflow.start_run() as run:

        # Log model parameters
        mlflow.log_params({
            "model_type": config["model"]["type"],
            "n_estimators": config["model"]["n_estimators"],
            "max_depth": config["model"]["max_depth"],
            "random_state": config["model"]["random_state"],
            "test_size": config["split"]["test_size"],
            "validation_size": config["split"]["validation_size"],
        })

        # Train model
        model.fit(X_train, y_train)

        print("\nModel training completed.")

        # Generate predictions
        val_predictions = model.predict(X_val)
        test_predictions = model.predict(X_test)

        # Calculate evaluation metrics
        val_accuracy = accuracy_score(
            y_val, val_predictions
        )

        test_accuracy = accuracy_score(
            y_test, test_predictions
        )

        print(f"\nValidation Accuracy: {val_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        print("\nTest Classification Report:")

        report = classification_report(
            y_test,
            test_predictions
        )

        print(report)

        # Log evaluation metrics to MLflow
        mlflow.log_metric(
            "validation_accuracy",
            val_accuracy
        )

        mlflow.log_metric(
            "test_accuracy",
            test_accuracy
        )

        # Log classification report
        mlflow.log_text(
            report,
            "classification_report.txt"
        )

        # Save trained model locally
        model_path = Path(
            config["output"]["model_path"]
        )

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(model, model_path)

        print(f"\nModel saved to: {model_path}")

        # Log trained model to MLflow
        # Use cloudpickle to avoid the skops
        # Random Forest trusted-type error.
        mlflow.sklearn.log_model(
            sk_model=model,
            name="penguin_model",
            serialization_format="cloudpickle",
            input_example=X_train.head(5),
        )

        print("\nExperiment logged to MLflow.")
        print(f"MLflow Run ID: {run.info.run_id}")

    return model


if __name__ == "__main__":
    train_model()
