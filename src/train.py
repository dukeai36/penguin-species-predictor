
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.download_data import download_data


PUBLIC_DATA_URL = (
    "https://raw.githubusercontent.com/allisonhorst/"
    "palmerpenguins/master/inst/extdata/penguins.csv"
)


def load_training_data(config):
    data_source = os.getenv("PENGUIN_DATA_SOURCE", "gcs").lower()

    processed_path = Path(config["data"]["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    features = config["features"]["numeric"]
    target = config["data"]["target"]
    required_columns = features + [target]

    if data_source == "gcs":
        print("Downloading cleaned data from Google Cloud Storage...")
        download_data()
        df = pd.read_csv(processed_path)

    elif data_source == "public":
        print("Downloading public Palmer Penguins dataset...")
        df = pd.read_csv(PUBLIC_DATA_URL)

        print("Preprocessing public dataset...")
        df = df[required_columns].dropna().copy()
        df.to_csv(processed_path, index=False)

        print(f"Cleaned dataset saved to: {processed_path}")

    else:
        raise ValueError(
            "PENGUIN_DATA_SOURCE must be 'gcs' or 'public'."
        )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    df = df[required_columns].dropna().copy()

    print(f"Data source: {data_source}")
    print(f"Dataset rows: {len(df)}")

    return df, data_source


def train_model(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    df, data_source = load_training_data(config)

    features = config["features"]["numeric"]
    target = config["data"]["target"]

    X = df[features]
    y = df[target]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=config["split"]["validation_size"],
        random_state=config["split"]["random_state"],
        stratify=y_temp,
    )

    print("\nDataset splits:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        random_state=config["model"]["random_state"],
    )

    mlflow.set_tracking_uri(
        config["tracking"]["tracking_uri"]
    )

    mlflow.set_experiment(
        config["tracking"]["experiment_name"]
    )

    with mlflow.start_run() as run:
        mlflow.log_params({
            "model_type": config["model"]["type"],
            "n_estimators": config["model"]["n_estimators"],
            "max_depth": config["model"]["max_depth"],
            "random_state": config["model"]["random_state"],
            "test_size": config["split"]["test_size"],
            "validation_size": config["split"]["validation_size"],
            "data_source": data_source,
        })

        model.fit(X_train, y_train)

        print("\nModel training completed.")

        val_predictions = model.predict(X_val)
        test_predictions = model.predict(X_test)

        val_accuracy = accuracy_score(
            y_val, val_predictions
        )

        test_accuracy = accuracy_score(
            y_test, test_predictions
        )

        print(f"\nValidation Accuracy: {val_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        report = classification_report(
            y_test,
            test_predictions
        )

        print("\nTest Classification Report:")
        print(report)

        mlflow.log_metric(
            "validation_accuracy", val_accuracy
        )

        mlflow.log_metric(
            "test_accuracy", test_accuracy
        )

        mlflow.log_text(
            report, "classification_report.txt"
        )

        model_path = Path(
            config["output"]["model_path"]
        )

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(model, model_path)

        print(f"\nModel saved to: {model_path}")

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
