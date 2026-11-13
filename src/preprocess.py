
from pathlib import Path

import pandas as pd
import yaml


def preprocess_data(config_path="config.yaml"):
    # Load pipeline configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Read dataset and configuration
    raw_path = Path(config["data"]["raw_path"])
    processed_path = Path(config["data"]["processed_path"])
    features = config["features"]["numeric"]
    target = config["data"]["target"]

    print("Loading penguin dataset...")
    df = pd.read_csv(raw_path)

    print(f"Original dataset shape: {df.shape}")

    # Keep only required features and target
    required_columns = features + [target]
    df = df[required_columns].copy()

    # Report missing values
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # Remove rows with missing required values
    df = df.dropna(subset=required_columns)

    print(f"\nCleaned dataset shape: {df.shape}")
    print("\nSpecies distribution:")
    print(df[target].value_counts())

    # Create output folder if necessary
    processed_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save cleaned data
    df.to_csv(processed_path, index=False)

    print(f"\nCleaned dataset saved to: {processed_path}")

    return df


if __name__ == "__main__":
    preprocess_data()
