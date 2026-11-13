
from pathlib import Path

import pandas as pd
from google.cloud import storage


PROJECT_ID = "penguin-species-predictor"
BUCKET_NAME = "penguin-species-predictor-data-338643374400"
BLOB_NAME = "penguins_clean.csv"

LOCAL_PATH = "data/processed/penguins_clean.csv"


def download_data():
    """Download cleaned penguin data from Google Cloud Storage."""

    print("Connecting to Google Cloud Storage...")

    # Authenticate with Application Default Credentials
    client = storage.Client(project=PROJECT_ID)

    # Access the private storage bucket
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(BLOB_NAME)

    # Prepare local destination
    local_path = Path(LOCAL_PATH)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Download the dataset
    blob.download_to_filename(str(local_path))

    print(f"Dataset downloaded to: {local_path}")

    # Verify the downloaded dataset
    df = pd.read_csv(local_path)

    print(f"Dataset shape: {df.shape}")

    print("\nSpecies distribution:")
    print(df["species"].value_counts())

    return df


if __name__ == "__main__":
    download_data()
