
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import yaml

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load configuration
CONFIG_PATH = BASE_DIR / "config.yaml"

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

# Define model location
MODEL_PATH = BASE_DIR / config["output"]["model_path"]

# Feature names from configuration
FEATURES = config["features"]["numeric"]

# Store loaded model
model_store = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model when the API starts."""

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Run python -m src.train first."
        )

    model_store["model"] = joblib.load(MODEL_PATH)

    print("Penguin prediction model loaded successfully.")

    yield

    model_store.clear()


# Initialize FastAPI
app = FastAPI(
    title="Penguin Species Prediction API",
    description=(
        "AIPI 510 Project 3. "
        "Predict penguin species using a Random Forest model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# Define input schema
class PenguinInput(BaseModel):
    bill_length_mm: float = Field(gt=0)
    bill_depth_mm: float = Field(gt=0)
    flipper_length_mm: float = Field(gt=0)
    body_mass_g: float = Field(gt=0)


# Define prediction response
class PredictionResponse(BaseModel):
    predicted_species: str
    probabilities: dict[str, float]


@app.get("/")
def home():
    """Check whether the API is running."""

    return {
        "message": "Penguin Species Prediction API",
        "status": "running",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(penguin: PenguinInput):
    """Predict the species of a penguin."""

    model = model_store.get("model")

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    # Convert request to DataFrame
    input_data = pd.DataFrame(
        [penguin.model_dump()]
    )

    # Ensure correct feature order
    input_data = input_data[FEATURES]

    # Generate prediction
    prediction = model.predict(input_data)[0]

    # Generate class probabilities
    probabilities = model.predict_proba(input_data)[0]

    probability_dict = {
        species: float(probability)
        for species, probability in zip(
            model.classes_,
            probabilities
        )
    }

    return PredictionResponse(
        predicted_species=str(prediction),
        probabilities=probability_dict,
    )
