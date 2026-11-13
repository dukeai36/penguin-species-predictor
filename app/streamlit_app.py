
import os

import pandas as pd
import requests
import streamlit as st


# API configuration
DEFAULT_API_URL = (
    "https://penguin-species-api-338643374400."
    "us-west1.run.app"
)

API_URL = os.getenv(
    "API_URL",
    DEFAULT_API_URL
).rstrip("/")


# Page configuration
st.set_page_config(
    page_title="Penguin Species Predictor",
    page_icon="🐧",
    layout="centered",
)


# Application title
st.title("Penguin Species Predictor")

st.write(
    "Predict whether a penguin is an Adelie, "
    "Chinstrap, or Gentoo using its physical measurements."
)

st.caption(
    "AIPI 510 Project 3 | Machine Learning Engineering"
)

st.divider()


# User input form
st.subheader("Enter Penguin Measurements")

with st.form("prediction_form"):

    bill_length = st.number_input(
        "Bill Length (mm)",
        min_value=1.0,
        max_value=100.0,
        value=39.1,
        step=0.1,
    )

    bill_depth = st.number_input(
        "Bill Depth (mm)",
        min_value=1.0,
        max_value=50.0,
        value=18.7,
        step=0.1,
    )

    flipper_length = st.number_input(
        "Flipper Length (mm)",
        min_value=1.0,
        max_value=300.0,
        value=181.0,
        step=1.0,
    )

    body_mass = st.number_input(
        "Body Mass (g)",
        min_value=100.0,
        max_value=10000.0,
        value=3750.0,
        step=50.0,
    )

    submitted = st.form_submit_button(
        "Predict Penguin Species",
        use_container_width=True,
    )


# Prediction request
if submitted:

    payload = {
        "bill_length_mm": bill_length,
        "bill_depth_mm": bill_depth,
        "flipper_length_mm": flipper_length,
        "body_mass_g": body_mass,
    }

    try:
        with st.spinner("Predicting penguin species..."):

            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )

            response.raise_for_status()

            result = response.json()

        predicted_species = result["predicted_species"]
        probabilities = result["probabilities"]

        st.success(
            f"Predicted Species: {predicted_species}"
        )

        st.subheader("Prediction Probabilities")

        probability_df = pd.DataFrame({
            "Species": list(probabilities.keys()),
            "Probability": list(probabilities.values()),
        })

        probability_df = probability_df.set_index(
            "Species"
        )

        st.bar_chart(probability_df)

        st.dataframe(
            probability_df.style.format(
                {"Probability": "{:.2%}"}
            ),
            use_container_width=True,
        )

    except requests.exceptions.RequestException as error:
        st.error(
            "Unable to connect to the prediction API. "
            "Please try again."
        )
        st.caption(str(error))

    except (KeyError, ValueError) as error:
        st.error(
            "The API returned an unexpected response."
        )
        st.caption(str(error))


st.divider()

st.caption(
    "Model: Random Forest Classifier | "
    "Dataset: Palmer Penguins | "
    "API: Google Cloud Run"
)

st.caption(
    "Predictions are estimates based on the training "
    "dataset and may not generalize to all penguins."
)
