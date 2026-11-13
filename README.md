# Penguin Species Predictor

AIPI 510 Project 3: Data/ML Engineering

An end-to-end machine learning application that predicts penguin species using physical measurements. The project demonstrates cloud data storage, reproducible training, MLflow experiment tracking, Docker containerization, API deployment, and an interactive web interface.

## Live Application and Links

| Resource | Link |
|---|---|
| Streamlit Application | [Launch App](https://penguin-species-predictor-duke.streamlit.app) |
| FastAPI Documentation | [API Swagger UI](https://penguin-species-api-338643374400.us-west1.run.app/docs) |
| Prediction API Endpoint | [POST /predict](https://penguin-species-api-338643374400.us-west1.run.app/predict) |
| GitHub Repository | [Source Code](https://github.com/dukeai36/penguin-species-predictor) |

The prediction endpoint accepts POST requests containing JSON data.

## Dataset and Problem

The project uses the public [Palmer Penguins dataset](https://github.com/allisonhorst/palmerpenguins), containing measurements of Adelie, Chinstrap, and Gentoo penguins.

The task is multiclass classification, with `species` as the target variable.

Input features:

1. `bill_length_mm`
2. `bill_depth_mm`
3. `flipper_length_mm`
4. `body_mass_g`

The original dataset contains 344 observations. After removing records with missing required measurements, 342 observations remain.

The cleaned dataset is stored in a private Google Cloud Storage bucket and downloaded programmatically during training.

## System Architecture

```text
Palmer Penguins Dataset
         |
         v
Data Preprocessing
         |
         v
Google Cloud Storage
         |
         v
Training Pipeline
         |
         +----> MLflow Tracking
         |
         v
Random Forest Model
         |
         v
Docker Container
         |
         v
Google Artifact Registry
         |
         v
Google Cloud Run (FastAPI)
         |
         v
Streamlit Web Application
```

## Technology Stack

| Component | Technology |
|---|---|
| Data processing | pandas |
| Machine learning | scikit-learn |
| Model | Random Forest Classifier |
| Experiment tracking | MLflow |
| Cloud storage | Google Cloud Storage |
| Configuration | YAML |
| API | FastAPI |
| Containers | Docker |
| Container registry | Google Artifact Registry |
| API hosting | Google Cloud Run |
| Frontend | Streamlit Community Cloud |
| Version control | GitHub |

## Model Training and Results

The model uses a Random Forest classifier with 100 trees, maximum depth of 5, and random seed 42.

Stratified sampling preserves the proportions of the three species across dataset splits.

| Dataset | Samples |
|---|---:|
| Training | 218 |
| Validation | 55 |
| Test | 69 |

Evaluation results:

| Metric | Result |
|---|---:|
| Validation accuracy | 100% |
| Test accuracy | 100% |
| Test macro F1 score | 1.00 |

Accuracy is the primary evaluation metric. These results apply to the configured dataset split and do not guarantee equivalent performance on new data.

MLflow records model parameters, evaluation metrics, classification reports, and trained model artifacts.

## Project Structure

```text
penguin-species-predictor/
├── api/
│   └── main.py
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
│   ├── preprocess.py
│   ├── download_data.py
│   └── train.py
├── config.yaml
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

## Installation and Local Training

Python 3.13 is recommended.

Clone the repository and install dependencies:

```bash
git clone https://github.com/dukeai36/penguin-species-predictor.git
cd penguin-species-predictor

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
```

Authenticate with Google Cloud using an account authorized to read the private dataset:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project penguin-species-predictor
```

Download cloud data and train the model:

```bash
python -m src.download_data
python -m src.train
```

Training automatically downloads the dataset from Google Cloud Storage.

To preprocess the original local dataset, run:

```bash
python -m src.preprocess
```

The trained model is saved to `models/penguin_model.joblib`.

Start MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

View experiment runs at http://localhost:5000.

## API and Docker

Start FastAPI locally after training:

```bash
uvicorn api.main:app --reload --port 8000
```

API documentation: http://localhost:8000/docs

The API exposes `GET /` for service status and `POST /predict` for predictions.

Example request:

```json
{
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181,
  "body_mass_g": 3750
}
```

Example prediction:

```json
{
  "predicted_species": "Adelie",
  "probabilities": {
    "Adelie": 0.9996,
    "Chinstrap": 0.0004,
    "Gentoo": 0.0
  }
}
```

Test the deployed API using curl:

```bash
curl -X POST \
  https://penguin-species-api-338643374400.us-west1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181,
    "body_mass_g": 3750
  }'
```

Build and run the Docker container after generating the local model file:

```bash
docker build -t penguin-species-api .
docker run --rm -p 8080:8080 penguin-species-api
```

The container serves the API at http://localhost:8080/docs.

Docker also supports training by overriding its default command:

```bash
docker run --rm \
  -v "$HOME/.config/gcloud:/root/.config/gcloud:ro" \
  -e GOOGLE_CLOUD_PROJECT=penguin-species-predictor \
  penguin-species-api \
  python -m src.train
```

Generated training artifacts remain inside the temporary container unless volumes are mounted.

## Cloud Deployment

The cleaned dataset is hosted in Google Cloud Storage.

The API image is built for Linux AMD64, uploaded to Google Artifact Registry, and deployed to Google Cloud Run in `us-west1`.

Authenticate Docker:

```bash
gcloud auth configure-docker us-west1-docker.pkg.dev
```

Build and upload the image:

```bash
docker buildx build \
  --platform linux/amd64 \
  -t us-west1-docker.pkg.dev/penguin-species-predictor/penguin-api-repo/penguin-api:v1 \
  --load .

docker push us-west1-docker.pkg.dev/penguin-species-predictor/penguin-api-repo/penguin-api:v1
```

Deploy to Cloud Run:

```bash
gcloud run deploy penguin-species-api \
  --image us-west1-docker.pkg.dev/penguin-species-predictor/penguin-api-repo/penguin-api:v1 \
  --region us-west1 \
  --port 8080 \
  --memory 1Gi \
  --allow-unauthenticated \
  --min-instances 0 \
  --max-instances 2 \
  --project penguin-species-predictor
```

The image includes a locally generated model file. Cloud deployment therefore requires building the image from a project directory containing that file.

Deployed API:

https://penguin-species-api-338643374400.us-west1.run.app

## Streamlit Frontend

The Streamlit application accepts four penguin measurements and sends a JSON request to the deployed FastAPI endpoint.

It displays the predicted species, probability chart, and probability table.

Run locally:

```bash
streamlit run app/streamlit_app.py
```

Public application:

https://penguin-species-predictor-duke.streamlit.app

The frontend is deployed through Streamlit Community Cloud using the GitHub `main` branch.

## Collaboration and Reproducibility

Development uses Git feature branches and pull requests. Changes to preprocessing, cloud ingestion, training, API serving, Docker, and Streamlit are maintained in the GitHub commit history.

Pipeline parameters are controlled through `config.yaml`, dependencies through `requirements.txt`, and experiments through MLflow.

Local credentials, generated model files, and tracking databases are excluded from version control.

Each team member must contribute at least one pull request, as required by the assignment.

## Ethical Considerations and Limitations

The dataset represents a limited sample of three penguin species. Predictions may not generalize to other populations or species.

Model performance depends on measurement quality. The 100% test accuracy reflects one dataset split, not guaranteed real-world accuracy.

Prediction probabilities are not necessarily calibrated. The application is intended for educational demonstration rather than scientific or conservation decision making.

## References

[Palmer Penguins](https://github.com/allisonhorst/palmerpenguins) | [scikit-learn](https://scikit-learn.org) | [MLflow](https://mlflow.org) | [FastAPI](https://fastapi.tiangolo.com) | [Docker](https://docs.docker.com) | [Google Cloud Run](https://cloud.google.com/run) | [Streamlit](https://streamlit.io)