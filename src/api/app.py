import os

import pandas as pd
from fastapi import FastAPI

from dotenv import load_dotenv

from src.api.mappings import COLUMN_MAPPING
from src.api.schemas import StudentPredictionRequest
from src.inference.predictor import Predictor

from utils.load_config import load_config

load_dotenv()

app = FastAPI(
    title="Student Dropout Prediction API",
    description="ML inference API for student academic outcomes.",
    version="1.0.0",
)


config = load_config()

MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not all(
    [
        MLFLOW_TRACKING_USERNAME,
        MLFLOW_TRACKING_PASSWORD,
    ]
):
    raise RuntimeError("MLflow environment variables are not configured.")


predictor = Predictor(
    tracking_uri=config.mlflow.tracking_uri,
    model_name=config.mlflow.registered_model_name,
    model_alias=config.mlflow.model_alias,
    encoder_path=config.artifacts.encoder_path,
)


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": config.mlflow.registered_model_name,
        "model_alias": config.mlflow.model_alias,
    }


@app.post("/predict")
def predict(
    request: StudentPredictionRequest,
):

    data = request.model_dump()

    data = {
        COLUMN_MAPPING[key]: value
        for key, value in data.items()
    }

    input_data = pd.DataFrame([data])

    prediction = predictor.predict(input_data)

    return {
        "prediction": str(prediction[0]),
        "model": config.mlflow.registered_model_name,
        "model_alias": config.mlflow.model_alias,
    }