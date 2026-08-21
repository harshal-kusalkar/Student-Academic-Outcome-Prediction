import os

import pandas as pd
from fastapi import FastAPI
import mlflow

from src.api.mappings import COLUMN_MAPPING
from src.api.schemas import StudentPredictionRequest
from src.inference.predictor import Predictor

from utils.load_config import load_config
from utils.logger import get_logger


logger = get_logger(__name__)


app = FastAPI(
    title="Student Dropout Prediction API",
    description="ML inference API for student academic outcomes.",
    version="1.0.0",
)


config = load_config()

client = mlflow.MlflowClient()

model_version = client.get_model_version_by_alias(
    config.mlflow.registered_model_name,
    config.mlflow.model_alias,
)


logger.info(
    "Using MLflow tracking URI: %s",
    config.mlflow.tracking_uri,
)


predictor = Predictor(
    tracking_uri=config.mlflow.tracking_uri,
    encoder_path=config.artifacts.encoder_path,
)


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": config.mlflow.registered_model_name,
        "model_alias": config.mlflow.model_alias,
        "version": f"version: {model_version.version}"
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