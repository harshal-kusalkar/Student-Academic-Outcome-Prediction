import os

import pandas as pd
from dotenv import load_dotenv

from src.inference.predictor import Predictor

from utils.io import load_csv


load_dotenv()

MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

def test_predictor(config):

    predictor = Predictor(
        config.mlflow.tracking_uri,
        config.mlflow.registered_model_name,
        config.mlflow.model_alias,
        config.artifacts.encoder_path

    )

    X_test = load_csv(
        config.data_paths.X_test_path
    )

    sample = X_test.head(5)

    predictions = predictor.predict(
        sample
    )

    assert len(predictions) == 5