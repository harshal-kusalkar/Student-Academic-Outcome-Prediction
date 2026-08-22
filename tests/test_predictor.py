import pandas as pd

from src.inference.predictor import Predictor

from utils.io import load_csv, load_model


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