import time
from pathlib import Path

import mlflow
import pandas as pd

from utils.io import load_model
from utils.logger import get_logger


logger = get_logger(__name__)


class Predictor:

    def __init__(
        self,
        tracking_uri: str,
        model_name: str,
        model_alias: str,
        encoder_path: Path,
    ):

        mlflow.set_tracking_uri(
            tracking_uri
        )

        self.model_uri = (
            f"models:/{model_name}@{model_alias}"
        )

        logger.info(
            "Loading model from MLflow: %s",
            self.model_uri,
        )

        start = time.time()

        self.model = mlflow.pyfunc.load_model(
            self.model_uri
        )

        self.model_loading_time = time.time()-start

        logger.info(
            f"Model Loading time: {self.model_loading_time}"
        )

        self.encoder = load_model(
            path=encoder_path
        )

        logger.info(
            "Model and label encoder loaded successfully."
        )

    def predict(
        self,
        data: pd.DataFrame,
    ):

        prediction = self.model.predict(
            data
        )

        prediction = self.encoder.inverse_transform(
            prediction
        )

        return prediction