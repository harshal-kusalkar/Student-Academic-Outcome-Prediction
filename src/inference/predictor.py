import mlflow
import pandas as pd

from utils.logger import get_logger


logger = get_logger(__name__)


class Predictor:
    """
    Load the champion model from MLflow
    and perform inference.
    """

    def __init__(
        self,
        tracking_uri: str,
        model_name: str,
        alias: str = "champion",
    ) -> None:

        self.model_uri = (
            f"models:/{model_name}@{alias}"
        )

        logger.info(
            "Loading model from MLflow: %s",
            self.model_uri,
        )

        mlflow.set_tracking_uri(
            tracking_uri
        )

        self.model = mlflow.sklearn.load_model(
            self.model_uri
        )

        logger.info(
            "Model loaded successfully."
        )

    def predict(
        self,
        data: pd.DataFrame,
    ):

        logger.info(
            "Running prediction. Input shape: %s",
            data.shape,
        )

        prediction = self.model.predict(
            data
        )

        logger.info(
            "Prediction completed."
        )

        return prediction