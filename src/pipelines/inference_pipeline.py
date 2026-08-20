import pandas as pd

from src.inference.predictor import Predictor

from utils.load_config import load_config
from utils.io import load_model
from utils.logger import get_logger


logger = get_logger(__name__)


def run(
    input_data: pd.DataFrame,
):

    logger.info(
        "======== INFERENCE PIPELINE STARTED ========"
    )

    config = load_config()

    predictor = Predictor(
        tracking_uri=config.mlflow.tracking_uri,
        model_name=(
            config.mlflow.registered_model_name
        ),
        alias=config.mlflow.model_alias,
        encoder_path=config.artifacts.encoder_path
    )

    predictions = predictor.predict(
        input_data
    )

    logger.info(
        "======== INFERENCE PIPELINE COMPLETED ========"
    )

    return predictions