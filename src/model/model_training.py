from sklearn.pipeline import Pipeline

from utils.logger import get_logger


logger = get_logger(__name__)


def train_final_model(
    pipeline: Pipeline,
    X_train,
    y_train,
    best_params: dict,
):
    """
    Apply the best hyperparameters and train the
    complete pipeline on the full training dataset.
    """

    logger.info(
        "Starting final model training."
    )

    pipeline.set_params(
        **{
            f"model__{key}": value
            for key, value in best_params.items()
        }
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    logger.info(
        "Final model training completed."
    )

    return pipeline