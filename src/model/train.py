from time import perf_counter

from lightgbm import LGBMClassifier

from utils.logger import get_logger

logger = get_logger(__name__)


def train_model(
    X_train,
    y_train,
    config,
) -> LGBMClassifier:
    """
    Train the LightGBM classifier using the configured hyperparameters.
    """

    logger.info("Starting LightGBM training...")
    logger.info(
        "Training data shape: %d samples, %d features",
        X_train.shape[0],
        X_train.shape[1],
    )

    params = config.model_params.model_dump()

    logger.debug("Model hyperparameters: %s", params)

    start = perf_counter()

    try:
        model = LGBMClassifier(
            **params,
            random_state=config.training.random_state,
            n_jobs=config.training.n_jobs,
        )

        model.fit(X_train, y_train)

    except Exception:
        logger.exception("Model training failed.")
        raise

    elapsed = perf_counter() - start

    logger.info("Training completed successfully.")
    logger.info("Training time: %.2f seconds", elapsed)

    return model