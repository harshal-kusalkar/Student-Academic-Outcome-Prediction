from model.model_evaluation import evaluate

from utils.io import (
    load_model,
    load_csv,
    load_numpy,
    save_json,
)

from utils.logger import get_logger
from utils.load_config import load_config

logger = get_logger(__name__)


def run():

    logger.info("========== Evaluation Started ==========")

    # ------------------------------------------------
    # Load configuration
    # ------------------------------------------------

    config = load_config()

    logger.info("Configuration loaded successfully.")

    # ------------------------------------------------
    # Load processed test data
    # ------------------------------------------------

    X_test = load_csv(config.processed_data.X_test_path)
    y_test = load_numpy(config.processed_data.y_test_path)

    logger.info(
        "Loaded test data: %d samples, %d features",
        X_test.shape[0],
        X_test.shape[1],
    )

    # ------------------------------------------------
    # Load trained model
    # ------------------------------------------------

    model = load_model(config.artifacts.model_path)

    logger.info("Model loaded successfully.")

    # ------------------------------------------------
    # Evaluate
    # ------------------------------------------------

    metrics, report = evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test,
        config=config,
    )

    # ------------------------------------------------
    # Save evaluation artifacts
    # ------------------------------------------------

    save_json(
        metrics,
        config.artifacts.eval_result_path,
    )

    save_json(
        report,
        config.artifacts.classification_report_path,
    )

    logger.info("Evaluation artifacts saved.")

    logger.info("========== Evaluation Completed ==========")

if __name__ == "__main__":
    run()