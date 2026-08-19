from src.model.final_evaluation import (
    evaluate_final_model,
)

from utils.io import (
    load_csv,
    load_model,
    load_numpy,
    save_json,
)

from utils.load_config import load_config
from utils.logger import get_logger


logger = get_logger(__name__)


def run():

    logger.info(
        "======== EVALUATION PIPELINE STARTED ========"
    )

    config = load_config()

    # -----------------------------------------
    # Load final model
    # -----------------------------------------

    model = load_model(
        path=config.artifacts.model_path
    )

    logger.info(
        "Final model loaded."
    )

    # -----------------------------------------
    # Load untouched test data
    # -----------------------------------------

    X_test = load_csv(
        path=config.data_paths.X_test_path
    )

    y_test = load_numpy(
        path=config.data_paths.y_test_path
    )

    logger.info(
        "Test data loaded. Shape: %s",
        X_test.shape,
    )

    # -----------------------------------------
    # Load class names
    # -----------------------------------------

    encoder = load_model(
        path=config.artifacts.encoder_path
    )

    class_names = list(
        encoder.classes_
    )

    # -----------------------------------------
    # Evaluate
    # -----------------------------------------

    result = evaluate_final_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
    )

    logger.info(
        "Evaluation metrics: %s",
        result["metrics"],
    )


    # -----------------------------------------
    # Save Artifacts
    # -----------------------------------------

    save_json(
        data={
            "metrics": result["metrics"],
            "classification_report": result[
                "classification_report"
            ],
            "confusion_matrix": result[
                "confusion_matrix"
            ],
        },
        path=config.artifacts.evaluation_path,
    )

    logger.info("Evaluation Artifacts Saved")


    logger.info(
        "======== EVALUATION PIPELINE COMPLETED ========"
    )

    return result


# if __name__ == "__main__":
#     run()