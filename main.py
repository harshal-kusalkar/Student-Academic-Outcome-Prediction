from src.pipelines import (
    data_pipeline,
    training_pipeline,
    evaluation_pipeline,
)

from utils.logger import get_logger


logger = get_logger(__name__)


def run():
    """
    Main pipeline orchestrator.

    Executes the complete ML pipeline in order:

    1. Data pipeline
    2. Training pipeline
    3. Evaluation pipeline
    """

    logger.info("=" * 70)
    logger.info("MAIN ML PIPELINE STARTED")
    logger.info("=" * 70)

    try:

        # -----------------------------------------
        # Data Pipeline
        # -----------------------------------------

        logger.info("Starting data pipeline.")

        data_pipeline.run()

        logger.info(
            "Data pipeline completed successfully."
        )

        # -----------------------------------------
        # Training Pipeline
        # -----------------------------------------

        logger.info("Starting training pipeline.")

        training_result = (
            training_pipeline.run()
        )

        logger.info(
            "Training pipeline completed successfully."
        )

        # -----------------------------------------
        # Evaluation Pipeline
        # -----------------------------------------

        logger.info("Starting evaluation pipeline.")

        evaluation_result = (
            evaluation_pipeline.run()
        )

        logger.info(
            "Evaluation pipeline completed successfully."
        )

        # -----------------------------------------
        # Complete
        # -----------------------------------------

        logger.info("=" * 70)
        logger.info(
            "MAIN ML PIPELINE COMPLETED SUCCESSFULLY"
        )
        logger.info("=" * 70)

        return {
            "training": training_result,
            "evaluation": evaluation_result,
        }

    except Exception:

        logger.exception(
            "MAIN ML PIPELINE FAILED."
        )

        raise


if __name__ == "__main__":
    run()