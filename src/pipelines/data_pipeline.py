from src.data.data_validation import DataValidation

from utils.io import load_csv, save_json
from utils.logger import get_logger
from utils.load_config import load_config


logger = get_logger(__name__)


def run():
    """Run the data validation pipeline."""

    logger.info("=" * 60)
    logger.info("DATA VALIDATION PIPELINE STARTED")
    logger.info("=" * 60)

    # -----------------------------------------
    # 1. Load configuration
    # -----------------------------------------

    config = load_config()

    logger.info("Configuration loaded successfully.")

    # -----------------------------------------
    # 2. Load raw data
    # -----------------------------------------

    data_path = config.DATA_PATHS.data_path

    logger.info(
        "Loading dataset from: %s",
        data_path,
    )

    df = load_csv(path=data_path)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace("\t", "", regex=False)
      )

    logger.info(
        "Dataset loaded. Shape: %s",
        df.shape,
    )

    # -----------------------------------------
    # 3. Data validation
    # -----------------------------------------

    validator = DataValidation(
        config=config
    )

    validation_result = validator.validate(df)

    # -----------------------------------------
    # 4. Save validation report
    # -----------------------------------------

    save_json(
        data=validation_result,
        path=config.ARTIFACTS.data_validation_path,
    )

    logger.info(
        "Validation report saved to: %s",
        config.ARTIFACTS.data_validation_path,
    )

    # -----------------------------------------
    # Complete
    # -----------------------------------------

    logger.info("=" * 60)
    logger.info(
        "DATA VALIDATION PIPELINE COMPLETED"
    )
    logger.info("=" * 60)

if __name__ == "__main__":
    run()