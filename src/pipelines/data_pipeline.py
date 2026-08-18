from src.data import data_cleaning
from src.data.data_validation import DataValidation

from utils.io import load_csv, save_json
from utils.logger import get_logger
from utils.load_config import load_config


logger = get_logger(__name__)


def run():

    logger.info("DATA PIPELINE STARTED")

    config = load_config()

    data_path = (
        config.DATA_PATHS.data_path
    )

    df = load_csv(path=data_path)

    # Normalize raw column formatting
    df.columns = df.columns.str.strip()

    # -----------------------------
    # Validation
    # -----------------------------

    validator = DataValidation(
        config=config
    )

    validation_result = validator.validate(df)

    save_json(
        data=validation_result,
        path=config.ARTIFACTS.data_validation_path,
    )

    # -----------------------------
    # Cleaning
    # -----------------------------

    df = data_cleaning.clean_data(
        data=df,
        config=config,
    )

    logger.info(
        "Final cleaned shape: %s",
        df.shape,
    )

    logger.info(
        "DATA PIPELINE COMPLETED"
    )

if __name__ == "__main__":
    run()