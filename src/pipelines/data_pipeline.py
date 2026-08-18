from src.data import data_cleaning
from src.data.data_validation import DataValidation
from src.data import split

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


    # ==================================================
    # 5. Train/Test split
    # ==================================================

    logger.info("Starting train/test split.")

    X_train, X_test, y_train, y_test = (
        split.split_data(
            df=df,
            config=config,
        )
    )

    logger.info(
        "Train/Test split completed."
    )

    logger.info(
        "X_train shape: %s",
        X_train.shape,
    )

    logger.info(
        "X_test shape: %s",
        X_test.shape,
    )

    logger.info(
        "y_train shape: %s",
        y_train.shape,
    )

    logger.info(
        "y_test shape: %s",
        y_test.shape,
    )

    # ==================================================
    # Pipeline completed
    # ==================================================
    
    logger.info("=" * 60)
    logger.info(
        "DATA PIPELINE COMPLETED SUCCESSFULLY"
    )
    logger.info("=" * 60)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


if __name__ == "__main__":
    run()