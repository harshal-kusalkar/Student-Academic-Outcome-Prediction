import pandas as pd

from utils.logger import get_logger


logger = get_logger(__name__)


def clean_data(
    data: pd.DataFrame,
    config,
) -> pd.DataFrame:
    """
    Clean raw student dataset.

    Responsibilities:
        - Normalize column names
        - Remove configured columns
        - Remove duplicate rows
        - Return cleaned dataframe
    """

    logger.info("Data cleaning started.")

    df = data.copy()

    # -----------------------------------------
    # 1. Normalize column names
    # -----------------------------------------

    df.columns = (
        df.columns
        .str.strip()
    )

    logger.debug(
        "Column names normalized."
    )

    # -----------------------------------------
    # 2. Remove configured columns
    # -----------------------------------------

    drop_columns = config.PREPROCESSING.drop_columns

    existing_columns = [
        column
        for column in drop_columns
        if column in df.columns
    ]

    if existing_columns:

        logger.info(
            "Dropping columns: %s",
            existing_columns,
        )

        df = df.drop(
            columns=existing_columns
        )

    # -----------------------------------------
    # 3. Remove duplicate rows
    # -----------------------------------------

    duplicate_count = int(
        df.duplicated().sum()
    )

    if duplicate_count > 0:

        logger.info(
            "Removing %d duplicate rows.",
            duplicate_count,
        )

        df = df.drop_duplicates(
            ignore_index=True
        )

    # -----------------------------------------
    # 4. Final result
    # -----------------------------------------

    logger.info(
        "Data cleaning completed. Shape: %s",
        df.shape,
    )

    return df

