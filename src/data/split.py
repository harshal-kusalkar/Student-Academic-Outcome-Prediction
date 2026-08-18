import pandas as pd

from sklearn.model_selection import train_test_split

from utils.logger import get_logger


logger = get_logger(__name__)


def split_data(
    df: pd.DataFrame,
    config,
):
    """
    Split dataset into training and testing sets.

    Returns:
        X_train
        X_test
        y_train
        y_test
    """

    logger.info("Data splitting started.")

    target_col = config.target.name

    test_size = config.data_split.test_size
    random_state = config.data_split.random_state

    # -----------------------------------------
    # Separate features and target
    # -----------------------------------------

    X = df.drop(
        columns=[target_col]
    )

    y = df[target_col]

    logger.info(
        "Features shape: %s",
        X.shape,
    )

    logger.info(
        "Target shape: %s",
        y.shape,
    )

    # -----------------------------------------
    # Train/Test split
    # -----------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
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

    logger.info(
        "Data splitting completed."
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )