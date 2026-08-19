from sklearn.preprocessing import LabelEncoder

from utils.logger import get_logger


logger = get_logger(__name__)


def encode_target(
    y_train,
    y_test,
):
    """
    Encode target labels into integer classes.

    The encoder is fitted ONLY on y_train.
    """

    encoder = LabelEncoder()

    y_train_encoded = encoder.fit_transform(
        y_train
    )

    y_test_encoded = encoder.transform(
        y_test
    )

    logger.info(
        "Target classes: %s",
        list(encoder.classes_),
    )

    return (
        y_train_encoded,
        y_test_encoded,
        encoder,
    )