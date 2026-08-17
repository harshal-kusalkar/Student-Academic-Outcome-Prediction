import pytest
from pydantic import ValidationError

from utils.load_config import load_config


def test_load_config():

    config = load_config()

    assert config is not None


def test_numerical_preprocessing_config():

    config = load_config()

    numerical = config.PREPROCESSING.numerical

    assert numerical.imputer.strategy == "median"
    assert numerical.scaler.type == "standard"


def test_categorical_preprocessing_config():

    config = load_config()

    categorical = config.PREPROCESSING.categorical

    assert categorical.imputer.strategy == "most_frequent"
    assert categorical.encoder.type == "onehot"
    assert categorical.encoder.handle_unknown == "ignore"


def test_binary_preprocessing_config():

    config = load_config()

    binary = config.PREPROCESSING.binary

    assert binary.imputer.strategy == "most_frequent"


def test_target_config():

    config = load_config()

    assert config.TARGET.name == "Target"

    assert set(config.TARGET.classes) == {
        "Dropout",
        "Enrolled",
        "Graduate",
    }


def test_data_path():

    config = load_config()

    assert (
        config.DATA_PATHS.data_path
        == "data/raw/student_academic_data/data.csv"
    )