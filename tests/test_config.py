from utils.load_config import load_config

def test_load_config():

    config = load_config()

    assert config is not None


def test_numerical_preprocessing_config():

    config = load_config()

    numerical = config.preprocessing.numerical

    assert numerical.imputer.strategy == "median"
    assert numerical.scaler.type == "standard"


def test_categorical_preprocessing_config():

    config = load_config()

    categorical = config.preprocessing.categorical

    assert categorical.imputer.strategy == "most_frequent"
    assert categorical.encoder.type == "onehot"
    assert categorical.encoder.handle_unknown == "ignore"


def test_binary_preprocessing_config():

    config = load_config()

    binary = config.preprocessing.binary

    assert binary.imputer.strategy == "most_frequent"


def test_target_config():

    config = load_config()

    assert config.target.name == "Target"

    assert set(config.target.classes) == {
        "Dropout",
        "Enrolled",
        "Graduate",
    }


def test_data_path():

    config = load_config()

    assert (
        config.data_paths.data_path
        == "data/raw/student_academic_data/data.csv"
    )