import pandas as pd

from utils.load_config import load_config
from src.features.preprocessing import create_preprocessor


def test_preprocessor_fit_transform(
    config,
    sample_dataframe,
):

    preprocessor = create_preprocessor(
        config
    )

    transformed = preprocessor.fit_transform(
        sample_dataframe
    )

    assert transformed.shape[0] == (
        sample_dataframe.shape[0]
    )

    assert transformed.shape[1] > 0





