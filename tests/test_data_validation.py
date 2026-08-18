import pandas as pd
import pytest
from types import SimpleNamespace

from src.data.data_validation import DataValidation


@pytest.fixture
def config():
    """
    Minimal configuration required by DataValidation.
    """

    return SimpleNamespace(
        DATA_PATHS=SimpleNamespace(
            data_path="data/raw/student_academic_data/data.csv"
        ),

        TARGET=SimpleNamespace(
            name="Target",
            classes=[
                "Dropout",
                "Enrolled",
                "Graduate",
            ],
        ),

        FEATURES=SimpleNamespace(
            numerical=[
                "Age",
                "Admission grade",
            ],

            categorical=[
                "Course",
            ],

            binary=[
                "Gender",
                "Scholarship holder",
            ],
        ),
    )


@pytest.fixture
def validator(config):
    return DataValidation(config)


@pytest.fixture
def valid_dataframe():
    """
    Small valid dataset used for testing.
    """

    return pd.DataFrame(
        {
            "Age": [20, 21, 22],
            "Admission grade": [120.0, 130.0, 140.0],
            "Course": [1, 2, 1],
            "Gender": [0, 1, 0],
            "Scholarship holder": [1, 0, 1],
            "Target": [
                "Dropout",
                "Enrolled",
                "Graduate",
            ],
        }
    )

def test_validate_valid_data(
    validator,
    valid_dataframe,
):

    result = validator.validate(
        valid_dataframe
    )

    assert result["status"] == "passed"

    assert result["data_shape"] == {
        "rows": 3,
        "columns": 6,
    }

    assert result["duplicate_values"] == 0

    assert result["missing_values"] == {}

    assert set(
        result["target"]["classes"]
    ) == {
        "Dropout",
        "Enrolled",
        "Graduate",
    }

def test_empty_dataframe(
    validator,
):

    df = pd.DataFrame()

    with pytest.raises(ValueError, match="Dataset is empty"):
        validator.validate(df)

def test_missing_column(
    validator,
    valid_dataframe,
):

    df = valid_dataframe.drop(
        columns=["Age"]
    )

    with pytest.raises(
        ValueError,
        match="Missing columns",
    ):
        validator.validate(df)

def test_missing_values(
    validator,
    valid_dataframe,
):

    df = valid_dataframe.copy()

    df.loc[0, "Age"] = None

    result = validator.validate(df)

    assert (
        result["missing_values"]["Age"]
        == 1
    )

def test_duplicate_rows(
    validator,
    valid_dataframe,
):

    df = pd.concat(
        [
            valid_dataframe,
            valid_dataframe.iloc[[0]],
        ],
        ignore_index=True,
    )

    result = validator.validate(df)

    assert result["duplicate_values"] == 1

def test_invalid_target_class(
    validator,
    valid_dataframe,
):

    df = valid_dataframe.copy()

    df.loc[0, "Target"] = "Unknown"

    with pytest.raises(
        ValueError,
        match="Target classes mismatch",
    ):
        validator.validate(df)

def test_invalid_numerical_feature(
    validator,
    valid_dataframe,
):

    df = valid_dataframe.copy()

    df["Age"] = [
        "twenty",
        "twenty-one",
        "twenty-two",
    ]

    with pytest.raises(
        TypeError,
        match="Expected numerical columns",
    ):
        validator.validate(df)

def test_invalid_binary_feature(
    validator,
    valid_dataframe,
):

    df = valid_dataframe.copy()

    df.loc[0, "Gender"] = 2

    with pytest.raises(
        ValueError,
        match="Binary features must contain",
    ):
        validator.validate(df)

def test_invalid_input_type(
    validator,
):

    with pytest.raises(
        TypeError,
        match="Input data must be a pandas DataFrame",
    ):
        validator.validate(
            {"Age": [20, 21]}
        )

def test_validate_path(
    validator,
    tmp_path,
):

    data_file = tmp_path / "data.csv"

    data_file.write_text(
        "Age,Target\n20,Dropout\n"
    )

    validator.data_path = data_file

    validator.validate_path()
