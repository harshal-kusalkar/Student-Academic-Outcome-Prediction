import numpy as np
import pandas as pd

from src.features.feature_engineering import (
    StudentFeatureEngineer,
)


def test_feature_engineering_creates_expected_features():

    df = pd.DataFrame(
        {
            "Curricular units 1st sem (enrolled)": [10, 20],
            "Curricular units 1st sem (approved)": [8, 15],
            "Curricular units 1st sem (evaluations)": [9, 18],
            "Curricular units 1st sem (without evaluations)": [1, 2],

            "Curricular units 2nd sem (enrolled)": [10, 20],
            "Curricular units 2nd sem (approved)": [9, 10],
            "Curricular units 2nd sem (evaluations)": [10, 15],
            "Curricular units 2nd sem (without evaluations)": [0, 5],

            "Curricular units 1st sem (grade)": [12.0, 14.0],
            "Curricular units 2nd sem (grade)": [14.0, 13.0],
        }
    )

    transformer = StudentFeatureEngineer()

    result = transformer.fit_transform(df)

    expected_columns = [
        "sem1_approval_rate",
        "sem1_evaluation_rate",
        "sem1_without_eval_rate",
        "sem2_approval_rate",
        "sem2_evaluation_rate",
        "sem2_without_eval_rate",
        "total_enrolled",
        "total_approved",
        "total_evaluations",
        "overall_approval_rate",
        "grade_change",
        "approval_change",
        "enrollment_change",
    ]

    for column in expected_columns:
        assert column in result.columns


def test_approval_rate():

    df = pd.DataFrame(
        {
            "Curricular units 1st sem (enrolled)": [10],
            "Curricular units 1st sem (approved)": [8],
            "Curricular units 1st sem (evaluations)": [9],
            "Curricular units 1st sem (without evaluations)": [1],

            "Curricular units 2nd sem (enrolled)": [10],
            "Curricular units 2nd sem (approved)": [9],
            "Curricular units 2nd sem (evaluations)": [10],
            "Curricular units 2nd sem (without evaluations)": [0],

            "Curricular units 1st sem (grade)": [12.0],
            "Curricular units 2nd sem (grade)": [14.0],
        }
    )

    transformer = StudentFeatureEngineer()

    result = transformer.fit_transform(df)

    assert result["sem1_approval_rate"].iloc[0] == 0.8
    assert result["sem2_approval_rate"].iloc[0] == 0.9
    assert result["grade_change"].iloc[0] == 2.0


def test_zero_enrollment_produces_nan():

    df = pd.DataFrame(
        {
            "Curricular units 1st sem (enrolled)": [0],
            "Curricular units 1st sem (approved)": [0],
            "Curricular units 1st sem (evaluations)": [0],
            "Curricular units 1st sem (without evaluations)": [0],

            "Curricular units 2nd sem (enrolled)": [10],
            "Curricular units 2nd sem (approved)": [8],
            "Curricular units 2nd sem (evaluations)": [9],
            "Curricular units 2nd sem (without evaluations)": [1],

            "Curricular units 1st sem (grade)": [0.0],
            "Curricular units 2nd sem (grade)": [12.0],
        }
    )

    transformer = StudentFeatureEngineer()

    result = transformer.fit_transform(df)

    assert np.isnan(
        result["sem1_approval_rate"].iloc[0]
    )


def test_original_dataframe_is_not_modified():

    df = pd.DataFrame(
        {
            "Curricular units 1st sem (enrolled)": [10],
            "Curricular units 1st sem (approved)": [8],
            "Curricular units 1st sem (evaluations)": [9],
            "Curricular units 1st sem (without evaluations)": [1],

            "Curricular units 2nd sem (enrolled)": [10],
            "Curricular units 2nd sem (approved)": [9],
            "Curricular units 2nd sem (evaluations)": [10],
            "Curricular units 2nd sem (without evaluations)": [0],

            "Curricular units 1st sem (grade)": [12.0],
            "Curricular units 2nd sem (grade)": [14.0],
        }
    )

    original_columns = df.columns.tolist()

    transformer = StudentFeatureEngineer()

    transformer.fit_transform(df)

    assert df.columns.tolist() == original_columns