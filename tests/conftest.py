import pytest
import pandas as pd

from utils.load_config import load_config


@pytest.fixture
def config():
    return load_config()

@pytest.fixture
def sample_dataframe():

    return pd.DataFrame(
        {
            # -----------------------------
            # Numerical
            # -----------------------------

            "Application order": [1, 2, 3],
            "Previous qualification (grade)": [
                120.0,
                130.0,
                140.0,
            ],
            "Admission grade": [
                125.0,
                135.0,
                145.0,
            ],
            "Age at enrollment": [
                20,
                21,
                22,
            ],

            "Curricular units 1st sem (credited)": [
                0, 1, 2
            ],
            "Curricular units 1st sem (enrolled)": [
                10, 12, 15
            ],
            "Curricular units 1st sem (evaluations)": [
                8, 10, 13
            ],
            "Curricular units 1st sem (approved)": [
                7, 9, 11
            ],
            "Curricular units 1st sem (grade)": [
                12.5, 14.0, 15.5
            ],
            "Curricular units 1st sem (without evaluations)": [
                2, 2, 2
            ],

            "Curricular units 2nd sem (credited)": [
                0, 1, 2
            ],
            "Curricular units 2nd sem (enrolled)": [
                10, 12, 15
            ],
            "Curricular units 2nd sem (evaluations)": [
                9, 11, 14
            ],
            "Curricular units 2nd sem (approved)": [
                8, 10, 12
            ],
            "Curricular units 2nd sem (grade)": [
                13.0, 14.5, 16.0
            ],
            "Curricular units 2nd sem (without evaluations)": [
                1, 1, 1
            ],

            "Unemployment rate": [
                10.8, 11.2, 12.0
            ],
            "Inflation rate": [
                1.5, 2.0, 2.5
            ],
            "GDP": [
                1.74, 1.80, 1.90
            ],

            # -----------------------------
            # Engineered numerical features
            # -----------------------------

            "sem1_approval_rate": [
                0.70, 0.75, 0.73
            ],
            "sem1_evaluation_rate": [
                0.80, 0.83, 0.87
            ],
            "sem1_without_eval_rate": [
                0.20, 0.17, 0.13
            ],
            "sem2_approval_rate": [
                0.80, 0.83, 0.80
            ],
            "sem2_evaluation_rate": [
                0.90, 0.92, 0.93
            ],
            "sem2_without_eval_rate": [
                0.10, 0.08, 0.07
            ],
            "total_enrolled": [
                20, 24, 30
            ],
            "total_approved": [
                15, 19, 23
            ],
            "total_evaluations": [
                17, 21, 27
            ],
            "overall_approval_rate": [
                0.75, 0.79, 0.77
            ],
            "grade_change": [
                0.5, 0.5, 0.5
            ],
            "approval_change": [
                1, 1, 1
            ],
            "enrollment_change": [
                0, 0, 0
            ],

            # -----------------------------
            # Categorical
            # -----------------------------

            "Marital status": [
                1, 1, 2
            ],
            "Application mode": [
                17, 18, 39
            ],
            "Course": [
                171, 9254, 9070
            ],
            "Previous qualification": [
                1, 1, 2
            ],
            "Nacionality": [
                1, 1, 1
            ],
            "Mother's qualification": [
                13, 1, 22
            ],
            "Father's qualification": [
                10, 1, 14
            ],
            "Mother's occupation": [
                6, 4, 10
            ],
            "Father's occupation": [
                10, 8, 5
            ],

            # -----------------------------
            # Binary
            # -----------------------------

            "Daytime/evening attendance": [
                1, 1, 0
            ],
            "Displaced": [
                0, 1, 0
            ],
            "Educational special needs": [
                0, 0, 0
            ],
            "Debtor": [
                0, 0, 1
            ],
            "Tuition fees up to date": [
                1, 1, 0
            ],
            "Gender": [
                1, 0, 1
            ],
            "Scholarship holder": [
                0, 1, 0
            ],
            "International": [
                0, 0, 1
            ],
        }
    )