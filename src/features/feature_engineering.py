import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from utils.logger import get_logger


logger = get_logger(__name__)


class StudentFeatureEngineer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Feature engineering for student academic data.

    Creates:
        - Semester approval rates
        - Semester evaluation rates
        - Overall academic performance
        - Semester-to-semester progression
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        logger.info("Feature engineering started.")

        X = X.copy()

        # ==================================================
        # Semester 1 features
        # ==================================================

        sem1_enrolled = (
            X["Curricular units 1st sem (enrolled)"]
            .replace(0, np.nan)
        )

        X["sem1_approval_rate"] = (
            X["Curricular units 1st sem (approved)"]
            / sem1_enrolled
        )

        X["sem1_evaluation_rate"] = (
            X["Curricular units 1st sem (evaluations)"]
            / sem1_enrolled
        )

        X["sem1_without_eval_rate"] = (
            X["Curricular units 1st sem (without evaluations)"]
            / sem1_enrolled
        )

        # ==================================================
        # Semester 2 features
        # ==================================================

        sem2_enrolled = (
            X["Curricular units 2nd sem (enrolled)"]
            .replace(0, np.nan)
        )

        X["sem2_approval_rate"] = (
            X["Curricular units 2nd sem (approved)"]
            / sem2_enrolled
        )

        X["sem2_evaluation_rate"] = (
            X["Curricular units 2nd sem (evaluations)"]
            / sem2_enrolled
        )

        X["sem2_without_eval_rate"] = (
            X["Curricular units 2nd sem (without evaluations)"]
            / sem2_enrolled
        )

        # ==================================================
        # Overall academic features
        # ==================================================

        X["total_enrolled"] = (
            X["Curricular units 1st sem (enrolled)"]
            + X["Curricular units 2nd sem (enrolled)"]
        )

        X["total_approved"] = (
            X["Curricular units 1st sem (approved)"]
            + X["Curricular units 2nd sem (approved)"]
        )

        X["total_evaluations"] = (
            X["Curricular units 1st sem (evaluations)"]
            + X["Curricular units 2nd sem (evaluations)"]
        )

        X["overall_approval_rate"] = (
            X["total_approved"]
            / X["total_enrolled"].replace(0, np.nan)
        )

        # ==================================================
        # Academic progression
        # ==================================================

        X["grade_change"] = (
            X["Curricular units 2nd sem (grade)"]
            - X["Curricular units 1st sem (grade)"]
        )

        X["approval_change"] = (
            X["Curricular units 2nd sem (approved)"]
            - X["Curricular units 1st sem (approved)"]
        )

        X["enrollment_change"] = (
            X["Curricular units 2nd sem (enrolled)"]
            - X["Curricular units 1st sem (enrolled)"]
        )

        # ==================================================
        # Handle invalid mathematical results
        # ==================================================

        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        logger.info(
            "Feature engineering completed. Shape: %s",
            X.shape,
        )

        return X