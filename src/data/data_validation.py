from pathlib import Path

import pandas as pd

from utils.logger import get_logger


logger = get_logger(__name__)


class DataValidation:
    """
    Validate the input dataset before preprocessing.

    Checks:
        1. Dataset path exists
        2. Dataset is not empty
        3. Expected columns exist
        4. Unexpected columns are detected
        5. Missing values
        6. Duplicate rows
        7. Target column exists
        8. Target classes are valid
        9. Numerical columns contain numeric data
        10. Binary columns contain only 0/1
    """

    def __init__(self, config) -> None:

        self.config = config

        # -------------------------
        # Paths
        # -------------------------

        self.data_path = Path(
            config.data_paths.data_path
        )

        # -------------------------
        # Target
        # -------------------------

        self.target_col = config.target.name
        self.target_classes = set(
            config.target.classes
        )

        # -------------------------
        # Features
        # -------------------------

        self.numerical_features = set(
            config.features.raw.numerical
        )

        self.categorical_features = set(
            config.features.raw.categorical
        )

        self.binary_features = set(
            config.features.raw.binary
        )

        self.expected_features = (
            self.numerical_features
            | self.categorical_features
            | self.binary_features
        )

    # =====================================================
    # PATH VALIDATION
    # =====================================================

    def validate_path(self) -> None:

        logger.info(
            "Checking dataset path: %s",
            self.data_path,
        )

        if not self.data_path.exists():

            logger.error(
                "Dataset not found: %s",
                self.data_path,
            )

            raise FileNotFoundError(
                f"Dataset not found: {self.data_path}"
            )

        if not self.data_path.is_file():

            logger.error(
                "Dataset path is not a file: %s",
                self.data_path,
            )

            raise ValueError(
                f"Dataset path is not a file: "
                f"{self.data_path}"
            )

        logger.info(
            "Dataset path validation passed."
        )

    # =====================================================
    # COLUMN VALIDATION
    # =====================================================

    def validate_columns(self, df: pd.DataFrame) -> None:

        actual_columns = set(df.columns)

        missing_columns = (
            self.expected_features
            - actual_columns
        )

        if self.target_col not in actual_columns:

            missing_columns.add(
                self.target_col
            )

        unexpected_columns = (
            actual_columns
            - self.expected_features
            - {self.target_col}
        )

        if missing_columns:

            logger.error(
                "Missing columns: %s",
                sorted(missing_columns),
            )

            raise ValueError(
                f"Missing columns: "
                f"{sorted(missing_columns)}"
            )

        if unexpected_columns:

            logger.warning(
                "Unexpected columns detected: %s",
                sorted(unexpected_columns),
            )

        logger.info(
            "Column validation passed."
        )

    # =====================================================
    # MISSING VALUE VALIDATION
    # =====================================================

    def validate_missing_values(
        self,
        df: pd.DataFrame,
    ) -> dict:

        missing_values = {
            column: int(count)
            for column, count in (
                df.isnull().sum().items()
            )
            if count > 0
        }

        if missing_values:

            logger.warning(
                "Missing values detected: %s",
                missing_values,
            )

        else:

            logger.info(
                "No missing values detected."
            )

        return missing_values

    # =====================================================
    # DUPLICATE VALIDATION
    # =====================================================

    def validate_duplicates(
        self,
        df: pd.DataFrame,
    ) -> int:

        duplicate_count = int(
            df.duplicated().sum()
        )

        if duplicate_count > 0:

            logger.warning(
                "Duplicate rows detected: %d",
                duplicate_count,
            )

        else:

            logger.info(
                "No duplicate rows detected."
            )

        return duplicate_count

    # =====================================================
    # TARGET VALIDATION
    # =====================================================

    def validate_target(
        self,
        df: pd.DataFrame,
    ) -> list:

        actual_classes = set(
            df[self.target_col].unique()
        )

        logger.debug(
            "Expected target classes: %s",
            self.target_classes,
        )

        logger.debug(
            "Actual target classes: %s",
            actual_classes,
        )

        if actual_classes != self.target_classes:

            logger.error(
                "Target classes do not match."
            )

            raise ValueError(
                f"Target classes mismatch. "
                f"Expected: {self.target_classes}, "
                f"Got: {actual_classes}"
            )

        logger.info(
            "Target validation passed."
        )

        return sorted(actual_classes)

    # =====================================================
    # NUMERICAL FEATURE VALIDATION
    # =====================================================

    def validate_numerical_features(
        self,
        df: pd.DataFrame,
    ) -> None:

        invalid_columns = [
            column
            for column in self.numerical_features
            if not pd.api.types.is_numeric_dtype(
                df[column]
            )
        ]

        if invalid_columns:

            logger.error(
                "Invalid numerical columns: %s",
                invalid_columns,
            )

            raise TypeError(
                f"Expected numerical columns but "
                f"found non-numerical data in: "
                f"{invalid_columns}"
            )

        logger.info(
            "Numerical feature validation passed."
        )

    # =====================================================
    # BINARY FEATURE VALIDATION
    # =====================================================

    def validate_binary_features(
        self,
        df: pd.DataFrame,
    ) -> None:

        invalid_values = {}

        for column in self.binary_features:

            values = set(
                df[column].dropna().unique()
            )

            if not values.issubset({0, 1}):

                invalid_values[column] = sorted(
                    values
                )

        if invalid_values:

            logger.error(
                "Invalid binary feature values: %s",
                invalid_values,
            )

            raise ValueError(
                "Binary features must contain "
                f"only 0 and 1: {invalid_values}"
            )

        logger.info(
            "Binary feature validation passed."
        )

    # =====================================================
    # MAIN VALIDATION
    # =====================================================

    def validate(
        self,
        df: pd.DataFrame,
    ) -> dict:

        logger.info(
            "Data validation started."
        )

        # -------------------------
        # Basic validation
        # -------------------------

        if not isinstance(df, pd.DataFrame):

            raise TypeError(
                "Input data must be a pandas DataFrame."
            )

        if df.empty:

            logger.error(
                "Dataset is empty."
            )

            raise ValueError(
                "Dataset is empty."
            )

        # -------------------------
        # Dataset shape
        # -------------------------

        rows, columns = df.shape

        logger.info(
            "Dataset shape: rows=%d, columns=%d",
            rows,
            columns,
        )

        # -------------------------
        # Individual checks
        # -------------------------

        self.validate_columns(df)

        missing_values = (
            self.validate_missing_values(df)
        )

        duplicate_count = (
            self.validate_duplicates(df)
        )

        target_classes = (
            self.validate_target(df)
        )

        self.validate_numerical_features(df)

        self.validate_binary_features(df)

        # -------------------------
        # Validation report
        # -------------------------

        result = {
            "status": "passed",
            "data_shape": {
                "rows": rows,
                "columns": columns,
            },
            "missing_values": missing_values,
            "duplicate_values": duplicate_count,
            "target": {
                "column": self.target_col,
                "classes": target_classes,
            },
        }

        logger.info(
            "Data validation completed successfully."
        )

        return result