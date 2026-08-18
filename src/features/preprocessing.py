from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

from utils.logger import get_logger


logger = get_logger(__name__)


def create_imputer(config):

    return SimpleImputer(
        strategy=config.strategy
    )


def create_scaler(config):

    scaler_type = config.type

    if scaler_type == "standard":
        return StandardScaler()

    if scaler_type == "minmax":
        return MinMaxScaler()

    if scaler_type == "robust":
        return RobustScaler()

    if scaler_type == "none":
        return "passthrough"

    raise ValueError(
        f"Unsupported scaler: {scaler_type}"
    )


def create_encoder(config):

    if config.type == "onehot":

        return OneHotEncoder(
            handle_unknown=config.handle_unknown,
            sparse_output=False,
        )

    if config.type == "ordinal":

        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

    raise ValueError(
        f"Unsupported encoder: {config.type}"
    )


def create_preprocessor(config):

    logger.info(
        "Creating preprocessing pipeline."
    )

    # -----------------------------------------
    # Numerical pipeline
    # -----------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                create_imputer(
                    config.preprocessing.numerical.imputer
                ),
            ),
            (
                "scaler",
                create_scaler(
                    config.preprocessing.numerical.scaler
                ),
            ),
        ]
    )

    # -----------------------------------------
    # Categorical pipeline
    # -----------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                create_imputer(
                    config.preprocessing.categorical.imputer
                ),
            ),
            (
                "encoder",
                create_encoder(
                    config.preprocessing.categorical.encoder
                ),
            ),
        ]
    )

    # -----------------------------------------
    # Binary pipeline
    # -----------------------------------------

    binary_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                create_imputer(
                    config.preprocessing.binary.imputer
                ),
            ),
        ]
    )

    # -----------------------------------------
    # ColumnTransformer
    # -----------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                config.features.raw.numerical,
            ),
            (
                "categorical",
                categorical_pipeline,
                config.features.raw.categorical,
            ),
            (
                "binary",
                binary_pipeline,
                config.features.raw.binary,
            ),
        ],
        remainder="drop",
    )

    logger.info(
        "ColumnTransformer created successfully."
    )

    return preprocessor