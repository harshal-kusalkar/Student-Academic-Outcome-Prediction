from typing import Literal

from pydantic import BaseModel


class DataPathsConfig(BaseModel):
    data_path: str


class TargetConfig(BaseModel):
    name: str
    classes: list[str]


class ImputerConfig(BaseModel):
    strategy: Literal[
        "mean",
        "median",
        "most_frequent",
        "constant",
    ]


class ScalerConfig(BaseModel):
    type: Literal[
        "standard",
        "minmax",
        "robust",
        "none",
    ]


class EncoderConfig(BaseModel):
    type: Literal[
        "onehot",
        "ordinal",
    ]

    handle_unknown: Literal[
        "ignore",
        "error",
    ] = "ignore"


class NumericalPreprocessingConfig(BaseModel):
    imputer: ImputerConfig
    scaler: ScalerConfig


class CategoricalPreprocessingConfig(BaseModel):
    imputer: ImputerConfig
    encoder: EncoderConfig


class BinaryPreprocessingConfig(BaseModel):
    imputer: ImputerConfig


class PreprocessingConfig(BaseModel):
    numerical: NumericalPreprocessingConfig
    categorical: CategoricalPreprocessingConfig
    binary: BinaryPreprocessingConfig


class FeaturesConfig(BaseModel):
    numerical: list[str]
    categorical: list[str]
    binary: list[str]


class DataSplitConfig(BaseModel):
    test_size: float
    random_state: int
    stratify: bool


class Config(BaseModel):
    DATA_PATHS: DataPathsConfig
    TARGET: TargetConfig
    FEATURES: FeaturesConfig
    PREPROCESSING: PreprocessingConfig
    DATA_SPLIT: DataSplitConfig