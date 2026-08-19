from typing import Literal
from pathlib import Path

from pydantic import BaseModel, Field


class DataPathsConfig(BaseModel):
    data_path: Path
    X_train_path: Path 
    X_test_path: Path 
    y_train_path: Path 
    y_test_path: Path

class ArtifactsConfig(BaseModel):
    data_validation_path: Path
    model_comparison_path: Path 
    encoder_path: Path
    model_path: Path

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
    drop_columns: list[str] = Field(
        default_factory=list
    )


class RawFeaturesConfig(BaseModel):
    numerical: list[str]
    categorical: list[str]
    binary: list[str]


class EngineeredFeaturesConfig(BaseModel):
    numerical: list[str]


class FeaturesConfig(BaseModel):
    raw: RawFeaturesConfig
    engineered: EngineeredFeaturesConfig


class DataSplitConfig(BaseModel):
    test_size: float
    random_state: int
    stratify: bool

class CVConfig(BaseModel):
    n_splits: int = Field(
        default=5,
        ge=2,
        )
    random_state: int = 42

class ModelSelectionConfig(BaseModel):
    cv: CVConfig

class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str

class TuningConfig(BaseModel):
    n_trials: int = Field(
        default=50,
        ge=1,
    )


class Config(BaseModel):
    data_paths: DataPathsConfig
    artifacts: ArtifactsConfig
    target: TargetConfig
    features: FeaturesConfig
    preprocessing: PreprocessingConfig
    data_split: DataSplitConfig
    model_selection: ModelSelectionConfig
    mlflow: MLflowConfig
    tuning: TuningConfig