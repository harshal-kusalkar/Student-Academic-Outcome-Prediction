from pathlib import Path
from typing import Any

from pydantic import BaseModel


# -----------------------------
# Preprocessing
# -----------------------------

class PreprocessingConfig(BaseModel):
    drop_columns: list[str]
    target_col: str
    target: list[str]


# -----------------------------
# Data Split
# -----------------------------

class DataSplitConfig(BaseModel):
    test_size: float
    random_state: int


# -----------------------------
# Paths
# -----------------------------

class DataPathConfig(BaseModel):
    data_dir_path: Path


class ArtifactPathConfig(BaseModel):
    model_path: Path
    encoder_path: Path
    feature_names_path: Path
    data_validation_path: Path
    eval_result_path: Path
    classification_report_path: Path


class ProcessedDataConfig(BaseModel):
    X_train_path: Path
    X_test_path: Path
    y_train_path: Path
    y_test_path: Path


# -----------------------------
# Model
# -----------------------------

class ModelConfig(BaseModel):
    name: str
    version: str


class TrainingConfig(BaseModel):
    random_state: int
    n_jobs: int


class ModelParamsConfig(BaseModel):
    objective: str = "multi:softprob"
    eval_metric: str = "mlogloss"
    booster: str = "gbtree"
    verbosity: int = 0

    n_estimators: int
    learning_rate: float
    max_depth: int
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    gamma: float                       # Formerly min_split_gain
    reg_alpha: float
    reg_lambda: float


class ExperimentConfig(BaseModel):
    model_name: str = "XGBoost"
    best_trial: int
    cv_metric: str
    best_cv_score: float


# -----------------------------
# Root Config
# -----------------------------

class Config(BaseModel):
    preprocessing: PreprocessingConfig
    data_split: DataSplitConfig
    data_paths: DataPathConfig

    artifacts: ArtifactPathConfig
    processed_data: ProcessedDataConfig

    model: ModelConfig
    training: TrainingConfig
    model_params: ModelParamsConfig
    experiment: ExperimentConfig