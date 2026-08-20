from src.model.model_factory import (
    get_candidate_models,
)

from src.model.pipeline_builder import (
    build_pipeline,
)

from src.model.model_selection import (
    select_best_model,
)

from src.tracking.mlflow_tracker import (
    setup_mlflow,
    log_optuna_best_trial,
    register_final_model,
)

from src.tuning.optuna_tuner import (
    tune_random_forest,
)

from src.model.model_training import (
    train_final_model,
)

from utils.io import (
    save_csv,
    save_model,
    load_csv,
    load_numpy
)
from utils.logger import get_logger
from utils.load_config import load_config



logger = get_logger(__name__)


def run():

    logger.info(
        "======== TRAINING PIPELINE STARTED ========"
    )

    config = load_config()

    setup_mlflow(config=config)

    # -----------------------------------------
    # Load training data
    # -----------------------------------------

    X_train = load_csv(path=config.data_paths.X_train_path)
    y_train = load_numpy(path=config.data_paths.y_train_path)

    # -----------------------------------------
    # Create candidate pipelines
    # -----------------------------------------

    models = get_candidate_models()

    pipelines = {
        name: build_pipeline(
            model=model,
            config=config,
        )
        for name, model in models.items()
    }

    # -----------------------------------------
    # Model selection
    # -----------------------------------------

    best_model_name ,best_pipeline, results = (
        select_best_model(
            pipelines=pipelines,
            X_train=X_train,
            y_train=y_train,
            cv_config=config.model_selection.cv,
        )
    )

    logger.info(
        "Best model: %s",
        best_model_name,
    )

    # -----------------------------------------
    # Hyperparameter Tuning
    # -----------------------------------------

    if best_model_name == "random_forest":

        study = tune_random_forest(
            pipeline=best_pipeline,
            X_train=X_train,
            y_train=y_train,
            config=config,
        )

        log_optuna_best_trial(study)

        logger.info(
            "Optuna best score: %.4f",
            study.best_value,
        )

        logger.info(
            "Optuna best parameters: %s",
            study.best_params,
        )

    # -----------------------------------------
    # Final Model Training
    # -----------------------------------------

    final_pipeline = train_final_model(
        pipeline=best_pipeline,
        X_train=X_train,
        y_train=y_train,
        best_params=study.best_params,
    )

    # -----------------------------------------
    # Save model and result
    # -----------------------------------------

    save_model(
        model=final_pipeline,
        path=config.artifacts.model_path,
    )

    save_csv(
        data=results,
        path=config.artifacts.model_comparison_path
    )

    # -----------------------------------------
    # Resister Model To Mlflow
    # -----------------------------------------

    register_final_model(
        model=final_pipeline,
        model_name=config.mlflow.registered_model_name,
        X_train=X_train,
    )

    logger.info(
        "======== TRAINING PIPELINE COMPLETED ========"
    )

if __name__ == "__main__":
    run()


