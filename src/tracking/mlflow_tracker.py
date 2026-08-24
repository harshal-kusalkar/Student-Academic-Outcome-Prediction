import os 

import mlflow
import mlflow.sklearn
import dagshub

from utils.logger import get_logger


logger = get_logger(__name__)


def setup_mlflow(config):
    """
    Configure MLflow tracking.
    """

    dagshub.init(repo_owner='harshal-kusalkar', repo_name='mlflow-server', mlflow=True)

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        config.mlflow.tracking_uri,
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        config.mlflow.experiment_name
    )

    logger.info(
        "MLflow configured."
    )

    logger.info(
        "Tracking URI: %s",
        config.mlflow.tracking_uri,
    )

    logger.info(
        "Experiment: %s",
        config.mlflow.experiment_name,
    )


def log_model_run(
    model_name,
    model,
    metrics,
):
    """
    Log one model comparison experiment.
    """

    with mlflow.start_run(
        run_name=model_name
    ):

        # -----------------------------
        # Model information
        # -----------------------------

        mlflow.set_tag(
            "model_name",
            model_name,
        )

        mlflow.set_tag(
            "stage",
            "model_comparison",
        )

        # -----------------------------
        # Parameters
        # -----------------------------

        model_params = model.get_params()

        mlflow.log_params(
            model_params
        )

        # -----------------------------
        # Metrics
        # -----------------------------

        mlflow.log_metrics(
            metrics
        )

        logger.info(
            "MLflow run logged: %s",
            model_name,
        )

def log_optuna_best_trial(study):

    with mlflow.start_run(
        run_name="random_forest_optuna_best"
    ):

        mlflow.set_tag(
            "stage",
            "hyperparameter_tuning",
        )

        mlflow.set_tag(
            "model_name",
            "random_forest",
        )

        mlflow.log_params(
            study.best_params
        )

        mlflow.log_metric(
            "best_cv_macro_f1",
            float(study.best_value),
        )


def register_final_model(
    model,
    model_name: str,
    X_train,
):
    """
    Log and register the final sklearn pipeline
    with MLflow.
    """

    trusted_types = [
        "src.features.feature_engineering.StudentFeatureEngineer",
        "numpy.dtype",
    ]

    with mlflow.start_run(
        run_name="final_model"
    ):

        mlflow.set_tag(
            "stage",
            "final_training",
        )

        mlflow.set_tag(
            "model_name",
            model_name,
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train.head(5),
            registered_model_name=model_name,
            skops_trusted_types=trusted_types,
        )

        logger.info(
            "Final model registered."
        )

        logger.info(
            "Model URI: %s",
            model_info.model_uri,
        )

        return model_info