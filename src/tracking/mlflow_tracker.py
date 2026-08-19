import mlflow

from utils.logger import get_logger


logger = get_logger(__name__)


def setup_mlflow(config):
    """
    Configure MLflow tracking.
    """

    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
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