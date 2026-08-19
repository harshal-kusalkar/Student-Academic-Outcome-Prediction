import pandas as pd

from src.model.model_evaluation import evaluate_model
from src.tracking.mlflow_tracker import log_model_run

from utils.logger import get_logger


logger = get_logger(__name__)


def select_best_model(
    pipelines,
    X_train,
    y_train,
    cv_config,
):

    results = []

    for model_name, pipeline in pipelines.items():

        logger.info(
            "Evaluating model: %s",
            model_name,
        )

        metrics = evaluate_model(
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            cv_config=cv_config,
        )

        # -----------------------------
        # MLflow
        # -----------------------------

        model = pipeline.named_steps[
            "model"
        ]

        mlflow_metrics = {
            key: float(value)
            for key, value in metrics.items()
        }

        log_model_run(
            model_name=model_name,
            model=model,
            metrics=mlflow_metrics,
        )

        # -----------------------------
        # Results
        # -----------------------------

        results.append(
            {
                "model": model_name,
                **metrics,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="f1_macro_mean",
        ascending=False,
    ).reset_index(drop=True)

    best_model_name = (
        results_df.iloc[0]["model"]
    )

    best_pipeline = pipelines[
        best_model_name
    ]

    logger.info(
        "Best model: %s",
        best_model_name,
    )

    return (
        best_model_name,
        best_pipeline,
        results_df,
    )