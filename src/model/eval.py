import wandb

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from utils.logger import get_logger

logger = get_logger(__name__)


def evaluate(model, X_test, y_test, config):
    """
    Evaluate the trained model on the test set.
    """

    logger.info("Starting model evaluation...")

    try:
        predictions = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(
                y_test,
                predictions,
            ),
            "precision_macro": precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "recall_macro": recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "f1_macro": f1_score(
                y_test,
                predictions,
                average="macro",
            ),
            "f1_weighted": f1_score(
                y_test,
                predictions,
                average="weighted",
            ),
        }

        report = classification_report(
            y_test,
            predictions,
            target_names=config.preprocessing.target,
            zero_division=0,
            output_dict=True,
        )

        with wandb.init(
            project="student-drop-enroll-grad-preds",
            name=f"{config.model.name}-Evaluation",
            group=config.model.name,
            tags=["Evaluation"],
        ) as run:

            wandb.log(metrics)

            wandb.summary["model"] = config.model.name
            wandb.summary["version"] = config.model.version

            wandb.log({
                "classification_report": wandb.Table(
                    dataframe=__import__("pandas").DataFrame(report).transpose()
                )
            })

        logger.info(
            "Evaluation completed successfully. F1 Macro: %.4f",
            metrics["f1_macro"],
        )

        return metrics, report

    except Exception:
        logger.exception("Model evaluation failed.")
        raise