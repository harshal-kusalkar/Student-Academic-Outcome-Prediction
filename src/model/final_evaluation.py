import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from utils.logger import get_logger


logger = get_logger(__name__)


def evaluate_final_model(
    model,
    X_test: pd.DataFrame,
    y_test,
    class_names,
):
    """
    Evaluate the final trained model on the
    untouched test dataset.
    """

    logger.info(
        "Starting final model evaluation."
    )

    # -----------------------------
    # Predictions
    # -----------------------------

    y_pred = model.predict(X_test)

    # -----------------------------
    # Metrics
    # -----------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            y_pred,
        )
    )

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # -----------------------------
    # Classification report
    # -----------------------------

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # -----------------------------
    # Confusion matrix
    # -----------------------------

    matrix = confusion_matrix(
        y_test,
        y_pred,
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
    }

    logger.info(
        "Final Macro F1: %.4f",
        macro_f1,
    )

    logger.info(
        "Final Balanced Accuracy: %.4f",
        balanced_accuracy,
    )

    return {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "predictions": y_pred,
    }