import numpy as np

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)


def evaluate_model(
    pipeline,
    X_train,
    y_train,
    cv_config,
):

    cv = StratifiedKFold(
        n_splits=cv_config.n_splits,
        shuffle=True,
        random_state=cv_config.random_state,
    )

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "f1_macro": "f1_macro",
            "balanced_accuracy": "balanced_accuracy",
        },
        n_jobs=-1,
        return_train_score=False,
    )

    return {
        "f1_macro_mean": float(
            np.mean(
                scores["test_f1_macro"]
            )
        ),

        "f1_macro_std": float(
            np.std(
                scores["test_f1_macro"]
            )
        ),

        "balanced_accuracy_mean": float(
            np.mean(
                scores["test_balanced_accuracy"]
            )
        ),

        "balanced_accuracy_std": float(
            np.std(
                scores["test_balanced_accuracy"]
            )
        ),
    }