import optuna

from sklearn.model_selection import StratifiedKFold, cross_val_score

from utils.logger import get_logger


logger = get_logger(__name__)


def tune_random_forest(
    pipeline,
    X_train,
    y_train,
    config,
):
    """
    Tune Random Forest using Optuna.

    The complete sklearn pipeline is evaluated
    inside each CV fold.
    """

    cv = StratifiedKFold(
        n_splits=config.model_selection.cv.n_splits,
        shuffle=True,
        random_state=config.model_selection.cv.random_state,
    )

    def objective(trial):

        params = {
            "model__n_estimators": trial.suggest_int(
                "n_estimators",
                100,
                500,
            ),

            "model__max_depth": trial.suggest_int(
                "max_depth",
                5,
                30,
            ),

            "model__min_samples_split": trial.suggest_int(
                "min_samples_split",
                2,
                20,
            ),

            "model__min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                1,
                10,
            ),

            "model__max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2", None],
            ),
        }

        pipeline.set_params(**params)

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )

        score = scores.mean()

        logger.info(
            "Trial %d | Macro F1: %.4f",
            trial.number,
            score,
        )

        return score

    study = optuna.create_study(
        direction="maximize",
        study_name="random_forest_tuning",
    )

    study.optimize(
        objective,
        n_trials=config.tuning.n_trials,
    )

    logger.info(
        "Best Macro F1: %.4f",
        study.best_value,
    )

    logger.info(
        "Best parameters: %s",
        study.best_params,
    )

    return study