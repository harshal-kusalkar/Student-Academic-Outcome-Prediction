from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def get_candidate_models():

    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(
                random_state=42,
            )
        ),

        "xgboost": XGBClassifier(
            n_estimators=300,
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=-1,
        ),

        "lightgbm": LGBMClassifier(
            n_estimators=300,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        ),

        "catboost": CatBoostClassifier(
            iterations=300,
            verbose=False,
            random_seed=42,
        ),
    }