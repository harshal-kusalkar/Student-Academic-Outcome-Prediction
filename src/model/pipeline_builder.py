from sklearn.pipeline import Pipeline

from src.features.feature_engineering import (
    StudentFeatureEngineer,
)

from src.features.preprocessing import (
    create_preprocessor,
)


def build_pipeline(
    model,
    config,
):

    preprocessor = create_preprocessor(
        config
    )

    return Pipeline(
        steps=[
            (
                "feature_engineering",
                StudentFeatureEngineer(),
            ),
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )