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
)

from utils.io import load_csv, load_numpy
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

    logger.info(
        "======== TRAINING PIPELINE COMPLETED ========"
    )

    return best_model_name,best_pipeline, results

if __name__ == "__main__":
    model, pipeline, result = run()
    print(type(model))
    print(type(pipeline))
    print(result)

