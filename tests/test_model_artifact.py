from utils.io import (
    load_csv,
    load_model,
    load_numpy,
)


def test_saved_model_can_predict(config):

    model = load_model(
        path=config.artifacts.model_path
    )

    X_test = load_csv(
        path=config.data_paths.X_test_path
    )

    y_test = load_numpy(
        path=config.data_paths.y_test_path
    )

    predictions = model.predict(
        X_test
    )

    assert predictions.shape == y_test.shape

    assert len(predictions) > 0