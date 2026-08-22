import mlflow

from utils.load_config import load_config


def test_load_model_from_mlflow():

    config = load_config()

    # Local MLflow server
    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    model_uri = (
        "models:/student_dropout_model@champion"
    )

    print("Tracking URI:")
    print(mlflow.get_tracking_uri())

    print("Model URI:")
    print(model_uri)

    model = mlflow.pyfunc.load_model(
        model_uri
    )

    assert model is not None

    print(
        "Model loaded successfully:",
        type(model),
    )