import pandas as pd

from fastapi import FastAPI

from api.mappings import COLUMN_MAPPING
from api.schemas import StudentPredictionRequest
from src.inference.predictor import Predictor

from utils.load_config import load_config
from utils.logger import get_logger


logger = get_logger(__name__)


app = FastAPI(
    title="Student Dropout Prediction API",
    description=(
        "ML inference API for predicting "
        "student academic outcomes."
    ),
    version="1.0.0",
)

config = load_config()

predictor = Predictor(
    tracking_uri=config.mlflow.tracking_uri,
    model_name=(
        config.mlflow.registered_model_name
    ),
    alias=config.mlflow.model_alias,
    encoder_path=config.artifacts.encoder_path
)

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": config.mlflow.registered_model_name,
        "alias": config.mlflow.model_alias,
    }


@app.post("/predict")
def predict(
    request: StudentPredictionRequest,
):

    # Pydantic model → dictionary
    data = request.model_dump()

    # API names → dataset names
    data = {
        COLUMN_MAPPING[key]: value
        for key, value in data.items()
    }

    # Dictionary → DataFrame
    input_data = pd.DataFrame(
        [data]
    )

    prediction = predictor.predict(
        input_data
    )

    return {
        "prediction": str(prediction[0])
    }