from contextlib import asynccontextmanager

import pandas as pd

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from src.data.data_cleaning import clean_data
from src.inference.predict import Modelservice
from src.inference.validate import ValidateStudentData
from utils.load_config import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

config = load_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading application services...")

    app.state.model_service = Modelservice()
    app.state.validator = ValidateStudentData()

    logger.info("Application services loaded.")

    yield

    logger.info("Application shutting down.")


def get_model_service(request: Request) -> Modelservice:
    return request.app.state.model_service


def get_validator(request: Request) -> ValidateStudentData:
    return request.app.state.validator


def get_dataframe(file: UploadFile = File(...)) -> pd.DataFrame:
    df =  pd.read_csv(file.file, sep=";")
    return clean_data(df, config=config)

def create_app() -> FastAPI:

    app = FastAPI(
        title="Student Success Prediction API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/")
    def health():
        return {
            "status": "healthy",
            "message": "Application is working fine.",
        }

    @app.post("/predict/file")
    async def predict_file(
        file: UploadFile = File(...),
        df: pd.DataFrame = Depends(get_dataframe),
        validator: ValidateStudentData = Depends(get_validator),
        model_service: Modelservice = Depends(get_model_service),
    ):
        try:
            logger.info("Reading uploaded CSV: %s", file.filename)
            logger.info("Rows: %d Columns: %d", *df.shape)

            df = validator.validate_csv_file(df)

            predictions = model_service.predict(df)

            return JSONResponse(
                status_code=200,
                content={
                    "filename": file.filename,
                    "rows": len(predictions),
                    "predictions": predictions,
                },
            )

        except ValueError as e:
            logger.exception("Validation failed.")
            raise HTTPException(status_code=400, detail=str(e))

        except Exception:
            logger.exception("Prediction failed.")
            raise HTTPException(
                status_code=500,
                detail="Internal server error.",
            )

    return app


app = create_app()