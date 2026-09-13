from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.ml.inference import PredictionExecutionError
from app.ml.model_loader import (
    ModelArtifactUnavailableError,
    ModelLoadError,
    NoModelAvailableError,
)
from app.services.prediction import PredictionDatabaseError
from app.services.feedback import FeedbackDatabaseError


def register_exception_handlers(application: FastAPI) -> None:
    @application.exception_handler(NoModelAvailableError)
    async def no_model_handler(
        _: Request, __: NoModelAvailableError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "No trained model available."},
        )

    @application.exception_handler(ModelArtifactUnavailableError)
    @application.exception_handler(ModelLoadError)
    async def unavailable_artifact_handler(
        _: Request,
        __: ModelArtifactUnavailableError | ModelLoadError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "The trained model is currently unavailable."},
        )

    @application.exception_handler(PredictionExecutionError)
    async def prediction_failure_handler(
        _: Request, __: PredictionExecutionError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "The prediction could not be completed."},
        )

    @application.exception_handler(PredictionDatabaseError)
    @application.exception_handler(FeedbackDatabaseError)
    async def prediction_database_handler(
        _: Request, __: PredictionDatabaseError | FeedbackDatabaseError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "A database operation failed."},
        )
