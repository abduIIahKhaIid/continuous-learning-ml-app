import logging
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from app.ml.inference import PredictionExecutionError, run_inference
from app.ml.model_loader import (
    ModelArtifactUnavailableError,
    ModelLoadError,
    ModelLoader,
    NoModelAvailableError,
)
from app.repositories.prediction import PredictionRepository
from app.repositories.training import TrainingRepository
from app.schemas.prediction import (
    ModelMetrics,
    ModelStatusResponse,
    PredictionHistoryItem,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)


class PredictionNotFoundError(Exception):
    pass


class PredictionDatabaseError(RuntimeError):
    pass


def create_prediction(
    payload: PredictionRequest,
    *,
    prediction_repository: PredictionRepository,
    training_repository: TrainingRepository,
    loader: ModelLoader,
    model_dir: Path,
) -> PredictionResponse:
    try:
        loaded_model = loader.load(training_repository, model_dir=model_dir)
        result = run_inference(loaded_model, **payload.model_dump())
        record = prediction_repository.create(
            **payload.model_dump(),
            predicted_class=result.predicted_class,
            prediction_probability=result.probability,
            model_version=result.model_version,
        )
    except (NoModelAvailableError, ModelArtifactUnavailableError, ModelLoadError):
        raise
    except PredictionExecutionError:
        logger.exception("Prediction execution failed.")
        raise
    except SQLAlchemyError as error:
        logger.exception("Prediction database operation failed.")
        raise PredictionDatabaseError("Prediction could not be stored.") from error

    return PredictionResponse(
        prediction_id=record.id,
        prediction=record.predicted_class,
        predicted_class=record.predicted_class,
        probability=record.prediction_probability,
        model_version=record.model_version,
        created_at=record.created_at,
    )


def list_predictions(
    *,
    skip: int,
    limit: int,
    repository: PredictionRepository,
) -> list[PredictionHistoryItem]:
    try:
        records = repository.list(skip=skip, limit=limit)
    except SQLAlchemyError as error:
        logger.exception("Could not retrieve prediction history.")
        raise PredictionDatabaseError(
            "Prediction history could not be retrieved."
        ) from error
    return [PredictionHistoryItem.model_validate(record) for record in records]


def get_prediction(
    prediction_id: int,
    repository: PredictionRepository,
) -> PredictionHistoryItem:
    try:
        record = repository.get(prediction_id)
    except SQLAlchemyError as error:
        logger.exception("Could not retrieve prediction %s.", prediction_id)
        raise PredictionDatabaseError(
            "Prediction could not be retrieved."
        ) from error
    if record is None:
        raise PredictionNotFoundError(prediction_id)
    return PredictionHistoryItem.model_validate(record)


def get_model_status(
    *,
    repository: TrainingRepository,
    loader: ModelLoader,
    model_dir: Path,
) -> ModelStatusResponse:
    try:
        loaded_model = loader.load(repository, model_dir=model_dir)
    except NoModelAvailableError:
        return ModelStatusResponse(
            model_available=False,
            model_version=None,
            status="unavailable",
            detail="No trained model available.",
        )
    except (ModelArtifactUnavailableError, ModelLoadError) as error:
        return ModelStatusResponse(
            model_available=False,
            model_version=error.model_version,
            status="unavailable",
            detail="The registered model artifact is unavailable.",
        )
    except SQLAlchemyError as error:
        logger.exception("Could not read model registry status.")
        raise PredictionDatabaseError(
            "Model status could not be retrieved."
        ) from error

    descriptor = loaded_model.descriptor
    return ModelStatusResponse(
        model_available=True,
        model_version=descriptor.model_version,
        algorithm=descriptor.algorithm,
        created_at=descriptor.created_at,
        metrics=ModelMetrics(
            accuracy=descriptor.accuracy,
            f1_score=descriptor.f1_score,
            roc_auc=descriptor.roc_auc,
        ),
        status="ready",
    )
