from fastapi import APIRouter, HTTPException, Path, Query, status

from app.api.dependencies import (
    ModelDirectoryDependency,
    ModelLoaderDependency,
    PredictionRepositoryDependency,
    TrainingRepositoryDependency,
)
from app.schemas.prediction import (
    ModelStatusResponse,
    PredictionHistoryItem,
    PredictionRequest,
    PredictionResponse,
)
from app.services.prediction import (
    PredictionNotFoundError,
    create_prediction,
    get_model_status,
    get_prediction,
    list_predictions,
)

router = APIRouter(prefix="/api", tags=["predictions"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def predict(
    payload: PredictionRequest,
    prediction_repository: PredictionRepositoryDependency,
    training_repository: TrainingRepositoryDependency,
    loader: ModelLoaderDependency,
    model_dir: ModelDirectoryDependency,
) -> PredictionResponse:
    return create_prediction(
        payload,
        prediction_repository=prediction_repository,
        training_repository=training_repository,
        loader=loader,
        model_dir=model_dir,
    )


@router.get("/predictions", response_model=list[PredictionHistoryItem])
async def prediction_history(
    repository: PredictionRepositoryDependency,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[PredictionHistoryItem]:
    return list_predictions(skip=skip, limit=limit, repository=repository)


@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionHistoryItem,
)
async def prediction_by_id(
    repository: PredictionRepositoryDependency,
    prediction_id: int = Path(gt=0),
) -> PredictionHistoryItem:
    try:
        return get_prediction(prediction_id, repository)
    except PredictionNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found.",
        ) from error


@router.get("/model/status", response_model=ModelStatusResponse)
async def model_status(
    repository: TrainingRepositoryDependency,
    loader: ModelLoaderDependency,
    model_dir: ModelDirectoryDependency,
) -> ModelStatusResponse:
    return get_model_status(
        repository=repository,
        loader=loader,
        model_dir=model_dir,
    )
