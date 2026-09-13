import logging

from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Path,
    Query,
    status,
)

from app.api.dependencies import (
    FeedbackServiceDependency,
    ContinuousTrainingCoordinatorDependency,
    ModelDirectoryDependency,
    ModelLoaderDependency,
    PredictionRepositoryDependency,
    TrainingRepositoryDependency,
    TrainingJobRunnerDependency,
)
from app.schemas.prediction import (
    ModelStatusResponse,
    PredictionHistoryItem,
    PredictionRequest,
    PredictionResponse,
    PredictionFeedbackRequest,
    PredictionFeedbackResponse,
    PredictionFeedbackStatus,
    PredictionFeedbackSummary,
)
from app.services.feedback import (
    FeedbackAlreadySubmittedError,
    FeedbackNotFoundError,
    InvalidPredictionDataError,
)
from app.services.prediction import (
    PredictionNotFoundError,
    create_prediction,
    get_model_status,
    get_prediction,
    list_predictions,
)

router = APIRouter(prefix="/api", tags=["predictions"])
logger = logging.getLogger(__name__)


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
    "/predictions/feedback/summary",
    response_model=PredictionFeedbackSummary,
)
async def feedback_summary(
    service: FeedbackServiceDependency,
) -> PredictionFeedbackSummary:
    return service.get_summary()


@router.patch(
    "/predictions/{prediction_id}/feedback",
    response_model=PredictionFeedbackResponse,
)
async def submit_prediction_feedback(
    payload: PredictionFeedbackRequest,
    background_tasks: BackgroundTasks,
    service: FeedbackServiceDependency,
    training_coordinator: ContinuousTrainingCoordinatorDependency,
    training_job_runner: TrainingJobRunnerDependency,
    prediction_id: int = Path(gt=0),
) -> PredictionFeedbackResponse:
    try:
        response = service.submit(prediction_id, payload)
    except FeedbackNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found.",
        ) from error
    except FeedbackAlreadySubmittedError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Ground-truth feedback has already been submitted for this "
                "prediction."
            ),
        ) from error
    except InvalidPredictionDataError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error

    try:
        check = training_coordinator.check_and_reserve()
        if check.training_run_id is not None:
            background_tasks.add_task(
                training_job_runner, check.training_run_id
            )
    except Exception:
        logger.exception(
            "Post-feedback automatic-training check failed for prediction %s.",
            prediction_id,
        )
    return response


@router.get(
    "/predictions/{prediction_id}/feedback",
    response_model=PredictionFeedbackStatus,
)
async def prediction_feedback_status(
    service: FeedbackServiceDependency,
    prediction_id: int = Path(gt=0),
) -> PredictionFeedbackStatus:
    try:
        return service.get_status(prediction_id)
    except FeedbackNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found.",
        ) from error


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
