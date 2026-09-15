from fastapi import APIRouter, HTTPException, Query, Response, status

from app.api.dependencies import (
    ContinuousTrainingConfigDependency,
    ContinuousTrainingCoordinatorDependency,
    InfrastructureHealthDependency,
    TrainingDispatcherDependency,
    TrainingRepositoryDependency,
)
from app.schemas.training import (
    TrainingCheckResponse,
    TrainingRunRead,
    TrainingStatusResponse,
    ReconciliationResponse,
    ReconciliationItem,
)
from app.services.training import (
    get_training_status,
    list_training_runs,
    get_training_run,
    training_check_response,
)

router = APIRouter(prefix="/api/training", tags=["continuous training"])


@router.post("/check", response_model=TrainingCheckResponse)
async def check_training_eligibility(
    response: Response,
    coordinator: ContinuousTrainingCoordinatorDependency,
    dispatcher: TrainingDispatcherDependency,
) -> TrainingCheckResponse:
    result = coordinator.check_and_reserve()
    dispatch = None
    if result.training_run_id is not None:
        dispatch = dispatcher.dispatch(result.training_run_id)
        response.status_code = (
            status.HTTP_202_ACCEPTED
            if dispatch.dispatched
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
    return training_check_response(result, dispatch)


@router.get("/status", response_model=TrainingStatusResponse)
async def training_status(
    repository: TrainingRepositoryDependency,
    config: ContinuousTrainingConfigDependency,
    infrastructure: InfrastructureHealthDependency,
) -> TrainingStatusResponse:
    from app.core.config import get_settings

    return get_training_status(
        repository,
        config,
        infrastructure,
        stale_timeout_seconds=get_settings().training_stale_timeout_seconds,
    )


@router.get("/runs", response_model=list[TrainingRunRead])
async def training_runs(
    repository: TrainingRepositoryDependency,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[TrainingRunRead]:
    from app.core.config import get_settings

    return list_training_runs(
        repository,
        skip=skip,
        limit=limit,
        stale_timeout_seconds=get_settings().training_stale_timeout_seconds,
    )


@router.get("/runs/{training_run_id}", response_model=TrainingRunRead)
async def training_run_detail(
    training_run_id: int,
    repository: TrainingRepositoryDependency,
) -> TrainingRunRead:
    from app.core.config import get_settings

    result = get_training_run(
        repository,
        training_run_id,
        stale_timeout_seconds=get_settings().training_stale_timeout_seconds,
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training run not found.",
        )
    return result


@router.post("/reconcile", response_model=ReconciliationResponse)
async def reconcile_queued_training(
    dispatcher: TrainingDispatcherDependency,
) -> ReconciliationResponse:
    results = dispatcher.reconcile()
    return ReconciliationResponse(
        recovered_count=sum(result.dispatched for result in results),
        failed_count=sum(not result.dispatched for result in results),
        items=[ReconciliationItem(**result.__dict__) for result in results],
    )
