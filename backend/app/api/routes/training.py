from fastapi import APIRouter, BackgroundTasks, Query

from app.api.dependencies import (
    ContinuousTrainingConfigDependency,
    ContinuousTrainingCoordinatorDependency,
    TrainingJobRunnerDependency,
    TrainingRepositoryDependency,
)
from app.schemas.training import (
    TrainingCheckResponse,
    TrainingRunRead,
    TrainingStatusResponse,
)
from app.services.training import (
    get_training_status,
    list_training_runs,
    training_check_response,
)

router = APIRouter(prefix="/api/training", tags=["continuous training"])


@router.post("/check", response_model=TrainingCheckResponse)
async def check_training_eligibility(
    background_tasks: BackgroundTasks,
    coordinator: ContinuousTrainingCoordinatorDependency,
    job_runner: TrainingJobRunnerDependency,
) -> TrainingCheckResponse:
    result = coordinator.check_and_reserve()
    if result.training_run_id is not None:
        background_tasks.add_task(job_runner, result.training_run_id)
    return training_check_response(result)


@router.get("/status", response_model=TrainingStatusResponse)
async def training_status(
    repository: TrainingRepositoryDependency,
    config: ContinuousTrainingConfigDependency,
) -> TrainingStatusResponse:
    return get_training_status(repository, config)


@router.get("/runs", response_model=list[TrainingRunRead])
async def training_runs(
    repository: TrainingRepositoryDependency,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[TrainingRunRead]:
    return list_training_runs(repository, skip=skip, limit=limit)
