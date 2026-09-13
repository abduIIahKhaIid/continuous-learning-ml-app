from app.ml.config import ContinuousTrainingConfig
from app.ml.continuous_training import TrainingCheckResult
from app.models.training_run import TrainingRun
from app.repositories.training import SqlAlchemyTrainingRepository
from app.schemas.training import (
    TrainingCheckResponse,
    TrainingRunRead,
    TrainingRunSummary,
    TrainingStatusResponse,
)


def training_check_response(
    result: TrainingCheckResult,
) -> TrainingCheckResponse:
    return TrainingCheckResponse(**result.__dict__)


def get_training_status(
    repository: SqlAlchemyTrainingRepository,
    config: ContinuousTrainingConfig,
) -> TrainingStatusResponse:
    active = repository.get_model_for_inference()
    last_run = repository.get_last_run()
    return TrainingStatusResponse(
        auto_retrain_enabled=config.enabled,
        new_verified_samples=repository.count_unused_verified_samples(),
        retrain_threshold=config.retrain_min_new_samples,
        minimum_training_samples=config.minimum_training_samples,
        training_in_progress=repository.has_training_in_progress(),
        active_model_version=(
            active.model_version if active is not None else None
        ),
        last_training_run=(
            TrainingRunSummary.model_validate(last_run, from_attributes=True)
            if last_run is not None
            else None
        ),
    )


def list_training_runs(
    repository: SqlAlchemyTrainingRepository,
    *,
    skip: int,
    limit: int,
) -> list[TrainingRunRead]:
    runs = repository.list_runs(skip=skip, limit=limit)
    return [_training_run_read(run) for run in runs]


def _training_run_read(run: TrainingRun) -> TrainingRunRead:
    return TrainingRunRead.model_validate(
        {
            **{
                field: getattr(run, field)
                for field in TrainingRunRead.model_fields
                if field != "failure_reason"
            },
            "failure_reason": run.error_message,
        }
    )
