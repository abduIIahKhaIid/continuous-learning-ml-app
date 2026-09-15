from datetime import timedelta

from app.ml.config import ContinuousTrainingConfig
from app.ml.continuous_training import TrainingCheckResult
from app.models.training_run import TrainingRun
from app.repositories.training import SqlAlchemyTrainingRepository
from app.schemas.training import (
    TrainingCheckResponse,
    TrainingRunRead,
    TrainingJobRead,
    TrainingRunSummary,
    TrainingStatusResponse,
)
from app.models.sample import utc_now
from app.services.infrastructure_health import InfrastructureHealthService
from app.services.training_dispatch import DispatchResult


def training_check_response(
    result: TrainingCheckResult,
    dispatch: DispatchResult | None = None,
) -> TrainingCheckResponse:
    payload = result.__dict__.copy()
    if dispatch is not None:
        payload.update(
            training_scheduled=dispatch.dispatched,
            task_id=dispatch.task_id,
            status="queued",
            dispatch_error=dispatch.error,
            reason=dispatch.error or result.reason,
        )
    return TrainingCheckResponse(**payload)


def get_training_status(
    repository: SqlAlchemyTrainingRepository,
    config: ContinuousTrainingConfig,
    infrastructure: InfrastructureHealthService,
    *,
    stale_timeout_seconds: int,
) -> TrainingStatusResponse:
    active = repository.get_model_for_inference()
    last_run = repository.get_last_run()
    latest_terminal = repository.get_latest_terminal_run()
    current = repository.get_current_run()
    health = infrastructure.check()
    counts = repository.get_sample_counts()
    new_verified = repository.count_unused_verified_samples()
    new_needed = max(0, config.retrain_min_new_samples - new_verified)
    minimum_needed = max(
        0, config.minimum_training_samples - counts.verified
    )
    return TrainingStatusResponse(
        auto_retrain_enabled=config.enabled,
        total_samples=counts.total,
        labelled_samples=counts.labelled,
        unlabelled_samples=counts.total - counts.labelled,
        verified_feedback_samples=counts.verified,
        new_verified_samples=new_verified,
        retrain_threshold=config.retrain_min_new_samples,
        new_verified_samples_needed=new_needed,
        minimum_training_samples=config.minimum_training_samples,
        verified_samples_needed_for_minimum=minimum_needed,
        retraining_data_ready=(new_needed == 0 and minimum_needed == 0),
        training_in_progress=repository.has_training_in_progress(),
        active_model_version=(
            active.model_version if active is not None else None
        ),
        last_training_run=(
            TrainingRunSummary.model_validate(last_run, from_attributes=True)
            if last_run is not None
            else None
        ),
        latest_completed_run=(
            TrainingRunSummary.model_validate(
                latest_terminal, from_attributes=True
            )
            if latest_terminal is not None
            else None
        ),
        queued_jobs=repository.count_runs_by_status(("queued",)),
        running_jobs=repository.count_runs_by_status(("running", "training")),
        current_training_run=(
            _training_job_read(
                current,
                repository=repository,
                stale_timeout_seconds=stale_timeout_seconds,
            )
            if current is not None
            else None
        ),
        redis_available=health.redis_available,
        celery_worker_available=health.celery_worker_available,
    )


def list_training_runs(
    repository: SqlAlchemyTrainingRepository,
    *,
    skip: int,
    limit: int,
    stale_timeout_seconds: int,
) -> list[TrainingRunRead]:
    runs = repository.list_runs(skip=skip, limit=limit)
    return [
        _training_run_read(
            run,
            repository=repository,
            stale_timeout_seconds=stale_timeout_seconds,
        )
        for run in runs
    ]


def get_training_run(
    repository: SqlAlchemyTrainingRepository,
    run_id: int,
    *,
    stale_timeout_seconds: int,
) -> TrainingRunRead | None:
    run = repository.get_run(run_id)
    if run is None:
        return None
    return _training_run_read(
        run,
        repository=repository,
        stale_timeout_seconds=stale_timeout_seconds,
    )


def _training_run_read(
    run: TrainingRun,
    *,
    repository: SqlAlchemyTrainingRepository,
    stale_timeout_seconds: int,
) -> TrainingRunRead:
    return TrainingRunRead.model_validate(
        {
            **{
                field: getattr(run, field)
                for field in TrainingRunRead.model_fields
                if field
                not in {"failure_reason", "task_id", "is_stale"}
            },
            "failure_reason": run.error_message,
            "task_id": run.celery_task_id,
            "is_stale": _is_stale(
                run, repository, stale_timeout_seconds
            ),
        }
    )


def _training_job_read(
    run: TrainingRun,
    *,
    repository: SqlAlchemyTrainingRepository,
    stale_timeout_seconds: int,
) -> TrainingJobRead:
    return TrainingJobRead(
        id=run.id,
        task_id=run.celery_task_id,
        status=run.status,
        progress_stage=run.progress_stage,
        model_version=run.model_version,
        retry_count=run.retry_count,
        dispatch_error=run.dispatch_error,
        created_at=run.created_at,
        dispatched_at=run.dispatched_at,
        started_at=run.started_at,
        heartbeat_at=run.heartbeat_at,
        completed_at=run.completed_at,
        is_stale=_is_stale(run, repository, stale_timeout_seconds),
    )


def _is_stale(
    run: TrainingRun,
    repository: SqlAlchemyTrainingRepository,
    timeout_seconds: int,
) -> bool:
    cutoff = utc_now() - timedelta(seconds=timeout_seconds)
    return repository.is_run_stale(run, cutoff=cutoff)
