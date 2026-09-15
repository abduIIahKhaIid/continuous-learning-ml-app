from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

TrainingStatusValue = Literal[
    "queued",
    "running",
    "training",
    "completed",
    "failed",
    "rejected",
    "promoted",
]


class TrainingCheckResponse(BaseModel):
    eligible: bool
    new_samples: int
    threshold: int
    training_scheduled: bool
    training_run_id: int | None = None
    task_id: str | None = None
    status: str | None = None
    dispatch_error: str | None = None
    reason: str | None = None


class TrainingRunSummary(BaseModel):
    model_version: str
    status: TrainingStatusValue
    created_at: datetime

    @field_validator("created_at", mode="after")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class TrainingStatusResponse(BaseModel):
    auto_retrain_enabled: bool
    total_samples: int
    labelled_samples: int
    unlabelled_samples: int
    verified_feedback_samples: int
    new_verified_samples: int
    retrain_threshold: int
    new_verified_samples_needed: int
    minimum_training_samples: int
    verified_samples_needed_for_minimum: int
    retraining_data_ready: bool
    training_in_progress: bool
    active_model_version: str | None
    last_training_run: TrainingRunSummary | None
    latest_completed_run: TrainingRunSummary | None
    queued_jobs: int
    running_jobs: int
    current_training_run: "TrainingJobRead | None"
    redis_available: bool
    celery_worker_available: bool


class TrainingJobRead(BaseModel):
    id: int
    task_id: str | None
    status: TrainingStatusValue
    progress_stage: str
    model_version: str
    retry_count: int
    dispatch_error: str | None
    created_at: datetime
    dispatched_at: datetime | None
    started_at: datetime | None
    heartbeat_at: datetime | None
    completed_at: datetime | None
    is_stale: bool = False


class TrainingRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    model_version: str
    training_batch_id: str
    trigger_type: str | None
    trigger_new_sample_count: int
    trigger_sample_ids: list[int] | None
    training_sample_count: int
    train_sample_count: int | None
    test_sample_count: int | None
    evaluation_sample_ids: list[int] | None
    algorithm: str
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1_score: float | None
    roc_auc: float | None
    confusion_matrix: list[list[int]] | None
    active_model_version_before: str | None
    active_comparison_metrics: dict[str, object] | None
    status: TrainingStatusValue
    is_active: bool
    promoted_at: datetime | None
    rejection_reason: str | None
    failure_reason: str | None
    created_at: datetime
    completed_at: datetime | None
    task_id: str | None = None
    progress_stage: str
    retry_count: int
    dispatch_error: str | None
    dispatched_at: datetime | None
    started_at: datetime | None
    heartbeat_at: datetime | None
    is_stale: bool = False

    @field_validator(
        "created_at",
        "promoted_at",
        "completed_at",
        "dispatched_at",
        "started_at",
        "heartbeat_at",
        mode="after",
    )
    @classmethod
    def ensure_optional_utc(
        cls, value: datetime | None
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class ReconciliationItem(BaseModel):
    training_run_id: int
    dispatched: bool
    task_id: str | None
    error: str | None


class ReconciliationResponse(BaseModel):
    recovered_count: int
    failed_count: int
    items: list[ReconciliationItem]


class WorkerHealthResponse(BaseModel):
    redis_available: bool
    celery_worker_available: bool
