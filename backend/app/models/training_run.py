from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
    false,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base
from app.models.sample import utc_now


class TrainingRun(Base):
    __tablename__ = "training_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'training', 'completed', "
            "'failed', 'rejected', 'promoted')",
            name="ck_training_runs_status",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_batch_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False
    )
    model_version: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False
    )
    model_path: Mapped[str | None] = mapped_column(String(1024))
    artifact_checksum: Mapped[str | None] = mapped_column(String(64))
    artifact_format: Mapped[str | None] = mapped_column(String(32))
    algorithm: Mapped[str] = mapped_column(String(255), nullable=False)
    parameters: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    data_selection: Mapped[list[int] | None] = mapped_column(JSON)
    training_sample_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    trigger_type: Mapped[str | None] = mapped_column(String(32))
    trigger_new_sample_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    trigger_sample_ids: Mapped[list[int] | None] = mapped_column(JSON)
    evaluation_sample_ids: Mapped[list[int] | None] = mapped_column(JSON)
    active_model_version_before: Mapped[str | None] = mapped_column(String(64))
    active_comparison_metrics: Mapped[dict[str, Any] | None] = mapped_column(
        JSON
    )
    train_sample_count: Mapped[int | None] = mapped_column(Integer)
    test_sample_count: Mapped[int | None] = mapped_column(Integer)
    accuracy: Mapped[float | None] = mapped_column(Float)
    precision: Mapped[float | None] = mapped_column(Float)
    recall: Mapped[float | None] = mapped_column(Float)
    f1_score: Mapped[float | None] = mapped_column(Float)
    roc_auc: Mapped[float | None] = mapped_column(Float)
    confusion_matrix: Mapped[list[list[int]] | None] = mapped_column(JSON)
    random_state: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default="training", nullable=False
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=false(), nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(String(2000))
    rejection_reason: Mapped[str | None] = mapped_column(String(2000))
    concurrency_slot: Mapped[str | None] = mapped_column(
        String(32), unique=True
    )
    celery_task_id: Mapped[str | None] = mapped_column(
        String(255), unique=True, index=True
    )
    retry_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    progress_stage: Mapped[str] = mapped_column(
        String(32), default="queued", server_default="queued", nullable=False
    )
    dispatch_error: Mapped[str | None] = mapped_column(String(1000))
    dispatched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    last_retry_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    promoted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )
