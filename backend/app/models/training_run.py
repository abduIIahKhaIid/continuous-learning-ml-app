from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, Float, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base
from app.models.sample import utc_now


class TrainingRun(Base):
    __tablename__ = "training_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('training', 'completed', 'failed')",
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
    error_message: Mapped[str | None] = mapped_column(String(2000))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )
