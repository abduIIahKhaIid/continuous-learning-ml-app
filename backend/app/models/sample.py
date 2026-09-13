from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    false,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


class Sample(Base):
    __tablename__ = "samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    feature_1: Mapped[float] = mapped_column(Float, nullable=False)
    feature_2: Mapped[float] = mapped_column(Float, nullable=False)
    feature_3: Mapped[float] = mapped_column(Float, nullable=False)
    label: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )
    used_for_training: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
        nullable=False,
    )
    training_batch_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    model_version: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    source_prediction_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("predictions.id", ondelete="RESTRICT"),
        nullable=True,
        unique=True,
    )
    last_triggered_training_run_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("training_runs.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
