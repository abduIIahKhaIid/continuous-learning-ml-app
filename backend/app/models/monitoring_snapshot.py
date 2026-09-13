from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base
from app.models.sample import utc_now


class MonitoringSnapshot(Base):
    __tablename__ = "monitoring_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("training_runs.model_version", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    snapshot_type: Mapped[str] = mapped_column(String(32), nullable=False)
    overall_status: Mapped[str] = mapped_column(String(32), nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
