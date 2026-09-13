from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base
from app.models.sample import utc_now


class ModelEvent(Base):
    __tablename__ = "model_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    model_version: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("training_runs.model_version", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    previous_model_version: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("training_runs.model_version", ondelete="RESTRICT"),
        nullable=True,
    )
    reason: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
