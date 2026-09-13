from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    false,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base
from app.models.sample import utc_now


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint(
            "predicted_class IN (0, 1)",
            name="ck_predictions_predicted_class",
        ),
        CheckConstraint(
            "actual_label IS NULL OR actual_label IN (0, 1)",
            name="ck_predictions_actual_label",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    feature_1: Mapped[float] = mapped_column(Float, nullable=False)
    feature_2: Mapped[float] = mapped_column(Float, nullable=False)
    feature_3: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_class: Mapped[int] = mapped_column(Integer, nullable=False)
    prediction_probability: Mapped[float | None] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("training_runs.model_version", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    actual_label: Mapped[int | None] = mapped_column(Integer)
    feedback_received: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=false(), nullable=False
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
