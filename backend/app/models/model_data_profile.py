from datetime import datetime
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base
from app.models.sample import utc_now


class ModelDataProfile(Base):
    __tablename__ = "model_data_profiles"
    __table_args__ = (
        UniqueConstraint(
            "model_version", "feature_name", name="uq_model_profile_feature"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("training_runs.model_version", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    feature_name: Mapped[str] = mapped_column(String(64), nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    mean: Mapped[float] = mapped_column(Float, nullable=False)
    std: Mapped[float] = mapped_column(Float, nullable=False)
    min: Mapped[float] = mapped_column(Float, nullable=False)
    max: Mapped[float] = mapped_column(Float, nullable=False)
    median: Mapped[float] = mapped_column(Float, nullable=False)
    q25: Mapped[float] = mapped_column(Float, nullable=False)
    q75: Mapped[float] = mapped_column(Float, nullable=False)
    histogram_bins: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    histogram_counts: Mapped[list[int]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
