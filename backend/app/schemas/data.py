from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DataCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_1: float = Field(allow_inf_nan=False)
    feature_2: float = Field(allow_inf_nan=False)
    feature_3: float = Field(allow_inf_nan=False)
    label: int | None = Field(default=None, ge=0, le=1)


class DataRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    feature_1: float
    feature_2: float
    feature_3: float
    label: int | None
    created_at: datetime
    updated_at: datetime
    used_for_training: bool
    training_batch_id: str | None
    model_version: str | None

    @field_validator("created_at", "updated_at", mode="after")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
