from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_1: float = Field(allow_inf_nan=False)
    feature_2: float = Field(allow_inf_nan=False)
    feature_3: float = Field(allow_inf_nan=False)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    prediction_id: int
    prediction: int
    predicted_class: int
    probability: float | None
    model_version: str
    created_at: datetime

    @field_validator("created_at", mode="after")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class PredictionHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    feature_1: float
    feature_2: float
    feature_3: float
    predicted_class: int
    prediction_probability: float | None
    model_version: str
    actual_label: int | None
    feedback_received: bool
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at", mode="after")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class PredictionFeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actual_label: int = Field(strict=True, ge=0, le=1)


class PredictionFeedbackStatus(BaseModel):
    prediction_id: int
    predicted_class: int
    actual_label: int | None
    feedback_received: bool
    was_correct: bool | None
    model_version: str


class PredictionFeedbackResponse(PredictionFeedbackStatus):
    updated_at: datetime

    @field_validator("updated_at", mode="after")
    @classmethod
    def ensure_updated_at_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class PredictionFeedbackSummary(BaseModel):
    total_predictions: int
    feedback_received: int
    feedback_pending: int
    correct_predictions: int
    incorrect_predictions: int
    verified_accuracy: float | None


class ModelMetrics(BaseModel):
    accuracy: float | None
    f1_score: float | None
    roc_auc: float | None


class ModelStatusResponse(BaseModel):
    model_available: bool
    model_version: str | None
    algorithm: str | None = None
    created_at: datetime | None = None
    metrics: ModelMetrics | None = None
    status: Literal["ready", "unavailable"]
    detail: str | None = None

    @field_validator("created_at", mode="after")
    @classmethod
    def ensure_optional_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
