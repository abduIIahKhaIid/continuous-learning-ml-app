from datetime import datetime

from pydantic import BaseModel, Field


class ModelSummaryRead(BaseModel):
    model_version: str
    algorithm: str
    status: str
    is_active: bool
    trigger_type: str | None
    training_sample_count: int
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1_score: float | None
    roc_auc: float | None
    created_at: datetime
    promoted_at: datetime | None
    completed_at: datetime | None


class DataProfileRead(BaseModel):
    feature_name: str
    sample_count: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float


class ModelEventRead(BaseModel):
    event_type: str
    model_version: str
    previous_model_version: str | None
    reason: str | None
    created_at: datetime


class ModelDetailRead(ModelSummaryRead):
    parameters: dict[str, object] | None
    confusion_matrix: list[list[int]] | None
    data_profiles: list[DataProfileRead]
    events: list[ModelEventRead]


class ModelComparisonRead(BaseModel):
    model_a: ModelSummaryRead
    model_b: ModelSummaryRead
    metric_differences: dict[str, float | None]
    comparison_notice: str


class RollbackRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=500)


class RollbackRead(BaseModel):
    active_model_version: str
    previous_model_version: str | None
    event_id: int
    message: str
