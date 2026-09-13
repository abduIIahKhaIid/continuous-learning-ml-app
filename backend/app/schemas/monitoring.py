from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

MonitorStatus = Literal["stable", "warning", "critical", "insufficient_data"]


class FeatureDriftRead(BaseModel):
    feature_name: str
    psi: float | None
    status: MonitorStatus
    reference_samples: int
    current_samples: int


class DriftRead(BaseModel):
    model_version: str | None
    status: MonitorStatus
    sample_count: int
    minimum_samples: int
    window_size: int
    features: list[FeatureDriftRead]
    samples_analyzed: int
    overall_status: MonitorStatus


class PerformanceRead(BaseModel):
    model_version: str | None
    status: MonitorStatus
    total_predictions: int
    verified_samples: int
    minimum_samples: int
    window_size: int
    feedback_coverage: float
    primary_metric: str
    baseline_value: float | None
    current_value: float | None
    metric_drop: float | None
    metrics: dict[str, Any] | None
    samples_with_feedback: int
    baseline_metric: float | None
    current_metric: float | None


class MonitoringHealthRead(BaseModel):
    status: Literal["healthy", "warning", "unhealthy"]
    model_status: Literal["healthy", "warning", "unhealthy"]
    model_version: str | None
    artifact_available: bool
    data_drift_status: MonitorStatus
    performance_status: MonitorStatus
    feedback_coverage: float
    training_in_progress: bool
    last_training_status: str | None
    new_verified_samples: int
    retrain_threshold: int
    rollback_recommended: bool
    message: str


class MonitoringSnapshotRead(BaseModel):
    id: int
    snapshot_type: str
    status: str
    created_at: datetime


class MonitoringCheckRead(BaseModel):
    drift: DriftRead
    performance: PerformanceRead
    health: MonitoringHealthRead
    snapshots: list[MonitoringSnapshotRead]
