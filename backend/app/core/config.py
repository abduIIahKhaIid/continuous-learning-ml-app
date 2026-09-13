from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import AnyHttpUrl, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
PROJECT_ROOT = ROOT_ENV_FILE.parent


class Settings(BaseSettings):
    app_name: str = "Continuous Learning ML API"
    database_url: str
    min_training_samples: int = Field(default=100, ge=2)
    ml_test_size: float = Field(default=0.2, gt=0, lt=1)
    ml_random_state: int = 42
    model_dir: Path = Path("models")
    auto_retrain_enabled: bool = True
    retrain_min_new_samples: int = Field(default=50, ge=1)
    primary_promotion_metric: Literal[
        "accuracy", "precision", "recall", "f1_score", "roc_auc"
    ] = "f1_score"
    min_promotion_improvement: float = Field(default=0.01, ge=0)
    min_acceptable_f1: float = Field(default=0.6, ge=0, le=1)
    drift_window_size: int = Field(default=200, ge=1)
    drift_min_samples: int = Field(default=50, ge=1)
    drift_psi_warning: float = Field(default=0.10, ge=0)
    drift_psi_critical: float = Field(default=0.25, ge=0)
    performance_window_size: int = Field(default=100, ge=1)
    performance_min_feedback_samples: int = Field(default=30, ge=1)
    performance_warning_drop: float = Field(default=0.05, ge=0)
    performance_critical_drop: float = Field(default=0.10, ge=0)
    frontend_origin: AnyHttpUrl = "http://localhost:5173"
    local_frontend_origin: AnyHttpUrl = "http://localhost:5173"
    codespaces_origin_regex: str = (
        r"https://[a-z0-9-]+-5173\.app\.github\.dev"
    )

    model_config = SettingsConfigDict(
        env_file=ROOT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_monitoring_thresholds(self) -> Self:
        if self.drift_psi_warning > self.drift_psi_critical:
            raise ValueError(
                "DRIFT_PSI_WARNING must not exceed DRIFT_PSI_CRITICAL."
            )
        if self.performance_warning_drop > self.performance_critical_drop:
            raise ValueError(
                "PERFORMANCE_WARNING_DROP must not exceed "
                "PERFORMANCE_CRITICAL_DROP."
            )
        return self

    @property
    def resolved_model_dir(self) -> Path:
        if self.model_dir.is_absolute():
            return self.model_dir
        return (PROJECT_ROOT / self.model_dir).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
