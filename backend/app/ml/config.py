from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings


@dataclass(frozen=True)
class TrainingConfig:
    minimum_samples: int
    test_size: float
    random_state: int
    model_dir: Path

    @classmethod
    def from_settings(cls, settings: Settings) -> "TrainingConfig":
        return cls(
            minimum_samples=settings.min_training_samples,
            test_size=settings.ml_test_size,
            random_state=settings.ml_random_state,
            model_dir=settings.resolved_model_dir,
        )


@dataclass(frozen=True)
class ContinuousTrainingConfig:
    enabled: bool
    retrain_min_new_samples: int
    minimum_training_samples: int
    test_size: float
    random_state: int
    model_dir: Path
    primary_promotion_metric: str
    min_promotion_improvement: float
    min_acceptable_f1: float

    @classmethod
    def from_settings(cls, settings: Settings) -> "ContinuousTrainingConfig":
        return cls(
            enabled=settings.auto_retrain_enabled,
            retrain_min_new_samples=settings.retrain_min_new_samples,
            minimum_training_samples=settings.min_training_samples,
            test_size=settings.ml_test_size,
            random_state=settings.ml_random_state,
            model_dir=settings.resolved_model_dir,
            primary_promotion_metric=settings.primary_promotion_metric,
            min_promotion_improvement=settings.min_promotion_improvement,
            min_acceptable_f1=settings.min_acceptable_f1,
        )
