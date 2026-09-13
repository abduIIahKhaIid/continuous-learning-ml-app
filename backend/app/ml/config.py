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
