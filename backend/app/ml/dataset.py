from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.models.sample import Sample
from app.repositories.training import TrainingRepository

FEATURE_NAMES = ("feature_1", "feature_2", "feature_3")


class TrainingDataError(ValueError):
    pass


class InvalidLabelError(TrainingDataError):
    pass


class InvalidFeatureError(TrainingDataError):
    pass


class InsufficientTrainingDataError(TrainingDataError):
    pass


@dataclass(frozen=True)
class TrainingDataset:
    features: NDArray[np.float64]
    labels: NDArray[np.int64]
    sample_ids: list[int]

    @property
    def sample_count(self) -> int:
        return len(self.sample_ids)


def load_training_dataset(
    repository: TrainingRepository,
    *,
    minimum_samples: int,
) -> TrainingDataset:
    samples = repository.list_labelled_samples()
    return build_training_dataset(samples, minimum_samples=minimum_samples)


def build_training_dataset(
    samples: list[Sample],
    *,
    minimum_samples: int,
) -> TrainingDataset:
    invalid_labels = sorted(
        {sample.label for sample in samples if sample.label not in {0, 1}}
    )
    if invalid_labels:
        raise InvalidLabelError(
            f"Training labels must be 0 or 1; found {invalid_labels}."
        )

    if len(samples) < minimum_samples:
        raise InsufficientTrainingDataError(
            "Not enough labelled samples to train: "
            f"found {len(samples)}, require at least {minimum_samples}."
        )

    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    if np.unique(labels).size < 2:
        raise TrainingDataError(
            "Training requires labelled samples from both classes 0 and 1."
        )

    features = np.asarray(
        [
            [sample.feature_1, sample.feature_2, sample.feature_3]
            for sample in samples
        ],
        dtype=np.float64,
    )
    if not np.isfinite(features).all():
        raise InvalidFeatureError(
            "Training features must be present and finite."
        )
    return TrainingDataset(
        features=features,
        labels=labels,
        sample_ids=[sample.id for sample in samples],
    )
