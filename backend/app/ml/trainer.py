from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.ml.config import TrainingConfig
from app.ml.dataset import TrainingDataError, TrainingDataset
from app.ml.evaluator import EvaluationMetrics, evaluate_model
from app.ml.preprocessing import build_training_pipeline


@dataclass(frozen=True)
class TrainingSplit:
    x_train: NDArray[np.float64]
    x_evaluation: NDArray[np.float64]
    y_train: NDArray[np.int64]
    y_evaluation: NDArray[np.int64]
    train_sample_ids: list[int]
    evaluation_sample_ids: list[int]


@dataclass(frozen=True)
class TrainingResult:
    pipeline: Pipeline
    metrics: EvaluationMetrics
    train_sample_count: int
    test_sample_count: int
    evaluation_sample_ids: list[int]


def _ensure_both_classes_in_training_set(
    x_train: NDArray[np.float64],
    x_evaluation: NDArray[np.float64],
    y_train: NDArray[np.int64],
    y_evaluation: NDArray[np.int64],
    train_ids: NDArray[np.int64],
    evaluation_ids: NDArray[np.int64],
) -> None:
    if np.unique(y_train).size == 2:
        return

    missing_label = 1 - int(y_train[0])
    matching_indices = np.flatnonzero(y_evaluation == missing_label)
    if matching_indices.size == 0:
        raise TrainingDataError(
            "Unable to place both label classes in the training split."
        )

    evaluation_index = int(matching_indices[0])
    train_row = x_train[0].copy()
    train_label = y_train[0].copy()
    train_id = train_ids[0].copy()
    x_train[0] = x_evaluation[evaluation_index]
    y_train[0] = y_evaluation[evaluation_index]
    train_ids[0] = evaluation_ids[evaluation_index]
    x_evaluation[evaluation_index] = train_row
    y_evaluation[evaluation_index] = train_label
    evaluation_ids[evaluation_index] = train_id


def create_reproducible_split(
    dataset: TrainingDataset,
    config: TrainingConfig,
) -> TrainingSplit:
    if dataset.sample_count < 3:
        raise TrainingDataError(
            "At least three samples are required for a train/test split."
        )

    sample_ids = np.asarray(dataset.sample_ids, dtype=np.int64)
    try:
        split = train_test_split(
            dataset.features,
            dataset.labels,
            sample_ids,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=dataset.labels,
        )
    except ValueError:
        split = train_test_split(
            dataset.features,
            dataset.labels,
            sample_ids,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=None,
        )
        _ensure_both_classes_in_training_set(*split)

    (
        x_train,
        x_evaluation,
        y_train,
        y_evaluation,
        train_ids,
        evaluation_ids,
    ) = split
    if len(y_evaluation) == 0 or len(y_train) < 2:
        raise TrainingDataError(
            "Configured test size does not leave enough data to train."
        )
    return TrainingSplit(
        x_train=x_train,
        x_evaluation=x_evaluation,
        y_train=y_train,
        y_evaluation=y_evaluation,
        train_sample_ids=train_ids.tolist(),
        evaluation_sample_ids=evaluation_ids.tolist(),
    )


def train_candidate_on_split(
    split: TrainingSplit,
    config: TrainingConfig,
) -> TrainingResult:
    pipeline = build_training_pipeline(random_state=config.random_state)
    pipeline.fit(split.x_train, split.y_train)
    metrics = evaluate_model(
        pipeline, split.x_evaluation, split.y_evaluation
    )
    return TrainingResult(
        pipeline=pipeline,
        metrics=metrics,
        train_sample_count=len(split.y_train),
        test_sample_count=len(split.y_evaluation),
        evaluation_sample_ids=split.evaluation_sample_ids,
    )


def train_baseline_model(
    dataset: TrainingDataset,
    config: TrainingConfig,
) -> TrainingResult:
    split = create_reproducible_split(dataset, config)
    return train_candidate_on_split(split, config)
