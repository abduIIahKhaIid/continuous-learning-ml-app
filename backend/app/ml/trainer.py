from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.ml.config import TrainingConfig
from app.ml.dataset import TrainingDataError, TrainingDataset
from app.ml.evaluator import EvaluationMetrics, evaluate_model
from app.ml.preprocessing import build_training_pipeline


@dataclass(frozen=True)
class TrainingResult:
    pipeline: Pipeline
    metrics: EvaluationMetrics
    train_sample_count: int
    test_sample_count: int


def _ensure_both_classes_in_training_set(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    if np.unique(y_train).size == 2:
        return

    missing_label = 1 - int(y_train[0])
    matching_test_indices = np.flatnonzero(y_test == missing_label)
    if matching_test_indices.size == 0:
        raise TrainingDataError(
            "Unable to place both label classes in the training split."
        )

    test_index = int(matching_test_indices[0])
    train_row = x_train[0].copy()
    train_label = y_train[0].copy()
    x_train[0] = x_test[test_index]
    y_train[0] = y_test[test_index]
    x_test[test_index] = train_row
    y_test[test_index] = train_label


def train_baseline_model(
    dataset: TrainingDataset,
    config: TrainingConfig,
) -> TrainingResult:
    if dataset.sample_count < 3:
        raise TrainingDataError(
            "At least three samples are required for a train/test split."
        )

    stratify = dataset.labels
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.features,
            dataset.labels,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=stratify,
        )
    except ValueError:
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.features,
            dataset.labels,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=None,
        )
        _ensure_both_classes_in_training_set(
            x_train, x_test, y_train, y_test
        )

    if len(y_test) == 0 or len(y_train) < 2:
        raise TrainingDataError(
            "Configured test size does not leave enough data to train."
        )

    pipeline = build_training_pipeline(random_state=config.random_state)
    pipeline.fit(x_train, y_train)
    metrics = evaluate_model(pipeline, x_test, y_test)
    return TrainingResult(
        pipeline=pipeline,
        metrics=metrics,
        train_sample_count=len(y_train),
        test_sample_count=len(y_test),
    )
