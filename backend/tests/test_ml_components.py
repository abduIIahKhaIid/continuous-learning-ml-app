from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.orm import Session

from app.ml.config import TrainingConfig
from app.ml.dataset import (
    InsufficientTrainingDataError,
    InvalidLabelError,
    TrainingDataset,
    load_training_dataset,
)
from app.ml.preprocessing import build_training_pipeline
from app.ml.registry import generate_next_model_version, save_trained_model
from app.ml.trainer import train_baseline_model
from app.models.sample import Sample
from app.repositories.training import SqlAlchemyTrainingRepository


def add_sample(
    session: Session,
    *,
    index: int,
    label: int | None,
) -> Sample:
    sample = Sample(
        feature_1=float(index),
        feature_2=float(index % 3),
        feature_3=float(index * 2),
        label=label,
    )
    session.add(sample)
    session.commit()
    session.refresh(sample)
    return sample


def make_dataset(size: int = 20) -> TrainingDataset:
    labels = np.asarray([index % 2 for index in range(size)], dtype=np.int64)
    features = np.asarray(
        [
            [float(label), float(index), float(label + index / 10)]
            for index, label in enumerate(labels)
        ],
        dtype=np.float64,
    )
    return TrainingDataset(
        features=features,
        labels=labels,
        sample_ids=list(range(1, size + 1)),
    )


def test_loads_only_labelled_samples(db_session: Session) -> None:
    labelled = [
        add_sample(db_session, index=index, label=index % 2)
        for index in range(4)
    ]
    add_sample(db_session, index=99, label=None)
    repository = SqlAlchemyTrainingRepository(db_session)

    dataset = load_training_dataset(repository, minimum_samples=4)

    assert dataset.sample_count == 4
    assert dataset.sample_ids == [sample.id for sample in labelled]
    assert dataset.features.shape == (4, 3)
    assert dataset.labels.tolist() == [0, 1, 0, 1]


def test_invalid_labels_are_rejected(db_session: Session) -> None:
    add_sample(db_session, index=1, label=0)
    add_sample(db_session, index=2, label=2)
    repository = SqlAlchemyTrainingRepository(db_session)

    with pytest.raises(InvalidLabelError, match="must be 0 or 1"):
        load_training_dataset(repository, minimum_samples=2)


def test_insufficient_data_has_clear_error(db_session: Session) -> None:
    add_sample(db_session, index=1, label=0)
    add_sample(db_session, index=2, label=1)
    repository = SqlAlchemyTrainingRepository(db_session)

    with pytest.raises(
        InsufficientTrainingDataError,
        match="found 2, require at least 20",
    ):
        load_training_dataset(repository, minimum_samples=20)


def test_preprocessing_pipeline_handles_missing_values() -> None:
    features = np.asarray(
        [[0.0, np.nan, 1.0], [1.0, 2.0, np.nan], [0.2, 1.0, 0.0], [0.8, 3.0, 1.0]]
    )
    labels = np.asarray([0, 1, 0, 1])

    pipeline = build_training_pipeline(random_state=42)
    pipeline.fit(features, labels)

    assert pipeline.predict(features).shape == (4,)


def test_training_produces_metrics(tmp_path: Path) -> None:
    config = TrainingConfig(
        minimum_samples=20,
        test_size=0.2,
        random_state=42,
        model_dir=tmp_path,
    )

    result = train_baseline_model(make_dataset(), config)

    assert result.train_sample_count == 16
    assert result.test_sample_count == 4
    assert 0 <= result.metrics.accuracy <= 1
    assert 0 <= result.metrics.precision <= 1
    assert 0 <= result.metrics.recall <= 1
    assert 0 <= result.metrics.f1_score <= 1
    assert len(result.metrics.confusion_matrix) == 2


def test_training_handles_unstratifiable_minority_class(tmp_path: Path) -> None:
    labels = np.asarray([0] * 9 + [1], dtype=np.int64)
    dataset = TrainingDataset(
        features=np.asarray(
            [[float(index), float(label), 1.0] for index, label in enumerate(labels)]
        ),
        labels=labels,
        sample_ids=list(range(1, 11)),
    )
    config = TrainingConfig(
        minimum_samples=10,
        test_size=0.2,
        random_state=42,
        model_dir=tmp_path,
    )

    result = train_baseline_model(dataset, config)

    assert result.train_sample_count == 8
    assert result.test_sample_count == 2
    assert result.metrics.roc_auc is None


def test_model_artifact_is_saved_without_overwriting(tmp_path: Path) -> None:
    pipeline = build_training_pipeline(random_state=42)
    dataset = make_dataset()
    pipeline.fit(dataset.features, dataset.labels)

    artifact = save_trained_model(
        pipeline,
        model_version="model_v1",
        model_dir=tmp_path,
    )

    assert artifact.path == tmp_path / "model_v1.joblib"
    assert artifact.path.is_file()
    assert len(artifact.checksum) == 64
    with pytest.raises(FileExistsError):
        save_trained_model(
            pipeline,
            model_version="model_v1",
            model_dir=tmp_path,
        )
    assert artifact.path.is_file()
    assert artifact.path.stat().st_size > 0


def test_next_model_version_is_unique() -> None:
    assert generate_next_model_version([]) == "model_v1"
    assert (
        generate_next_model_version(["model_v1", "model_v3", "model_v2"])
        == "model_v4"
    )
