from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ml.config import TrainingConfig
from app.ml.dataset import InsufficientTrainingDataError
from app.ml.registry import list_available_model_versions
from app.ml.service import run_initial_training
from app.models.sample import Sample
from app.models.training_run import TrainingRun
from app.repositories.training import SqlAlchemyTrainingRepository


def add_training_samples(session: Session, count: int) -> list[Sample]:
    samples = [
        Sample(
            feature_1=float(index % 2),
            feature_2=float(index),
            feature_3=float((index % 2) + index / 10),
            label=index % 2,
        )
        for index in range(count)
    ]
    session.add_all(samples)
    session.commit()
    for sample in samples:
        session.refresh(sample)
    return samples


def training_config(model_dir: Path, minimum: int = 20) -> TrainingConfig:
    return TrainingConfig(
        minimum_samples=minimum,
        test_size=0.2,
        random_state=42,
        model_dir=model_dir,
    )


def test_successful_training_updates_samples_and_registry(
    db_session: Session,
    tmp_path: Path,
) -> None:
    samples = add_training_samples(db_session, 20)

    summary = run_initial_training(
        db_session,
        training_config(tmp_path / "artifacts"),
    )

    assert summary.model_version == "model_v1"
    assert summary.model_path.is_file()
    db_session.expire_all()
    stored_samples = list(
        db_session.scalars(select(Sample).order_by(Sample.id))
    )
    assert all(sample.used_for_training for sample in stored_samples)
    assert all(
        sample.training_batch_id == summary.training_batch_id
        for sample in stored_samples
    )
    assert all(
        sample.model_version == summary.model_version
        for sample in stored_samples
    )

    run = db_session.scalar(select(TrainingRun))
    assert run is not None
    assert run.status == "completed"
    assert run.training_sample_count == len(samples)
    assert run.train_sample_count == 16
    assert run.test_sample_count == 4
    assert run.data_selection == [sample.id for sample in samples]
    assert run.model_path == str(summary.model_path)
    assert run.artifact_format == "joblib"
    assert run.artifact_checksum is not None
    assert len(run.artifact_checksum) == 64
    assert run.confusion_matrix is not None
    assert run.parameters is not None
    assert run.accuracy is not None


def test_repeated_training_creates_new_model_version(
    db_session: Session,
    tmp_path: Path,
) -> None:
    add_training_samples(db_session, 20)
    config = training_config(tmp_path / "artifacts")

    first = run_initial_training(db_session, config)
    second = run_initial_training(db_session, config)
    repository = SqlAlchemyTrainingRepository(db_session)

    assert first.model_version == "model_v1"
    assert second.model_version == "model_v2"
    assert first.training_batch_id != second.training_batch_id
    assert first.model_path != second.model_path
    assert list_available_model_versions(repository) == [
        "model_v1",
        "model_v2",
    ]


def test_failed_training_does_not_mark_samples_used(
    db_session: Session,
    tmp_path: Path,
) -> None:
    add_training_samples(db_session, 4)

    with pytest.raises(InsufficientTrainingDataError):
        run_initial_training(
            db_session,
            training_config(tmp_path / "artifacts", minimum=20),
        )

    db_session.expire_all()
    samples = list(db_session.scalars(select(Sample)))
    assert all(not sample.used_for_training for sample in samples)
    assert all(sample.training_batch_id is None for sample in samples)
    assert all(sample.model_version is None for sample in samples)
    assert list((tmp_path / "artifacts").glob("*.joblib")) == []

    run = db_session.scalar(select(TrainingRun))
    assert run is not None
    assert run.status == "failed"
    assert run.error_message is not None
    assert "Not enough labelled samples" in run.error_message
