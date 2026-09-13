from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.ml.config import TrainingConfig
from app.ml.dataset import build_training_dataset
from app.ml.evaluator import EvaluationMetrics
from app.ml.registry import ArtifactInfo, reserve_training_run, save_trained_model
from app.ml.trainer import train_baseline_model
from app.monitoring.profiles import build_reference_profiles
from app.repositories.training import SqlAlchemyTrainingRepository

ALGORITHM = "LogisticRegression"


@dataclass(frozen=True)
class TrainingSummary:
    model_version: str
    training_batch_id: str
    model_path: Path
    training_sample_count: int
    train_sample_count: int
    test_sample_count: int
    metrics: EvaluationMetrics


def run_initial_training(
    session: Session,
    config: TrainingConfig,
) -> TrainingSummary:
    repository = SqlAlchemyTrainingRepository(session)
    samples = repository.list_labelled_samples()
    run = reserve_training_run(
        repository,
        algorithm=ALGORITHM,
        training_sample_count=len(samples),
        random_state=config.random_state,
    )
    artifact: ArtifactInfo | None = None

    try:
        dataset = build_training_dataset(
            samples,
            minimum_samples=config.minimum_samples,
        )
        result = train_baseline_model(dataset, config)
        train_id_set = set(result.train_sample_ids)
        training_features = dataset.features[
            [
                index
                for index, sample_id in enumerate(dataset.sample_ids)
                if sample_id in train_id_set
            ]
        ]
        reference_profiles = build_reference_profiles(training_features)
        artifact = save_trained_model(
            result.pipeline,
            model_version=run.model_version,
            model_dir=config.model_dir,
        )
        parameters = {
            "algorithm": ALGORITHM,
            "features": ["feature_1", "feature_2", "feature_3"],
            "imputer": "median",
            "scaler": "StandardScaler",
            "max_iter": 1000,
            "test_size": config.test_size,
            "random_state": config.random_state,
        }
        completed_run = repository.complete_training_run(
            run_id=run.id,
            sample_ids=dataset.sample_ids,
            model_path=str(artifact.path),
            artifact_checksum=artifact.checksum,
            artifact_format=artifact.format,
            parameters=parameters,
            train_sample_count=result.train_sample_count,
            test_sample_count=result.test_sample_count,
            accuracy=result.metrics.accuracy,
            precision=result.metrics.precision,
            recall=result.metrics.recall,
            f1_score=result.metrics.f1_score,
            roc_auc=result.metrics.roc_auc,
            confusion_matrix=result.metrics.confusion_matrix,
            reference_profiles=reference_profiles,
        )
    except Exception as error:
        if artifact is not None:
            artifact.path.unlink(missing_ok=True)
        repository.fail_training_run(run.id, str(error))
        raise

    return TrainingSummary(
        model_version=completed_run.model_version,
        training_batch_id=completed_run.training_batch_id,
        model_path=artifact.path,
        training_sample_count=dataset.sample_count,
        train_sample_count=result.train_sample_count,
        test_sample_count=result.test_sample_count,
        metrics=result.metrics,
    )
