import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sqlalchemy.orm import Session

from app.ml.config import ContinuousTrainingConfig, TrainingConfig
from app.ml.dataset import build_training_dataset
from app.ml.evaluator import EvaluationMetrics, evaluate_model
from app.ml.model_loader import ModelLoader
from app.ml.promotion import decide_promotion, metrics_to_dict
from app.ml.registry import ArtifactInfo, save_trained_model
from app.ml.trainer import create_reproducible_split, train_candidate_on_split
from app.repositories.training import SqlAlchemyTrainingRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingExecutionResult:
    run_id: int
    model_version: str | None
    status: str
    reason: str | None = None


class RetrainingService:
    """Execute a reserved run without holding a transaction during fitting."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session],
        config: ContinuousTrainingConfig,
        loader: ModelLoader,
    ) -> None:
        self._session_factory = session_factory
        self._config = config
        self._loader = loader

    def execute(self, run_id: int) -> TrainingExecutionResult:
        artifact: ArtifactInfo | None = None
        finalized = False
        model_version: str | None = None
        try:
            with self._session_factory() as session:
                run = SqlAlchemyTrainingRepository(session).mark_run_running(
                    run_id
                )
                if run is None:
                    return TrainingExecutionResult(
                        run_id=run_id,
                        model_version=None,
                        status="skipped",
                        reason="Run is not queued.",
                    )
                model_version = run.model_version
                training_sample_ids = list(run.data_selection or [])
                active_version = run.active_model_version_before

            logger.info(
                "Automatic training started: run_id=%s model_version=%s",
                run_id,
                model_version,
            )
            with self._session_factory() as session:
                repository = SqlAlchemyTrainingRepository(session)
                samples = repository.list_verified_samples_by_ids(
                    training_sample_ids
                )
                dataset = build_training_dataset(
                    samples,
                    minimum_samples=self._config.minimum_training_samples,
                )

            training_config = TrainingConfig(
                minimum_samples=self._config.minimum_training_samples,
                test_size=self._config.test_size,
                random_state=self._config.random_state,
                model_dir=self._config.model_dir,
            )
            split = create_reproducible_split(dataset, training_config)

            active_metrics: EvaluationMetrics | None = None
            if active_version is not None:
                with self._session_factory() as session:
                    active_run = SqlAlchemyTrainingRepository(
                        session
                    ).get_by_model_version(active_version)
                    if active_run is None:
                        raise RuntimeError(
                            "The previously active model registry row is missing."
                        )
                    active_model = self._loader.load_registered_run(
                        active_run,
                        model_dir=self._config.model_dir,
                        cache=False,
                    )
                active_metrics = evaluate_model(
                    active_model.pipeline,
                    split.x_evaluation,
                    split.y_evaluation,
                )

            candidate = train_candidate_on_split(split, training_config)
            decision = decide_promotion(
                candidate=candidate.metrics,
                active=active_metrics,
                primary_metric=self._config.primary_promotion_metric,
                minimum_improvement=self._config.min_promotion_improvement,
                minimum_acceptable_f1=self._config.min_acceptable_f1,
            )
            logger.info(
                "Candidate evaluated: model_version=%s metrics=%s",
                model_version,
                metrics_to_dict(candidate.metrics),
            )
            artifact = save_trained_model(
                candidate.pipeline,
                model_version=model_version,
                model_dir=self._config.model_dir,
            )
            parameters = {
                "algorithm": "LogisticRegression",
                "features": ["feature_1", "feature_2", "feature_3"],
                "imputer": "median",
                "scaler": "StandardScaler",
                "max_iter": 1000,
                "test_size": self._config.test_size,
                "random_state": self._config.random_state,
                "primary_promotion_metric": (
                    self._config.primary_promotion_metric
                ),
                "minimum_promotion_improvement": (
                    self._config.min_promotion_improvement
                ),
                "minimum_acceptable_f1": self._config.min_acceptable_f1,
            }
            candidate_metrics = metrics_to_dict(candidate.metrics)
            comparison_metrics = (
                metrics_to_dict(active_metrics)
                if active_metrics is not None
                else None
            )
            with self._session_factory() as session:
                completed_run = SqlAlchemyTrainingRepository(
                    session
                ).finalize_automatic_run(
                    run_id=run_id,
                    all_sample_ids=dataset.sample_ids,
                    evaluation_sample_ids=split.evaluation_sample_ids,
                    model_path=str(artifact.path),
                    artifact_checksum=artifact.checksum,
                    artifact_format=artifact.format,
                    parameters=parameters,
                    train_sample_count=candidate.train_sample_count,
                    test_sample_count=candidate.test_sample_count,
                    metrics=candidate_metrics,
                    active_metrics=comparison_metrics,
                    promoted=decision.promote,
                    rejection_reason=(
                        None if decision.promote else decision.reason
                    ),
                )
            finalized = True
            if decision.promote:
                self._loader.invalidate_cache()
                logger.info(
                    "Active model changed: model_version=%s", model_version
                )
            else:
                logger.info(
                    "Candidate rejected: model_version=%s reason=%s",
                    model_version,
                    decision.reason,
                )
            return TrainingExecutionResult(
                run_id=run_id,
                model_version=model_version,
                status=completed_run.status,
                reason=None if decision.promote else decision.reason,
            )
        except Exception as error:
            logger.exception(
                "Automatic training failed: run_id=%s model_version=%s",
                run_id,
                model_version,
            )
            if artifact is not None and not finalized:
                artifact.path.unlink(missing_ok=True)
            with self._session_factory() as session:
                SqlAlchemyTrainingRepository(session).fail_training_run(
                    run_id, _safe_failure_reason(error)
                )
            return TrainingExecutionResult(
                run_id=run_id,
                model_version=model_version,
                status="failed",
                reason=_safe_failure_reason(error),
            )


def _safe_failure_reason(error: Exception) -> str:
    message = str(error).strip()
    if not message:
        return type(error).__name__
    return f"{type(error).__name__}: {message}"[:1000]
