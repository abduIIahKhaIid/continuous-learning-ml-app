import logging
from dataclasses import dataclass
from threading import Lock
from uuid import uuid4

from app.ml.config import ContinuousTrainingConfig
from app.ml.registry import generate_next_model_version
from app.repositories.training import (
    AutomaticTrainingAlreadyReservedError,
    SqlAlchemyTrainingRepository,
)

logger = logging.getLogger(__name__)
_reservation_lock = Lock()
ALGORITHM = "LogisticRegression"


@dataclass(frozen=True)
class TrainingCheckResult:
    eligible: bool
    new_samples: int
    threshold: int
    training_scheduled: bool
    training_run_id: int | None = None
    reason: str | None = None


class ContinuousTrainingCoordinator:
    """Perform a quick threshold check and atomically reserve one run."""

    def __init__(
        self,
        repository: SqlAlchemyTrainingRepository,
        config: ContinuousTrainingConfig,
    ) -> None:
        self._repository = repository
        self._config = config

    def check_and_reserve(self) -> TrainingCheckResult:
        with _reservation_lock:
            return self._check_and_reserve_locked()

    def _check_and_reserve_locked(self) -> TrainingCheckResult:
        unused = self._repository.list_unused_verified_samples()
        all_verified = self._repository.list_verified_samples()
        new_count = len(unused)
        threshold_reached = new_count >= self._config.retrain_min_new_samples
        enough_total = (
            len(all_verified) >= self._config.minimum_training_samples
        )
        if not self._config.enabled:
            return self._result(
                new_count, False, "Automatic retraining is disabled."
            )
        if not threshold_reached:
            return self._result(
                new_count, False, "New-sample threshold not reached."
            )
        if not enough_total:
            return self._result(
                new_count,
                False,
                "Minimum total verified training-sample count not reached.",
            )
        if self._repository.has_training_in_progress():
            return self._result(
                new_count,
                False,
                "A training run is already queued or running.",
                eligible=True,
            )

        active = self._repository.get_model_for_inference()
        model_version = generate_next_model_version(
            self._repository.list_model_versions()
        )
        try:
            run = self._repository.reserve_automatic_training_run(
                training_batch_id=f"batch_{uuid4().hex}",
                model_version=model_version,
                trigger_sample_ids=[sample.id for sample in unused],
                training_sample_ids=[sample.id for sample in all_verified],
                total_training_sample_count=len(all_verified),
                active_model_version=(
                    active.model_version if active is not None else None
                ),
                random_state=self._config.random_state,
                algorithm=ALGORITHM,
            )
        except AutomaticTrainingAlreadyReservedError:
            logger.info("Automatic retraining reservation already exists.")
            return self._result(
                new_count,
                False,
                "A training run was reserved concurrently.",
                eligible=True,
            )

        logger.info(
            "Automatic retraining scheduled: run_id=%s model_version=%s "
            "new_samples=%s total_samples=%s",
            run.id,
            run.model_version,
            run.trigger_new_sample_count,
            run.training_sample_count,
        )
        return TrainingCheckResult(
            eligible=True,
            new_samples=new_count,
            threshold=self._config.retrain_min_new_samples,
            training_scheduled=True,
            training_run_id=run.id,
        )

    def _result(
        self,
        new_count: int,
        scheduled: bool,
        reason: str,
        *,
        eligible: bool = False,
    ) -> TrainingCheckResult:
        self._repository.rollback_read_transaction()
        return TrainingCheckResult(
            eligible=eligible,
            new_samples=new_count,
            threshold=self._config.retrain_min_new_samples,
            training_scheduled=scheduled,
            reason=reason,
        )
