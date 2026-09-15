import logging
from dataclasses import dataclass
from typing import Protocol

from app.repositories.training import SqlAlchemyTrainingRepository

logger = logging.getLogger(__name__)


class AsyncTaskResult(Protocol):
    id: str


class TrainingTaskSender(Protocol):
    def apply_async(
        self, *, args: list[int], queue: str
    ) -> AsyncTaskResult: ...


@dataclass(frozen=True)
class DispatchResult:
    training_run_id: int
    dispatched: bool
    task_id: str | None
    error: str | None = None


class TrainingDispatcher:
    """Publish a committed SQL training run and record dispatch traceability."""

    def __init__(
        self,
        *,
        repository: SqlAlchemyTrainingRepository,
        task_sender: TrainingTaskSender,
        queue_name: str,
    ) -> None:
        self._repository = repository
        self._task_sender = task_sender
        self._queue_name = queue_name

    def dispatch(self, training_run_id: int) -> DispatchResult:
        try:
            task = self._task_sender.apply_async(
                args=[training_run_id], queue=self._queue_name
            )
            self._repository.record_dispatch(training_run_id, task.id)
        except Exception as error:
            safe_error = _safe_dispatch_error(error)
            self._repository.record_dispatch_failure(
                training_run_id, safe_error
            )
            logger.exception(
                "Training dispatch failed: training_run_id=%s",
                training_run_id,
            )
            return DispatchResult(
                training_run_id=training_run_id,
                dispatched=False,
                task_id=None,
                error=safe_error,
            )
        logger.info(
            "Training task queued: training_run_id=%s celery_task_id=%s",
            training_run_id,
            task.id,
        )
        return DispatchResult(
            training_run_id=training_run_id,
            dispatched=True,
            task_id=task.id,
        )

    def reconcile(self) -> list[DispatchResult]:
        run_ids = [
            run.id for run in self._repository.list_recoverable_queued_runs()
        ]
        return [self.dispatch(run_id) for run_id in run_ids]


def _safe_dispatch_error(error: Exception) -> str:
    message = str(error).strip() or type(error).__name__
    return f"{type(error).__name__}: {message}"[:1000]

