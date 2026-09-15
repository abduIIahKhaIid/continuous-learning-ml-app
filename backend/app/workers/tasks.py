import logging
import random
from datetime import timedelta
from typing import Any

from billiard.exceptions import SoftTimeLimitExceeded
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from sqlalchemy.exc import OperationalError

from app.core.config import get_settings
from app.database.session import SessionLocal
from app.ml.config import ContinuousTrainingConfig
from app.ml.dataset import TrainingDataError
from app.ml.model_loader import model_loader
from app.ml.retraining_service import RetrainingService
from app.models.sample import utc_now
from app.repositories.training import SqlAlchemyTrainingRepository
from app.workers.celery_app import celery_app
from app.workers.redis_lock import RedisTrainingLock

logger = logging.getLogger(__name__)
TERMINAL_STATUSES = ("completed", "failed", "rejected", "promoted")


class TrainingLockUnavailableError(RuntimeError):
    pass


@celery_app.task(
    bind=True,
    name="app.workers.tasks.run_retraining_task",
    acks_late=True,
)
def run_retraining_task(
    task: Any, training_run_id: int
) -> dict[str, int | str | None]:
    settings = get_settings()
    task_id = str(task.request.id or "unknown")
    logger.info(
        "Worker received training task: training_run_id=%s celery_task_id=%s",
        training_run_id,
        task_id,
    )

    with SessionLocal() as session:
        repository = SqlAlchemyTrainingRepository(session)
        run = repository.get_run(training_run_id)
        if run is None:
            return _result(training_run_id, "missing", None)
        if run.status in TERMINAL_STATUSES:
            logger.info(
                "Duplicate terminal task skipped: training_run_id=%s status=%s",
                training_run_id,
                run.status,
            )
            return _result(training_run_id, run.status, run.model_version)
        model_version = run.model_version

    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    lock = RedisTrainingLock(
        redis_client,
        timeout_seconds=settings.training_lock_timeout_seconds,
    )
    execution_started = False
    try:
        if not lock.acquire():
            raise TrainingLockUnavailableError(
                "Another automatic training task owns the distributed lock."
            )
        logger.info(
            "Training lock acquired: training_run_id=%s celery_task_id=%s",
            training_run_id,
            task_id,
        )
        with SessionLocal() as session:
            repository = SqlAlchemyTrainingRepository(session)
            current = repository.get_run(training_run_id)
            if current is None:
                return _result(training_run_id, "missing", None)
            if current.status in TERMINAL_STATUSES:
                return _result(
                    training_run_id, current.status, current.model_version
                )
            if current.status == "running":
                stale_cutoff = utc_now() - timedelta(
                    seconds=settings.training_stale_timeout_seconds
                )
                if repository.is_run_stale(current, cutoff=stale_cutoff):
                    repository.prepare_retry(
                        training_run_id,
                        "Recovering stale redelivered task.",
                    )
                else:
                    raise TrainingLockUnavailableError(
                        "Training run is already active and not stale."
                    )
        execution_started = True
        execution = RetrainingService(
            session_factory=SessionLocal,
            config=ContinuousTrainingConfig.from_settings(settings),
            loader=model_loader,
        ).execute(training_run_id, raise_failures=True)
        logger.info(
            "Training task completed: training_run_id=%s status=%s",
            training_run_id,
            execution.status,
        )
        return _result(
            training_run_id, execution.status, execution.model_version
        )
    except SoftTimeLimitExceeded as error:
        return _fail_final(training_run_id, error, model_version)
    except TrainingLockUnavailableError as error:
        if task.request.retries < settings.training_task_max_retries:
            delay = _retry_delay(
                settings.training_task_retry_delay_seconds,
                task.request.retries,
            )
            with SessionLocal() as session:
                SqlAlchemyTrainingRepository(session).prepare_retry(
                    training_run_id,
                    _safe_error(error),
                    requeue=False,
                )
            raise task.retry(
                exc=error,
                countdown=delay,
                max_retries=settings.training_task_max_retries,
            )
        with SessionLocal() as session:
            repository = SqlAlchemyTrainingRepository(session)
            current = repository.get_run(training_run_id)
            if current is not None and current.status == "queued":
                repository.record_dispatch_failure(
                    training_run_id, _safe_error(error)
                )
        return _result(training_run_id, "lock_unavailable", model_version)
    except Exception as error:
        if _is_transient(error) and task.request.retries < settings.training_task_max_retries:
            delay = _retry_delay(
                settings.training_task_retry_delay_seconds,
                task.request.retries,
            )
            with SessionLocal() as session:
                SqlAlchemyTrainingRepository(session).prepare_retry(
                    training_run_id,
                    _safe_error(error),
                    requeue=execution_started,
                )
            logger.warning(
                "Training retry scheduled: training_run_id=%s task_id=%s delay=%s",
                training_run_id,
                task_id,
                delay,
            )
            raise task.retry(
                exc=error,
                countdown=delay,
                max_retries=settings.training_task_max_retries,
            )
        return _fail_final(training_run_id, error, model_version)
    finally:
        try:
            if lock.release():
                logger.info(
                    "Training lock released: training_run_id=%s", training_run_id
                )
        except Exception:
            logger.exception(
                "Training lock release failed: training_run_id=%s",
                training_run_id,
            )
        redis_client.close()


def _is_transient(error: Exception) -> bool:
    if isinstance(error, TrainingDataError):
        return False
    return isinstance(
        error,
        (
            TrainingLockUnavailableError,
            RedisConnectionError,
            RedisTimeoutError,
            OperationalError,
            OSError,
        ),
    )


def _retry_delay(base_seconds: int, retry_count: int) -> int:
    backoff = base_seconds * (2**retry_count)
    return backoff + random.randint(0, min(base_seconds, 10))


def _fail_final(
    training_run_id: int, error: Exception, model_version: str | None
) -> dict[str, int | str | None]:
    with SessionLocal() as session:
        SqlAlchemyTrainingRepository(session).fail_training_run(
            training_run_id, _safe_error(error)
        )
    logger.exception(
        "Training task failed: training_run_id=%s", training_run_id
    )
    return _result(training_run_id, "failed", model_version)


def _safe_error(error: Exception) -> str:
    message = str(error).strip() or type(error).__name__
    return f"{type(error).__name__}: {message}"[:1000]


def _result(
    training_run_id: int, status: str, model_version: str | None
) -> dict[str, int | str | None]:
    return {
        "training_run_id": training_run_id,
        "status": status,
        "model_version": model_version,
    }
