from pathlib import Path

import pytest
from celery.exceptions import Retry
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

import app.workers.tasks as task_module
from app.ml.dataset import TrainingDataError
from app.ml.retraining_service import TrainingExecutionResult
from app.models.training_run import TrainingRun
from app.repositories.training import SqlAlchemyTrainingRepository
from app.services.infrastructure_health import InfrastructureHealthService
from app.services.training_dispatch import TrainingDispatcher
from app.workers.redis_lock import RedisTrainingLock


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.closed = False

    def set(
        self, key: str, value: str, *, nx: bool, ex: int
    ) -> bool:
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    def eval(
        self, script: str, key_count: int, key: str, token: str
    ) -> int:
        if self.values.get(key) != token:
            return 0
        del self.values[key]
        return 1

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


def queued_run(session: Session, version: str = "model_v8") -> TrainingRun:
    run = TrainingRun(
        training_batch_id=f"batch_{version}",
        model_version=version,
        algorithm="LogisticRegression",
        training_sample_count=10,
        random_state=42,
        status="queued",
        progress_stage="queued",
    )
    session.add(run)
    session.commit()
    return run


def test_redis_lock_is_exclusive_expiring_and_owner_safe() -> None:
    redis = FakeRedis()
    first = RedisTrainingLock(
        redis, timeout_seconds=30, token="owner-one"
    )
    second = RedisTrainingLock(
        redis, timeout_seconds=30, token="owner-two"
    )

    assert first.acquire() is True
    assert second.acquire() is False
    second.acquired = True
    assert second.release() is False
    assert redis.values[first.key] == "owner-one"
    assert first.release() is True
    assert second.acquire() is True

    # Redis TTL expiry is represented by the server removing the key.
    redis.values.clear()
    assert first.acquire() is True


def test_dispatch_records_task_id_and_reconciles_failed_publish(
    db_session: Session,
) -> None:
    run = queued_run(db_session)

    class Result:
        id = "celery-123"

    class Sender:
        should_fail = True

        def apply_async(self, *, args: list[int], queue: str):
            assert args == [run.id]
            assert queue == "training"
            if self.should_fail:
                raise ConnectionError("Redis unavailable")
            return Result()

    sender = Sender()
    dispatcher = TrainingDispatcher(
        repository=SqlAlchemyTrainingRepository(db_session),
        task_sender=sender,
        queue_name="training",
    )

    failed = dispatcher.dispatch(run.id)
    db_session.refresh(run)
    assert failed.dispatched is False
    assert run.status == "queued"
    assert run.celery_task_id is None
    assert "Redis unavailable" in (run.dispatch_error or "")

    sender.should_fail = False
    recovered = dispatcher.reconcile()
    db_session.refresh(run)
    assert recovered[0].dispatched is True
    assert run.celery_task_id == "celery-123"
    assert run.dispatch_error is None


def configure_task_test(
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
    redis: FakeRedis,
    service_class: type,
) -> None:
    factory = sessionmaker(
        bind=db_session.get_bind(),
        autoflush=False,
        expire_on_commit=False,
    )

    class RedisFactory:
        @staticmethod
        def from_url(url: str, *, decode_responses: bool) -> FakeRedis:
            return redis

    monkeypatch.setattr(task_module, "SessionLocal", factory)
    monkeypatch.setattr(task_module, "Redis", RedisFactory)
    monkeypatch.setattr(task_module, "RetrainingService", service_class)
    monkeypatch.setattr(task_module.random, "randint", lambda start, end: 0)


def test_task_invokes_orchestrator_and_terminal_redelivery_is_idempotent(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = queued_run(db_session)
    redis = FakeRedis()
    calls: list[int] = []
    factory = sessionmaker(
        bind=db_session.get_bind(),
        autoflush=False,
        expire_on_commit=False,
    )

    class SuccessfulService:
        def __init__(self, **kwargs: object) -> None:
            pass

        def execute(
            self, run_id: int, *, raise_failures: bool
        ) -> TrainingExecutionResult:
            assert raise_failures is True
            calls.append(run_id)
            with factory() as session:
                repository = SqlAlchemyTrainingRepository(session)
                claimed = repository.mark_run_running(run_id)
                assert claimed is not None
                claimed.status = "promoted"
                claimed.progress_stage = "completed"
                claimed.concurrency_slot = None
                session.commit()
            return TrainingExecutionResult(run_id, run.model_version, "promoted")

    configure_task_test(monkeypatch, db_session, redis, SuccessfulService)

    first = task_module.run_retraining_task.run(run.id)
    second = task_module.run_retraining_task.run(run.id)

    assert first["status"] == "promoted"
    assert second["status"] == "promoted"
    assert calls == [run.id]
    assert "ml:training:lock" not in redis.values


def test_deterministic_task_failure_is_not_retried_and_releases_lock(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = queued_run(db_session)
    redis = FakeRedis()

    class InvalidDataService:
        def __init__(self, **kwargs: object) -> None:
            pass

        def execute(self, run_id: int, *, raise_failures: bool):
            with sessionmaker(bind=db_session.get_bind())() as session:
                SqlAlchemyTrainingRepository(session).mark_run_running(run_id)
            raise TrainingDataError("one class only")

    configure_task_test(monkeypatch, db_session, redis, InvalidDataService)

    result = task_module.run_retraining_task.run(run.id)

    db_session.expire_all()
    failed = db_session.get(TrainingRun, run.id)
    assert result["status"] == "failed"
    assert failed is not None and failed.status == "failed"
    assert failed.retry_count == 0
    assert "ml:training:lock" not in redis.values


def test_transient_task_failure_records_retry_and_releases_lock(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = queued_run(db_session)
    redis = FakeRedis()

    class TransientService:
        def __init__(self, **kwargs: object) -> None:
            pass

        def execute(self, run_id: int, *, raise_failures: bool):
            with sessionmaker(bind=db_session.get_bind())() as session:
                SqlAlchemyTrainingRepository(session).mark_run_running(run_id)
            raise OperationalError("temporary", {}, ConnectionError())

    configure_task_test(monkeypatch, db_session, redis, TransientService)

    def retry(**kwargs: object):
        raise Retry()

    monkeypatch.setattr(task_module.run_retraining_task, "retry", retry)
    with pytest.raises(Retry):
        task_module.run_retraining_task.run(run.id)

    db_session.expire_all()
    retrying = db_session.get(TrainingRun, run.id)
    assert retrying is not None and retrying.status == "queued"
    assert retrying.retry_count == 1
    assert retrying.last_retry_at is not None
    assert "ml:training:lock" not in redis.values


def test_worker_health_keeps_redis_and_worker_states_separate() -> None:
    redis = FakeRedis()

    class Inspector:
        def ping(self):
            return {"worker@local": {"ok": "pong"}}

    class Control:
        def inspect(self, *, timeout: float) -> Inspector:
            assert timeout == 0.5
            return Inspector()

    class CeleryApp:
        control = Control()

    health = InfrastructureHealthService(
        redis_client=redis,
        celery_app=CeleryApp(),  # type: ignore[arg-type]
        timeout_seconds=0.5,
    ).check()

    assert health.redis_available is True
    assert health.celery_worker_available is True


def test_worker_health_caches_dependency_checks() -> None:
    class CountingRedis(FakeRedis):
        def __init__(self) -> None:
            super().__init__()
            self.ping_count = 0

        def ping(self) -> bool:
            self.ping_count += 1
            return True

    class Inspector:
        def ping(self):
            return {"worker@local": {"ok": "pong"}}

    class Control:
        def __init__(self) -> None:
            self.inspect_count = 0

        def inspect(self, *, timeout: float) -> Inspector:
            self.inspect_count += 1
            return Inspector()

    class CeleryApp:
        control = Control()

    redis = CountingRedis()
    celery = CeleryApp()
    service = InfrastructureHealthService(
        redis_client=redis,
        celery_app=celery,  # type: ignore[arg-type]
        timeout_seconds=0.5,
        cache_seconds=5.0,
    )

    assert service.check() == service.check()
    assert redis.ping_count == 1
    assert celery.control.inspect_count == 1


def test_worker_health_reports_dependency_outage_without_raising() -> None:
    class UnavailableRedis(FakeRedis):
        def ping(self) -> bool:
            raise ConnectionError("offline")

    class BrokenControl:
        def inspect(self, *, timeout: float):
            raise ConnectionError("no broker")

    class CeleryApp:
        control = BrokenControl()

    health = InfrastructureHealthService(
        redis_client=UnavailableRedis(),
        celery_app=CeleryApp(),  # type: ignore[arg-type]
        timeout_seconds=0.5,
    ).check()

    assert health.redis_available is False
    assert health.celery_worker_available is False
