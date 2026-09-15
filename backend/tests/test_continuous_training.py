from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

import app.ml.retraining_service as retraining_module
from app.api.dependencies import (
    get_continuous_training_config,
    get_continuous_training_coordinator,
    get_training_dispatcher,
    get_infrastructure_health_service,
    get_model_directory,
    get_model_loader,
)
from app.database.session import get_db
from app.main import app
from app.ml.config import ContinuousTrainingConfig
from app.ml.continuous_training import ContinuousTrainingCoordinator
from app.ml.model_loader import ModelLoader
from app.ml.preprocessing import build_training_pipeline
from app.ml.promotion import decide_promotion
from app.ml.registry import save_trained_model
from app.ml.retraining_service import RetrainingService
from app.models.prediction import Prediction
from app.models.sample import Sample
from app.models.training_run import TrainingRun
from app.repositories.training import SqlAlchemyTrainingRepository
from app.services.infrastructure_health import WorkerHealth
from app.services.training_dispatch import TrainingDispatcher


def continuous_config(
    model_dir: Path,
    *,
    threshold: int = 5,
    minimum: int = 10,
    improvement: float = 0.01,
    minimum_f1: float = 0.0,
    enabled: bool = True,
) -> ContinuousTrainingConfig:
    return ContinuousTrainingConfig(
        enabled=enabled,
        retrain_min_new_samples=threshold,
        minimum_training_samples=minimum,
        test_size=0.25,
        random_state=42,
        model_dir=model_dir,
        primary_promotion_metric="f1_score",
        min_promotion_improvement=improvement,
        min_acceptable_f1=minimum_f1,
    )


def ensure_source_run(session: Session) -> TrainingRun:
    existing = session.scalar(
        select(TrainingRun).where(TrainingRun.model_version == "source_model")
    )
    if existing is not None:
        return existing
    run = TrainingRun(
        training_batch_id="source_batch",
        model_version="source_model",
        algorithm="LogisticRegression",
        training_sample_count=0,
        random_state=42,
        status="failed",
    )
    session.add(run)
    session.flush()
    return run


def add_verified_samples(
    session: Session,
    count: int,
    *,
    used: bool = False,
    start: int | None = None,
) -> list[Sample]:
    source = ensure_source_run(session)
    offset = start if start is not None else int(
        session.scalar(select(func.count(Prediction.id))) or 0
    )
    samples: list[Sample] = []
    for index in range(offset, offset + count):
        label = index % 2
        prediction = Prediction(
            feature_1=float(label) + index / 1000,
            feature_2=float(label),
            feature_3=float(label) + (index % 3) / 100,
            predicted_class=label,
            prediction_probability=0.8,
            model_version=source.model_version,
        )
        session.add(prediction)
        session.flush()
        sample = Sample(
            feature_1=prediction.feature_1,
            feature_2=prediction.feature_2,
            feature_3=prediction.feature_3,
            label=label,
            source_prediction_id=prediction.id,
            used_for_training=used,
        )
        session.add(sample)
        samples.append(sample)
    session.commit()
    return samples


def create_active_model(
    session: Session,
    model_dir: Path,
    samples: list[Sample],
    *,
    poor: bool,
) -> TrainingRun:
    pipeline = build_training_pipeline(random_state=42)
    features = np.asarray(
        [[row.feature_1, row.feature_2, row.feature_3] for row in samples]
    )
    labels = np.asarray([row.label for row in samples])
    pipeline.fit(features, 1 - labels if poor else labels)
    artifact = save_trained_model(
        pipeline,
        model_version="model_v1",
        model_dir=model_dir,
    )
    run = TrainingRun(
        training_batch_id="active_batch",
        model_version="model_v1",
        model_path=str(artifact.path),
        artifact_checksum=artifact.checksum,
        artifact_format=artifact.format,
        algorithm="LogisticRegression",
        training_sample_count=len(samples),
        random_state=42,
        status="promoted",
        is_active=True,
    )
    session.add(run)
    session.commit()
    return run


def make_factory(session: Session):
    return sessionmaker(
        bind=session.get_bind(),
        autoflush=False,
        expire_on_commit=False,
    )


def reserve(
    session: Session,
    config: ContinuousTrainingConfig,
):
    return ContinuousTrainingCoordinator(
        SqlAlchemyTrainingRepository(session), config
    ).check_and_reserve()


def test_only_new_verified_labelled_samples_count_toward_threshold(
    db_session: Session,
    tmp_path: Path,
) -> None:
    add_verified_samples(db_session, 3)
    db_session.add(
        Sample(feature_1=1, feature_2=1, feature_3=1, label=1)
    )
    db_session.commit()
    result = reserve(
        db_session,
        continuous_config(tmp_path, threshold=4, minimum=3),
    )

    assert result.eligible is False
    assert result.new_samples == 3
    assert result.training_scheduled is False


def test_disabled_automatic_retraining_never_reserves_a_run(
    db_session: Session,
    tmp_path: Path,
) -> None:
    add_verified_samples(db_session, 10)

    result = reserve(
        db_session,
        continuous_config(tmp_path, enabled=False),
    )

    assert result.eligible is False
    assert result.training_scheduled is False
    assert "disabled" in (result.reason or "")


def test_threshold_reserves_once_and_snapshots_complete_verified_dataset(
    db_session: Session,
    tmp_path: Path,
) -> None:
    historical = add_verified_samples(db_session, 6, used=True)
    new = add_verified_samples(db_session, 4)
    config = continuous_config(tmp_path, threshold=4, minimum=10)

    first = reserve(db_session, config)
    second = reserve(db_session, config)

    assert first.eligible is True
    assert first.training_scheduled is True
    assert second.eligible is True
    assert second.training_scheduled is False
    run = db_session.get(TrainingRun, first.training_run_id)
    assert run is not None
    assert run.status == "queued"
    assert run.trigger_type == "automatic_threshold"
    assert run.trigger_sample_ids == [sample.id for sample in new]
    assert run.data_selection == [sample.id for sample in historical + new]
    assert run.training_sample_count == 10


def test_running_training_prevents_duplicate_reservation(
    db_session: Session,
    tmp_path: Path,
) -> None:
    add_verified_samples(db_session, 10)
    config = continuous_config(tmp_path)
    first = reserve(db_session, config)
    SqlAlchemyTrainingRepository(db_session).mark_run_running(
        first.training_run_id
    )

    second = reserve(db_session, config)

    assert second.eligible is True
    assert second.training_scheduled is False
    assert "already" in (second.reason or "")
    assert db_session.scalar(select(func.count(TrainingRun.id))) == 2


class CountingLoader(ModelLoader):
    def __init__(self) -> None:
        super().__init__()
        self.invalidation_count = 0

    def invalidate_cache(self) -> None:
        self.invalidation_count += 1
        super().invalidate_cache()


def test_candidate_trains_promotes_checkpoints_and_reloads_cache(
    db_session: Session,
    tmp_path: Path,
) -> None:
    samples = add_verified_samples(db_session, 20)
    config = continuous_config(tmp_path / "models", minimum=20)
    check = reserve(db_session, config)
    loader = CountingLoader()

    result = RetrainingService(
        session_factory=make_factory(db_session),
        config=config,
        loader=loader,
    ).execute(check.training_run_id)

    assert result.status == "promoted"
    assert loader.invalidation_count == 1
    db_session.expire_all()
    run = db_session.get(TrainingRun, check.training_run_id)
    assert run is not None
    assert run.is_active is True
    assert run.status == "promoted"
    assert run.completed_at is not None
    assert run.promoted_at is not None
    assert run.model_path is not None and Path(run.model_path).is_file()
    assert run.evaluation_sample_ids
    stored_samples = list(db_session.scalars(select(Sample)))
    assert all(sample.used_for_training for sample in stored_samples)
    assert all(
        sample.last_triggered_training_run_id == run.id
        for sample in stored_samples
    )
    loaded = loader.load(
        SqlAlchemyTrainingRepository(db_session),
        model_dir=config.model_dir,
    )
    assert loaded.descriptor.model_version == run.model_version
    assert run.data_selection == [sample.id for sample in samples]


def test_active_and_candidate_use_the_same_evaluation_rows(
    db_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = add_verified_samples(db_session, 20)
    create_active_model(db_session, tmp_path / "models", samples, poor=True)
    config = continuous_config(tmp_path / "models", minimum=20)
    check = reserve(db_session, config)
    captured: dict[str, np.ndarray] = {}
    original_train = retraining_module.train_candidate_on_split
    original_evaluate = retraining_module.evaluate_model

    def capture_candidate(split, training_config):
        captured["candidate"] = split.x_evaluation.copy()
        return original_train(split, training_config)

    def capture_active(pipeline, features, labels):
        captured["active"] = features.copy()
        return original_evaluate(pipeline, features, labels)

    monkeypatch.setattr(
        retraining_module, "train_candidate_on_split", capture_candidate
    )
    monkeypatch.setattr(retraining_module, "evaluate_model", capture_active)

    result = RetrainingService(
        session_factory=make_factory(db_session),
        config=config,
        loader=ModelLoader(),
    ).execute(check.training_run_id)

    assert result.status == "promoted"
    np.testing.assert_array_equal(captured["active"], captured["candidate"])
    db_session.expire_all()
    active_count = db_session.scalar(
        select(func.count(TrainingRun.id)).where(
            TrainingRun.is_active.is_(True)
        )
    )
    assert active_count == 1


def test_candidate_rejection_keeps_active_model_and_consumes_trigger_batch(
    db_session: Session,
    tmp_path: Path,
) -> None:
    samples = add_verified_samples(db_session, 20)
    active = create_active_model(
        db_session, tmp_path / "models", samples, poor=False
    )
    config = continuous_config(
        tmp_path / "models", minimum=20, improvement=0.1
    )
    check = reserve(db_session, config)

    result = RetrainingService(
        session_factory=make_factory(db_session),
        config=config,
        loader=ModelLoader(),
    ).execute(check.training_run_id)

    assert result.status == "rejected"
    db_session.expire_all()
    assert db_session.get(TrainingRun, active.id).is_active is True
    candidate = db_session.get(TrainingRun, check.training_run_id)
    assert candidate is not None
    assert candidate.is_active is False
    assert candidate.rejection_reason
    assert candidate.model_path is not None
    assert Path(candidate.model_path).is_file()
    assert SqlAlchemyTrainingRepository(
        db_session
    ).count_unused_verified_samples() == 0


def test_minimum_quality_rejection_is_explicit() -> None:
    from app.ml.evaluator import EvaluationMetrics

    candidate = EvaluationMetrics(0.5, 0.5, 0.5, 0.5, 0.5, [[1, 1], [1, 1]])
    decision = decide_promotion(
        candidate=candidate,
        active=None,
        primary_metric="f1_score",
        minimum_improvement=0.01,
        minimum_acceptable_f1=0.6,
    )

    assert decision.promote is False
    assert "below" in decision.reason


def test_unavailable_primary_metric_is_rejected() -> None:
    from app.ml.evaluator import EvaluationMetrics

    candidate = EvaluationMetrics(0.9, 0.9, 0.9, 0.9, None, [[2, 0], [0, 2]])
    decision = decide_promotion(
        candidate=candidate,
        active=None,
        primary_metric="roc_auc",
        minimum_improvement=0.01,
        minimum_acceptable_f1=0.6,
    )

    assert decision.promote is False
    assert "unavailable" in decision.reason


def test_failed_training_keeps_active_and_trigger_data_retryable(
    db_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = add_verified_samples(db_session, 20)
    active = create_active_model(
        db_session, tmp_path / "models", samples, poor=False
    )
    config = continuous_config(tmp_path / "models", minimum=20)
    check = reserve(db_session, config)

    def fail_training(*_: object, **__: object):
        raise RuntimeError("simulated fit failure")

    monkeypatch.setattr(
        retraining_module, "train_candidate_on_split", fail_training
    )
    result = RetrainingService(
        session_factory=make_factory(db_session),
        config=config,
        loader=ModelLoader(),
    ).execute(check.training_run_id)

    assert result.status == "failed"
    db_session.expire_all()
    assert db_session.get(TrainingRun, active.id).is_active is True
    failed = db_session.get(TrainingRun, check.training_run_id)
    assert failed is not None
    assert failed.error_message and "simulated fit failure" in failed.error_message
    assert all(not sample.used_for_training for sample in samples)
    assert all(sample.last_triggered_training_run_id is None for sample in samples)
    retry = reserve(db_session, config)
    assert retry.training_scheduled is True


def test_feedback_arriving_after_reservation_remains_new_for_next_cycle(
    db_session: Session,
    tmp_path: Path,
) -> None:
    reserved_samples = add_verified_samples(db_session, 20)
    config = continuous_config(tmp_path / "models", minimum=20)
    check = reserve(db_session, config)
    later = add_verified_samples(db_session, 1)

    RetrainingService(
        session_factory=make_factory(db_session),
        config=config,
        loader=ModelLoader(),
    ).execute(check.training_run_id)

    db_session.expire_all()
    assert all(sample.used_for_training for sample in reserved_samples)
    later_sample = db_session.get(Sample, later[0].id)
    assert later_sample is not None
    assert later_sample.used_for_training is False
    assert later_sample.last_triggered_training_run_id is None


@pytest.fixture
async def training_client(
    db_session: Session,
    tmp_path: Path,
) -> AsyncIterator[tuple[AsyncClient, list[int]]]:
    config = continuous_config(tmp_path / "models", threshold=5, minimum=10)
    coordinator = ContinuousTrainingCoordinator(
        SqlAlchemyTrainingRepository(db_session), config
    )
    scheduled: list[int] = []

    class FakeTaskResult:
        def __init__(self, task_id: str) -> None:
            self.id = task_id

    class FakeTaskSender:
        def apply_async(self, *, args: list[int], queue: str):
            scheduled.append(args[0])
            return FakeTaskResult(f"task-{args[0]}")

    class FakeInfrastructure:
        def check(self) -> WorkerHealth:
            return WorkerHealth(True, True)

    async def override_get_db() -> AsyncIterator[Session]:
        yield db_session

    async def override_coordinator() -> ContinuousTrainingCoordinator:
        return coordinator

    async def override_config() -> ContinuousTrainingConfig:
        return config

    async def override_dispatcher() -> TrainingDispatcher:
        return TrainingDispatcher(
            repository=SqlAlchemyTrainingRepository(db_session),
            task_sender=FakeTaskSender(),
            queue_name="training",
        )

    async def override_infrastructure() -> FakeInfrastructure:
        return FakeInfrastructure()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[
        get_continuous_training_coordinator
    ] = override_coordinator
    app.dependency_overrides[get_continuous_training_config] = override_config
    app.dependency_overrides[get_training_dispatcher] = override_dispatcher
    app.dependency_overrides[
        get_infrastructure_health_service
    ] = override_infrastructure
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            yield client, scheduled
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_training_check_status_and_history_endpoints(
    training_client: tuple[AsyncClient, list[int]],
    db_session: Session,
) -> None:
    client, scheduled = training_client
    add_verified_samples(db_session, 10)

    check = await client.post("/api/training/check")
    status = await client.get("/api/training/status")
    history = await client.get("/api/training/runs?skip=0&limit=1")
    detail = await client.get(
        f"/api/training/runs/{check.json()['training_run_id']}"
    )
    worker_health = await client.get("/api/system/worker-health")

    assert check.status_code == 202
    assert check.json()["eligible"] is True
    assert check.json()["training_scheduled"] is True
    assert scheduled == [check.json()["training_run_id"]]
    assert check.json()["task_id"] == f"task-{check.json()['training_run_id']}"
    assert status.status_code == 200
    assert status.json()["training_in_progress"] is True
    assert status.json()["new_verified_samples"] == 10
    assert status.json()["queued_jobs"] == 1
    assert status.json()["redis_available"] is True
    assert status.json()["celery_worker_available"] is True
    assert status.json()["current_training_run"]["progress_stage"] == "queued"
    assert history.status_code == 200
    assert len(history.json()) == 1
    assert history.json()[0]["status"] == "queued"
    assert "model_path" not in history.json()[0]
    assert "artifact_checksum" not in history.json()[0]
    assert detail.status_code == 200
    assert detail.json()["task_id"] == check.json()["task_id"]
    assert worker_health.json() == {
        "redis_available": True,
        "celery_worker_available": True,
    }


@pytest.mark.anyio
async def test_training_check_does_not_force_below_threshold(
    training_client: tuple[AsyncClient, list[int]],
    db_session: Session,
) -> None:
    client, scheduled = training_client
    add_verified_samples(db_session, 5, used=True)
    add_verified_samples(db_session, 4)

    response = await client.post("/api/training/check")

    assert response.status_code == 200
    assert response.json()["eligible"] is False
    assert response.json()["new_samples"] == 4
    assert response.json()["training_scheduled"] is False
    assert scheduled == []


@pytest.mark.anyio
async def test_committed_feedback_schedules_threshold_check_once(
    training_client: tuple[AsyncClient, list[int]],
    db_session: Session,
) -> None:
    client, scheduled = training_client
    add_verified_samples(db_session, 5, used=True)
    add_verified_samples(db_session, 4)
    source = ensure_source_run(db_session)
    pending = Prediction(
        feature_1=1.0,
        feature_2=1.0,
        feature_3=1.0,
        predicted_class=1,
        prediction_probability=0.8,
        model_version=source.model_version,
    )
    db_session.add(pending)
    db_session.commit()

    response = await client.patch(
        f"/api/predictions/{pending.id}/feedback",
        json={"actual_label": 1},
    )

    assert response.status_code == 200
    assert len(scheduled) == 1
    run = db_session.get(TrainingRun, scheduled[0])
    assert run is not None
    assert run.trigger_new_sample_count == 5


@pytest.mark.anyio
async def test_feedback_persists_when_celery_dispatch_is_unavailable(
    training_client: tuple[AsyncClient, list[int]],
    db_session: Session,
) -> None:
    client, _ = training_client
    add_verified_samples(db_session, 5, used=True)
    add_verified_samples(db_session, 4)
    source = ensure_source_run(db_session)
    pending = Prediction(
        feature_1=1.0,
        feature_2=1.0,
        feature_3=1.0,
        predicted_class=1,
        prediction_probability=0.8,
        model_version=source.model_version,
    )
    db_session.add(pending)
    db_session.commit()

    class UnavailableSender:
        def apply_async(self, *, args: list[int], queue: str):
            raise ConnectionError("Redis unavailable")

    async def unavailable_dispatcher() -> TrainingDispatcher:
        return TrainingDispatcher(
            repository=SqlAlchemyTrainingRepository(db_session),
            task_sender=UnavailableSender(),
            queue_name="training",
        )

    app.dependency_overrides[
        get_training_dispatcher
    ] = unavailable_dispatcher
    response = await client.patch(
        f"/api/predictions/{pending.id}/feedback",
        json={"actual_label": 1},
    )

    assert response.status_code == 200
    db_session.refresh(pending)
    assert pending.feedback_received is True
    run = SqlAlchemyTrainingRepository(db_session).get_current_run()
    assert run is not None and run.status == "queued"
    assert "Redis unavailable" in (run.dispatch_error or "")


@pytest.mark.anyio
async def test_reconciliation_redispatches_only_recoverable_queued_runs(
    training_client: tuple[AsyncClient, list[int]],
    db_session: Session,
) -> None:
    client, scheduled = training_client
    add_verified_samples(db_session, 10)
    checked = await client.post("/api/training/check")
    run = db_session.get(TrainingRun, checked.json()["training_run_id"])
    assert run is not None
    run.celery_task_id = None
    run.dispatch_error = "Simulated publish gap"
    db_session.commit()

    response = await client.post("/api/training/reconcile")

    assert response.status_code == 200
    assert response.json()["recovered_count"] == 1
    assert scheduled == [run.id, run.id]


@pytest.mark.anyio
async def test_next_prediction_uses_newly_promoted_model(
    db_session: Session,
    tmp_path: Path,
) -> None:
    add_verified_samples(db_session, 20)
    config = continuous_config(tmp_path / "models", minimum=20)
    check = reserve(db_session, config)
    loader = ModelLoader()
    result = RetrainingService(
        session_factory=make_factory(db_session),
        config=config,
        loader=loader,
    ).execute(check.training_run_id)
    factory = make_factory(db_session)

    async def override_get_db() -> AsyncIterator[Session]:
        with factory() as session:
            yield session

    async def override_loader() -> ModelLoader:
        return loader

    async def override_model_dir() -> Path:
        return config.model_dir

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_model_loader] = override_loader
    app.dependency_overrides[get_model_directory] = override_model_dir
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/predict",
                json={
                    "feature_1": 1.0,
                    "feature_2": 1.0,
                    "feature_3": 1.0,
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    assert response.json()["model_version"] == result.model_version


@pytest.mark.anyio
async def test_api_cache_detects_promotion_from_separate_worker_process(
    db_session: Session,
    tmp_path: Path,
) -> None:
    samples = add_verified_samples(db_session, 20)
    model_dir = tmp_path / "models"
    create_active_model(db_session, model_dir, samples, poor=True)
    api_loader = ModelLoader()
    api_loader.load(
        SqlAlchemyTrainingRepository(db_session), model_dir=model_dir
    )
    assert api_loader.loaded_model_version == "model_v1"

    config = continuous_config(model_dir, minimum=20)
    check = reserve(db_session, config)
    worker_loader = ModelLoader()
    result = RetrainingService(
        session_factory=make_factory(db_session),
        config=config,
        loader=worker_loader,
    ).execute(check.training_run_id)
    assert result.status == "promoted"
    assert api_loader.loaded_model_version == "model_v1"

    factory = make_factory(db_session)

    async def override_get_db() -> AsyncIterator[Session]:
        with factory() as session:
            yield session

    async def override_loader() -> ModelLoader:
        return api_loader

    async def override_model_dir() -> Path:
        return model_dir

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_model_loader] = override_loader
    app.dependency_overrides[get_model_directory] = override_model_dir
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/predict",
                json={
                    "feature_1": 1.0,
                    "feature_2": 1.0,
                    "feature_3": 1.0,
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    assert response.json()["model_version"] == result.model_version
    assert api_loader.loaded_model_version == result.model_version
