from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.dependencies import (
    get_model_directory,
    get_model_loader,
    get_model_service,
    get_monitoring_service,
)
from app.database.session import get_db
from app.core.config import get_settings
from app.main import app
from app.ml.model_loader import ModelLoader
from app.ml.preprocessing import build_training_pipeline
from app.ml.registry import save_trained_model
from app.models.model_data_profile import ModelDataProfile
from app.models.model_event import ModelEvent
from app.models.monitoring_snapshot import MonitoringSnapshot
from app.models.prediction import Prediction
from app.models.training_run import TrainingRun
from app.monitoring.drift import calculate_psi, classify_psi, worst_status
from app.monitoring.performance import (
    calculate_verified_metrics,
    classify_performance_drop,
)
from app.monitoring.profiles import build_reference_profiles
from app.repositories.models import SqlAlchemyModelRepository
from app.repositories.monitoring import SqlAlchemyMonitoringRepository
from app.repositories.training import SqlAlchemyTrainingRepository
from app.services.models import ModelService
from app.services.monitoring import MonitoringService

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _register_model(
    session: Session,
    model_dir: Path,
    version: str,
    *,
    active: bool,
) -> TrainingRun:
    features = np.asarray(
        [[0, 0, 0], [0.2, 0.1, 0.2], [1, 1, 1], [1.2, 1.1, 1.2]],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    pipeline = build_training_pipeline(random_state=42)
    pipeline.fit(features, labels)
    artifact = save_trained_model(
        pipeline, model_version=version, model_dir=model_dir
    )
    run = TrainingRun(
        training_batch_id=f"batch_{version}",
        model_version=version,
        model_path=str(artifact.path),
        artifact_checksum=artifact.checksum,
        artifact_format=artifact.format,
        algorithm="LogisticRegression",
        parameters={"features": ["feature_1", "feature_2", "feature_3"]},
        training_sample_count=4,
        train_sample_count=4,
        test_sample_count=0,
        accuracy=1.0,
        precision=1.0,
        recall=1.0,
        f1_score=1.0,
        roc_auc=1.0,
        confusion_matrix=[[2, 0], [0, 2]],
        random_state=42,
        status="promoted" if active else "completed",
        is_active=active,
    )
    session.add(run)
    session.flush()
    session.add_all(
        [
            ModelDataProfile(model_version=version, **profile)
            for profile in build_reference_profiles(features)
        ]
    )
    session.commit()
    return run


def _add_predictions(session: Session, version: str) -> None:
    rows = []
    for index in range(5):
        rows.append(
            Prediction(
                feature_1=float(index),
                feature_2=float(index),
                feature_3=float(index),
                predicted_class=1,
                prediction_probability=0.9,
                model_version=version,
                actual_label=0 if index < 2 else None,
                feedback_received=index < 2,
            )
        )
    session.add_all(rows)
    session.commit()


@pytest.fixture
async def phase7_client(
    db_session: Session, tmp_path: Path
) -> AsyncIterator[tuple[AsyncClient, ModelLoader, Path]]:
    model_dir = tmp_path / "models"
    loader = ModelLoader()
    settings = get_settings().model_copy(
        update={
            "model_dir": model_dir,
            "drift_window_size": 5,
            "drift_min_samples": 5,
            "performance_window_size": 5,
            "performance_min_feedback_samples": 2,
        }
    )
    monitoring = MonitoringService(
        monitoring_repository=SqlAlchemyMonitoringRepository(db_session),
        training_repository=SqlAlchemyTrainingRepository(db_session),
        loader=loader,
        settings=settings,
        model_dir=model_dir,
    )
    models = ModelService(
        repository=SqlAlchemyModelRepository(db_session),
        loader=loader,
        model_dir=model_dir,
    )

    async def override_monitoring() -> MonitoringService:
        return monitoring

    async def override_models() -> ModelService:
        return models

    async def override_db() -> AsyncIterator[Session]:
        yield db_session

    async def override_loader() -> ModelLoader:
        return loader

    async def override_model_dir() -> Path:
        return model_dir

    app.dependency_overrides[get_monitoring_service] = override_monitoring
    app.dependency_overrides[get_model_service] = override_models
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_model_loader] = override_loader
    app.dependency_overrides[get_model_directory] = override_model_dir
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            yield client, loader, model_dir
    finally:
        app.dependency_overrides.clear()


def test_psi_is_stable_for_same_distribution_and_safe_with_empty_bins() -> None:
    profile_data = build_reference_profiles(
        np.asarray([[0, 0, 0], [0, 1, 1], [1, 2, 2], [1, 3, 3]], dtype=float)
    )[0]
    profile = ModelDataProfile(model_version="unused", **profile_data)

    same = calculate_psi(profile, [0, 0, 1, 1])
    shifted = calculate_psi(profile, [100, 100, 101, 101])

    assert same == pytest.approx(0.0)
    assert classify_psi(same, warning_threshold=0.1, critical_threshold=0.25) == "stable"
    assert classify_psi(shifted, warning_threshold=0.1, critical_threshold=0.25) == "critical"
    assert np.isfinite(shifted)


@pytest.mark.parametrize(
    ("psi", "expected"),
    [(0.05, "stable"), (0.10, "warning"), (0.24, "warning"), (0.25, "critical")],
)
def test_psi_thresholds_and_overall_worst_status(
    psi: float, expected: str
) -> None:
    assert classify_psi(
        psi, warning_threshold=0.10, critical_threshold=0.25
    ) == expected
    assert worst_status(["stable", "warning", "critical"]) == "critical"
    assert worst_status(["insufficient_data", "warning"]) == "warning"
    assert worst_status(["insufficient_data"]) == "insufficient_data"


@pytest.mark.parametrize(
    ("current", "expected"),
    [(0.96, "stable"), (0.94, "warning"), (0.89, "critical")],
)
def test_performance_drop_thresholds(current: float, expected: str) -> None:
    status, drop = classify_performance_drop(
        1.0, current, warning_drop=0.05, critical_drop=0.10
    )
    assert status == expected
    assert drop == pytest.approx(1.0 - current)


def test_roc_auc_is_unavailable_without_both_actual_classes() -> None:
    records = [
        Prediction(
            feature_1=0,
            feature_2=0,
            feature_3=0,
            predicted_class=0,
            prediction_probability=0.1,
            model_version="model_v1",
            actual_label=0,
            feedback_received=True,
        )
        for _ in range(3)
    ]
    metrics = calculate_verified_metrics(records)

    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] is None


async def test_monitoring_gets_are_read_only_and_check_persists_snapshots(
    phase7_client: tuple[AsyncClient, ModelLoader, Path], db_session: Session
) -> None:
    client, _, model_dir = phase7_client
    _register_model(db_session, model_dir, "model_v1", active=True)
    _add_predictions(db_session, "model_v1")

    drift = await client.get("/api/monitoring/drift")
    performance = await client.get("/api/monitoring/performance")
    health = await client.get("/api/monitoring/health")

    assert drift.status_code == performance.status_code == health.status_code == 200
    assert drift.json()["sample_count"] == 5
    assert len(drift.json()["features"]) == 3
    assert performance.json()["verified_samples"] == 2
    assert performance.json()["feedback_coverage"] == pytest.approx(0.4)
    assert performance.json()["status"] == "critical"
    assert health.json()["status"] == "unhealthy"
    assert db_session.scalar(select(func.count(MonitoringSnapshot.id))) == 0

    checked = await client.post("/api/monitoring/check")

    assert checked.status_code == 200
    assert len(checked.json()["snapshots"]) == 3
    assert db_session.scalar(select(func.count(MonitoringSnapshot.id))) == 3


async def test_model_history_detail_compare_and_atomic_rollback(
    phase7_client: tuple[AsyncClient, ModelLoader, Path], db_session: Session
) -> None:
    client, loader, model_dir = phase7_client
    _register_model(db_session, model_dir, "model_v1", active=False)
    _register_model(db_session, model_dir, "model_v2", active=True)
    loader.load(SqlAlchemyTrainingRepository(db_session), model_dir=model_dir)

    history = await client.get("/api/models")
    detail = await client.get("/api/models/model_v1")
    comparison = await client.get(
        "/api/models/compare", params={"model_a": "model_v1", "model_b": "model_v2"}
    )
    rollback = await client.post(
        "/api/models/model_v1/rollback", json={"reason": "Known stable release"}
    )
    prediction = await client.post(
        "/api/predict",
        json={"feature_1": 0.1, "feature_2": 0.1, "feature_3": 0.1},
    )

    assert history.status_code == 200 and len(history.json()) == 2
    assert detail.status_code == 200 and len(detail.json()["data_profiles"]) == 3
    assert "model_path" not in detail.json()
    assert "artifact_checksum" not in detail.json()
    assert comparison.status_code == 200
    assert "different evaluation datasets" in comparison.json()["comparison_notice"]
    assert rollback.status_code == 200
    assert prediction.status_code == 201
    assert prediction.json()["model_version"] == "model_v1"
    assert rollback.json()["previous_model_version"] == "model_v2"
    assert db_session.scalar(
        select(TrainingRun.model_version).where(TrainingRun.is_active.is_(True))
    ) == "model_v1"
    event = db_session.scalar(select(ModelEvent).where(ModelEvent.event_type == "rollback"))
    assert event is not None and event.reason == "Known stable release"
    assert loader.loaded_model_version == "model_v1"
    assert (model_dir / "model_v1.joblib").exists()
    assert (model_dir / "model_v2.joblib").exists()


async def test_rollback_rejects_missing_artifact_without_changing_active_model(
    phase7_client: tuple[AsyncClient, ModelLoader, Path], db_session: Session
) -> None:
    client, _, model_dir = phase7_client
    target = _register_model(db_session, model_dir, "model_v1", active=False)
    _register_model(db_session, model_dir, "model_v2", active=True)
    Path(target.model_path or "").unlink()

    response = await client.post("/api/models/model_v1/rollback", json={})

    assert response.status_code == 409
    assert db_session.scalar(
        select(TrainingRun.model_version).where(TrainingRun.is_active.is_(True))
    ) == "model_v2"
    assert db_session.scalar(select(func.count(ModelEvent.id))) == 0


async def test_missing_model_and_corrupt_rollback_fail_safely(
    phase7_client: tuple[AsyncClient, ModelLoader, Path], db_session: Session
) -> None:
    client, _, model_dir = phase7_client
    _register_model(db_session, model_dir, "model_v1", active=False)
    active = _register_model(db_session, model_dir, "model_v2", active=True)
    Path(active.model_path or "").write_bytes(b"corrupt")

    missing = await client.post("/api/models/model_v404/rollback", json={})
    corrupt_health = await client.get("/api/monitoring/health")

    assert missing.status_code == 404
    assert corrupt_health.status_code == 200
    assert corrupt_health.json()["model_status"] == "unhealthy"


async def test_corrupt_and_incompatible_targets_cannot_be_activated(
    phase7_client: tuple[AsyncClient, ModelLoader, Path], db_session: Session
) -> None:
    client, _, model_dir = phase7_client
    corrupt = _register_model(db_session, model_dir, "model_v1", active=False)
    _register_model(db_session, model_dir, "model_v2", active=True)
    Path(corrupt.model_path or "").write_bytes(b"not a joblib artifact")

    corrupt_response = await client.post(
        "/api/models/model_v1/rollback", json={}
    )

    incompatible_pipeline = build_training_pipeline(random_state=42)
    incompatible_pipeline.fit(
        np.asarray([[0, 0], [0.1, 0.1], [1, 1], [1.1, 1.1]]),
        np.asarray([0, 0, 1, 1]),
    )
    artifact = save_trained_model(
        incompatible_pipeline,
        model_version="model_bad",
        model_dir=model_dir,
    )
    db_session.add(
        TrainingRun(
            training_batch_id="batch_bad",
            model_version="model_bad",
            model_path=str(artifact.path),
            artifact_checksum=artifact.checksum,
            artifact_format=artifact.format,
            algorithm="LogisticRegression",
            training_sample_count=4,
            random_state=42,
            status="completed",
        )
    )
    db_session.commit()
    incompatible_response = await client.post(
        "/api/models/model_bad/rollback", json={}
    )

    assert corrupt_response.status_code == 409
    assert incompatible_response.status_code == 409
    assert db_session.scalar(
        select(TrainingRun.model_version).where(TrainingRun.is_active.is_(True))
    ) == "model_v2"
