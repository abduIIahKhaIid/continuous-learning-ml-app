import json
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sqlalchemy import select
from sqlalchemy.orm import Session

import app.ml.model_loader as model_loader_module
from app.api.dependencies import get_model_directory, get_model_loader
from app.database.session import get_db
from app.main import app
from app.ml.model_loader import ModelLoader
from app.ml.preprocessing import build_training_pipeline
from app.ml.registry import save_trained_model
from app.models.prediction import Prediction
from app.models.training_run import TrainingRun

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def prediction_client(
    db_session: Session,
    tmp_path: Path,
) -> AsyncIterator[tuple[AsyncClient, ModelLoader, Path]]:
    model_dir = tmp_path / "models"
    loader = ModelLoader()

    async def override_get_db() -> AsyncIterator[Session]:
        yield db_session

    async def override_model_loader() -> ModelLoader:
        return loader

    async def override_model_directory() -> Path:
        return model_dir

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_model_loader] = override_model_loader
    app.dependency_overrides[get_model_directory] = override_model_directory
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            yield client, loader, model_dir
    finally:
        app.dependency_overrides.clear()


def fitted_pipeline(*, probability: bool = True) -> Pipeline:
    features = np.asarray(
        [
            [0.0, 0.1, 0.2],
            [0.2, 0.0, 0.1],
            [1.0, 1.1, 0.9],
            [1.2, 0.8, 1.1],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    if probability:
        pipeline = build_training_pipeline(random_state=42)
    else:
        pipeline = Pipeline([("classifier", LinearSVC(random_state=42))])
    pipeline.fit(features, labels)
    return pipeline


def register_completed_model(
    session: Session,
    model_dir: Path,
    *,
    version: str = "model_v1",
    is_active: bool = False,
    probability: bool = True,
) -> TrainingRun:
    artifact = save_trained_model(
        fitted_pipeline(probability=probability),
        model_version=version,
        model_dir=model_dir,
    )
    run = TrainingRun(
        training_batch_id=f"batch_{version}",
        model_version=version,
        model_path=str(artifact.path),
        artifact_checksum=artifact.checksum,
        artifact_format=artifact.format,
        algorithm=("LogisticRegression" if probability else "LinearSVC"),
        parameters={},
        data_selection=[],
        training_sample_count=4,
        train_sample_count=4,
        test_sample_count=0,
        accuracy=1.0,
        precision=1.0,
        recall=1.0,
        f1_score=1.0,
        roc_auc=1.0 if probability else None,
        confusion_matrix=[[2, 0], [0, 2]],
        random_state=42,
        status="completed",
        is_active=is_active,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def register_missing_model(session: Session, model_dir: Path) -> None:
    run = TrainingRun(
        training_batch_id="batch_missing",
        model_version="model_missing",
        model_path=str(model_dir / "model_missing.joblib"),
        artifact_checksum="0" * 64,
        artifact_format="joblib",
        algorithm="LogisticRegression",
        training_sample_count=4,
        random_state=42,
        status="completed",
    )
    session.add(run)
    session.commit()


async def make_prediction(client: AsyncClient, value: float = 1.0):
    return await client.post(
        "/api/predict",
        json={
            "feature_1": value,
            "feature_2": value + 0.1,
            "feature_3": value - 0.1,
        },
    )


async def test_successful_prediction_returns_version_probability_and_stores_record(
    prediction_client: tuple[AsyncClient, ModelLoader, Path],
    db_session: Session,
) -> None:
    client, _, model_dir = prediction_client
    register_completed_model(db_session, model_dir)

    response = await make_prediction(client)

    assert response.status_code == 201
    body = response.json()
    assert body["prediction_id"] == 1
    assert body["prediction"] == body["predicted_class"]
    assert body["predicted_class"] in {0, 1}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] == "model_v1"
    assert body["created_at"].endswith("Z")

    stored = db_session.scalar(select(Prediction))
    assert stored is not None
    assert stored.id == body["prediction_id"]
    assert stored.model_version == body["model_version"]
    assert stored.actual_label is None
    assert stored.feedback_received is False


async def test_active_model_wins_over_newer_completed_fallback(
    prediction_client: tuple[AsyncClient, ModelLoader, Path],
    db_session: Session,
) -> None:
    client, _, model_dir = prediction_client
    register_completed_model(
        db_session, model_dir, version="model_v1", is_active=True
    )
    register_completed_model(db_session, model_dir, version="model_v2")

    response = await make_prediction(client)

    assert response.status_code == 201
    assert response.json()["model_version"] == "model_v1"


async def test_latest_completed_model_is_safe_fallback(
    prediction_client: tuple[AsyncClient, ModelLoader, Path],
    db_session: Session,
) -> None:
    client, _, model_dir = prediction_client
    register_completed_model(db_session, model_dir, version="model_v1")
    register_completed_model(db_session, model_dir, version="model_v2")

    response = await make_prediction(client)

    assert response.status_code == 201
    assert response.json()["model_version"] == "model_v2"


async def test_probability_is_null_when_estimator_has_no_predict_proba(
    prediction_client: tuple[AsyncClient, ModelLoader, Path],
    db_session: Session,
) -> None:
    client, _, model_dir = prediction_client
    register_completed_model(db_session, model_dir, probability=False)

    response = await make_prediction(client)

    assert response.status_code == 201
    assert response.json()["probability"] is None


async def test_no_model_available_returns_clean_service_error(
    prediction_client: tuple[AsyncClient, ModelLoader, Path],
) -> None:
    client, _, _ = prediction_client

    response = await make_prediction(client)
    status_response = await client.get("/api/model/status")

    assert response.status_code == 503
    assert response.json() == {"detail": "No trained model available."}
    assert status_response.status_code == 200
    assert status_response.json() == {
        "model_available": False,
        "model_version": None,
        "algorithm": None,
        "created_at": None,
        "metrics": None,
        "status": "unavailable",
        "detail": "No trained model available.",
    }


async def test_missing_model_artifact_returns_clean_error(
    prediction_client: tuple[AsyncClient, ModelLoader, Path],
    db_session: Session,
) -> None:
    client, _, model_dir = prediction_client
    register_missing_model(db_session, model_dir)

    response = await make_prediction(client)
    status_response = await client.get("/api/model/status")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The trained model is currently unavailable."
    }
    assert status_response.status_code == 200
    assert status_response.json()["model_available"] is False
    assert status_response.json()["model_version"] == "model_missing"


async def test_corrupt_model_artifact_returns_clean_error(
    prediction_client: tuple[AsyncClient, ModelLoader, Path],
    db_session: Session,
) -> None:
    client, _, model_dir = prediction_client
    run = register_completed_model(db_session, model_dir)
    Path(run.model_path or "").write_bytes(b"corrupted")

    response = await make_prediction(client)

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The trained model is currently unavailable."
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"feature_1": 1, "feature_2": 2},
        {"feature_1": "bad", "feature_2": 2, "feature_3": 3},
        {"feature_1": "NaN", "feature_2": 2, "feature_3": 3},
        {"feature_1": "Infinity", "feature_2": 2, "feature_3": 3},
        {"feature_1": 1, "feature_2": 2, "feature_3": 3, "label": 1},
    ],
)
async def test_invalid_prediction_input_returns_422(
    prediction_client: tuple[AsyncClient, ModelLoader, Path],
    payload: dict[str, object],
) -> None:
    client, _, _ = prediction_client

    response = await client.post("/api/predict", json=payload)

    assert response.status_code == 422


async def test_prediction_history_pagination_and_get_by_id(
    prediction_client: tuple[AsyncClient, ModelLoader, Path],
    db_session: Session,
) -> None:
    client, _, model_dir = prediction_client
    register_completed_model(db_session, model_dir)
    first = (await make_prediction(client, 0.0)).json()
    second = (await make_prediction(client, 0.5)).json()
    third = (await make_prediction(client, 1.0)).json()

    history = await client.get(
        "/api/predictions", params={"skip": 1, "limit": 1}
    )
    by_id = await client.get(f"/api/predictions/{first['prediction_id']}")

    assert history.status_code == 200
    assert [item["id"] for item in history.json()] == [
        second["prediction_id"]
    ]
    assert by_id.status_code == 200
    assert by_id.json()["id"] == first["prediction_id"]
    assert by_id.json()["model_version"] == "model_v1"
    assert third["prediction_id"] == 3

    excessive_limit = await client.get(
        "/api/predictions", params={"limit": 101}
    )
    assert excessive_limit.status_code == 422


async def test_nonexistent_prediction_returns_404(
    prediction_client: tuple[AsyncClient, ModelLoader, Path],
) -> None:
    client, _, _ = prediction_client

    response = await client.get("/api/predictions/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Prediction not found."}


async def test_model_status_returns_metadata_without_artifact_path(
    prediction_client: tuple[AsyncClient, ModelLoader, Path],
    db_session: Session,
) -> None:
    client, _, model_dir = prediction_client
    register_completed_model(db_session, model_dir, is_active=True)

    response = await client.get("/api/model/status")

    assert response.status_code == 200
    body = response.json()
    assert body["model_available"] is True
    assert body["model_version"] == "model_v1"
    assert body["algorithm"] == "LogisticRegression"
    assert body["status"] == "ready"
    assert body["metrics"] == {
        "accuracy": 1.0,
        "f1_score": 1.0,
        "roc_auc": 1.0,
    }
    assert "model_path" not in json.dumps(body)


async def test_model_artifact_is_loaded_once_while_registry_version_is_unchanged(
    prediction_client: tuple[AsyncClient, ModelLoader, Path],
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, loader, model_dir = prediction_client
    register_completed_model(db_session, model_dir)
    original_load = model_loader_module.joblib.load
    load_count = 0

    def counting_load(path: Path):
        nonlocal load_count
        load_count += 1
        return original_load(path)

    monkeypatch.setattr(model_loader_module.joblib, "load", counting_load)

    assert (await make_prediction(client, 0.2)).status_code == 201
    assert (await make_prediction(client, 0.8)).status_code == 201

    assert load_count == 1
    assert loader.loaded_model_version == "model_v1"


async def test_loader_lazily_reloads_when_selected_version_changes(
    prediction_client: tuple[AsyncClient, ModelLoader, Path],
    db_session: Session,
) -> None:
    client, loader, model_dir = prediction_client
    register_completed_model(db_session, model_dir, version="model_v1")
    first = await make_prediction(client, 0.2)

    register_completed_model(db_session, model_dir, version="model_v2")
    second = await make_prediction(client, 0.8)

    assert first.status_code == 201
    assert first.json()["model_version"] == "model_v1"
    assert second.status_code == 201
    assert second.json()["model_version"] == "model_v2"
    assert loader.loaded_model_version == "model_v2"
