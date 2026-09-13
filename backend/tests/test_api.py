from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.orm import Session, sessionmaker

from app.database.base import Base
from app.database.session import create_db_engine, get_db
from app.main import app
from app.models.sample import Sample  # noqa: F401

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[AsyncClient]:
    database_path = tmp_path / "test.db"
    test_engine = create_db_engine(f"sqlite:///{database_path}")
    test_session_factory = sessionmaker(
        bind=test_engine,
        autoflush=False,
        expire_on_commit=False,
    )
    Base.metadata.create_all(bind=test_engine)

    async def override_get_db() -> AsyncIterator[Session]:
        with test_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        test_engine.dispose()


def sample_payload(index: int = 1) -> dict[str, int | float]:
    return {
        "feature_1": index + 0.1,
        "feature_2": index + 0.2,
        "feature_3": index + 0.3,
        "label": index,
    }


async def create_sample(
    client: AsyncClient,
    index: int = 1,
) -> dict[str, Any]:
    response = await client.post("/api/data", json=sample_payload(index))
    assert response.status_code == 201
    return response.json()


async def test_health_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_create_record(client: AsyncClient) -> None:
    payload = sample_payload()
    response = await client.post("/api/data", json=payload)

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == 1
    assert {key: body[key] for key in payload} == payload
    assert datetime.fromisoformat(body["created_at"]).tzinfo is not None
    assert datetime.fromisoformat(body["updated_at"]).tzinfo is not None
    assert body["used_for_training"] is False
    assert body["training_batch_id"] is None
    assert body["model_version"] is None


async def test_retrieve_stored_records(client: AsyncClient) -> None:
    first = await create_sample(client, 1)
    second = await create_sample(client, 2)

    response = await client.get("/api/data")

    assert response.status_code == 200
    assert [record["id"] for record in response.json()] == [
        first["id"],
        second["id"],
    ]


async def test_retrieve_record_by_id(client: AsyncClient) -> None:
    created = await create_sample(client)

    response = await client.get(f"/api/data/{created['id']}")

    assert response.status_code == 200
    assert response.json() == created


async def test_nonexistent_record_returns_404(client: AsyncClient) -> None:
    response = await client.get("/api/data/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Data record not found."}


async def test_invalid_data_returns_422(client: AsyncClient) -> None:
    response = await client.post(
        "/api/data",
        json={
            "feature_1": "not-a-number",
            "feature_2": 2.5,
            "feature_3": 3.75,
        },
    )

    assert response.status_code == 422


async def test_list_pagination(client: AsyncClient) -> None:
    for index in range(1, 6):
        await create_sample(client, index)

    response = await client.get("/api/data", params={"skip": 1, "limit": 2})

    assert response.status_code == 200
    assert [record["id"] for record in response.json()] == [2, 3]

    excessive_limit = await client.get("/api/data", params={"limit": 101})
    assert excessive_limit.status_code == 422


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:5173",
        "https://example-codespace-5173.app.github.dev",
    ],
)
async def test_cors_allows_frontend_origin(
    client: AsyncClient,
    origin: str,
) -> None:
    response = await client.options(
        "/api/data",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
