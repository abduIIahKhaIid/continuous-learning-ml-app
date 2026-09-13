from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as test_client:
        yield test_client


async def test_health_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_data_returns_validated_payload(client: AsyncClient) -> None:
    payload = {
        "feature_1": 1.25,
        "feature_2": 2.5,
        "feature_3": -3.75,
        "label": 1,
    }
    response = await client.post(
        "/api/data",
        json=payload,
    )

    assert response.status_code == 200
    assert response.json() == payload


async def test_data_accepts_omitted_label(client: AsyncClient) -> None:
    response = await client.post(
        "/api/data",
        json={"feature_1": 1, "feature_2": 2, "feature_3": 3},
    )

    assert response.status_code == 200
    assert response.json() == {
        "feature_1": 1.0,
        "feature_2": 2.0,
        "feature_3": 3.0,
        "label": None,
    }


async def test_data_rejects_invalid_payload(client: AsyncClient) -> None:
    response = await client.post(
        "/api/data",
        json={
            "feature_1": "not-a-number",
            "feature_2": 2.5,
            "feature_3": 3.75,
        },
    )

    assert response.status_code == 422


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
