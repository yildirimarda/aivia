"""API security regression tests: auth, body limits and rate limits.

Enumerates every route and asserts that protected routes fail closed without an
API key, public routes stay open, oversized bodies are rejected with 413, and
clients exceeding the rate limit receive 429.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from aivia.api import create_app
from aivia.core.config import ApiConfig
from aivia.core.db import create_database_engine, upgrade_database

KEY = "test-api-key"
AUTH = {"X-API-Key": KEY}

# Routes that must remain reachable without an API key.
PUBLIC_ROUTES = [
    ("GET", "/health"),
    ("GET", "/openapi.json"),
    ("GET", "/docs"),
]

# Every route that is not public. Auth middleware runs before routing, so these
# all return 401 without a key regardless of whether their handler would touch
# the database or a media toolkit.
PROTECTED_ROUTES = [
    ("GET", "/api/v1/ping"),
    ("POST", "/api/v1/uploads"),
    ("POST", "/api/v1/jobs"),
    ("GET", "/api/v1/jobs/{job_id}"),
    ("GET", "/api/v1/jobs/{job_id}/status"),
    ("POST", "/api/v1/runs"),
    ("GET", "/api/v1/runs/{run_id}"),
    ("GET", "/api/v1/runs/{run_id}/costs"),
    ("POST", "/api/v1/media/jobs"),
    ("POST", "/api/v1/publish/jobs"),
    ("GET", "/api/v1/media/jobs/{job_id}"),
    ("GET", "/api/v1/media/jobs/{job_id}/status"),
    ("GET", "/api/v1/media/{job_id}"),
    ("GET", "/api/v1/publish/jobs/{job_id}"),
    ("GET", "/api/v1/publish/jobs/{job_id}/status"),
    ("GET", "/api/v1/publish/jobs/{job_id}/result"),
    ("GET", "/api/v1/media/jobs/{job_id}/result"),
    ("GET", "/api/v1/media/jobs/{job_id}/download"),
    ("GET", "/api/v1/media/jobs/{job_id}/download/{index}"),
    ("GET", "/api/v1/media/jobs/{job_id}/result/{index}"),
    ("GET", "/api/v1/media/jobs/{job_id}/files/{index}"),
    ("POST", "/api/v1/orders"),
    ("GET", "/api/v1/orders"),
    ("GET", "/api/v1/orders/{order_id}"),
    ("POST", "/api/v1/sessions"),
    ("POST", "/api/v1/sessions/{session_id}/message"),
    ("GET", "/api/v1/sessions/{session_id}/preview"),
    ("GET", "/api/v1/sessions/{session_id}/history"),
    ("GET", "/api/v1/sessions/{session_id}/assets"),
    ("GET", "/api/v1/sessions/{session_id}/assets/{asset_label}"),
]


def _fill(path: str) -> str:
    """Substitute path placeholders with dummy values."""
    return path.format(
        job_id="job_0000000000000001",
        run_id="run_0000000000000001",
        order_id="order_0000000000000001",
        session_id="session_0000000000000001",
        asset_label="cover",
        kind="image",
        index=0,
    )


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    created = create_database_engine(path=tmp_path / "aivia.db")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


def make_client(engine: Engine, **overrides: object) -> TestClient:
    values: dict[str, object] = {
        "require_auth": True,
        "api_key_ref": "AIVA_API_KEY",
        "max_body_bytes": 10_000,
        "rate_limit_per_minute": 0,
    }
    values.update(overrides)
    config = ApiConfig.model_validate(values)
    return TestClient(
        create_app(config, engine, environ={"AIVA_API_KEY": KEY}),
    )


@pytest.fixture
def client(engine: Engine) -> TestClient:
    return make_client(engine)


@pytest.mark.parametrize("method,path", PUBLIC_ROUTES)
def test_public_route_is_accessible_without_api_key(
    client: TestClient, method: str, path: str
) -> None:
    response = client.request(method, path)

    assert response.status_code == 200


@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_protected_route_returns_401_without_api_key(
    client: TestClient, method: str, path: str
) -> None:
    response = client.request(method, _fill(path))

    assert response.status_code == 401
    assert response.json()["detail"] == "invalid or missing API key"


def test_valid_api_key_passes_auth(client: TestClient) -> None:
    assert client.get("/api/v1/ping", headers=AUTH).status_code == 200
    assert (
        client.post("/api/v1/sessions", json={"title": "s"}, headers=AUTH).status_code
        == 201
    )
    assert (
        client.post(
            "/api/v1/jobs", json={"kind": "run", "payload": {}}, headers=AUTH
        ).status_code
        == 202
    )
    assert client.get("/api/v1/orders", headers=AUTH).status_code == 200
    # An unknown route is handled by the router (404), not rejected as unauthenticated.
    assert client.get("/api/v1/jobs/does-not-exist", headers=AUTH).status_code == 404


def test_wrong_api_key_is_rejected(client: TestClient) -> None:
    assert client.get("/api/v1/ping", headers={"X-API-Key": "wrong"}).status_code == 401
    assert (
        client.get(
            "/api/v1/ping", headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )


def test_bearer_token_is_accepted(client: TestClient) -> None:
    assert (
        client.get(
            "/api/v1/ping", headers={"Authorization": f"Bearer {KEY}"}
        ).status_code
        == 200
    )


@pytest.mark.parametrize("path", ["/api/v1/jobs", "/api/v1/uploads"])
def test_body_limit_rejects_oversized_body_on_protected_routes(
    engine: Engine, path: str
) -> None:
    client = make_client(engine, max_body_bytes=64)

    if path.endswith("uploads"):
        response = client.post(
            _fill(path),
            headers=AUTH,
            files={"file": ("big.txt", b"x" * 200, "text/plain")},
        )
    else:
        response = client.post(_fill(path), headers=AUTH, json={"payload": "x" * 200})

    assert response.status_code == 413
    assert response.json()["detail"] == "request body too large"


def test_body_limit_allows_small_body_with_auth(engine: Engine) -> None:
    client = make_client(engine, max_body_bytes=64)

    response = client.post(
        "/api/v1/jobs", json={"kind": "run", "payload": {}}, headers=AUTH
    )

    assert response.status_code != 413


def test_rate_limit_returns_429_and_exempts_health(engine: Engine) -> None:
    client = make_client(engine, rate_limit_per_minute=2)

    assert client.get("/api/v1/ping", headers=AUTH).status_code == 200
    assert client.get("/api/v1/ping", headers=AUTH).status_code == 200
    response = client.get("/api/v1/ping", headers=AUTH)

    assert response.status_code == 429
    assert response.json()["detail"] == "rate limit exceeded"
    assert int(response.headers["Retry-After"]) >= 1

    # /health stays public and is exempt from rate limiting.
    assert client.get("/health").status_code == 200
