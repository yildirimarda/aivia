"""Publish HTTP routes: submission, status and the dry run flag."""

from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from aivia.api import create_app
from aivia.core.config import ApiConfig, CapabilityChainConfig
from aivia.core.cost import CostTracker
from aivia.core.db import JobRepository, create_database_engine, upgrade_database
from aivia.core.router import Router
from aivia.media import MediaToolkit
from aivia.providers.mock import MockProvider
from aivia.providers.storage_local import LocalStorageProvider

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def provider() -> MockProvider:
    return MockProvider(name="mock")


@pytest.fixture
def toolkit(
    provider: MockProvider,
    engine: Engine,
    tmp_path: Path,
) -> MediaToolkit:
    router = Router(
        {provider.name: provider},
        config={
            "publish": CapabilityChainConfig(chain=[provider.name]),
        },
    )
    return MediaToolkit(
        router,
        tracker=CostTracker(engine),
        engine=engine,
        storage=LocalStorageProvider(tmp_path / "media"),
    )


@pytest.fixture
def client(engine: Engine, toolkit: MediaToolkit) -> TestClient:
    config = ApiConfig(require_auth=False)
    return TestClient(
        create_app(config, engine, environ={}, toolkit=toolkit),
    )


def publish_body(*, dry_run: bool = False) -> dict[str, Any]:
    return {
        "kind": "publish",
        "platform": "x",
        "account": "myaccount",
        "caption": "Hello world",
        "media": [
            {
                "data": base64.b64encode(PNG_1X1).decode("ascii"),
                "media_type": "image/png",
            }
        ],
        "dry_run": dry_run,
    }


def test_submitting_a_publish_job_runs_it_and_exposes_the_result(
    client: TestClient,
    provider: MockProvider,
) -> None:
    response = client.post("/api/v1/publish/jobs", json=publish_body())

    assert response.status_code == 202
    submitted = response.json()
    assert submitted["job_id"].startswith("job_")
    assert submitted["id"] == submitted["job_id"]
    assert submitted["kind"] == "publish"
    assert submitted["status"] == "pending"

    status = client.get(f"/api/v1/publish/jobs/{submitted['job_id']}")

    assert status.status_code == 200
    body = status.json()
    assert body["state"] == "succeeded"
    assert body["status"] == "succeeded"
    assert body["kind"] == "publish"
    assert body["dry_run"] is False
    assert body["remote_id"].startswith("mock-post-")
    assert body["url"].startswith("https://example.com/posts/")
    assert body["provider"] == "mock"
    assert body["platform"] == "x"
    assert body["account"] == "myaccount"
    assert body["downloads"] == []
    assert body["download_url"] is None
    assert len(provider.calls) == 1

    result = client.get(f"/api/v1/publish/jobs/{submitted['job_id']}/result")

    assert result.status_code == 200
    body = result.json()
    assert body["job_id"] == submitted["job_id"]
    assert body["kind"] == "publish"
    assert body["status"] == "succeeded"
    assert body["remote_id"] == status.json()["remote_id"]
    assert body["dry_run"] is False


def test_a_dry_run_publish_is_marked_and_has_no_remote_post(
    client: TestClient,
    provider: MockProvider,
) -> None:
    response = client.post("/api/v1/publish/jobs", json=publish_body(dry_run=True))

    assert response.status_code == 202
    submitted = response.json()
    assert submitted["kind"] == "publish"

    status = client.get(f"/api/v1/publish/jobs/{submitted['job_id']}").json()

    assert status["state"] == "succeeded"
    assert status["status"] == "succeeded"
    assert status["dry_run"] is True
    assert status["remote_id"] == ""
    assert status["url"] is None
    assert status["provider"] == "mock"

    result = client.get(f"/api/v1/publish/jobs/{submitted['job_id']}/result").json()
    assert result["dry_run"] is True
    assert result["remote_id"] == ""
    assert len(provider.calls) == 1


def test_a_data_url_media_item_is_decoded_before_publishing(
    client: TestClient,
    provider: MockProvider,
) -> None:
    data_url = "data:image/png;base64," + base64.b64encode(PNG_1X1).decode("ascii")
    response = client.post(
        "/api/v1/publish/jobs",
        json={
            "kind": "publish",
            "platform": "x",
            "account": "myaccount",
            "caption": "Hello world",
            "media": [{"data": data_url}],
            "dry_run": True,
        },
    )

    assert response.status_code == 202
    submitted = response.json()
    body = client.get(f"/api/v1/publish/jobs/{submitted['job_id']}").json()
    assert body["status"] == "succeeded"
    assert body["dry_run"] is True
    assert len(provider.calls) == 1


def test_an_invalid_publish_submission_is_rejected_before_it_is_queued(
    client: TestClient,
    engine: Engine,
) -> None:
    response = client.post(
        "/api/v1/publish/jobs",
        json={"account": "myaccount", "caption": "Hello world"},
    )

    assert response.status_code == 422
    assert JobRepository(engine).list(kind="publish") == []


def test_a_media_item_without_a_source_is_rejected(
    client: TestClient,
    engine: Engine,
) -> None:
    response = client.post(
        "/api/v1/publish/jobs",
        json={
            "kind": "publish",
            "platform": "x",
            "account": "myaccount",
            "caption": "Hello world",
            "media": [{"media_type": "image/png"}],
        },
    )

    assert response.status_code == 422
    assert JobRepository(engine).list(kind="publish") == []


def test_a_repeated_idempotency_key_returns_the_existing_job(
    client: TestClient,
    engine: Engine,
    provider: MockProvider,
) -> None:
    submission = publish_body(dry_run=True)
    submission["idempotency_key"] = "publish-request-42"

    first = client.post("/api/v1/publish/jobs", json=submission)
    second = client.post("/api/v1/publish/jobs", json=submission)

    assert first.status_code == 202
    assert second.status_code == 202
    assert second.json()["job_id"] == first.json()["job_id"]
    assert len(JobRepository(engine).list(kind="publish")) == 1
    assert len(provider.calls) == 1


def test_a_different_payload_with_the_same_key_conflicts(
    client: TestClient,
    engine: Engine,
) -> None:
    first = client.post(
        "/api/v1/publish/jobs",
        json={
            **publish_body(),
            "idempotency_key": "publish-request-42",
        },
    )
    assert first.status_code == 202

    response = client.post(
        "/api/v1/publish/jobs",
        json={
            **publish_body(),
            "caption": "A different caption",
            "idempotency_key": "publish-request-42",
        },
    )

    assert response.status_code == 409
    assert "different payload" in response.json()["detail"]
    assert len(JobRepository(engine).list(kind="publish")) == 1


def test_an_unknown_publish_job_is_not_found(client: TestClient) -> None:
    assert client.get("/api/v1/publish/jobs/job_missing").status_code == 404
    assert client.get("/api/v1/publish/jobs/job_missing/result").status_code == 404


def test_publish_routes_require_authentication(
    engine: Engine,
    toolkit: MediaToolkit,
) -> None:
    client = TestClient(
        create_app(
            ApiConfig(require_auth=True, api_key_ref="AIVA_API_KEY"),
            engine,
            environ={"AIVA_API_KEY": "test-api-key"},
            toolkit=toolkit,
        )
    )

    assert client.post("/api/v1/publish/jobs", json=publish_body()).status_code == 401
    authorized = client.post(
        "/api/v1/publish/jobs",
        json=publish_body(),
        headers={"X-API-Key": "test-api-key"},
    )
    assert authorized.status_code == 202


def test_a_publish_job_result_is_publish_request_shaped(
    client: TestClient,
) -> None:
    submitted = client.post("/api/v1/publish/jobs", json=publish_body()).json()

    result = client.get(f"/api/v1/publish/jobs/{submitted['job_id']}/result").json()

    assert result["job_id"] == submitted["job_id"]
    assert result["kind"] == "publish"
    assert result["status"] == "succeeded"
    assert result["remote_id"].startswith("mock-post-")
    assert result["url"].startswith("https://example.com/posts/")
    assert result["provider"] == "mock"
    assert result["dry_run"] is False
    assert isinstance(result["metadata"], dict)
