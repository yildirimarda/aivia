"""Media toolkit HTTP routes: submission, status and result download."""

from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from aivia.api import create_app
from aivia.core.config import ApiConfig
from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.cost import CostTracker
from aivia.core.db import JobRepository, create_database_engine, upgrade_database
from aivia.core.jobs import JobQueue
from aivia.core.models import Job, JobState
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
        config=CapabilityChainConfig(chain=[provider.name]),
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


def test_submitting_an_image_job_runs_it_and_exposes_the_result(
    client: TestClient,
    provider: MockProvider,
) -> None:
    response = client.post(
        "/api/v1/media/jobs",
        json={"kind": "image", "prompt": "A lighthouse at sunset"},
    )

    assert response.status_code == 202
    submitted = response.json()
    assert submitted["job_id"].startswith("job_")
    assert submitted["id"] == submitted["job_id"]
    assert submitted["kind"] == "image"
    assert submitted["status"] == "pending"

    status = client.get(f"/api/v1/media/jobs/{submitted['job_id']}")

    assert status.status_code == 200
    body = status.json()
    assert body["state"] == "succeeded"
    assert body["status"] == "succeeded"
    assert body["kind"] == "image"
    assert body["provider"] == "mock"
    assert body["cost_usd"] == pytest.approx(0.02)
    assert body["cached"] is False
    assert body["result"]["kind"] == "image"
    assert body["result"]["cost_usd"] == pytest.approx(0.02)
    assert len(body["paths"]) == 1
    assert len(body["downloads"]) == 1
    assert body["download_url"] == body["downloads"][0]["url"]
    assert len(provider.calls) == 1

    download = client.get(body["download_url"])

    assert download.status_code == 200
    assert download.content.startswith(b"\x89PNG")
    assert download.headers["content-disposition"].startswith("attachment")


def test_the_result_route_returns_the_same_summary(
    client: TestClient,
) -> None:
    submitted = client.post(
        "/api/v1/media/jobs",
        json={"kind": "image", "prompt": "A quiet harbour"},
    ).json()

    response = client.get(f"/api/v1/media/jobs/{submitted['job_id']}/result")

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == submitted["job_id"]
    assert body["kind"] == "image"
    assert body["status"] == "succeeded"
    assert (
        body["download_url"] == f"/api/v1/media/jobs/{submitted['job_id']}/download/0"
    )


def test_a_second_identical_submission_is_cached_and_free(
    client: TestClient,
    provider: MockProvider,
) -> None:
    first = client.post(
        "/api/v1/media/jobs",
        json={"kind": "image", "prompt": "A lighthouse at sunset"},
    ).json()
    second = client.post(
        "/api/v1/media/jobs",
        json={"kind": "image", "prompt": "A lighthouse at sunset"},
    ).json()

    first_status = client.get(f"/api/v1/media/jobs/{first['job_id']}").json()
    second_status = client.get(f"/api/v1/media/jobs/{second['job_id']}").json()

    assert first_status["cached"] is False
    assert second_status["cached"] is True
    assert second_status["cost_usd"] == pytest.approx(0.0)
    assert second_status["paths"] == first_status["paths"]
    assert len(provider.calls) == 1


def test_a_failed_job_records_the_error_and_has_no_download(
    client: TestClient,
    toolkit: MediaToolkit,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(toolkit, "generate", fail)

    submitted = client.post(
        "/api/v1/media/jobs",
        json={"kind": "image", "prompt": "A storm"},
    ).json()

    status = client.get(f"/api/v1/media/jobs/{submitted['job_id']}")

    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "failed"
    assert body["error_code"] == "media_job_failed"
    assert "provider exploded" in body["error_message"]
    assert body["error"]["message"] == body["error_message"]
    assert body["result"] is None
    assert body["download_url"] is None

    assert (
        client.get(f"/api/v1/media/jobs/{submitted['job_id']}/download").status_code
        == 404
    )
    assert (
        client.get(f"/api/v1/media/jobs/{submitted['job_id']}/result").status_code
        == 404
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "image"},
        {"kind": "image", "prompt": ""},
        {"kind": "hologram", "prompt": "A lighthouse"},
        {"prompt": "A lighthouse"},
    ],
)
def test_an_invalid_submission_is_rejected_before_it_is_queued(
    client: TestClient,
    engine: Engine,
    payload: dict[str, object],
) -> None:
    response = client.post("/api/v1/media/jobs", json=payload)

    assert response.status_code == 422
    assert JobRepository(engine).list() == []


def test_an_unknown_job_is_not_found(client: TestClient) -> None:
    assert client.get("/api/v1/media/jobs/job_missing").status_code == 404
    assert client.get("/api/v1/media/jobs/job_missing/result").status_code == 404
    assert client.get("/api/v1/media/jobs/job_missing/download").status_code == 404


def test_a_download_index_outside_the_result_is_not_found(
    client: TestClient,
) -> None:
    submitted = client.post(
        "/api/v1/media/jobs",
        json={"kind": "image", "prompt": "A quiet harbour"},
    ).json()

    response = client.get(f"/api/v1/media/jobs/{submitted['job_id']}/download/1")

    assert response.status_code == 404
    assert "index 1" in response.json()["detail"]


def test_a_capability_can_be_selected_from_the_path(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/media/image/jobs",
        json={"prompt": "A lighthouse at sunset"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["kind"] == "image"

    status = client.get(f"/api/v1/media/jobs/{body['job_id']}").json()
    assert status["status"] == "succeeded"


def test_an_edit_job_accepts_an_inline_base64_image(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/media/jobs",
        json={
            "kind": "edit",
            "operation": "upscale",
            "scale": 2.0,
            "image": base64.b64encode(PNG_1X1).decode("ascii"),
        },
    )

    assert response.status_code == 202
    status = client.get(f"/api/v1/media/jobs/{response.json()['job_id']}").json()

    assert status["status"] == "succeeded"
    assert status["kind"] == "edit"
    assert status["cost_usd"] == pytest.approx(0.03)
    assert len(status["downloads"]) == 1


def test_media_routes_require_authentication(
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

    assert (
        client.post(
            "/api/v1/media/jobs",
            json={"kind": "image", "prompt": "A lighthouse"},
        ).status_code
        == 401
    )
    authorized = client.post(
        "/api/v1/media/jobs",
        json={"kind": "image", "prompt": "A lighthouse"},
        headers={"X-API-Key": "test-api-key"},
    )
    assert authorized.status_code == 202


def test_a_repeated_idempotency_key_returns_the_existing_job(
    client: TestClient,
    engine: Engine,
    provider: MockProvider,
) -> None:
    submission = {
        "kind": "image",
        "prompt": "A lighthouse at sunset",
        "idempotency_key": "request-42",
    }

    first = client.post("/api/v1/media/jobs", json=submission)
    second = client.post("/api/v1/media/jobs", json=submission)

    assert first.status_code == 202
    assert second.status_code == 202
    assert second.json()["job_id"] == first.json()["job_id"]
    assert len(JobRepository(engine).list(kind="media.image")) == 1
    assert len(provider.calls) == 1


def test_a_different_payload_with_the_same_key_conflicts(
    client: TestClient,
    engine: Engine,
) -> None:
    first = client.post(
        "/api/v1/media/jobs",
        json={
            "kind": "image",
            "prompt": "A lighthouse",
            "idempotency_key": "request-42",
        },
    )
    assert first.status_code == 202

    response = client.post(
        "/api/v1/media/jobs",
        json={
            "kind": "image",
            "prompt": "A storm",
            "idempotency_key": "request-42",
        },
    )

    assert response.status_code == 409
    assert "different payload" in response.json()["detail"]
    assert len(JobRepository(engine).list(kind="media.image")) == 1


def test_api_startup_runs_pending_media_jobs(
    engine: Engine,
    toolkit: MediaToolkit,
    provider: MockProvider,
) -> None:
    queue = JobQueue(engine)
    first = queue.enqueue(
        Job(
            kind="media.image",
            payload={
                "kind": "image",
                "request": {"prompt": "A stranded lighthouse"},
            },
        )
    )
    second = queue.enqueue(
        Job(
            kind="media.image",
            payload={
                "kind": "image",
                "request": {"prompt": "A stranded harbour"},
            },
        )
    )

    app = create_app(ApiConfig(require_auth=False), engine, environ={}, toolkit=toolkit)

    with TestClient(app) as client:
        first_status = client.get(f"/api/v1/media/jobs/{first.id}").json()
        second_status = client.get(f"/api/v1/media/jobs/{second.id}").json()

    assert first_status["status"] == "succeeded"
    assert second_status["status"] == "succeeded"
    assert len(provider.calls) == 2
    stored_first = JobQueue(engine).get(first.id)
    stored_second = JobQueue(engine).get(second.id)
    assert stored_first is not None
    assert stored_second is not None
    assert stored_first.state is JobState.SUCCEEDED
    assert stored_second.state is JobState.SUCCEEDED


def test_api_startup_leaves_other_pending_jobs_for_the_worker(
    engine: Engine,
    toolkit: MediaToolkit,
    provider: MockProvider,
) -> None:
    queued = JobQueue(engine).enqueue(
        Job(kind="run", payload={"prompt": "Not a media job"})
    )
    app = create_app(ApiConfig(require_auth=False), engine, environ={}, toolkit=toolkit)

    with TestClient(app):
        pass

    stored = JobQueue(engine).get(queued.id)
    assert stored is not None
    assert stored.state is JobState.PENDING
    assert not provider.calls
