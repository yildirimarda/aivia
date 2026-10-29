"""Generic job submission, job status, run details and run costs over HTTP."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from aivia.api import create_app
from aivia.core.config import ApiConfig
from aivia.core.cost import CostTracker
from aivia.core.db import (
    RunRepository,
    StageRunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.jobs import JobQueue
from aivia.core.models import JobState, Run, RunKind, RunState, StageResult, StageState

KEY = "test-api-key"
AUTH = {"X-API-Key": KEY}


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    created = create_database_engine(path=tmp_path / "aivia.db")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def client(engine: Engine) -> TestClient:
    config = ApiConfig(require_auth=False)
    return TestClient(create_app(config, engine, environ={}))


def test_submitting_a_job_creates_a_pending_job_and_a_queued_run(
    client: TestClient,
    engine: Engine,
) -> None:
    response = client.post(
        "/api/v1/jobs",
        json={
            "kind": "run",
            "payload": {"prompt": "a lighthouse"},
            "metadata": {"source": "test"},
            "profile": "daily",
            "format": "short",
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert body["id"].startswith("job_")
    assert body["job_id"] == body["id"]
    assert body["state"] == "pending"
    assert body["status"] == "pending"
    assert body["payload"] == {"prompt": "a lighthouse"}
    assert body["metadata"] == {"source": "test"}
    assert body["run_id"].startswith("run_")
    assert body["run"]["id"] == body["run_id"]
    assert body["run"]["state"] == "queued"

    stored_job = JobQueue(engine).get(body["id"])
    assert stored_job is not None
    assert stored_job.state is JobState.PENDING
    assert stored_job.run_id == body["run_id"]

    stored_run = RunRepository(engine).get(body["run_id"])
    assert stored_run is not None
    assert stored_run.kind is RunKind.PIPELINE
    assert stored_run.state is RunState.QUEUED
    assert stored_run.profile == "daily"
    assert stored_run.format == "short"


def test_submitting_a_run_returns_the_run_and_its_job(
    client: TestClient,
    engine: Engine,
) -> None:
    response = client.post(
        "/api/v1/runs",
        json={"id": "run_explicit", "kind": "media", "profile": "one-off"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["id"] == "run_explicit"
    assert body["run_id"] == "run_explicit"
    assert body["kind"] == "media"
    assert body["state"] == "queued"
    assert body["job_id"].startswith("job_")
    assert body["job"]["run_id"] == "run_explicit"

    run = RunRepository(engine).get("run_explicit")
    assert run is not None
    assert run.kind is RunKind.MEDIA
    assert run.profile == "one-off"


def test_a_submission_can_attach_to_an_existing_run(
    client: TestClient,
    engine: Engine,
) -> None:
    run = RunRepository(engine).create(Run(kind=RunKind.ORDER, profile="existing"))

    response = client.post(
        "/api/v1/jobs",
        json={"kind": "run", "run_id": run.id, "payload": {"stage": "write"}},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["run_id"] == run.id
    assert len(RunRepository(engine).list(profile="existing")) == 1


def test_job_status_reports_the_current_claim(
    client: TestClient,
    engine: Engine,
) -> None:
    submitted = client.post("/api/v1/jobs", json={"kind": "run"}).json()
    job_id = submitted["id"]
    claimed = JobQueue(engine).claim("worker-1")
    assert claimed is not None

    response = client.get(f"/api/v1/jobs/{job_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == job_id
    assert body["state"] == "running"
    assert body["status"] == "running"
    assert body["claimed_by"] == "worker-1"
    assert body["attempts"] == 1
    assert body["run"]["id"] == body["run_id"]

    alias = client.get(f"/api/v1/jobs/{job_id}/status")
    assert alias.status_code == 200
    assert alias.json()["state"] == "running"


def test_job_status_is_not_found_for_an_unknown_id(client: TestClient) -> None:
    response = client.get("/api/v1/jobs/job_missing")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_run_details_include_ordered_stages_and_artifacts(
    client: TestClient,
    engine: Engine,
) -> None:
    run = RunRepository(engine).create(Run(profile="daily"))
    stages = StageRunRepository(engine)
    stages.create(
        StageResult(
            run_id=run.id,
            stage="render",
            status=StageState.SUCCEEDED,
            index=2,
            artifacts=["clip.mp4", "cover.png"],
        )
    )
    stages.create(
        StageResult(
            run_id=run.id,
            stage="write",
            status=StageState.SUCCEEDED,
            index=1,
            artifacts=["script.md", "cover.png"],
        )
    )

    response = client.get(f"/api/v1/runs/{run.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == run.id
    assert body["run_id"] == run.id
    assert body["state"] == "queued"
    assert [stage["stage"] for stage in body["stages"]] == ["write", "render"]
    assert body["artifacts"] == ["script.md", "cover.png", "clip.mp4"]
    assert body["run"]["stages"] == body["stages"]


def test_run_details_are_not_found_for_an_unknown_id(client: TestClient) -> None:
    response = client.get("/api/v1/runs/run_missing")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_run_costs_report_effective_totals_and_breakdowns(
    client: TestClient,
    engine: Engine,
) -> None:
    run = RunRepository(engine).create(Run())
    tracker = CostTracker(engine)
    tracker.record_estimate(
        run_id=run.id,
        provider="acme",
        model="image-1",
        operation="image_gen",
        units=1.0,
        amount_usd=0.20,
    )
    tracker.record_actual(
        run_id=run.id,
        provider="acme",
        model="image-1",
        operation="image_gen",
        units=1.0,
        amount_usd=0.15,
    )
    tracker.record_actual(
        run_id=run.id,
        provider="other",
        model="video-1",
        operation="video_gen",
        units=2.0,
        amount_usd=0.05,
    )

    response = client.get(f"/api/v1/runs/{run.id}/costs")

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == run.id
    assert body["total_usd"] == pytest.approx(0.20)
    assert body["estimated_usd"] == pytest.approx(0.0)
    assert body["actual_usd"] == pytest.approx(0.20)
    assert body["count"] == 2
    assert len(body["events"]) == 3
    assert [bucket["key"] for bucket in body["by_provider"]] == ["acme", "other"]
    assert [bucket["key"] for bucket in body["by_stage"]] == [
        "image_gen",
        "video_gen",
    ]
    assert body["by_model"][0]["key"] == "image-1"


def test_run_costs_are_not_found_for_an_unknown_run(client: TestClient) -> None:
    response = client.get("/api/v1/runs/run_missing/costs")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_job_routes_work_with_an_in_memory_engine() -> None:
    engine = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=engine)
    try:
        config = ApiConfig(require_auth=False)
        client = TestClient(create_app(config, engine, environ={}))

        response = client.post("/api/v1/jobs", json={"kind": "run"})

        assert response.status_code == 202
        assert JobQueue(engine).get(response.json()["id"]) is not None
    finally:
        engine.dispose()


def test_job_routes_require_authentication(engine: Engine) -> None:
    client = TestClient(
        create_app(
            ApiConfig(require_auth=True, api_key_ref="AIVA_API_KEY"),
            engine,
            environ={"AIVA_API_KEY": KEY},
        )
    )

    assert client.post("/api/v1/jobs", json={"kind": "run"}).status_code == 401
    authorized = client.post("/api/v1/jobs", json={"kind": "run"}, headers=AUTH)
    assert authorized.status_code == 202
