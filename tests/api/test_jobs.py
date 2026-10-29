"""Generic job submission, job status, run details and run costs over HTTP."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from aivia.api import create_app
from aivia.core.config import ApiConfig, JobsConfig
from aivia.core.cost import CostTracker
from aivia.core.db import (
    JobRepository,
    RunRepository,
    StageRunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.errors import QuotaError
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


def test_a_repeated_idempotency_key_returns_the_existing_run_and_job(
    client: TestClient,
    engine: Engine,
) -> None:
    submission = {
        "kind": "run",
        "format": "short",
        "idempotency_key": "request-42",
    }

    first = client.post("/api/v1/runs", json=submission)
    second = client.post("/api/v1/runs", json=submission)

    assert first.status_code == 202
    assert second.status_code == 202
    body = second.json()
    assert body["id"] == first.json()["id"]
    assert body["job_id"] == first.json()["job_id"]
    assert len(RunRepository(engine).list()) == 1
    assert len(JobRepository(engine).list(run_id=body["id"])) == 1


def test_a_different_payload_with_the_same_key_conflicts(
    client: TestClient,
    engine: Engine,
) -> None:
    first = client.post(
        "/api/v1/runs",
        json={"kind": "run", "format": "short", "idempotency_key": "request-42"},
    )
    assert first.status_code == 202

    response = client.post(
        "/api/v1/runs",
        json={"kind": "run", "format": "long", "idempotency_key": "request-42"},
    )

    assert response.status_code == 409
    assert "different payload" in response.json()["detail"]
    assert len(RunRepository(engine).list()) == 1


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


@pytest.mark.parametrize("path", ["/api/v1/jobs", "/api/v1/runs"])
def test_a_full_queue_rejects_a_submission_without_creating_a_run(
    path: str,
    engine: Engine,
) -> None:
    JobQueue(engine, max_queue_depth=1).enqueue(kind="seed")
    client = TestClient(
        create_app(
            ApiConfig(require_auth=False),
            engine,
            environ={},
            jobs=JobsConfig(max_queue_depth=1),
        )
    )

    response = client.post(path, json={"kind": "run", "payload": {"prompt": "x"}})

    assert response.status_code == 429
    detail = response.json()["detail"]
    assert "queue is full" in detail["message"]
    assert detail["depth"] == 1
    assert detail["max_queue_depth"] == 1
    assert int(response.headers["Retry-After"]) >= 1
    assert RunRepository(engine).list() == []
    assert len(JobRepository(engine).list()) == 1


def test_a_repeated_idempotent_submission_succeeds_when_the_queue_is_full(
    engine: Engine,
) -> None:
    client = TestClient(
        create_app(
            ApiConfig(require_auth=False),
            engine,
            environ={},
            jobs=JobsConfig(max_queue_depth=1),
        )
    )
    submission = {"kind": "run", "idempotency_key": "request-full"}

    first = client.post("/api/v1/runs", json=submission)
    second = client.post("/api/v1/runs", json=submission)

    assert first.status_code == 202
    assert second.status_code == 202
    assert second.json()["id"] == first.json()["id"]
    assert second.json()["job_id"] == first.json()["job_id"]
    assert len(RunRepository(engine).list()) == 1
    assert len(JobRepository(engine).list()) == 1


@pytest.mark.parametrize("path", ["/api/v1/jobs", "/api/v1/runs"])
def test_a_rejected_enqueue_rolls_back_the_run_created_for_the_submission(
    path: str,
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(
        create_app(
            ApiConfig(require_auth=False),
            engine,
            environ={},
            jobs=JobsConfig(max_queue_depth=1),
        )
    )

    def reject(self: JobQueue, *args: object, **kwargs: object) -> None:
        raise QuotaError(
            "queue is full: depth 1 reached limit 1",
            details={"depth": 1, "max_queue_depth": 1},
        )

    monkeypatch.setattr(JobQueue, "enqueue", reject)

    response = client.post(path, json={"kind": "run", "payload": {"prompt": "x"}})

    assert response.status_code == 429
    assert RunRepository(engine).list() == []
    assert JobRepository(engine).list() == []


def test_concurrent_submissions_leave_a_run_for_every_queued_job(
    tmp_path: Path,
) -> None:
    database = tmp_path / "aivia.db"
    setup = create_database_engine(database)
    upgrade_database(engine=setup)
    setup.dispose()

    threads = 8
    barrier = threading.Barrier(threads)
    statuses: list[int] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def submit() -> None:
        engine = create_database_engine(database)
        try:
            client = TestClient(
                create_app(
                    ApiConfig(require_auth=False),
                    engine,
                    environ={},
                    jobs=JobsConfig(max_queue_depth=1),
                )
            )
            barrier.wait(timeout=5)
            response = client.post("/api/v1/runs", json={"kind": "run"})
            with lock:
                statuses.append(response.status_code)
        except BaseException as exc:  # pragma: no cover - reported below
            with lock:
                errors.append(exc)
        finally:
            engine.dispose()

    workers = [threading.Thread(target=submit) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(20)
    assert not any(worker.is_alive() for worker in workers)

    check = create_database_engine(database)
    try:
        runs = RunRepository(check).list()
        jobs = JobRepository(check).list()
    finally:
        check.dispose()

    assert errors == []
    assert statuses.count(202) == 1
    assert statuses.count(429) == threads - 1
    assert len(runs) == len(jobs) == 1
    assert jobs[0].run_id == runs[0].id
