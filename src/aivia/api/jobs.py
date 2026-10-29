"""Generic job submission and run inspection routes for the local API.

The API is intentionally thin: a submission becomes a durable row in the job
queue and, when it describes a run, a queued run record. Status, run details and
costs are read straight from the same repositories the CLI and workers use, so
the HTTP surface never duplicates queue or ledger logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import Engine

from aivia.core.config import JobsConfig
from aivia.core.cost import CostAggregator, CostTracker
from aivia.core.db import (
    JobRepository,
    RunRepository,
    StageRunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.errors import QuotaError
from aivia.core.idempotency import IdempotencyConflict, create_run_once
from aivia.core.jobs import JobQueue
from aivia.core.models import Job, Run, RunKind, StageResult

__all__ = ["JobSubmission", "register_job_routes"]

_RUN_FIELDS = (
    "kind",
    "state",
    "idempotency_key",
    "toolchain_fingerprint",
    "profile",
    "format",
    "run_dir",
    "config_snapshot",
    "estimated_cost_usd",
    "error_code",
    "error_message",
)

_RUN_KINDS = frozenset(kind.value for kind in RunKind)


class JobSubmission(BaseModel):
    """A generic queue submission, optionally carrying a run description.

    ``kind`` is the job kind. Run-specific fields may be supplied directly,
    through :attr:`run`, or through aliases the run model accepts. A submission
    without an explicit run id creates a new run so the caller can immediately
    inspect it and its costs.
    """

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    kind: str = "run"
    run_id: str = ""
    run_kind: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_attempts: int = Field(default=3, ge=1)
    not_before: datetime | None = None
    run: dict[str, Any] | None = None
    profile: str = ""
    format: str = ""
    idempotency_key: str = ""
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    estimated_cost_usd: float | None = None
    toolchain_fingerprint: dict[str, Any] | str | None = None

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("kind must be a non-empty name")
        return text

    @field_validator("run_id", "profile", "format", "idempotency_key")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()


def register_job_routes(
    app: FastAPI,
    *,
    engine: Engine | None,
    environ: Mapping[str, str] | None,
    database_path: str | Path | None,
    data_dir: str | Path | None,
    jobs: JobsConfig | Mapping[str, Any] | None = None,
) -> None:
    """Attach the generic job and run routes to *app*.

    A caller may inject an engine, or let the API open the configured local
    database on the first request. Keeping the lazy path means health and
    OpenAPI stay usable before the database exists. *jobs* supplies the queue
    depth limit every submission is checked against.
    """

    def database() -> Engine:
        existing = getattr(app.state, "engine", None)
        if isinstance(existing, Engine):
            return existing
        created = create_database_engine(
            path=database_path,
            data_dir=data_dir,
            environ=environ,
        )
        upgrade_database(engine=created)
        app.state.engine = created
        app.state.api_engine_owned = True
        return created

    def queue_for(target: Engine) -> JobQueue:
        return JobQueue(target, config=jobs)

    @app.post(
        "/api/v1/jobs",
        status_code=202,
        tags=["jobs"],
        summary="Submit a generic job",
    )
    def submit_job(submission: JobSubmission) -> dict[str, Any]:
        """Queue a job and return its identifier and initial status."""
        target = database()
        queue = queue_for(target)
        try:
            if not _reuses_existing_job(target, submission):
                queue.ensure_capacity()
            run, _created, reused = _create_or_attach_run(
                target,
                submission,
                requested_run_id=submission.run_id or None,
            )
            job = _enqueue_job(
                queue,
                submission,
                run,
                job_id=submission.id,
                reuse=reused,
            )
        except QuotaError as exc:
            raise _queue_full(exc) from exc
        payload = _job_payload(job)
        payload["job"] = dict(payload)
        if run is not None:
            payload["run"] = _run_payload(run)
        return payload

    @app.post(
        "/api/v1/runs",
        status_code=202,
        tags=["runs"],
        summary="Submit a run",
    )
    def submit_run(submission: JobSubmission) -> dict[str, Any]:
        """Create a run and queue the job that will execute it."""
        target = database()
        queue = queue_for(target)
        requested = submission.run_id or submission.id or None
        try:
            if not _reuses_existing_job(target, submission):
                queue.ensure_capacity()
            run, _created, reused = _create_or_attach_run(
                target,
                submission,
                requested_run_id=requested,
            )
            if run is None:
                raise HTTPException(
                    status_code=422,
                    detail="a run submission needs a kind",
                )
            job = _enqueue_job(queue, submission, run, job_id=None, reuse=reused)
        except QuotaError as exc:
            raise _queue_full(exc) from exc
        payload = _run_payload(run)
        payload["run"] = dict(payload)
        payload["job_id"] = job.id
        payload["job"] = _job_payload(job)
        return payload

    @app.get(
        "/api/v1/jobs/{job_id}",
        tags=["jobs"],
        summary="Get job status",
    )
    @app.get(
        "/api/v1/jobs/{job_id}/status",
        tags=["jobs"],
        summary="Get job status",
        include_in_schema=False,
    )
    def job_status(job_id: str) -> dict[str, Any]:
        """Return one job's current state and its run, when it has one."""
        target = database()
        job = JobQueue(target).get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
        payload = _job_payload(job)
        payload["job"] = dict(payload)
        if job.run_id:
            run = RunRepository(target).get(job.run_id)
            if run is not None:
                payload["run_state"] = run.state.value
                payload["run"] = _run_payload(run)
        return payload

    @app.get(
        "/api/v1/runs/{run_id}",
        tags=["runs"],
        summary="Get run details",
    )
    def run_details(run_id: str) -> dict[str, Any]:
        """Return the run record, its stages and the artifacts they produced."""
        target = database()
        run = RunRepository(target).get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
        payload = _run_payload(run, _stage_payloads(target, run_id))
        payload["run"] = dict(payload)
        return payload

    @app.get(
        "/api/v1/runs/{run_id}/costs",
        tags=["runs"],
        summary="Get run costs",
    )
    @app.get(
        "/api/v1/runs/{run_id}/cost",
        tags=["runs"],
        summary="Get run costs",
        include_in_schema=False,
    )
    def run_costs(run_id: str) -> dict[str, Any]:
        """Return the run's spend, its events and per-stage/provider/model totals."""
        target = database()
        run = RunRepository(target).get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")

        tracker = CostTracker(target)
        aggregator = CostAggregator(target)
        totals = tracker.totals(run_id=run_id)
        events = tracker.for_run(run_id)
        by_stage = aggregator.by_stage(run_id=run_id)
        by_provider = aggregator.by_provider(run_id=run_id)
        by_model = aggregator.by_model(run_id=run_id)

        return {
            "run_id": run.id,
            "total_usd": totals["total"],
            "estimated_usd": totals["estimated"],
            "actual_usd": totals["actual"],
            "count": totals["count"],
            "total": totals["total"],
            "estimated": totals["estimated"],
            "actual": totals["actual"],
            "totals": totals,
            "events": [event.model_dump(mode="json") for event in events],
            "stages": _stage_payloads(target, run_id),
            "by_stage": [bucket.to_dict() for bucket in by_stage],
            "by_provider": [bucket.to_dict() for bucket in by_provider],
            "by_model": [bucket.to_dict() for bucket in by_model],
        }


def _create_or_attach_run(
    engine: Engine,
    submission: JobSubmission,
    *,
    requested_run_id: str | None,
) -> tuple[Run | None, bool, bool]:
    """Return the target run, whether this call created it and whether reused."""
    repository = RunRepository(engine)
    if requested_run_id:
        existing = repository.get(requested_run_id)
        if existing is not None:
            return existing, False, False

    values = _run_values(submission)
    if requested_run_id:
        values["id"] = requested_run_id
    if not values.get("kind"):
        return None, False, False
    run = Run.model_validate(values)
    try:
        result = create_run_once(repository, run)
    except IdempotencyConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    reused = not result.created and run.has_idempotency_key
    return result.run, result.created, reused


def _run_values(submission: JobSubmission) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if submission.run:
        values.update(submission.run)

    for name in _RUN_FIELDS:
        if name == "kind":
            continue
        value = getattr(submission, name, None)
        if value not in (None, ""):
            values.setdefault(name, value)
    if submission.run_kind:
        values.setdefault("kind", submission.run_kind)

    extra = submission.model_extra or {}
    for name in _RUN_FIELDS:
        if name in extra:
            values.setdefault(name, extra[name])

    kind = values.get("kind")
    if not kind:
        kind = (
            submission.kind if submission.kind in _RUN_KINDS else RunKind.PIPELINE.value
        )
    values["kind"] = kind
    return values


def _reuses_existing_job(engine: Engine, submission: JobSubmission) -> bool:
    if not submission.idempotency_key:
        return False
    existing = RunRepository(engine).find_by_idempotency_key(submission.idempotency_key)
    return existing is not None


def _queue_full(error: QuotaError) -> HTTPException:
    detail = {"message": error.message, "code": error.code, **error.details}
    return HTTPException(
        status_code=429,
        detail=detail,
        headers={"Retry-After": "1"},
    )


def _enqueue_job(
    queue: JobQueue,
    submission: JobSubmission,
    run: Run | None,
    *,
    job_id: str | None,
    reuse: bool = False,
) -> Job:
    if reuse and run is not None:
        existing = JobRepository(queue.engine).list(run_id=run.id)
        if existing:
            return existing[0]
    values: dict[str, Any] = {
        "kind": submission.kind,
        "run_id": run.id if run is not None else submission.run_id,
        "payload": submission.payload,
        "metadata": submission.metadata,
        "max_attempts": submission.max_attempts,
        "not_before": submission.not_before,
    }
    if job_id:
        values["id"] = job_id
    return queue.enqueue(Job(**values))


def _job_payload(job: Job) -> dict[str, Any]:
    payload = job.model_dump(mode="json")
    payload["job_id"] = job.id
    payload["status"] = job.state.value
    return payload


def _run_payload(
    run: Run,
    stages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = run.model_dump(mode="json")
    payload["run_id"] = run.id
    payload["status"] = run.state.value
    if stages is not None:
        payload["stages"] = stages
        payload["artifacts"] = _artifacts(stages)
    return payload


def _stage_payloads(engine: Engine, run_id: str) -> list[dict[str, Any]]:
    stages: list[StageResult] = StageRunRepository(engine).list(run_id=run_id)
    stages.sort(key=lambda stage: (stage.index is None, stage.index or 0, stage.stage))
    return [stage.model_dump(mode="json") for stage in stages]


def _artifacts(stages: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    artifacts: list[str] = []
    for stage in stages:
        for artifact in stage.get("artifacts", []):
            if isinstance(artifact, str) and artifact not in seen:
                seen.add(artifact)
                artifacts.append(artifact)
    return artifacts
