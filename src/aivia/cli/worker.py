"""``aivia worker``: claim and run queued jobs until stopped.

The worker command connects the durable job queue to the execution code. It
loads the configured database, builds a :class:`~aivia.core.jobs.Worker` with
the jobs section settings and a handler that executes queued run jobs through
the orchestrator. ``SIGINT`` and ``SIGTERM`` request a graceful stop: the job
already running finishes and no new job is claimed.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

import aivia.cli.run as run_command
from aivia.cli.registry import register_command
from aivia.cli.signals import stop_on_signals
from aivia.core import paths
from aivia.core.config.database import DatabaseConfig
from aivia.core.config.formats import FormatRecipeConfig, FormatsConfig
from aivia.core.config.ops import JobsConfig
from aivia.core.config.resolve import resolve_app_config
from aivia.core.cost import CostTracker
from aivia.core.db import RunRepository, create_database_engine, upgrade_database
from aivia.core.dry_run import apply_dry_run
from aivia.core.errors import AiviaError, ConfigError
from aivia.core.jobs import JobQueue, Worker
from aivia.core.models import Job, Run
from aivia.core.orchestrator import Orchestrator

__all__ = [
    "JobExecutionError",
    "configure",
    "execute_job",
    "job_handler",
    "run_worker",
]


class JobExecutionError(AiviaError):
    """A queued job could not be executed."""

    code = "job_execution"
    retryable = False


def _configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--database",
        "--db",
        dest="database",
        metavar="PATH",
        help="database file to use instead of the one in the user data directory",
    )
    parser.add_argument(
        "--worker-id",
        dest="worker_id",
        metavar="NAME",
        help="name this worker records on every claim",
    )
    parser.add_argument(
        "--kind",
        metavar="NAME",
        help="only claim jobs of this kind",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="run at most one due job and exit",
    )
    parser.add_argument(
        "--max-jobs",
        dest="max_jobs",
        type=int,
        metavar="N",
        help="stop after running N jobs",
    )
    parser.set_defaults(_worker_handler=run_worker)


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the worker command's options."""
    _configure(parser)


def _database_path(args: argparse.Namespace, config: object) -> Path:
    explicit = getattr(args, "database", None)
    if explicit:
        return Path(explicit).expanduser()
    database = getattr(config, "database", None)
    if not isinstance(database, DatabaseConfig):
        database = None
    filename = database.filename if database is not None else DatabaseConfig().filename
    return paths.resolve_paths(args.config).data_dir / filename


def _open_database(path: Path) -> Engine:
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_database_engine(path=path)
    try:
        upgrade_database(engine=engine)
    except BaseException:
        engine.dispose()
        raise
    return engine


def _jobs_config(config: object) -> JobsConfig:
    section = getattr(config, "jobs", None)
    return section if isinstance(section, JobsConfig) else JobsConfig()


def _recipe(
    config: object,
    job: Job,
    run: Run,
) -> FormatRecipeConfig:
    payload = job.payload if isinstance(job.payload, Mapping) else {}
    name = str(payload.get("format") or run.format or "").strip()
    if not name:
        raise JobExecutionError(
            f"run job {job.id} has no format to execute",
            details={"job_id": job.id, "run_id": run.id},
        )
    section = getattr(config, "formats", None)
    formats = section.formats if isinstance(section, FormatsConfig) else {}
    recipe = formats.get(name)
    if recipe is None:
        available = ", ".join(sorted(formats)) or "none"
        raise JobExecutionError(
            f"unknown format: {name} (available: {available})",
            details={"job_id": job.id, "run_id": run.id, "format": name},
        )
    return recipe


def _data_dir(run: Run, database_path: Path) -> Path:
    if run.run_dir:
        return Path(run.run_dir).expanduser().parent.parent
    return database_path.parent


def _resolution_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "config_file": getattr(args, "config", None),
        "environment": getattr(args, "environment", None),
        "set_values": getattr(args, "set_values", None),
    }


def _config_for_run(args: argparse.Namespace, run: Run) -> object:
    """Resolve the config a run should execute with.

    The profile recorded on the run is authoritative: it may differ from the
    worker's own profile when one worker drains jobs queued by many callers.
    Worker-level ``--set`` values still apply after the profile, matching the
    normal resolution order, and a run without a profile keeps the worker's
    selected profile as its default.
    """
    options = _resolution_options(args)
    profile = (run.profile or "").strip() or (
        getattr(args, "profile", None) or ""
    ).strip()
    options["profile"] = profile or None
    try:
        config = resolve_app_config(**options)
    except ConfigError as exc:
        raise JobExecutionError(
            f"cannot resolve config for run {run.id}: {exc}",
            details={
                "run_id": run.id,
                "profile": profile,
            },
        ) from exc
    if getattr(args, "dry_run", False):
        config = apply_dry_run(config)
    return config


async def execute_job(
    engine: Engine,
    config: object,
    database_path: Path,
    job: Job,
    *,
    config_for_run: Callable[[Run], object] | None = None,
) -> None:
    """Execute one claimed job, raising when it cannot complete.

    Run jobs execute their stored run through the orchestrator. Any other kind
    is rejected as a permanent failure so the queue does not retry work this
    worker cannot perform. When *config_for_run* is given, it supplies the
    config for the job's run so per-run profile overrides take effect.
    """
    if job.kind != "run":
        raise JobExecutionError(
            f"unsupported job kind: {job.kind}",
            details={"job_id": job.id, "kind": job.kind},
        )
    if not job.run_id:
        raise JobExecutionError(
            f"run job {job.id} has no run id",
            details={"job_id": job.id},
        )
    run = RunRepository(engine).get(job.run_id)
    if run is None:
        raise JobExecutionError(
            f"run not found: {job.run_id}",
            details={"job_id": job.id, "run_id": job.run_id},
        )

    if config_for_run is not None:
        config = config_for_run(run)
    recipe = _recipe(config, job, run)
    stages = run_command.build_stages(recipe)
    orchestrator = Orchestrator(
        stages=stages,
        run=run,
        repository=engine,
        run_dir=run.run_dir or None,
        data_dir=_data_dir(run, database_path),
        config=config,
        cost=CostTracker(engine),
        raise_on_failure=False,
    )
    result = await orchestrator.execute()
    if result.failed:
        raise JobExecutionError(
            f"run {run.id} failed at stage {result.failed_stage!r}: "
            f"{result.error_message}",
            details={
                "job_id": job.id,
                "run_id": run.id,
                "stage": result.failed_stage or "",
                "error_code": result.error_code,
            },
        )


def job_handler(
    engine: Engine,
    config: object,
    database_path: Path,
    *,
    config_for_run: Callable[[Run], object] | None = None,
) -> Callable[[Job], Awaitable[None]]:
    """Return an async handler that executes jobs with :func:`execute_job`."""

    async def handler(job: Job) -> None:
        await execute_job(
            engine,
            config,
            database_path,
            job,
            config_for_run=config_for_run,
        )

    return handler


def _max_jobs(args: argparse.Namespace) -> int | None:
    if getattr(args, "once", False):
        return 1
    value = getattr(args, "max_jobs", None)
    return value


def _resolve(args: argparse.Namespace) -> tuple[object, Path, JobsConfig]:
    config = resolve_app_config(
        config_file=getattr(args, "config", None),
        environment=getattr(args, "environment", None),
        profile=getattr(args, "profile", None),
        set_values=getattr(args, "set_values", None),
    )
    if getattr(args, "dry_run", False):
        config = apply_dry_run(config)
    database_path = _database_path(args, config)
    return config, database_path, _jobs_config(config)


class _WorkerGroup:
    """Stop every claim loop with one graceful stop request."""

    def __init__(self, workers: Sequence[Worker]) -> None:
        self.workers = workers

    def request_stop(self) -> None:
        for worker in self.workers:
            worker.request_stop()


def _worker_ids(base: str | None, count: int) -> list[str | None]:
    if count == 1:
        return [base]
    stem = (base or f"worker-{time.time_ns():x}").strip()
    return [f"{stem}-{index + 1}" for index in range(count)]


def _quotas(max_jobs: int | None, count: int) -> list[int | None]:
    if max_jobs is None:
        return [None] * count
    if max_jobs < 0:
        raise ValueError("max_jobs must be a non-negative integer or None")
    base, extra = divmod(max_jobs, count)
    return [base + (1 if index < extra else 0) for index in range(count)]


async def _run_until_stopped(
    workers: Sequence[Worker],
    max_jobs: int | None,
) -> int:
    quotas = _quotas(max_jobs, len(workers))
    with stop_on_signals(_WorkerGroup(workers)):
        results = await asyncio.gather(
            *(
                worker.run(max_jobs=quota)
                for worker, quota in zip(workers, quotas, strict=True)
            )
        )
    return sum(results)


def run_worker(args: argparse.Namespace) -> int:
    """Claim and run queued jobs until stopped, then print a summary."""
    try:
        config, database_path, jobs = _resolve(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        engine = _open_database(database_path)
    except (OSError, SQLAlchemyError) as exc:
        print(f"error: cannot open the database: {exc}", file=sys.stderr)
        return 1

    try:
        queue = JobQueue(engine, config=jobs)
        handler = job_handler(
            engine,
            config,
            database_path,
            config_for_run=lambda run: _config_for_run(args, run),
        )
        workers = [
            Worker(
                queue,
                handler,
                worker_id=worker_id,
                config=jobs,
                kind=getattr(args, "kind", None),
            )
            for worker_id in _worker_ids(
                getattr(args, "worker_id", None), jobs.concurrency
            )
        ]
        processed = asyncio.run(_run_until_stopped(workers, _max_jobs(args)))
    except (OSError, SQLAlchemyError, ValueError) as exc:
        print(f"error: cannot run the worker: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    succeeded = sum(worker.succeeded for worker in workers)
    failed = sum(worker.failed for worker in workers)
    dead_lettered = sum(worker.dead_lettered for worker in workers)
    if len(workers) == 1:
        label = f"Worker {workers[0].worker_id}"
    else:
        names = ", ".join(worker.worker_id for worker in workers)
        label = f"Workers {names}"
    print(
        f"{label} processed {processed} job(s): "
        f"{succeeded} succeeded, {failed} failed, "
        f"{dead_lettered} dead-lettered"
    )
    return 0


register_command(
    "worker",
    run_worker,
    help="Claim and run queued jobs until stopped",
    configure=configure,
)
