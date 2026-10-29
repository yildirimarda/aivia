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
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path

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


async def execute_job(
    engine: Engine,
    config: object,
    database_path: Path,
    job: Job,
) -> None:
    """Execute one claimed job, raising when it cannot complete.

    Run jobs execute their stored run through the orchestrator. Any other kind
    is rejected as a permanent failure so the queue does not retry work this
    worker cannot perform.
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
) -> Callable[[Job], Awaitable[None]]:
    """Return an async handler that executes jobs with :func:`execute_job`."""

    async def handler(job: Job) -> None:
        await execute_job(engine, config, database_path, job)

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


async def _run_until_stopped(worker: Worker, max_jobs: int | None) -> int:
    with stop_on_signals(worker):
        return await worker.run(max_jobs=max_jobs)


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
        worker = Worker(
            queue,
            job_handler(engine, config, database_path),
            worker_id=getattr(args, "worker_id", None),
            config=jobs,
            kind=getattr(args, "kind", None),
        )
        processed = asyncio.run(_run_until_stopped(worker, _max_jobs(args)))
    except (OSError, SQLAlchemyError, ValueError) as exc:
        print(f"error: cannot run the worker: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    print(
        f"Worker {worker.worker_id} processed {processed} job(s): "
        f"{worker.succeeded} succeeded, {worker.failed} failed, "
        f"{worker.dead_lettered} dead-lettered"
    )
    return 0


register_command(
    "worker",
    run_worker,
    help="Claim and run queued jobs until stopped",
    configure=configure,
)
