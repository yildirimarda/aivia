"""Health reporting for the job queue, workers, disk space and providers.

The health report gathers four independent signals into one snapshot:

* queue depth - jobs waiting in ``pending``
* worker liveness - whether running jobs are still being heartbeated
* disk space - free bytes on the data directory versus a guard threshold
* provider status - the health of every configured provider

A :class:`DiskGuard` refuses new runs when free disk space falls below a
configured minimum so a full disk cannot silently swallow submissions.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import Engine, select

from aivia.core.db.tables import jobs
from aivia.core.errors import AiviaError
from aivia.core.jobs import JobQueue
from aivia.core.models.base import utcnow
from aivia.core.models.job import JobState
from aivia.providers.base import Provider

__all__ = [
    "DEFAULT_MIN_FREE_BYTES",
    "DiskFullError",
    "DiskGuard",
    "DiskSpace",
    "HealthReport",
    "ProviderStatus",
    "WorkerLiveness",
    "check_health",
    "guard_disk_space",
]

DEFAULT_MIN_FREE_BYTES: int = 1_000_000_000
"""Default minimum free bytes the disk guard requires before a new run."""


class DiskFullError(AiviaError):
    """A new run cannot start because free disk space is below the guard."""

    code = "disk_full"
    retryable = False


def _coerce_min_free_bytes(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("min_free_bytes must be a non-negative integer")
    if value < 0:
        raise ValueError("min_free_bytes must be a non-negative integer")
    return value


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class DiskSpace:
    path: str
    total_bytes: int
    free_bytes: int
    used_bytes: int
    min_free_bytes: int
    ok: bool


@dataclass(frozen=True, slots=True)
class WorkerLiveness:
    running_jobs: int
    fresh_count: int
    stale_count: int

    @property
    def live(self) -> bool:
        """Whether every in-flight job is still being heartbeated.

        An idle queue (no running jobs) is live; a stale heartbeat means a
        worker died while a job was in flight.
        """
        return self.stale_count == 0


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    name: str
    status: str
    message: str
    latency_ms: float | None


@dataclass(frozen=True, slots=True)
class HealthReport:
    queue_depth: int
    worker_liveness: WorkerLiveness
    disk_space: DiskSpace
    provider_status: list[ProviderStatus]

    @property
    def healthy(self) -> bool:
        return (
            self.disk_space.ok
            and self.worker_liveness.live
            and all(status.status == "healthy" for status in self.provider_status)
        )


class DiskGuard:
    """Refuse new runs when free disk space falls below a minimum."""

    def __init__(self, path: str | Path, min_free_bytes: int) -> None:
        self.path = Path(path)
        self.min_free_bytes = _coerce_min_free_bytes(min_free_bytes)

    def space(self) -> DiskSpace:
        """Report free space without refusing anything."""
        self.path.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(self.path)
        ok = usage.free >= self.min_free_bytes
        return DiskSpace(
            path=str(self.path),
            total_bytes=usage.total,
            free_bytes=usage.free,
            used_bytes=usage.used,
            min_free_bytes=self.min_free_bytes,
            ok=ok,
        )

    def check(self) -> DiskSpace:
        """Return the disk space, raising :class:`DiskFullError` when low."""
        space = self.space()
        if not space.ok:
            raise DiskFullError(
                "not enough free disk space for a new run: "
                f"{space.free_bytes} bytes available, "
                f"{space.min_free_bytes} required",
                details={
                    "path": space.path,
                    "free_bytes": space.free_bytes,
                    "min_free_bytes": space.min_free_bytes,
                    "total_bytes": space.total_bytes,
                    "used_bytes": space.used_bytes,
                },
            )
        return space


def guard_disk_space(path: str | Path, min_free_bytes: int) -> DiskSpace:
    """Convenience for :meth:`DiskGuard.check`."""
    return DiskGuard(path, min_free_bytes).check()


def _worker_liveness(engine: Engine, stale_after_seconds: float) -> WorkerLiveness:
    cutoff = utcnow() - timedelta(seconds=stale_after_seconds)
    running_jobs = 0
    fresh_count = 0
    with engine.connect() as connection:
        rows = (
            connection.execute(
                select(jobs.c.heartbeat_at).where(
                    jobs.c.state == JobState.RUNNING.value
                )
            )
            .scalars()
            .all()
        )
    running_jobs = len(rows)
    for heartbeat in rows:
        observed = _as_utc(heartbeat)
        if observed is not None and observed >= cutoff:
            fresh_count += 1
    return WorkerLiveness(
        running_jobs=running_jobs,
        fresh_count=fresh_count,
        stale_count=running_jobs - fresh_count,
    )


async def _provider_status(provider: Provider) -> ProviderStatus:
    try:
        health = await provider.health()
    except Exception as exc:  # defensive: a probe must never crash the report
        message = str(exc) or type(exc).__name__
        return ProviderStatus(
            name=provider.name,
            status="unhealthy",
            message=message,
            latency_ms=None,
        )
    return ProviderStatus(
        name=provider.name,
        status=health.status.value,
        message=health.message,
        latency_ms=health.latency_ms,
    )


async def check_health(
    engine: Engine,
    data_dir: str | Path,
    providers: Iterable[Provider] = (),
    *,
    stale_after_seconds: float = 120.0,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
) -> HealthReport:
    """Build one health snapshot from queue, workers, disk and providers."""
    queue = JobQueue(engine)
    depth = queue.depth()
    liveness = _worker_liveness(engine, stale_after_seconds)
    disk = DiskGuard(data_dir, min_free_bytes).space()
    provider_status = [await _provider_status(provider) for provider in providers]
    return HealthReport(
        queue_depth=depth,
        worker_liveness=liveness,
        disk_space=disk,
        provider_status=provider_status,
    )
