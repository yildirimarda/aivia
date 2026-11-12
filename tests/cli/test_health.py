"""``aivia health``: queue depth, worker liveness, disk space and provider status."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest
from sqlalchemy import insert

from aivia.cli import main
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.db.tables import jobs
from aivia.core.health import (
    DEFAULT_MIN_FREE_BYTES,
    DiskFullError,
    DiskGuard,
    check_health,
    guard_disk_space,
)
from aivia.core.jobs import JobQueue
from aivia.core.models import Job
from aivia.providers.base import HealthStatus, Provider, ProviderHealth
from aivia.providers.mock import MockProvider


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip(), encoding="utf-8")
    return path


class _Usage:
    def __init__(self, *, total: int, used: int, free: int) -> None:
        self.total = total
        self.used = used
        self.free = free


def _usage(*, total: int, used: int, free: int) -> _Usage:
    return _Usage(total=total, used=used, free=free)


@pytest.fixture
def config(tmp_path: Path) -> Path:
    return write_config(
        tmp_path,
        """
formats:
  sample:
    description: A small recipe.
    stages:
      - stage: write
    output_type: video
    handler: save
providers:
  mock:
    type: mock
""",
    )


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "aivia.db"
    upgrade_database(path)
    return path


def health_command(
    capsys: pytest.CaptureFixture[str],
    config: Path,
    database: Path,
    *args: str,
) -> tuple[int, str, str]:
    code = main(
        [
            "health",
            *args,
            "--config",
            str(config),
            "--database",
            str(database),
        ]
    )
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def enqueue_jobs(database: Path, count: int) -> None:
    engine = create_database_engine(database)
    try:
        queue = JobQueue(engine)
        for _ in range(count):
            queue.enqueue(Job(kind="run", payload={"format": "sample"}))
    finally:
        engine.dispose()


def insert_running_job(
    database: Path,
    job_id: str,
    heartbeat_at: datetime | None,
) -> None:
    engine = create_database_engine(database)
    try:
        with engine.begin() as connection:
            connection.execute(
                insert(jobs).values(
                    id=job_id,
                    kind="run",
                    state="running",
                    heartbeat_at=heartbeat_at,
                    claimed_by="worker-1",
                )
            )
    finally:
        engine.dispose()


def test_health_reports_queue_depth(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    enqueue_jobs(database, 3)

    code, out, err = health_command(capsys, config, database)

    assert code == 0
    assert err == ""
    assert "depth: 3" in out
    assert "overall: healthy" in out


def test_health_reports_zero_depth_when_idle(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = health_command(capsys, config, database)

    assert code == 0
    assert err == ""
    assert "depth: 0" in out


def test_health_reports_worker_liveness_when_idle(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, _err = health_command(capsys, config, database)

    assert code == 0
    assert "running: 0" in out
    assert "stale heartbeats: 0" in out
    assert "live: True" in out


def test_health_reports_fresh_heartbeat_as_live(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    insert_running_job(database, "fresh-job", datetime.now(tz=UTC))

    code, out, _err = health_command(capsys, config, database)

    assert code == 0
    assert "running: 1" in out
    assert "fresh heartbeats: 1" in out
    assert "stale heartbeats: 0" in out
    assert "live: True" in out


def test_health_reports_stale_heartbeat_as_dead_worker(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    insert_running_job(database, "stale-job", datetime(2020, 1, 1, tzinfo=UTC))

    code, out, err = health_command(capsys, config, database)

    assert code == 1
    assert err == ""
    assert "running: 1" in out
    assert "fresh heartbeats: 0" in out
    assert "stale heartbeats: 1" in out
    assert "live: False" in out
    assert "overall: unhealthy" in out


def test_health_reports_disk_space(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = health_command(capsys, config, database)

    assert code == 0
    assert err == ""
    assert "DISK" in out
    assert "ok: True" in out
    assert "free: " in out
    assert "min required: 1000000000 bytes" in out


def test_health_reports_provider_status(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = health_command(capsys, config, database)

    assert code == 0
    assert err == ""
    assert "mock: healthy" in out
    assert "overall: healthy" in out


def test_health_reports_an_unhealthy_provider(
    tmp_path: Path,
    database: Path,
) -> None:
    import asyncio

    class BrokenProvider(Provider):
        default_metadata: ClassVar[Mapping[str, Any]] = {
            "name": "broken",
            "capabilities": ["llm"],
        }

        async def check_health(self) -> ProviderHealth:
            return ProviderHealth(
                status=HealthStatus.UNHEALTHY, message="service is down"
            )

    engine = create_database_engine(database)
    try:
        report = asyncio.run(
            check_health(
                engine,
                database.parent,
                [BrokenProvider()],
                min_free_bytes=0,
            )
        )
    finally:
        engine.dispose()

    assert len(report.provider_status) == 1
    status = report.provider_status[0]
    assert status.name == "broken"
    assert status.status == "unhealthy"
    assert status.message == "service is down"
    assert report.healthy is False


def test_health_is_unhealthy_when_disk_minimum_is_not_met(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = health_command(
        capsys, config, database, "--min-free-bytes", "999999999999"
    )

    assert code == 1
    assert err == ""
    assert "ok: False" in out
    assert "overall: unhealthy" in out


def test_check_health_reports_depth_and_liveness(
    config: Path,
    database: Path,
) -> None:
    enqueue_jobs(database, 2)
    insert_running_job(database, "live-job", datetime.now(tz=UTC))

    import asyncio

    engine = create_database_engine(database)
    try:
        report = asyncio.run(
            check_health(
                engine,
                database.parent,
                [],
                stale_after_seconds=120.0,
                min_free_bytes=0,
            )
        )
    finally:
        engine.dispose()

    assert report.queue_depth == 2
    assert report.worker_liveness.running_jobs == 1
    assert report.worker_liveness.fresh_count == 1
    assert report.worker_liveness.stale_count == 0
    assert report.worker_liveness.live is True
    assert report.disk_space.ok is True
    assert report.healthy is True


def test_check_health_reports_stale_worker_as_not_live(
    config: Path,
    database: Path,
) -> None:
    insert_running_job(database, "stale-job", datetime(2020, 1, 1, tzinfo=UTC))

    import asyncio

    engine = create_database_engine(database)
    try:
        report = asyncio.run(
            check_health(
                engine,
                database.parent,
                [],
                stale_after_seconds=120.0,
                min_free_bytes=0,
            )
        )
    finally:
        engine.dispose()

    assert report.worker_liveness.live is False
    assert report.worker_liveness.stale_count == 1
    assert report.healthy is False


def test_check_health_reports_provider_status(
    config: Path,
    database: Path,
) -> None:
    import asyncio

    engine = create_database_engine(database)
    try:
        report = asyncio.run(
            check_health(
                engine,
                database.parent,
                [MockProvider()],
                min_free_bytes=0,
            )
        )
    finally:
        engine.dispose()

    assert len(report.provider_status) == 1
    status = report.provider_status[0]
    assert status.name == "mock"
    assert status.status == "healthy"
    assert report.healthy is True


def test_disk_guard_space_reports_free_space(tmp_path: Path) -> None:
    space = DiskGuard(tmp_path, 0).space()

    assert space.path == str(tmp_path)
    assert space.total_bytes > 0
    assert space.free_bytes > 0
    assert space.ok is True
    assert space.min_free_bytes == 0


def test_disk_guard_raises_when_free_space_is_below_minimum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "shutil.disk_usage",
        lambda path: _Usage(total=1_000_000_000, used=1_000_000_000, free=0),
    )
    guard = DiskGuard(tmp_path, DEFAULT_MIN_FREE_BYTES + 1)

    with pytest.raises(DiskFullError) as exc_info:
        guard.check()

    assert exc_info.value.code == "disk_full"
    assert "not enough free disk space" in str(exc_info.value)


def test_guard_disk_space_raises_when_low(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "shutil.disk_usage",
        lambda path: _Usage(total=1_000_000_000, used=1_000_000_000, free=0),
    )
    with pytest.raises(DiskFullError):
        guard_disk_space(tmp_path, DEFAULT_MIN_FREE_BYTES + 1)


def test_guard_disk_space_returns_space_when_ok(tmp_path: Path) -> None:
    space = guard_disk_space(tmp_path, 0)

    assert space.ok is True


def test_run_is_refused_when_disk_guard_fails(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_disk_usage(path: object) -> object:
        class Usage:
            total = 1_000_000_000
            used = 1_000_000_000
            free = 0

        return Usage()

    monkeypatch.setattr("shutil.disk_usage", fake_disk_usage)

    code = main(
        [
            "run",
            "--format",
            "sample",
            "--enqueue",
            "--config",
            str(config),
            "--database",
            str(database),
        ]
    )
    captured = capsys.readouterr()
    err = captured.err

    assert code == 1
    assert "disk_full" in err or "not enough free disk space" in err
    assert captured.out == ""
