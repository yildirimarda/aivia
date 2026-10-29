"""``aivia worker`` claims queued jobs and runs them until stopped."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aivia.cli import main
from aivia.core.db import (
    JobRepository,
    RunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.jobs import JobQueue
from aivia.core.models import Job, JobState, RunState, StageResult
from aivia.core.stages import Stage, StageContext


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip(), encoding="utf-8")
    return path


def write_environment(config: Path, name: str, timezone: str) -> None:
    path = config.with_name(f"config.{name}.yaml")
    path.write_text(f"pipeline:\n  timezone: {timezone}\n", encoding="utf-8")


@pytest.fixture
def config(tmp_path: Path) -> Path:
    return write_config(
        tmp_path,
        """
formats:
  sample:
    description: A small recipe the worker can execute.
    stages:
      - stage: write
    output_type: video
    handler: save
profiles:
  late:
    overrides:
      pipeline:
        timezone: Europe/Istanbul
  early:
    overrides:
      pipeline:
        timezone: UTC
""",
    )


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "aivia.db"
    upgrade_database(path)
    return path


def enqueue_run(
    capsys: pytest.CaptureFixture[str],
    config: Path,
    database: Path,
    *args: str,
) -> str:
    code = main(
        [
            "run",
            "--format",
            "sample",
            "--enqueue",
            *args,
            "--config",
            str(config),
            "--database",
            str(database),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    line = next(line for line in captured.out.splitlines() if line.startswith("Run: "))
    return line.removeprefix("Run: ").strip()


def worker_command(
    capsys: pytest.CaptureFixture[str],
    config: Path,
    database: Path,
    *args: str,
) -> tuple[int, str, str]:
    code = main(
        [
            "worker",
            *args,
            "--config",
            str(config),
            "--database",
            str(database),
        ]
    )
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class Write(Stage):
    name = "write"
    artifacts = ("script.md",)

    async def run(self, context: StageContext) -> StageResult:
        (context.run_dir / "script.md").write_text("script", encoding="utf-8")
        return StageResult(stage=self.name, artifacts=["script.md"])


class WriteTimezone(Stage):
    name = "write"
    artifacts = ("script.md",)

    async def run(self, context: StageContext) -> StageResult:
        timezone = context.config.pipeline.timezone
        (context.run_dir / "script.md").write_text(timezone, encoding="utf-8")
        return StageResult(stage=self.name, artifacts=["script.md"])


def test_worker_claims_and_runs_a_queued_run_job(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aivia.cli.run.build_stages",
        lambda recipe: [Write() for _stage in recipe.stages],
    )
    run_id = enqueue_run(capsys, config, database)

    code, out, err = worker_command(capsys, config, database, "--once")

    assert code == 0
    assert err == ""
    assert "processed 1 job(s)" in out
    assert "1 succeeded" in out

    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).get(run_id)
        assert run is not None
        assert run.state is RunState.SUCCEEDED
        jobs = JobRepository(engine).list(run_id=run_id)
        assert len(jobs) == 1
        assert jobs[0].state is JobState.SUCCEEDED
        assert jobs[0].attempts == 1
    finally:
        engine.dispose()

    assert run is not None and run.run_dir
    assert Path(run.run_dir, "script.md").is_file()


def test_worker_runs_jobs_until_the_maximum_is_reached(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aivia.cli.run.build_stages",
        lambda recipe: [Write() for _stage in recipe.stages],
    )
    first = enqueue_run(capsys, config, database)
    second = enqueue_run(capsys, config, database)

    code, out, err = worker_command(
        capsys,
        config,
        database,
        "--max-jobs",
        "2",
    )

    assert code == 0
    assert err == ""
    assert "processed 2 job(s)" in out

    engine = create_database_engine(database)
    try:
        for run_id in (first, second):
            run = RunRepository(engine).get(run_id)
            assert run is not None
            assert run.state is RunState.SUCCEEDED
            jobs = JobRepository(engine).list(run_id=run_id)
            assert [job.state for job in jobs] == [JobState.SUCCEEDED]
    finally:
        engine.dispose()


def test_once_leaves_later_jobs_pending(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aivia.cli.run.build_stages",
        lambda recipe: [Write() for _stage in recipe.stages],
    )
    first = enqueue_run(capsys, config, database)
    second = enqueue_run(capsys, config, database)

    code, out, _err = worker_command(capsys, config, database, "--once")

    assert code == 0
    assert "processed 1 job(s)" in out

    engine = create_database_engine(database)
    try:
        first_run = RunRepository(engine).get(first)
        second_run = RunRepository(engine).get(second)
        assert first_run is not None and first_run.state is RunState.SUCCEEDED
        assert second_run is not None and second_run.state is RunState.QUEUED
        jobs = JobRepository(engine).list(run_id=second)
        assert [job.state for job in jobs] == [JobState.PENDING]
    finally:
        engine.dispose()


def test_worker_fails_an_unsupported_job_kind_without_stopping(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = create_database_engine(database)
    try:
        job = Job(kind="unknown", payload={})
        stored = JobQueue(engine).enqueue(job)
    finally:
        engine.dispose()

    code, out, err = worker_command(capsys, config, database, "--once")

    assert code == 0
    assert err == ""
    assert "1 failed" in out

    engine = create_database_engine(database)
    try:
        result = JobRepository(engine).get(stored.id)
        assert result is not None
        assert result.state is JobState.FAILED
        assert result.error_code == "job_execution"
        assert "unsupported job kind" in result.error_message
    finally:
        engine.dispose()


def test_worker_applies_the_profile_recorded_on_the_run(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aivia.cli.run.build_stages",
        lambda recipe: [WriteTimezone() for _stage in recipe.stages],
    )
    run_id = enqueue_run(capsys, config, database, "--profile", "late")

    code, out, err = worker_command(capsys, config, database, "--once")

    assert code == 0
    assert err == ""
    assert "1 succeeded" in out

    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).get(run_id)
        assert run is not None
        assert run.state is RunState.SUCCEEDED
        assert run.profile == "late"
    finally:
        engine.dispose()

    assert run is not None and run.run_dir
    script = Path(run.run_dir, "script.md")
    assert script.read_text(encoding="utf-8") == "Europe/Istanbul"


def test_worker_prefers_the_run_profile_over_its_own_profile(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aivia.cli.run.build_stages",
        lambda recipe: [WriteTimezone() for _stage in recipe.stages],
    )
    run_id = enqueue_run(capsys, config, database, "--profile", "late")

    code, out, err = worker_command(
        capsys,
        config,
        database,
        "--profile",
        "early",
        "--once",
    )

    assert code == 0
    assert err == ""
    assert "1 succeeded" in out

    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).get(run_id)
        assert run is not None
        assert run.state is RunState.SUCCEEDED
    finally:
        engine.dispose()

    assert run is not None and run.run_dir
    script = Path(run.run_dir, "script.md")
    assert script.read_text(encoding="utf-8") == "Europe/Istanbul"


def test_worker_applies_the_environment_recorded_on_the_run(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_environment(config, "staging", "Asia/Tokyo")
    monkeypatch.setattr(
        "aivia.cli.run.build_stages",
        lambda recipe: [WriteTimezone() for _stage in recipe.stages],
    )
    run_id = enqueue_run(capsys, config, database, "--env", "staging")

    code, out, err = worker_command(capsys, config, database, "--once")

    assert code == 0
    assert err == ""
    assert "1 succeeded" in out

    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).get(run_id)
        assert run is not None
        assert run.state is RunState.SUCCEEDED
        assert run.environment == "staging"
    finally:
        engine.dispose()

    assert run is not None and run.run_dir
    script = Path(run.run_dir, "script.md")
    assert script.read_text(encoding="utf-8") == "Asia/Tokyo"


def test_worker_prefers_the_run_environment_over_its_own(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_environment(config, "staging", "Asia/Tokyo")
    write_environment(config, "other", "America/New_York")
    monkeypatch.setattr(
        "aivia.cli.run.build_stages",
        lambda recipe: [WriteTimezone() for _stage in recipe.stages],
    )
    run_id = enqueue_run(capsys, config, database, "--env", "staging")

    code, out, err = worker_command(
        capsys,
        config,
        database,
        "--env",
        "other",
        "--once",
    )

    assert code == 0
    assert err == ""
    assert "1 succeeded" in out

    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).get(run_id)
        assert run is not None
        assert run.state is RunState.SUCCEEDED
    finally:
        engine.dispose()

    assert run is not None and run.run_dir
    script = Path(run.run_dir, "script.md")
    assert script.read_text(encoding="utf-8") == "Asia/Tokyo"


class Overlap(Stage):
    name = "write"
    artifacts = ("script.md",)

    def __init__(self, state: dict[str, int]) -> None:
        super().__init__()
        self.state = state

    async def run(self, context: StageContext) -> StageResult:
        state = self.state
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        try:
            for _ in range(200):
                if state["max_active"] >= 2:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("jobs did not run concurrently")
            await asyncio.sleep(0.02)
        finally:
            state["active"] -= 1
        (context.run_dir / "script.md").write_text("script", encoding="utf-8")
        return StageResult(stage=self.name, artifacts=["script.md"])


def concurrency_config(tmp_path: Path) -> Path:
    return write_config(
        tmp_path,
        """
jobs:
  concurrency: 2
formats:
  sample:
    description: A small recipe the worker can execute.
    stages:
      - stage: write
    output_type: video
    handler: save
""",
    )


def test_worker_runs_jobs_on_the_configured_number_of_claim_loops(
    tmp_path: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = concurrency_config(tmp_path)
    state = {"active": 0, "max_active": 0}
    monkeypatch.setattr(
        "aivia.cli.run.build_stages",
        lambda recipe: [Overlap(state) for _stage in recipe.stages],
    )
    run_ids = [enqueue_run(capsys, config, database) for _ in range(4)]

    code, out, err = worker_command(
        capsys,
        config,
        database,
        "--max-jobs",
        "4",
    )

    assert code == 0
    assert err == ""
    assert "processed 4 job(s)" in out
    assert "4 succeeded" in out
    assert state["max_active"] == 2

    engine = create_database_engine(database)
    try:
        claimed_by: set[str] = set()
        for run_id in run_ids:
            run = RunRepository(engine).get(run_id)
            assert run is not None
            assert run.state is RunState.SUCCEEDED
            jobs = JobRepository(engine).list(run_id=run_id)
            assert [job.state for job in jobs] == [JobState.SUCCEEDED]
            claimed_by.add(jobs[0].claimed_by)
        assert len(claimed_by) == 2
    finally:
        engine.dispose()


def test_once_claims_one_job_even_when_concurrency_is_higher(
    tmp_path: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = concurrency_config(tmp_path)
    monkeypatch.setattr(
        "aivia.cli.run.build_stages",
        lambda recipe: [Write() for _stage in recipe.stages],
    )
    first = enqueue_run(capsys, config, database)
    second = enqueue_run(capsys, config, database)

    code, out, err = worker_command(capsys, config, database, "--once")

    assert code == 0
    assert err == ""
    assert "processed 1 job(s)" in out
    assert "1 succeeded" in out

    engine = create_database_engine(database)
    try:
        first_run = RunRepository(engine).get(first)
        second_run = RunRepository(engine).get(second)
        assert first_run is not None and first_run.state is RunState.SUCCEEDED
        assert second_run is not None and second_run.state is RunState.QUEUED
        jobs = JobRepository(engine).list(run_id=second)
        assert [job.state for job in jobs] == [JobState.PENDING]
    finally:
        engine.dispose()
