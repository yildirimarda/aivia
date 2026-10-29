"""``aivia run``: create, execute or enqueue a run and inspect its outcome."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.cli import main
from aivia.core.cost import CostTracker
from aivia.core.db import (
    JobRepository,
    RunRepository,
    StageRunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.jobs import JobQueue
from aivia.core.models import Run, RunState, StageResult, StageState
from aivia.core.run_dirs import RunDirectory
from aivia.core.stages import Stage, StageContext


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip(), encoding="utf-8")
    return path


@pytest.fixture
def config(tmp_path: Path) -> Path:
    return write_config(
        tmp_path,
        """
providers:
  text:
    type: mock-llm
formats:
  sample:
    description: A small priced recipe.
    stages:
      - stage: write
    capabilities: [llm]
    output_type: video
    handler: save
""",
    )


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "aivia.db"
    upgrade_database(path)
    return path


def run_command(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, str, str]:
    code = main(["run", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def run_id_from(output: str) -> str:
    line = next(line for line in output.splitlines() if line.startswith("Run: "))
    return line.removeprefix("Run: ").strip()


def test_run_creates_and_executes_a_run_by_default(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Write(Stage):
        name = "write"
        artifacts = ("script.md",)

        async def run(self, context: StageContext) -> StageResult:
            (context.run_dir / "script.md").write_text("script", encoding="utf-8")
            return StageResult(stage=self.name, artifacts=["script.md"], cost_usd=0.1)

    monkeypatch.setattr(
        "aivia.cli.run.build_stages",
        lambda recipe: [Write() for _stage in recipe.stages],
    )

    code, out, err = run_command(
        capsys,
        "--format",
        "sample",
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    run_id = run_id_from(out)
    assert "Estimate: $0.000500" in out
    assert "State: succeeded" in out

    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).get(run_id)
        assert run is not None
        assert run.state is RunState.SUCCEEDED
        assert run.format == "sample"
        assert run.estimated_cost_usd == pytest.approx(0.0005)
        assert run.cost_usd == pytest.approx(0.1)
        stage = StageRunRepository(engine).get(run_id, "write")
        assert stage is not None
        assert stage.status is StageState.SUCCEEDED
        assert stage.artifacts == ["script.md"]
    finally:
        engine.dispose()

    assert Path(run.run_dir, "script.md").is_file()


def test_run_enqueues_a_job_without_executing(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = run_command(
        capsys,
        "--format",
        "sample",
        "--enqueue",
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    run_id = run_id_from(out)
    assert "Estimate: $0.000500" in out
    assert "State: queued" in out
    assert "Job: job_" in out

    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).get(run_id)
        assert run is not None
        assert run.state is RunState.QUEUED
        assert run.format == "sample"
        assert run.estimated_cost_usd == pytest.approx(0.0005)
        assert run.run_dir

        jobs = JobRepository(engine).list(run_id=run_id)
        assert len(jobs) == 1
        assert jobs[0].state.value == "pending"
        assert jobs[0].kind == "run"
        assert jobs[0].payload["format"] == "sample"
        assert StageRunRepository(engine).list(run_id=run_id) == []
    finally:
        engine.dispose()


def test_run_enqueues_when_no_stage_implementation_is_registered(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = run_command(
        capsys,
        "--format",
        "sample",
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    run_id = run_id_from(out)
    assert "State: queued" in out

    engine = create_database_engine(database)
    try:
        stored = RunRepository(engine).get(run_id)
        assert stored is not None
        assert stored.state is RunState.QUEUED
        assert len(JobRepository(engine).list(run_id=run_id)) == 1
    finally:
        engine.dispose()


def test_run_requires_a_format(
    config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run_command(capsys, "--config", str(config))

    assert code == 2
    assert out == ""
    assert "--format is required" in err


def seed_run(database: Path) -> str:
    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).create(
            Run(
                id="run_show",
                state=RunState.SUCCEEDED,
                profile="daily",
                format="sample",
                cost_usd=0.25,
                estimated_cost_usd=0.5,
            )
        )
        run_dir = RunDirectory(database.parent / "runs" / run.id, run_id=run.id)
        run_dir.create()
        (run_dir.path / "script.md").write_text("script", encoding="utf-8")
        run_dir.register("write", ["script.md"])
        RunRepository(engine).update(run.id, run_dir=str(run_dir.path))

        StageRunRepository(engine).create(
            StageResult(
                run_id=run.id,
                stage="write",
                status=StageState.SUCCEEDED,
                index=0,
                artifacts=["script.md"],
                cost_usd=0.25,
                estimated_cost_usd=0.5,
            )
        )
        return run.id
    finally:
        engine.dispose()


def test_run_show_prints_stages_cost_and_artifacts(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = seed_run(database)

    code, out, err = run_command(capsys, "show", run_id, "--database", str(database))

    assert code == 0
    assert err == ""
    assert f"Run: {run_id}" in out
    assert "Format: sample" in out
    assert "State: succeeded" in out
    assert "Cost: $0.250000" in out
    assert "Estimate: $0.500000" in out
    assert "STAGES" in out
    assert "write\tsucceeded\t$0.250000\t$0.500000" in out
    assert "ARTIFACTS" in out
    assert "write\tscript.md\t6" in out


def test_run_show_uses_the_cost_ledger_when_the_run_has_no_total(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = seed_run(database)
    engine = create_database_engine(database)
    try:
        RunRepository(engine).update(run_id, cost_usd=0.0)
        CostTracker(engine).record(
            provider="acme",
            model="llm-1",
            operation="write",
            units=10.0,
            unit_kind="token",
            amount_usd=0.65,
            run_id=run_id,
            estimated=False,
        )
    finally:
        engine.dispose()

    code, out, err = run_command(capsys, "show", run_id, "--database", str(database))

    assert code == 0
    assert err == ""
    assert "Cost: $0.650000" in out


def test_run_show_reports_a_missing_run(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = run_command(
        capsys,
        "show",
        "run_missing",
        "--database",
        str(database),
    )

    assert code == 1
    assert out == ""
    assert "run not found: run_missing" in err


def test_run_show_requires_a_run_id(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = run_command(capsys, "show", "--database", str(database))

    assert code == 2
    assert out == ""
    assert "a run id is required" in err


def test_run_show_reports_a_missing_database(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.db"

    code, out, err = run_command(
        capsys,
        "show",
        "run_1",
        "--database",
        str(missing),
    )

    assert code == 1
    assert out == ""
    assert f"database not found: {missing}" in err


def test_run_enqueue_rejects_a_full_queue_without_creating_a_run(
    tmp_path: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(
        tmp_path,
        """
jobs:
  max_queue_depth: 1
providers:
  text:
    type: mock-llm
formats:
  sample:
    description: A small priced recipe.
    stages:
      - stage: write
    capabilities: [llm]
    output_type: video
    handler: save
""",
    )
    setup = create_database_engine(database)
    try:
        JobQueue(setup).enqueue(kind="seed")
    finally:
        setup.dispose()

    code, out, err = run_command(
        capsys,
        "--format",
        "sample",
        "--enqueue",
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 1
    assert out == ""
    assert "queue is full" in err

    engine = create_database_engine(database)
    try:
        assert RunRepository(engine).list() == []
        assert len(JobRepository(engine).list()) == 1
    finally:
        engine.dispose()
