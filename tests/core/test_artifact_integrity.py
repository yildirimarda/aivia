"""A resume only skips stages whose registered artifacts are still intact."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine

from aivia.core.db import RunRepository, StageRunRepository, create_database_engine
from aivia.core.db.migrate import upgrade_database
from aivia.core.models.run import Run, StageResult, StageState
from aivia.core.orchestrator import Orchestrator
from aivia.core.resume import artifacts_intact, should_skip_stage
from aivia.core.run_dirs import RunDirectory
from aivia.core.run_state import RunStateMachine
from aivia.core.stages import Stage, StageContext


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


class _WriteStage(Stage):
    name = "write"
    artifacts = ("script.md",)

    def __init__(self, events: list[str], *, text: str = "script") -> None:
        super().__init__()
        self.events = events
        self.text = text

    async def run(self, context: StageContext) -> StageResult:
        self.events.append(self.name)
        (context.run_dir / "script.md").write_text(self.text, encoding="utf-8")
        return StageResult(
            run_id=context.run.id if context.run is not None else "",
            stage=self.name,
            artifacts=["script.md"],
        )


class _AfterStage(Stage):
    name = "after"

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    async def run(self, context: StageContext) -> StageResult:
        self.events.append(self.name)
        return StageResult(stage=self.name)


def _orchestrator(
    engine: Engine,
    run: Run | str,
    stages: list[Stage],
    run_dir: Path,
) -> Orchestrator:
    return Orchestrator(
        stages=stages,
        run=run,
        repository=StageRunRepository(engine),
        run_dir=run_dir,
        raise_on_failure=False,
    )


def _first_run(engine: Engine, tmp_path: Path) -> tuple[Run, list[str]]:
    machine = RunStateMachine(RunRepository(engine))
    run = machine.create(Run())
    events: list[str] = []
    result = asyncio.run(
        _orchestrator(
            engine,
            run,
            [_WriteStage(events), _AfterStage(events)],
            tmp_path,
        ).execute()
    )
    assert result.ok is True
    assert events == ["write", "after"]
    return run, events


def test_a_changed_artifact_hash_forces_the_stage_to_run_again(
    engine: Engine,
    tmp_path: Path,
) -> None:
    run, events = _first_run(engine, tmp_path)
    artifact = tmp_path / "script.md"
    artifact.write_text("SCRIPT", encoding="utf-8")
    assert artifact.stat().st_size == len("script")

    events.clear()
    result = asyncio.run(
        _orchestrator(
            engine,
            run.id,
            [_WriteStage(events), _AfterStage(events)],
            tmp_path,
        ).execute()
    )

    assert events == ["write"]
    assert [item.stage for item in result.skipped_stages] == ["after"]
    assert artifact.read_text(encoding="utf-8") == "script"


def test_a_vanished_artifact_forces_the_stage_to_run_again(
    engine: Engine,
    tmp_path: Path,
) -> None:
    run, events = _first_run(engine, tmp_path)
    (tmp_path / "script.md").unlink()

    events.clear()
    result = asyncio.run(
        _orchestrator(
            engine,
            run.id,
            [_WriteStage(events), _AfterStage(events)],
            tmp_path,
        ).execute()
    )

    assert events == ["write"]
    assert [item.stage for item in result.skipped_stages] == ["after"]
    assert (tmp_path / "script.md").read_text(encoding="utf-8") == "script"


def test_intact_artifacts_still_skip_the_stage(
    engine: Engine,
    tmp_path: Path,
) -> None:
    run, events = _first_run(engine, tmp_path)
    events.clear()

    result = asyncio.run(
        _orchestrator(
            engine,
            run.id,
            [_WriteStage(events), _AfterStage(events)],
            tmp_path,
        ).execute()
    )

    assert events == []
    assert [item.stage for item in result.skipped_stages] == ["write", "after"]


def test_a_stage_without_artifacts_can_still_be_skipped(
    engine: Engine,
    tmp_path: Path,
) -> None:
    machine = RunStateMachine(RunRepository(engine))
    run = machine.create(Run())
    events: list[str] = []
    first = asyncio.run(
        _orchestrator(engine, run, [_AfterStage(events)], tmp_path).execute()
    )
    assert first.ok is True
    events.clear()

    result = asyncio.run(
        _orchestrator(engine, run.id, [_AfterStage(events)], tmp_path).execute()
    )

    assert events == []
    assert result.stages[0].status is StageState.SKIPPED


def test_a_missing_manifest_is_not_trusted_on_resume(
    engine: Engine,
    tmp_path: Path,
) -> None:
    run, events = _first_run(engine, tmp_path)
    (tmp_path / "artifacts.json").unlink()

    events.clear()
    result = asyncio.run(
        _orchestrator(
            engine,
            run.id,
            [_WriteStage(events), _AfterStage(events)],
            tmp_path,
        ).execute()
    )

    assert events == ["write"]
    assert [item.stage for item in result.skipped_stages] == ["after"]
    assert (tmp_path / "artifacts.json").is_file()


def test_the_helpers_reject_changed_and_vanished_artifacts(
    engine: Engine,
    tmp_path: Path,
) -> None:
    run, _ = _first_run(engine, tmp_path)
    stored = StageRunRepository(engine).get(run.id, "write")
    assert stored is not None
    directory = RunDirectory(tmp_path, run_id=run.id)

    assert artifacts_intact(stored, directory) is True
    assert should_skip_stage(stored, stored.input_hash, directory=directory) is True

    (tmp_path / "script.md").write_text("tampered", encoding="utf-8")
    assert artifacts_intact(stored, directory) is False
    assert should_skip_stage(stored, stored.input_hash, directory=directory) is False
    assert [record.path for record in directory.verify("write")] == ["script.md"]

    (tmp_path / "script.md").unlink()
    assert artifacts_intact(stored, directory) is False
    assert should_skip_stage(stored, stored.input_hash, directory=directory) is False
    assert artifacts_intact(None, directory) is False
