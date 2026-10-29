"""``aivia replay``: re-execute a stored run and report artifact drift."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.cli import main
from aivia.core.db import (
    RunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.models import Run, RunState, StageResult
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
planner:
  defaults:
    seed: 7
formats:
  sample:
    description: A small replayable recipe.
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


def command(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, str, str]:
    code = main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def run_id_from(output: str) -> str:
    line = next(line for line in output.splitlines() if line.startswith("Run: "))
    return line.removeprefix("Run: ").strip()


def replay_ids_from(output: str) -> tuple[str, str]:
    line = next(line for line in output.splitlines() if line.startswith("Replay: "))
    original, replay = line.removeprefix("Replay: ").split(" -> ")
    return original.strip(), replay.strip()


class _SeedStage(Stage):
    name = "write"
    artifacts = ("script.md",)

    async def run(self, context: StageContext) -> StageResult:
        config = (
            context.config.model_dump()
            if hasattr(context.config, "model_dump")
            else context.config
        )
        planner = config.get("planner", {})
        defaults = planner.get("defaults", {}) if isinstance(planner, dict) else {}
        seed = defaults.get("seed", 0)
        (context.run_dir / "script.md").write_text(f"seed={seed}", encoding="utf-8")
        return StageResult(stage=self.name, artifacts=["script.md"])


class _OriginalStage(Stage):
    name = "write"
    artifacts = ("script.md",)

    async def run(self, context: StageContext) -> StageResult:
        (context.run_dir / "script.md").write_text("original", encoding="utf-8")
        return StageResult(stage=self.name, artifacts=["script.md"])


class _DifferentStage(Stage):
    name = "write"
    artifacts = ("script.md",)

    async def run(self, context: StageContext) -> StageResult:
        (context.run_dir / "script.md").write_text("replayed", encoding="utf-8")
        return StageResult(stage=self.name, artifacts=["script.md"])


class _OriginalPairStage(Stage):
    name = "write"
    artifacts = ("keep.txt", "gone.txt")

    async def run(self, context: StageContext) -> StageResult:
        (context.run_dir / "keep.txt").write_text("same", encoding="utf-8")
        (context.run_dir / "gone.txt").write_text("old", encoding="utf-8")
        return StageResult(
            stage=self.name,
            artifacts=["keep.txt", "gone.txt"],
        )


class _ReplayPairStage(Stage):
    name = "write"
    artifacts = ("keep.txt", "new.txt")

    async def run(self, context: StageContext) -> StageResult:
        (context.run_dir / "keep.txt").write_text("same", encoding="utf-8")
        (context.run_dir / "new.txt").write_text("new", encoding="utf-8")
        return StageResult(
            stage=self.name,
            artifacts=["keep.txt", "new.txt"],
        )


def _run_original(
    capsys: pytest.CaptureFixture[str],
    config: Path,
    database: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: type[Stage],
) -> str:
    monkeypatch.setattr(
        "aivia.cli.run.build_stages",
        lambda recipe: [stage() for _stage in recipe.stages],
    )
    code, out, err = command(
        capsys,
        "run",
        "--format",
        "sample",
        "--config",
        str(config),
        "--database",
        str(database),
    )
    assert code == 0
    assert err == ""
    return run_id_from(out)


def _stored_run(database: Path, run_id: str) -> Run | None:
    engine = create_database_engine(database)
    try:
        return RunRepository(engine).get(run_id)
    finally:
        engine.dispose()


def test_replay_reproduces_artifacts_without_touching_the_original(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_id = _run_original(capsys, config, database, monkeypatch, _SeedStage)
    original = _stored_run(database, original_id)
    assert original is not None
    original_dir = Path(original.run_dir)
    original_artifact = (original_dir / "script.md").read_text(encoding="utf-8")
    original_manifest = (original_dir / "artifacts.json").read_text(encoding="utf-8")
    original_snapshot = (original_dir / "snapshot.json").read_text(encoding="utf-8")

    monkeypatch.setattr(
        "aivia.cli.replay.build_stages",
        lambda recipe: [_SeedStage() for _stage in recipe.stages],
    )
    write_config(
        config.parent,
        """
planner:
  defaults:
    seed: 99
formats:
  sample:
    stages:
      - stage: write
    capabilities: [llm]
    output_type: video
    handler: save
""",
    )

    code, out, err = command(
        capsys,
        "replay",
        original_id,
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert "Drift: none" in out
    assert "Artifacts: 1 unchanged, 0 changed, 0 added, 0 removed" in out
    replayed_id, replay_id = replay_ids_from(out)
    assert replayed_id == original_id
    assert replay_id != original_id

    replay = _stored_run(database, replay_id)
    assert replay is not None
    assert replay.state is RunState.SUCCEEDED
    assert replay.run_dir != original.run_dir
    assert replay.config_snapshot["run_id"] == replay_id
    assert Path(replay.run_dir, "script.md").read_text(encoding="utf-8") == "seed=7"

    assert (original_dir / "script.md").read_text(encoding="utf-8") == original_artifact
    assert (original_dir / "artifacts.json").read_text(
        encoding="utf-8"
    ) == original_manifest
    assert (original_dir / "snapshot.json").read_text(
        encoding="utf-8"
    ) == original_snapshot
    stored_original = _stored_run(database, original_id)
    assert stored_original is not None
    assert stored_original.run_dir == original.run_dir
    assert stored_original.state is RunState.SUCCEEDED


def test_replay_reports_changed_artifacts_as_drift(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_id = _run_original(capsys, config, database, monkeypatch, _OriginalStage)
    original = _stored_run(database, original_id)
    assert original is not None
    original_dir = Path(original.run_dir)
    original_manifest = (original_dir / "artifacts.json").read_text(encoding="utf-8")

    monkeypatch.setattr(
        "aivia.cli.replay.build_stages",
        lambda recipe: [_DifferentStage() for _stage in recipe.stages],
    )

    code, out, err = command(
        capsys,
        "replay",
        original_id,
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert "Drift: 1" in out
    assert "Artifacts: 0 unchanged, 1 changed, 0 added, 0 removed" in out
    assert "write\tscript.md\tchanged\t" in out
    _, replay_id = replay_ids_from(out)
    replay = _stored_run(database, replay_id)
    assert replay is not None
    assert Path(replay.run_dir, "script.md").read_text(encoding="utf-8") == "replayed"

    assert (original_dir / "script.md").read_text(encoding="utf-8") == "original"
    assert (original_dir / "artifacts.json").read_text(
        encoding="utf-8"
    ) == original_manifest


def test_fail_on_drift_makes_a_changed_replay_exit_non_zero(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_id = _run_original(capsys, config, database, monkeypatch, _OriginalStage)
    monkeypatch.setattr(
        "aivia.cli.replay.build_stages",
        lambda recipe: [_DifferentStage() for _stage in recipe.stages],
    )

    code, out, err = command(
        capsys,
        "replay",
        original_id,
        "--fail-on-drift",
        "--database",
        str(database),
    )

    assert code == 1
    assert err == ""
    assert "Drift: 1" in out


def test_replay_reports_added_and_removed_artifacts(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_id = _run_original(
        capsys, config, database, monkeypatch, _OriginalPairStage
    )
    monkeypatch.setattr(
        "aivia.cli.replay.build_stages",
        lambda recipe: [_ReplayPairStage() for _stage in recipe.stages],
    )

    code, out, err = command(
        capsys,
        "replay",
        original_id,
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert "Artifacts: 1 unchanged, 0 changed, 1 added, 1 removed" in out
    assert "Drift: 2" in out
    assert "write\tgone.txt\tremoved\t" in out
    assert "write\tnew.txt\tadded\t" in out


def test_replay_requires_a_run_id(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = command(capsys, "replay")

    assert code == 2
    assert out == ""
    assert "a run id is required" in err


def test_replay_reports_a_missing_run(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = command(
        capsys,
        "replay",
        "run_missing",
        "--database",
        str(database),
    )

    assert code == 1
    assert out == ""
    assert "error: run not found: run_missing" in err


def test_replay_reports_a_run_without_a_snapshot(
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = create_database_engine(database)
    try:
        RunRepository(engine).create(
            Run(id="run_without_snapshot", state=RunState.SUCCEEDED, format="sample")
        )
    finally:
        engine.dispose()

    code, out, err = command(
        capsys,
        "replay",
        "run_without_snapshot",
        "--database",
        str(database),
    )

    assert code == 1
    assert out == ""
    assert "run has no replay snapshot: run_without_snapshot" in err
