"""Provided text and script files replace research and writing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aivia.cli import main
from aivia.core.db import (
    JobRepository,
    RunRepository,
    StageRunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.models import RunState, StageResult, StageState
from aivia.core.stages import Stage, StageContext
from aivia.formats.input_topic import InputTopicStage


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
  search:
    type: mock-search
formats:
  byo:
    description: A recipe with research, input and writing.
    stages:
      - stage: research
      - stage: input_topic
      - stage: write
    capabilities: [search, llm]
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


def estimate_rows(output: str) -> dict[str, list[str]]:
    rows = [line.split("\t") for line in output.splitlines()]
    return {row[0]: row for row in rows[2:]}


def test_a_text_file_skips_research_but_still_pays_for_writing(
    config: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = config.parent / "facts.txt"
    source.write_text(
        "Honey never spoils.\nBees recognise human faces.\n",
        encoding="utf-8",
    )

    code, out, err = run_command(
        capsys,
        "--format",
        "byo",
        "--estimate",
        "--text-file",
        str(source),
        "--config",
        str(config),
    )

    assert code == 0
    assert err == ""
    rows = estimate_rows(out)
    assert rows["research"][5] == "$0.000000"
    assert rows["research"][6] == "no paid call"
    assert rows["write"][5] == "$0.000500"
    assert rows["TOTAL"][5] == "$0.000500"


def test_a_script_file_skips_research_and_writing(
    config: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = config.parent / "script.en.json"
    source.write_text(
        json.dumps(
            {
                "language": "en",
                "title": "Honey",
                "hook": "Honey keeps forever.",
                "beats": [{"text": "Honey never spoils."}],
            }
        ),
        encoding="utf-8",
    )

    code, out, err = run_command(
        capsys,
        "--format",
        "byo",
        "--estimate",
        "--script-file",
        str(source),
        "--config",
        str(config),
    )

    assert code == 0
    assert err == ""
    rows = estimate_rows(out)
    assert rows["research"][5] == "$0.000000"
    assert rows["write"][5] == "$0.000000"
    assert rows["TOTAL"][5] == "$0.000000"


def test_an_empty_text_file_is_rejected_before_a_run_is_created(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = config.parent / "empty.txt"
    source.write_text("   \n", encoding="utf-8")

    code, out, err = run_command(
        capsys,
        "--format",
        "byo",
        "--text-file",
        str(source),
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 1
    assert out == ""
    assert "text file is empty" in err
    engine = create_database_engine(database)
    try:
        assert RunRepository(engine).list() == []
    finally:
        engine.dispose()


def test_a_missing_script_file_names_the_path(
    config: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = config.parent / "missing.json"

    code, out, err = run_command(
        capsys,
        "--format",
        "byo",
        "--script-file",
        str(missing),
        "--config",
        str(config),
    )

    assert code == 1
    assert out == ""
    assert "script file not found" in err
    assert str(missing) in err


def test_an_invalid_script_file_is_rejected(
    config: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = config.parent / "script.json"
    source.write_text("{not json", encoding="utf-8")

    code, out, err = run_command(
        capsys,
        "--format",
        "byo",
        "--script-file",
        str(source),
        "--config",
        str(config),
    )

    assert code == 1
    assert out == ""
    assert "invalid provided script JSON" in err


def test_a_script_without_content_is_rejected(
    config: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = config.parent / "script.json"
    source.write_text(
        json.dumps({"language": "en", "beats": []}),
        encoding="utf-8",
    )

    code, out, err = run_command(
        capsys,
        "--format",
        "byo",
        "--script-file",
        str(source),
        "--config",
        str(config),
    )

    assert code == 1
    assert out == ""
    assert "script has no content" in err


class RecordingWrite(Stage):
    """A write stage that records whether the orchestrator called it."""

    name = "write"
    artifacts = ("script.en.json",)

    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__(
            name="write",
            options=options,
            artifacts=("script.en.json",),
        )
        self.called = False

    async def run(self, context: StageContext) -> StageResult:
        self.called = True
        (context.run_dir / "script.en.json").write_text(
            json.dumps({"language": "en", "beats": [{"text": "written"}]}),
            encoding="utf-8",
        )
        return StageResult(
            run_id=context.run.id if context.run is not None else "",
            stage=self.name,
            artifacts=["script.en.json"],
            cost_usd=0.0005,
        )


class NoopStage(Stage):
    """An implemented stand-in for a stage the execution test does not inspect."""

    async def run(self, context: StageContext) -> StageResult:
        return StageResult(
            run_id=context.run.id if context.run is not None else "",
            stage=self.name,
            status=StageState.SUCCEEDED,
            cost_usd=0.0,
        )


def _build_recording_stages(
    recipe: Any,
    writes: list[RecordingWrite],
) -> list[Stage]:
    stages: list[Stage] = []
    for item in recipe.stages:
        if item.stage == "input_topic":
            stages.append(InputTopicStage(options=dict(item.options)))
        elif item.stage == "write":
            stage = RecordingWrite(dict(item.options))
            writes.append(stage)
            stages.append(stage)
        else:
            stages.append(NoopStage(item.stage, **item.to_stage_kwargs()))
    return stages


def _run_dir(database: Path, run_id: str) -> Path:
    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).get(run_id)
        assert run is not None
        assert run.run_dir
        return Path(run.run_dir)
    finally:
        engine.dispose()


def test_a_text_file_feeds_the_topic_stage_and_writing_still_runs(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = config.parent / "facts.txt"
    source.write_text(
        "Honey never spoils.\nBees recognise human faces.\n",
        encoding="utf-8",
    )
    writes: list[RecordingWrite] = []
    monkeypatch.setattr(
        "aivia.cli.run.build_stages",
        lambda recipe: _build_recording_stages(recipe, writes),
    )

    code, out, err = run_command(
        capsys,
        "--format",
        "byo",
        "--text-file",
        str(source),
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert writes[0].called is True
    run_id = run_id_from(out)
    run_dir = _run_dir(database, run_id)
    facts = json.loads((run_dir / "facts.json").read_text(encoding="utf-8"))
    assert [fact["text"] for fact in facts["facts"]] == [
        "Honey never spoils.",
        "Bees recognise human faces.",
    ]


def test_a_script_file_is_seeded_and_the_write_stage_is_skipped(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = config.parent / "script.json"
    source.write_text(
        json.dumps(
            {
                "language": "en",
                "title": "Honey",
                "hook": "Honey keeps forever.",
                "beats": [{"text": "Honey never spoils."}],
            }
        ),
        encoding="utf-8",
    )
    writes: list[RecordingWrite] = []
    monkeypatch.setattr(
        "aivia.cli.run.build_stages",
        lambda recipe: _build_recording_stages(recipe, writes),
    )

    code, out, err = run_command(
        capsys,
        "--format",
        "byo",
        "--script-file",
        str(source),
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert writes[0].called is False
    run_id = run_id_from(out)
    run_dir = _run_dir(database, run_id)
    script = json.loads((run_dir / "script.en.json").read_text(encoding="utf-8"))
    assert script["hook"] == "Honey keeps forever."
    assert script["beats"][0]["text"] == "Honey never spoils."

    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).get(run_id)
        assert run is not None
        assert run.state is RunState.SUCCEEDED
        assert run.cost_usd == pytest.approx(0.0)
        stage = StageRunRepository(engine).get(run_id, "write")
        assert stage is not None
        assert stage.status is StageState.SKIPPED
        assert stage.cost_usd == pytest.approx(0.0)
    finally:
        engine.dispose()


def test_an_enqueued_run_keeps_the_provided_input_options(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = config.parent / "script.json"
    source.write_text(
        json.dumps({"language": "en", "beats": [{"text": "Honey never spoils."}]}),
        encoding="utf-8",
    )

    code, out, err = run_command(
        capsys,
        "--format",
        "byo",
        "--script-file",
        str(source),
        "--enqueue",
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    run_id = run_id_from(out)
    run_dir = _run_dir(database, run_id)
    assert (run_dir / "script.en.json").is_file()

    engine = create_database_engine(database)
    try:
        jobs = JobRepository(engine).list(run_id=run_id)
        assert len(jobs) == 1
        assert jobs[0].payload["stage_options"]["write"] == {"skip": True}
        assert jobs[0].payload["stage_options"]["research"] == {"skip": True}
    finally:
        engine.dispose()
