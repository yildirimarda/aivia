"""``aivia write``: run the write stage for an existing run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aivia.cli import main
from aivia.core.db import (
    RunRepository,
    StageRunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.models import Run, RunState, StageResult
from aivia.core.run_dirs import RunDirectories
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
languages:
  default: en
  en:
    name: English
  de:
    name: German
providers:
  text:
    type: mock-llm
formats:
  sample:
    description: A small writing recipe.
    stages:
      - stage: write
        options:
          languages: [en, de]
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


def seed_run(
    database: Path,
    tmp_path: Path,
    *,
    run_id: str = "run_write",
    format_name: str = "sample",
) -> str:
    engine = create_database_engine(database)
    try:
        run = Run(id=run_id, format=format_name, state=RunState.QUEUED)
        RunDirectories(tmp_path / "data").create(run)
        RunRepository(engine).create(run)
        return run.id
    finally:
        engine.dispose()


def write_command(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, str, str]:
    code = main(["write", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class FakeWrite(Stage):
    """A write stage that records one priced token call."""

    name = "write"
    artifacts = ("script.json",)

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        *,
        metadata_tokens: bool = True,
        amount_usd: float = 0.0015,
    ) -> None:
        super().__init__(
            name="write",
            options=options,
            artifacts=("script.json",),
        )
        self.calls: list[dict[str, Any]] = []
        self.metadata_tokens = metadata_tokens
        self.amount_usd = amount_usd

    async def run(self, context: StageContext) -> StageResult:
        assert context.run is not None
        assert context.cost is not None
        self.calls.append(dict(self.options))
        (context.run_dir / "script.json").write_text(
            '{"language": "en"}', encoding="utf-8"
        )
        event = context.cost.record(
            provider="mock",
            model="mock-llm",
            operation="write",
            units=30.0,
            unit_kind="token",
            amount_usd=self.amount_usd,
            run_id=context.run.id,
            metadata={"input_tokens": 20, "output_tokens": 10},
        )
        metadata = (
            {"input_tokens": 20, "output_tokens": 10} if self.metadata_tokens else {}
        )
        return StageResult(
            run_id=context.run.id,
            stage=self.name,
            artifacts=["script.json"],
            cost_usd=event.amount_usd,
            metadata=metadata,
        )


def test_write_runs_the_stage_for_an_existing_run_with_a_language_filter(
    config: Path,
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = seed_run(database, tmp_path)
    stage = FakeWrite(options={"languages": ["en", "de"]})
    monkeypatch.setattr(
        "aivia.cli.write.build_stages",
        lambda recipe: [stage],
    )

    code, out, err = write_command(
        capsys,
        run_id,
        "--language",
        "en",
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert "Run: run_write" in out
    assert "Stage: write" in out
    assert "Languages: en" in out
    assert "Cost: $0.001500" in out
    assert "Tokens: 30 (input 20, output 10)" in out
    assert stage.calls == [{"languages": ["en"], "language": "en"}]

    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).get(run_id)
        assert run is not None
        assert run.state is RunState.SUCCEEDED
        assert run.cost_usd == pytest.approx(0.0015)
        result = StageRunRepository(engine).get(run_id, "write")
        assert result is not None
        assert result.artifacts == ["script.json"]
    finally:
        engine.dispose()


def test_write_without_a_filter_keeps_the_stage_languages(
    config: Path,
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = seed_run(database, tmp_path)
    stage = FakeWrite(options={"languages": ["en", "de"]})
    monkeypatch.setattr(
        "aivia.cli.write.build_stages",
        lambda recipe: [stage],
    )

    code, out, err = write_command(
        capsys,
        run_id,
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert "Languages: en, de" in out
    assert stage.calls == [{"languages": ["en", "de"]}]


def test_write_reads_tokens_from_cost_events_when_the_stage_omits_them(
    config: Path,
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = seed_run(database, tmp_path)
    stage = FakeWrite(options={"languages": ["en", "de"]}, metadata_tokens=False)
    monkeypatch.setattr(
        "aivia.cli.write.build_stages",
        lambda recipe: [stage],
    )

    code, out, err = write_command(
        capsys,
        run_id,
        "--languages",
        "de",
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert "Languages: de" in out
    assert "Tokens: 30 (input 20, output 10)" in out
    assert stage.calls == [{"languages": ["de"], "language": "de"}]


def test_write_rejects_an_unknown_language_before_running_the_stage(
    config: Path,
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = seed_run(database, tmp_path)
    stage = FakeWrite(options={"languages": ["en", "de"]})
    monkeypatch.setattr(
        "aivia.cli.write.build_stages",
        lambda recipe: [stage],
    )

    code, out, err = write_command(
        capsys,
        run_id,
        "--language",
        "fr",
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 1
    assert out == ""
    assert "unknown language(s): fr" in err
    assert stage.calls == []


def test_write_reports_a_missing_run_and_a_recipe_without_a_write_stage(
    config: Path,
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, out, err = write_command(
        capsys,
        "run_missing",
        "--config",
        str(config),
        "--database",
        str(database),
    )
    assert code == 1
    assert out == ""
    assert "run not found: run_missing" in err

    run_id = seed_run(database, tmp_path, run_id="run_render", format_name="render")
    render_config = write_config(
        tmp_path / "other",
        """
formats:
  render:
    description: Rendering only.
    stages:
      - stage: render
    output_type: image
    handler: save
""",
    )
    monkeypatch.setattr(
        "aivia.cli.write.build_stages",
        lambda recipe: [Stage(name=stage_name) for stage_name in ["render"]],
    )
    code, out, err = write_command(
        capsys,
        run_id,
        "--config",
        str(render_config),
        "--database",
        str(database),
    )
    assert code == 1
    assert out == ""
    assert "format has no write stage" in err
