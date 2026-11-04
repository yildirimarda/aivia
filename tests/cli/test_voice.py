"""``aivia voice``: run the voice, subtitle and timeline stages of a run."""

from __future__ import annotations

import json
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
formats:
  sample:
    description: A small audio recipe.
    stages:
      - stage: voice
        options:
          languages: [en, de]
      - stage: subtitle
        options:
          languages: [en, de]
      - stage: timeline
        options:
          languages: [en, de]
    capabilities: [tts]
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
    run_id: str = "run_voice",
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


def voice_command(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, str, str]:
    code = main(["voice", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class FakeVoice(Stage):
    """A voice stage that records a measured duration per language."""

    name = "voice"
    artifacts = ("timeline.json",)

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        *,
        durations: dict[str, float] | None = None,
        timeline_durations: dict[str, float] | None = None,
        amount_usd: float = 0.02,
    ) -> None:
        super().__init__(
            name="voice",
            options=options,
            artifacts=("timeline.json",),
        )
        self.calls: list[dict[str, Any]] = []
        self.durations = durations if durations is not None else {"en": 4.25, "de": 6.5}
        self.timeline_durations = timeline_durations
        self.amount_usd = amount_usd

    async def run(self, context: StageContext) -> StageResult:
        assert context.run is not None
        assert context.cost is not None
        self.calls.append(dict(self.options))
        languages = [
            str(code)
            for code in self.options.get("languages", self.durations)
            if str(code) in self.durations
        ]
        payload = {
            "id": context.run.id,
            "languages": [
                {
                    "language": code,
                    "voices": [
                        {
                            "source": f"audio/{code}.mp3",
                            "start_seconds": 0.0,
                            "end_seconds": self.durations[code],
                        }
                    ],
                }
                for code in languages
            ],
        }
        (context.run_dir / "timeline.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        event = context.cost.record(
            provider="mock",
            model="mock-tts",
            operation="voice",
            units=100.0,
            unit_kind="character",
            amount_usd=self.amount_usd,
            run_id=context.run.id,
        )
        metadata: dict[str, Any] = {}
        if self.timeline_durations is None:
            metadata["durations"] = {code: self.durations[code] for code in languages}
        return StageResult(
            run_id=context.run.id,
            stage=self.name,
            artifacts=["timeline.json"],
            cost_usd=event.amount_usd,
            metadata=metadata,
        )


class FakeSubtitle(Stage):
    """A subtitle stage that writes one ASS file per language."""

    name = "subtitle"
    artifacts = ("subtitle.en.ass",)

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(name="subtitle", options=options)
        self.calls: list[dict[str, Any]] = []

    async def run(self, context: StageContext) -> StageResult:
        assert context.run is not None
        self.calls.append(dict(self.options))
        languages = [str(code) for code in self.options.get("languages", ["en"])]
        artifacts = []
        for code in languages:
            relative = f"subtitle.{code}.ass"
            (context.run_dir / relative).write_text("[Script Info]\n", encoding="utf-8")
            artifacts.append(relative)
        return StageResult(
            run_id=context.run.id,
            stage=self.name,
            artifacts=artifacts,
        )


class FakeTimeline(Stage):
    """A timeline stage that records the languages it received."""

    name = "timeline"
    artifacts = ("timeline.composed.json",)

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(name="timeline", options=options)
        self.calls: list[dict[str, Any]] = []

    async def run(self, context: StageContext) -> StageResult:
        assert context.run is not None
        self.calls.append(dict(self.options))
        (context.run_dir / "timeline.composed.json").write_text("{}", encoding="utf-8")
        return StageResult(
            run_id=context.run.id,
            stage=self.name,
            artifacts=["timeline.composed.json"],
        )


def _stages(
    *,
    durations: dict[str, float] | None = None,
    timeline_durations: dict[str, float] | None = None,
) -> tuple[FakeVoice, FakeSubtitle, FakeTimeline]:
    return (
        FakeVoice(
            options={"languages": ["en", "de"]},
            durations=durations,
            timeline_durations=timeline_durations,
        ),
        FakeSubtitle(options={"languages": ["en", "de"]}),
        FakeTimeline(options={"languages": ["en", "de"]}),
    )


def test_voice_runs_the_audio_stages_with_a_language_filter_and_reports_cost(
    config: Path,
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = seed_run(database, tmp_path)
    voice, subtitle, timeline = _stages()
    monkeypatch.setattr(
        "aivia.cli.voice.build_stages",
        lambda recipe: [voice, subtitle, timeline],
    )

    code, out, err = voice_command(
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
    assert "Run: run_voice" in out
    assert "Stages: voice, subtitle, timeline" in out
    assert "Stage: voice" in out
    assert "Stage: subtitle" in out
    assert "Stage: timeline" in out
    assert "Languages: en" in out
    assert "Cost: $0.020000" in out
    assert "Durations: en 4.250s" in out
    assert "Duration: 4.250s" in out
    assert "Timeline: timeline.json" in out
    expected = [{"languages": ["en"], "language": "en"}]
    assert voice.calls == expected
    assert subtitle.calls == expected
    assert timeline.calls == expected

    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).get(run_id)
        assert run is not None
        assert run.state is RunState.SUCCEEDED
        assert run.cost_usd == pytest.approx(0.02)
        assert StageRunRepository(engine).get(run_id, "voice") is not None
        assert StageRunRepository(engine).get(run_id, "subtitle") is not None
        assert StageRunRepository(engine).get(run_id, "timeline") is not None
    finally:
        engine.dispose()


def test_voice_without_a_filter_keeps_every_stage_language(
    config: Path,
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = seed_run(database, tmp_path)
    voice, subtitle, timeline = _stages()
    monkeypatch.setattr(
        "aivia.cli.voice.build_stages",
        lambda recipe: [voice, subtitle, timeline],
    )

    code, out, err = voice_command(
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
    assert "Durations: en 4.250s, de 6.500s" in out
    assert "Duration: 6.500s" in out
    expected = [{"languages": ["en", "de"]}]
    assert voice.calls == expected
    assert subtitle.calls == expected
    assert timeline.calls == expected


def test_voice_reads_durations_from_the_timeline_when_metadata_omits_them(
    config: Path,
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = seed_run(database, tmp_path)
    voice, subtitle, timeline = _stages(timeline_durations={"en": 4.25, "de": 6.5})
    monkeypatch.setattr(
        "aivia.cli.voice.build_stages",
        lambda recipe: [voice, subtitle, timeline],
    )

    code, out, err = voice_command(
        capsys,
        run_id,
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert "Durations: en 4.250s, de 6.500s" in out
    assert "Duration: 6.500s" in out


def test_voice_rejects_an_unknown_language_before_running_a_stage(
    config: Path,
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = seed_run(database, tmp_path)
    voice, subtitle, timeline = _stages()
    monkeypatch.setattr(
        "aivia.cli.voice.build_stages",
        lambda recipe: [voice, subtitle, timeline],
    )

    code, out, err = voice_command(
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
    assert voice.calls == []
    assert subtitle.calls == []
    assert timeline.calls == []


def test_voice_reports_a_missing_run_and_a_recipe_without_a_voice_stage(
    config: Path,
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, out, err = voice_command(
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
        "aivia.cli.voice.build_stages",
        lambda recipe: [Stage(name=stage_name) for stage_name in ["render"]],
    )
    code, out, err = voice_command(
        capsys,
        run_id,
        "--config",
        str(render_config),
        "--database",
        str(database),
    )
    assert code == 1
    assert out == ""
    assert "format has no voice stage" in err


def test_voice_requires_a_run_id(
    config: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = voice_command(capsys, "--config", str(config))

    assert code == 2
    assert out == ""
    assert "a run id is required" in err
