"""``aivia add-language``: extend a run without recharging its visuals."""

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
from aivia.core.run_dirs import RunDirectories, RunDirectory
from aivia.core.stages import Stage, StageContext
from aivia.visuals.reuse import (
    DEFAULT_SHARED_VISUALS_FILENAME,
    DEFAULT_VISUAL_REUSE_FILENAME,
)


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
  tr:
    name: Turkish
providers:
  text:
    type: mock-llm
formats:
  sample:
    description: A small writing recipe with visuals.
    stages:
      - stage: write
        options:
          languages: [en, de]
      - stage: visuals
        options:
          strategy: image
    capabilities: [llm, image_gen]
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
    run_id: str = "run_add_language",
    format_name: str = "sample",
) -> tuple[str, Path]:
    engine = create_database_engine(database)
    try:
        run = Run(id=run_id, format=format_name, state=RunState.QUEUED)
        RunDirectories(tmp_path / "data").create(run)
        RunRepository(engine).create(run)
        assert run.run_dir
        return run.id, Path(run.run_dir)
    finally:
        engine.dispose()


def seed_existing_language(
    run_dir: Path,
    language: str,
    beats: int,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "facts.json").write_text(
        json.dumps(
            {
                "facts": [
                    {"text": "Honey never spoils."},
                    {"text": "Bees recognise human faces."},
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / f"script.{language}.json").write_text(
        json.dumps(
            {
                "language": language,
                "title": f"{language} title",
                "beats": [
                    {"text": f"{language} beat {index}"} for index in range(beats)
                ],
            }
        ),
        encoding="utf-8",
    )


def seed_visuals(run_dir: Path, *, beats: int = 2) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": 1,
        "stage": "visual_reuse",
        "source_language": "en",
        "languages": ["en", "de"],
        "beat_counts": {"en": beats, "de": beats},
        "cost_usd": 0.08,
        "beats": [
            {
                "index": index,
                "languages": ["en", "de"],
                "source_language": "en",
                "cost_usd": 0.04,
                "asset": {
                    "kind": "image",
                    "path": f"file:///tmp/visual-{index}.png",
                    "provider": "fake-image",
                },
            }
            for index in range(beats)
        ],
        "metadata": {
            "beats": beats,
            "languages": ["en", "de"],
            "reused_across": 2,
            "cost_usd": 0.08,
        },
    }
    for name in (
        DEFAULT_VISUAL_REUSE_FILENAME,
        DEFAULT_SHARED_VISUALS_FILENAME,
    ):
        (run_dir / name).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    RunDirectory(run_dir, run_id="run_add_language").register(
        "visual_reuse",
        [DEFAULT_VISUAL_REUSE_FILENAME, DEFAULT_SHARED_VISUALS_FILENAME],
    )
    return payload


class FakeWrite(Stage):
    """A write stage that records the languages it was asked to write."""

    name = "write"
    artifacts = ("script.tr.json",)

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(
            name="write",
            options=options,
            artifacts=("script.tr.json",),
        )
        self.calls: list[dict[str, Any]] = []
        self.fact_texts: list[str] = []

    async def run(self, context: StageContext) -> StageResult:
        assert context.run is not None
        assert context.cost is not None
        self.calls.append(dict(self.options))
        facts = json.loads((context.run_dir / "facts.json").read_text(encoding="utf-8"))
        self.fact_texts = [fact["text"] for fact in facts["facts"]]
        languages = list(self.options.get("languages") or ["tr"])
        artifacts: list[str] = []
        for language in languages:
            relative = f"script.{language}.json"
            (context.run_dir / relative).write_text(
                json.dumps(
                    {
                        "language": language,
                        "title": f"{language} title",
                        "beats": [
                            {"text": f"{language} beat 0"},
                            {"text": f"{language} beat 1"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            artifacts.append(relative)
        event = context.cost.record(
            provider="mock",
            model="mock-llm",
            operation="write",
            units=30.0,
            unit_kind="token",
            amount_usd=0.0015,
            run_id=context.run.id,
            metadata={"input_tokens": 20, "output_tokens": 10},
        )
        return StageResult(
            run_id=context.run.id,
            stage=self.name,
            artifacts=artifacts,
            cost_usd=event.amount_usd,
            metadata={"input_tokens": 20, "output_tokens": 10},
        )


class FakeVisuals(Stage):
    """A visual stage that must never be called when adding a language."""

    name = "visuals"
    artifacts = ("visual-reuse.json",)

    def __init__(self) -> None:
        super().__init__(name="visuals", artifacts=("visual-reuse.json",))
        self.calls = 0

    async def run(self, context: StageContext) -> StageResult:
        self.calls += 1
        raise AssertionError("adding a language must not recharge visuals")


def add_language_command(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, str, str]:
    code = main(["add-language", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_add_language_writes_only_the_new_language_and_reuses_visuals(
    config: Path,
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id, run_dir = seed_run(database, tmp_path)
    seed_existing_language(run_dir, "en", 2)
    seed_existing_language(run_dir, "de", 2)
    seed_visuals(run_dir)
    facts_before = (run_dir / "facts.json").read_text(encoding="utf-8")

    write = FakeWrite(options={"languages": ["en", "de"]})
    visuals = FakeVisuals()
    monkeypatch.setattr(
        "aivia.cli.add_language.build_stages",
        lambda recipe: [write, visuals],
    )

    code, out, err = add_language_command(
        capsys,
        run_id,
        "--language",
        "tr",
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert "Run: run_add_language" in out
    assert "Stage: write" in out
    assert "Languages: tr" in out
    assert "Cost: $0.001500" in out
    assert "Tokens: 30 (input 20, output 10)" in out
    assert "Visuals: reused without a provider call" in out

    assert write.calls == [{"languages": ["tr"], "language": "tr"}]
    assert write.fact_texts == [
        "Honey never spoils.",
        "Bees recognise human faces.",
    ]
    assert visuals.calls == 0
    assert (run_dir / "facts.json").read_text(encoding="utf-8") == facts_before
    assert (run_dir / "script.en.json").is_file()
    assert (run_dir / "script.de.json").is_file()
    assert (run_dir / "script.tr.json").is_file()

    for name in (
        DEFAULT_VISUAL_REUSE_FILENAME,
        DEFAULT_SHARED_VISUALS_FILENAME,
    ):
        manifest = json.loads((run_dir / name).read_text(encoding="utf-8"))
        assert manifest["languages"] == ["en", "de", "tr"]
        assert manifest["beat_counts"]["tr"] == 2
        assert manifest["cost_usd"] == pytest.approx(0.08)
        assert all("tr" in beat["languages"] for beat in manifest["beats"])

    directory = RunDirectory(run_dir, run_id=run_id)
    assert directory.verify("visual_reuse") == ()

    engine = create_database_engine(database)
    try:
        run = RunRepository(engine).get(run_id)
        assert run is not None
        assert run.state is RunState.SUCCEEDED
        assert run.cost_usd == pytest.approx(0.0015)
        result = StageRunRepository(engine).get(run_id, "write")
        assert result is not None
        assert result.artifacts == ["script.tr.json"]
    finally:
        engine.dispose()


def test_add_language_accepts_several_languages_and_skips_missing_visuals(
    config: Path,
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id, run_dir = seed_run(database, tmp_path)
    seed_existing_language(run_dir, "en", 2)
    write = FakeWrite(options={"languages": ["en"]})
    visuals = FakeVisuals()
    monkeypatch.setattr(
        "aivia.cli.add_language.build_stages",
        lambda recipe: [write, visuals],
    )

    code, out, err = add_language_command(
        capsys,
        run_id,
        "--languages",
        "de,tr",
        "--config",
        str(config),
        "--database",
        str(database),
    )

    assert code == 0
    assert err == ""
    assert "Languages: de, tr" in out
    assert "Visuals: none" in out
    assert write.calls == [{"languages": ["de", "tr"]}]
    assert visuals.calls == 0
    assert (run_dir / "script.de.json").is_file()
    assert (run_dir / "script.tr.json").is_file()
    assert not (run_dir / DEFAULT_VISUAL_REUSE_FILENAME).exists()


def test_add_language_rejects_an_unknown_language_before_writing(
    config: Path,
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id, run_dir = seed_run(database, tmp_path)
    seed_existing_language(run_dir, "en", 1)
    write = FakeWrite(options={"languages": ["en"]})
    visuals = FakeVisuals()
    monkeypatch.setattr(
        "aivia.cli.add_language.build_stages",
        lambda recipe: [write, visuals],
    )

    code, out, err = add_language_command(
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
    assert write.calls == []
    assert visuals.calls == 0


def test_add_language_reports_a_missing_run_and_a_recipe_without_a_write_stage(
    config: Path,
    database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, out, err = add_language_command(
        capsys,
        "run_missing",
        "--language",
        "tr",
        "--config",
        str(config),
        "--database",
        str(database),
    )
    assert code == 1
    assert out == ""
    assert "run not found: run_missing" in err

    run_id, _run_dir = seed_run(
        database,
        tmp_path,
        run_id="run_render",
        format_name="render",
    )
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
    visuals = FakeVisuals()
    monkeypatch.setattr(
        "aivia.cli.add_language.build_stages",
        lambda recipe: [visuals],
    )
    code, out, err = add_language_command(
        capsys,
        run_id,
        "--language",
        "tr",
        "--config",
        str(render_config),
        "--database",
        str(database),
    )
    assert code == 1
    assert out == ""
    assert "format has no write stage" in err
    assert visuals.calls == 0
