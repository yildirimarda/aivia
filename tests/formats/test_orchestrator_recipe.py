"""A run executes the stages of its recipe loaded from format files."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

from aivia.core.db import (
    RunRepository,
    StageRunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.errors import ConfigError
from aivia.core.models.run import Run, StageResult, StageState
from aivia.core.orchestrator import Orchestrator, execute_recipe
from aivia.core.run_state import RunStateMachine
from aivia.core.stages import Stage, StageContext
from aivia.formats import (
    FormatCatalog,
    StageContract,
    StageRegistry,
    load_format_file,
    load_formats,
)


class Topic:
    """A stand-in for the topic a recipe starts from."""


class Script:
    """A stand-in for a written script."""


class Audio:
    """A stand-in for a rendered voice track."""


class Video:
    """A stand-in for a rendered video."""


class Recording:
    events: ClassVar[list[str]] = []


class WriteStage(Stage):
    name = "write"
    artifacts = ("script.md",)

    async def run(self, context: StageContext) -> StageResult:
        Recording.events.append(self.name)
        (context.run_dir / "script.md").write_text("script", encoding="utf-8")
        return StageResult(
            run_id=context.run.id if context.run is not None else "",
            stage=self.name,
            artifacts=["script.md"],
            cost_usd=0.25,
        )


class VoiceStage(Stage):
    name = "voice"
    artifacts = ("voice.wav",)

    async def run(self, context: StageContext) -> StageResult:
        Recording.events.append(self.name)
        (context.run_dir / "voice.wav").write_text("voice", encoding="utf-8")
        return StageResult(
            run_id=context.run.id if context.run is not None else "",
            stage=self.name,
            artifacts=["voice.wav"],
            cost_usd=1.5,
        )


class RenderStage(Stage):
    name = "render"
    artifacts = ("clip.mp4",)

    async def run(self, context: StageContext) -> StageResult:
        Recording.events.append(self.name)
        (context.run_dir / "clip.mp4").write_text("clip", encoding="utf-8")
        return StageResult(
            run_id=context.run.id if context.run is not None else "",
            stage=self.name,
            artifacts=["clip.mp4"],
            cost_usd=0.75,
        )


@pytest.fixture(autouse=True)
def reset_events() -> Iterator[None]:
    Recording.events = []
    yield
    Recording.events = []


def _registry() -> StageRegistry:
    stages = StageRegistry()
    stages.register(
        StageContract("write", inputs=(Topic,), outputs=(Script,), stage=WriteStage)
    )
    stages.register(
        StageContract("voice", inputs=(Script,), outputs=(Audio,), stage=VoiceStage)
    )
    stages.register(
        StageContract(
            "render",
            inputs=(Script, Audio),
            outputs=(Video,),
            stage=RenderStage,
        )
    )
    return stages


def _write_recipe(
    directory: Path,
    name: str,
    stages: list[object],
    **values: object,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    payload = {"stages": stages, **values}
    import yaml

    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _catalog(tmp_path: Path, *, user: bool = True) -> FormatCatalog:
    packaged = tmp_path / "packaged"
    _write_recipe(
        packaged / "formats",
        "base",
        ["write", "voice", "render"],
        description="packaged",
    )
    config_dir = tmp_path / "config"
    if user:
        _write_recipe(
            config_dir / "formats",
            "custom",
            ["write", "voice", "render"],
            description="user",
        )
        _write_recipe(
            config_dir / "formats",
            "base",
            ["write", "render"],
            description="user override",
        )
    return load_formats(
        config_dir=config_dir,
        defaults_dir=packaged,
    )


def _orchestrator(
    tmp_path: Path,
    catalog: FormatCatalog,
    *,
    format_name: str = "custom",
    run: Run | str | None = None,
    **options: Any,
) -> Orchestrator:
    return Orchestrator(
        run=Run(format=format_name) if run is None else run,
        run_dir=tmp_path / "run",
        catalog=catalog,
        registry=_registry(),
        config={},
        inputs=(Topic,),
        **options,
    )


def test_a_run_executes_the_stages_of_its_loaded_recipe(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)

    result = asyncio.run(_orchestrator(tmp_path, catalog).execute())

    assert Recording.events == ["write", "voice", "render"]
    assert [item.stage for item in result.stages] == ["write", "voice", "render"]
    assert [item.status for item in result.stages] == [
        StageState.SUCCEEDED,
        StageState.SUCCEEDED,
        StageState.SUCCEEDED,
    ]
    assert result.run.cost_usd == pytest.approx(2.5)
    assert result.ok is True


def test_user_files_override_packaged_files_and_packaged_remain_available(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)

    assert catalog.names() == ("base", "custom")
    assert [item.stage for item in catalog.get("base").stages] == [
        "write",
        "render",
    ]
    assert catalog.get("base").description == "user override"
    assert catalog.get("custom").description == "user"


def test_packaged_files_load_when_the_user_directory_is_missing(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path, user=False)

    assert catalog.names() == ("base",)
    assert [item.stage for item in catalog.get("base").stages] == [
        "write",
        "voice",
        "render",
    ]


def test_stages_outside_the_recipe_never_run_and_cost_nothing(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    _write_recipe(config_dir / "formats", "small", ["write"])
    catalog = load_formats(
        config_dir=config_dir,
        defaults_dir=tmp_path / "empty",
    )

    result = asyncio.run(
        _orchestrator(tmp_path, catalog, format_name="small").execute()
    )

    assert Recording.events == ["write"]
    assert [item.stage for item in result.stages] == ["write"]
    assert result.run.cost_usd == pytest.approx(0.25)


def test_an_unused_stage_is_skipped_with_zero_cost(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)

    result = asyncio.run(
        _orchestrator(tmp_path, catalog, skip_stages=["voice"]).execute()
    )

    assert Recording.events == ["write", "render"]
    statuses = {item.stage: item.status for item in result.stages}
    assert statuses == {
        "write": StageState.SUCCEEDED,
        "voice": StageState.SKIPPED,
        "render": StageState.SUCCEEDED,
    }
    skipped = next(item for item in result.stages if item.stage == "voice")
    assert skipped.cost_usd == 0.0
    assert skipped.artifacts == []
    assert skipped.metadata["unused"] is True
    assert result.run.cost_usd == pytest.approx(1.0)


def test_a_recipe_can_disable_a_stage_at_zero_cost(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    _write_recipe(
        config_dir / "formats",
        "quiet",
        [
            "write",
            {"stage": "voice", "options": {"enabled": False}},
            "render",
        ],
    )
    catalog = load_formats(
        config_dir=config_dir,
        defaults_dir=tmp_path / "empty",
    )

    result = asyncio.run(
        _orchestrator(tmp_path, catalog, format_name="quiet").execute()
    )

    assert Recording.events == ["write", "render"]
    assert result.run.cost_usd == pytest.approx(1.0)
    skipped = next(item for item in result.stages if item.stage == "voice")
    assert skipped.status is StageState.SKIPPED
    assert skipped.cost_usd == 0.0


def test_a_provided_output_skips_the_stage_that_would_produce_it(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)

    result = asyncio.run(
        _orchestrator(
            tmp_path,
            catalog,
            provided_inputs=(Script,),
        ).execute()
    )

    assert Recording.events == ["voice", "render"]
    assert [item.stage for item in result.stages] == ["write", "voice", "render"]
    skipped = next(item for item in result.stages if item.stage == "write")
    assert skipped.status is StageState.SKIPPED
    assert skipped.cost_usd == 0.0
    assert result.run.cost_usd == pytest.approx(2.25)


def test_editing_a_provided_input_invalidates_downstream_resume(
    tmp_path: Path,
) -> None:
    engine = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=engine)
    try:
        catalog = _catalog(tmp_path)
        run = RunStateMachine(RunRepository(engine)).create(Run(format="custom"))
        repository = StageRunRepository(engine)
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        script = run_dir / "script.md"
        script.write_text("first", encoding="utf-8")

        first = asyncio.run(
            _orchestrator(
                tmp_path,
                catalog,
                run=run,
                repository=repository,
                provided_inputs=(Script,),
            ).execute()
        )

        assert Recording.events == ["voice", "render"]
        first_write = next(item for item in first.stages if item.stage == "write")
        first_voice = next(item for item in first.stages if item.stage == "voice")
        assert first_write.status is StageState.SKIPPED
        assert first_write.output_hash
        assert first_voice.status is StageState.SUCCEEDED

        Recording.events.clear()
        script.write_text("second", encoding="utf-8")
        second = asyncio.run(
            _orchestrator(
                tmp_path,
                catalog,
                run=run.id,
                repository=repository,
                provided_inputs=(Script,),
            ).execute()
        )

        assert Recording.events == ["voice", "render"]
        second_write = next(item for item in second.stages if item.stage == "write")
        second_voice = next(item for item in second.stages if item.stage == "voice")
        assert second_write.status is StageState.SKIPPED
        assert second_write.output_hash != first_write.output_hash
        assert second_voice.status is StageState.SUCCEEDED
        assert second_voice.input_hash != first_voice.input_hash

        Recording.events.clear()
        third = asyncio.run(
            _orchestrator(
                tmp_path,
                catalog,
                run=run.id,
                repository=repository,
                provided_inputs=(Script,),
            ).execute()
        )

        assert Recording.events == []
        assert [item.status for item in third.stages] == [
            StageState.SKIPPED,
            StageState.SKIPPED,
            StageState.SKIPPED,
        ]
    finally:
        engine.dispose()


def test_a_format_name_can_be_passed_to_execute_recipe(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    run = Run()

    result = asyncio.run(
        execute_recipe(
            run,
            "custom",
            run_dir=tmp_path / "run",
            catalog=catalog,
            registry=_registry(),
            config={},
            inputs=(Topic,),
        )
    )

    assert Recording.events == ["write", "voice", "render"]
    assert result.run.format == ""
    assert result.ok is True


def test_a_single_recipe_file_is_named_by_its_stem(tmp_path: Path) -> None:
    path = _write_recipe(tmp_path, "single", ["write", "render"])

    loaded = load_format_file(path)

    assert list(loaded) == ["single"]
    assert [item.stage for item in loaded["single"].stages] == ["write", "render"]


def test_a_named_format_file_can_hold_several_recipes(tmp_path: Path) -> None:
    path = tmp_path / "formats.yaml"
    path.write_text(
        "formats:\n  first:\n    stages: [write]\n  second:\n    stages: [render]\n",
        encoding="utf-8",
    )

    loaded = load_format_file(path)

    assert sorted(loaded) == ["first", "second"]
    assert [item.stage for item in loaded["second"].stages] == ["render"]


def test_an_invalid_format_file_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("stages: [write, write]\n", encoding="utf-8")

    with pytest.raises(ConfigError) as caught:
        load_format_file(path)

    assert str(path) in str(caught.value)


def test_an_unknown_format_lists_the_available_names(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)

    with pytest.raises(ConfigError) as caught:
        catalog.get("missing")

    assert "missing" in str(caught.value)
    assert "base" in caught.value.details["available"]
    assert "custom" in caught.value.details["available"]


def test_the_recipe_is_validated_against_the_stage_registry(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    _write_recipe(config_dir / "formats", "backwards", ["voice", "write"])
    catalog = load_formats(
        config_dir=config_dir,
        defaults_dir=tmp_path / "empty",
    )

    with pytest.raises(ConfigError) as caught:
        asyncio.run(_orchestrator(tmp_path, catalog, format_name="backwards").execute())

    assert "stage 'voice'" in str(caught.value)
    assert Recording.events == []
