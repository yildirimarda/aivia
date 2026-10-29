"""Stage base class and the context a stage runs with."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import cast

import pytest

from aivia.core.cost import CostTracker
from aivia.core.estimate import EstimateSource, StageEstimate
from aivia.core.models.run import Run, RunKind, StageResult
from aivia.core.router import Router
from aivia.core.stages import Artifact, ArtifactDeclaration, Stage, StageContext
from aivia.providers.protocols.storage import StorageProvider


def test_stage_defaults_to_no_name_no_artifacts_and_no_options() -> None:
    stage = Stage()

    assert stage.name == ""
    assert stage.artifacts == ()
    assert stage.artifact_declarations == ()
    assert stage.artifact_paths == ()
    assert stage.options == {}


def test_stage_declares_a_name_and_artifacts_on_the_subclass() -> None:
    class RenderStage(Stage):
        name = "render"
        artifacts = ("video.mp4", "thumbnail.png")

    stage = RenderStage()

    assert stage.name == "render"
    assert stage.stage_name == "render"
    assert stage.artifacts == ("video.mp4", "thumbnail.png")
    assert stage.artifact_paths == ("video.mp4", "thumbnail.png")


def test_stage_constructor_overrides_name_options_and_artifacts() -> None:
    stage = Stage(
        "  write  ",
        options={"mode": "facts"},
        artifacts=["script.md", "  facts.json  ", "script.md"],
    )

    assert stage.name == "write"
    assert stage.options == {"mode": "facts"}
    assert stage.artifacts == ("script.md", "facts.json")
    assert stage.artifact_paths == ("script.md", "facts.json")


def test_stage_options_are_copied_not_shared() -> None:
    options = {"mode": "facts"}
    stage = Stage("write", options=options)
    options["mode"] = "quote"

    assert stage.options == {"mode": "facts"}


@pytest.mark.parametrize("name", ["", "   "])
def test_stage_rejects_an_empty_name(name: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Stage(name)


def test_stage_rejects_a_non_string_name() -> None:
    with pytest.raises(TypeError, match="string"):
        Stage(cast(str, 42))


def test_stage_rejects_non_string_artifact_declarations() -> None:
    with pytest.raises(TypeError, match="strings"):
        Stage("render", artifacts=cast(list[str], [42]))


def test_artifact_aliases_are_declaration_strings() -> None:
    declaration: ArtifactDeclaration = "video.mp4"
    artifact: Artifact = declaration

    assert artifact == "video.mp4"
    assert isinstance(artifact, str)


def test_base_stage_estimate_is_free_and_zero() -> None:
    estimate = Stage(name="render").estimate()

    assert isinstance(estimate, StageEstimate)
    assert estimate.stage == "render"
    assert estimate.amount_usd == 0.0
    assert estimate.source is EstimateSource.FREE
    assert estimate.free is True
    assert estimate.paid is False


def test_a_stage_can_override_its_estimate() -> None:
    class PaidStage(Stage):
        name = "write"

        def estimate(self, context: StageContext | None = None) -> StageEstimate:
            return StageEstimate(
                stage=self.name,
                capability="llm",
                provider="example",
                units=2.0,
                unit_kind="call",
                amount_usd=0.25,
                source=EstimateSource.METADATA,
            )

    estimate = PaidStage().estimate()

    assert estimate.capability == "llm"
    assert estimate.provider == "example"
    assert estimate.amount_usd == pytest.approx(0.25)
    assert estimate.paid is True


def test_base_stage_run_raises_until_implemented(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="does not implement run"):
        asyncio.run(Stage("render").run(StageContext(run_dir=tmp_path)))


def test_a_stage_runs_with_every_context_collaborator(tmp_path: Path) -> None:
    cost = cast(CostTracker, object())
    storage = cast(StorageProvider, object())
    router = Router()
    logger = logging.getLogger("test.stage")
    run = Run(kind=RunKind.MEDIA, run_dir=str(tmp_path))

    class ArtifactStage(Stage):
        name = "render"
        artifacts = ("out.txt",)

        async def run(self, context: StageContext) -> StageResult:
            (context.run_dir / "out.txt").write_text("rendered", encoding="utf-8")
            return StageResult(
                run_id=context.run.id if context.run is not None else "",
                stage=self.name,
                artifacts=list(self.artifacts),
            )

    context = StageContext(
        run_dir=cast(Path, str(tmp_path)),
        config={"format": "sample"},
        cost=cost,
        router=router,
        storage=storage,
        logger=logger,
        run=run,
    )
    result = asyncio.run(ArtifactStage().run(context))

    assert context.run_dir == tmp_path
    assert context.run_directory == tmp_path
    assert context.config == {"format": "sample"}
    assert context.cost is cost
    assert context.cost_tracker is cost
    assert context.router is router
    assert context.storage is storage
    assert context.logger is logger
    assert context.run is run
    assert result.stage == "render"
    assert result.run_id == run.id
    assert result.artifacts == ["out.txt"]
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "rendered"


def test_context_defaults_are_usable_for_a_deterministic_stage(tmp_path: Path) -> None:
    context = StageContext(run_dir=tmp_path)

    assert context.run_dir == tmp_path
    assert context.config == {}
    assert context.cost is None
    assert context.router is None
    assert context.storage is None
    assert context.logger.name == "aivia.stages"
    assert context.run is None


def test_context_keeps_the_run_record_when_one_is_supplied(tmp_path: Path) -> None:
    run = Run(run_dir=str(tmp_path))
    context = StageContext(run_dir=tmp_path, run=run)

    assert context.run_dir == tmp_path
    assert context.run is run
