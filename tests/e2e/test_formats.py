"""Every built-in format runs its packaged recipe through a mock pipeline.

The recipe, stage order and stage options come from the packaged format files.
Only the stages themselves are local mocks: each one writes one artifact,
records one paid call and writes the matching cost sidecar. The test then
checks the same durable records a real run leaves behind: the artifact
manifest, the stage rows, the cost ledger and the per-output sidecars.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine

from aivia.core.config.formats import FormatRecipeConfig
from aivia.core.cost import CostTracker
from aivia.core.db import (
    RunRepository,
    StageRunRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.models import (
    CostEvent,
    Run,
    RunKind,
    RunState,
    StageResult,
    StageState,
    ToolchainFingerprint,
)
from aivia.core.orchestrator import OrchestrationResult, Orchestrator
from aivia.core.run_dirs import (
    ARTIFACT_MANIFEST_NAME,
    RunDirectories,
    RunDirectory,
)
from aivia.core.sidecar import sidecar_path, write_cost_sidecar
from aivia.core.stages import Stage, StageContext
from aivia.formats import (
    FormatCatalog,
    StageContract,
    StageRegistry,
    load_packaged_formats,
)

FINGERPRINT = ToolchainFingerprint(
    renderer_version="renderer-1",
    ffmpeg_version="ffmpeg-7",
    font_set="fonts-abc",
)
PRICING_VERSION = "2026-01"
BUILT_IN_FORMATS = load_packaged_formats().names()


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    created = create_database_engine(path=tmp_path / "data" / "aivia.db")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


class _MockStage(Stage):
    """Write one artifact, record one paid call and write its cost sidecar."""

    stage_cost_usd: float = 0.0

    async def run(self, context: StageContext) -> StageResult:
        run = context.run
        cost = context.cost
        if run is None or cost is None:
            raise AssertionError("the mock stage needs a run and a cost tracker")

        output = context.run_dir / self.artifacts[0]
        output.write_text(f"{run.format}:{self.name}", encoding="utf-8")
        event: CostEvent = cost.record(
            provider="mock",
            model="mock-model",
            operation=self.name,
            units=1.0,
            unit_kind="call",
            amount_usd=self.stage_cost_usd,
            run_id=run.id,
            pricing_version=PRICING_VERSION,
            estimated=False,
        )
        write_cost_sidecar(
            output,
            [event],
            toolchain_fingerprint=run.toolchain_fingerprint,
            run_id=run.id,
        )
        return StageResult(
            run_id=run.id,
            stage=self.name,
            artifacts=list(self.artifacts),
            cost_usd=event.amount_usd,
        )


def _stage_class(name: str, artifact: str, cost_usd: float) -> type[Stage]:
    return type(
        f"Mock_{name}",
        (_MockStage,),
        {
            "name": name,
            "artifacts": (artifact,),
            "stage_cost_usd": cost_usd,
        },
    )


def _registry(
    recipe: FormatRecipeConfig,
) -> tuple[StageRegistry, dict[str, tuple[str, float]]]:
    registry = StageRegistry()
    produced: tuple[str, ...] = ()
    expected: dict[str, tuple[str, float]] = {}
    for index, stage in enumerate(recipe.stages):
        artifact = f"{index:02d}-{stage.stage}.txt"
        cost_usd = round(0.01 * (index + 1), 6)
        stage_class = _stage_class(stage.stage, artifact, cost_usd)
        output = f"out:{stage.stage}"
        registry.register(
            StageContract(
                stage.stage,
                inputs=produced,
                outputs=(output,),
                stage=stage_class,
            )
        )
        produced = (*produced, output)
        expected[stage.stage] = (artifact, cost_usd)
    return registry, expected


def _execute(
    engine: Engine,
    tmp_path: Path,
    catalog: FormatCatalog,
    format_name: str,
) -> tuple[OrchestrationResult, Run, RunDirectory, dict[str, tuple[str, float]]]:
    recipe = catalog.get(format_name)
    registry, expected = _registry(recipe)
    data_dir = tmp_path / "data"
    estimated_cost = sum(cost for _, cost in expected.values())
    run = Run(
        kind=RunKind.PIPELINE,
        profile="smoke",
        format=format_name,
        toolchain_fingerprint=FINGERPRINT,
        estimated_cost_usd=estimated_cost,
    )
    directory = RunDirectories(data_dir).create(run)
    RunRepository(engine).create(run)

    result = asyncio.run(
        Orchestrator(
            run=run.id,
            repository=engine,
            run_dir=directory,
            data_dir=data_dir,
            catalog=catalog,
            registry=registry,
            config={"format": format_name},
            cost=CostTracker(engine),
            raise_on_failure=False,
        ).execute()
    )
    return result, run, directory, expected


@pytest.mark.parametrize("format_name", BUILT_IN_FORMATS)
def test_every_built_in_format_produces_artifacts_costs_and_sidecars(
    engine: Engine,
    tmp_path: Path,
    format_name: str,
) -> None:
    catalog = load_packaged_formats()
    recipe = catalog.get(format_name)
    stage_names = [stage.stage for stage in recipe.stages]
    result, run, directory, expected = _execute(
        engine,
        tmp_path,
        catalog,
        format_name,
    )

    assert result.ok is True
    assert result.failed_stage == ""
    assert result.run.format == format_name
    assert result.run.state is RunState.SUCCEEDED
    assert [stage.stage for stage in result.stages] == stage_names
    assert [stage.status for stage in result.stages] == [StageState.SUCCEEDED] * len(
        stage_names
    )

    expected_total = sum(cost for _, cost in expected.values())
    stored_run = RunRepository(engine).get(run.id)
    assert stored_run is not None
    assert stored_run.state is RunState.SUCCEEDED
    assert stored_run.cost_usd == pytest.approx(expected_total)
    assert result.run.cost_usd == pytest.approx(expected_total)

    stages = {
        stage.stage: stage for stage in StageRunRepository(engine).list(run_id=run.id)
    }
    assert set(stages) == set(stage_names)
    for index, stage_name in enumerate(stage_names):
        artifact, cost = expected[stage_name]
        stage = stages[stage_name]
        assert stage.status is StageState.SUCCEEDED
        assert stage.index == index
        assert stage.artifacts == [artifact]
        assert stage.cost_usd == pytest.approx(cost)
        assert stage.input_hash
        assert stage.output_hash

    manifest = json.loads(
        (directory.path / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["run_id"] == run.id
    assert set(manifest["stages"]) == set(stage_names)
    registered = directory.artifacts()
    assert {record.path for record in registered} == {
        artifact for artifact, _ in expected.values()
    }
    assert all(record.sha256 for record in registered)

    events = CostTracker(engine).for_run(run.id)
    assert len(events) == len(stage_names)
    assert {event.operation for event in events} == set(stage_names)
    assert {event.amount_kind for event in events} == {"actual"}
    assert {event.pricing_version for event in events} == {PRICING_VERSION}
    assert sum(event.amount_usd for event in events) == pytest.approx(expected_total)
    for event in events:
        assert event.amount_usd == pytest.approx(expected[event.operation][1])

    for stage_name in stage_names:
        artifact, cost = expected[stage_name]
        output = directory.path / artifact
        assert output.is_file()
        assert output.read_text(encoding="utf-8") == f"{format_name}:{stage_name}"
        sidecar_file = sidecar_path(output)
        assert sidecar_file.is_file()
        payload = json.loads(sidecar_file.read_text(encoding="utf-8"))
        assert payload["run_id"] == run.id
        assert payload["output"] == str(output)
        assert [item["operation"] for item in payload["items"]] == [stage_name]
        assert payload["totals"]["count"] == 1
        assert payload["totals"]["actual_usd"] == pytest.approx(cost)
        assert payload["pricing_version"] == PRICING_VERSION
        assert payload["providers"] == ["mock"]
        assert payload["models"] == ["mock-model"]
        assert payload["toolchain_fingerprint"]["digest"] == FINGERPRINT.digest
