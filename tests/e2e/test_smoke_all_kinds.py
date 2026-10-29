"""Every run kind produces the complete durable record of a real run.

The smoke test drives the same collaborators the application uses: a file
database, a run directory below the data directory, the orchestrator, the cost
ledger and the snapshot/sidecar writers. Only the stage is local to the test,
because the built-in pipeline stages arrive in a later milestone.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine

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
    RUNS_DIR_NAME,
    RunDirectories,
    RunDirectory,
)
from aivia.core.sidecar import sidecar_path, write_cost_sidecar
from aivia.core.snapshot import SNAPSHOT_FILENAME, read_config_snapshot
from aivia.core.stages import Stage, StageContext

FINGERPRINT = ToolchainFingerprint(
    renderer_version="renderer-1",
    ffmpeg_version="ffmpeg-7",
    font_set="fonts-abc",
)
PRICING_VERSION = "2026-01"
STAGE_COST_USD = 0.01


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    created = create_database_engine(path=tmp_path / "data" / "aivia.db")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


class _PaidStage(Stage):
    """Write one output, record its paid call and write its cost sidecar."""

    def __init__(self, name: str, output: str) -> None:
        super().__init__(name=name, artifacts=(output,))
        self.output = output

    async def run(self, context: StageContext) -> StageResult:
        run = context.run
        cost = context.cost
        if run is None or cost is None:
            raise AssertionError("the smoke stage needs a run and a cost tracker")

        output = context.run_dir / self.output
        output.write_text(f"{run.kind.value}:{self.name}", encoding="utf-8")
        event: CostEvent = cost.record(
            provider="mock",
            model="mock-model",
            operation=self.name,
            units=1.0,
            unit_kind="call",
            amount_usd=STAGE_COST_USD,
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
            artifacts=[self.output],
            cost_usd=event.amount_usd,
        )


def _outputs(kind: RunKind) -> dict[str, str]:
    return {
        "generate": f"{kind.value}-asset.txt",
        "finalize": f"{kind.value}-final.txt",
    }


def _execute(
    engine: Engine,
    tmp_path: Path,
    kind: RunKind,
) -> tuple[OrchestrationResult, Run, RunDirectory]:
    data_dir = tmp_path / "data"
    format_name = f"smoke-{kind.value}"
    outputs = _outputs(kind)
    run = Run(
        kind=kind,
        profile="smoke",
        format=format_name,
        toolchain_fingerprint=FINGERPRINT,
        estimated_cost_usd=STAGE_COST_USD * 2,
    )
    directory = RunDirectories(data_dir).create(run)
    RunRepository(engine).create(run)

    result = asyncio.run(
        Orchestrator(
            stages=[_PaidStage(name, output) for name, output in outputs.items()],
            run=run.id,
            repository=engine,
            run_dir=directory,
            data_dir=data_dir,
            config={
                "format": format_name,
                "planner": {"defaults": {"seed": 7}},
                "providers": {"mock": {"type": "mock-llm"}},
            },
            cost=CostTracker(engine),
            raise_on_failure=False,
        ).execute()
    )
    return result, run, directory


@pytest.mark.parametrize("kind", list(RunKind), ids=lambda kind: kind.value)
def test_every_run_kind_produces_a_run_directory_stages_costs_snapshot_and_sidecars(
    engine: Engine,
    tmp_path: Path,
    kind: RunKind,
) -> None:
    result, run, directory = _execute(engine, tmp_path, kind)
    data_dir = (tmp_path / "data").resolve()

    assert result.ok is True
    assert result.failed_stage == ""
    assert result.run.kind is kind
    assert result.run.state is RunState.SUCCEEDED
    assert result.run.run_dir == str(directory.path)

    assert directory.path.is_dir()
    assert directory.path.parent == data_dir / RUNS_DIR_NAME
    assert directory.run_id == run.id

    stored_run = RunRepository(engine).get(run.id)
    assert stored_run is not None
    assert stored_run.kind is kind
    assert stored_run.state is RunState.SUCCEEDED
    assert stored_run.cost_usd == pytest.approx(STAGE_COST_USD * 2)

    stages = StageRunRepository(engine).list(run_id=run.id)
    assert {stage.stage for stage in stages} == {"generate", "finalize"}
    assert [stage.stage for stage in result.stages] == ["generate", "finalize"]
    for stage in stages:
        assert stage.status is StageState.SUCCEEDED
        assert stage.index is not None
        assert stage.input_hash
        assert stage.output_hash
        assert stage.artifacts == [_outputs(kind)[stage.stage]]

    manifest = json.loads(
        (directory.path / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["run_id"] == run.id
    assert set(manifest["stages"]) == {"generate", "finalize"}
    registered = directory.artifacts()
    assert {record.path for record in registered} == set(_outputs(kind).values())
    assert all(record.sha256 for record in registered)

    events = CostTracker(engine).for_run(run.id)
    assert len(events) == 2
    assert {event.operation for event in events} == {"generate", "finalize"}
    assert {event.amount_kind for event in events} == {"actual"}
    assert {event.pricing_version for event in events} == {PRICING_VERSION}
    assert result.run.cost_usd == pytest.approx(
        CostTracker(engine).total(run_id=run.id)
    )

    snapshot_file = directory.path / SNAPSHOT_FILENAME
    assert snapshot_file.is_file()
    snapshot = read_config_snapshot(directory.path)
    assert snapshot.run_id == run.id
    assert snapshot.profile == "smoke"
    assert snapshot.seeds == {"planner.defaults.seed": 7}
    assert snapshot.fingerprint == FINGERPRINT.digest
    assert snapshot["config"]["format"] == f"smoke-{kind.value}"
    assert stored_run.config_snapshot == snapshot.to_dict()

    for name in _outputs(kind).values():
        output = directory.path / name
        sidecar_file = sidecar_path(output)
        assert output.is_file()
        assert sidecar_file.is_file()
        payload = json.loads(sidecar_file.read_text(encoding="utf-8"))
        assert payload["run_id"] == run.id
        assert payload["output"] == str(output)
        assert payload["totals"]["count"] == 1
        assert payload["totals"]["actual_usd"] == pytest.approx(STAGE_COST_USD)
        assert payload["pricing_version"] == PRICING_VERSION
        assert payload["providers"] == ["mock"]
        assert payload["models"] == ["mock-model"]
        assert payload["toolchain_fingerprint"]["digest"] == FINGERPRINT.digest
