"""The orchestrator executes stages in order, persists them and stops on failure."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy import Engine

from aivia.core.cost import CostTracker
from aivia.core.db import RunRepository, StageRunRepository, create_database_engine
from aivia.core.db.migrate import upgrade_database
from aivia.core.errors import AiviaError, ConfigError
from aivia.core.models.run import Run, RunState, StageResult, StageState
from aivia.core.orchestrator import (
    OrchestrationResult,
    Orchestrator,
    StageFailure,
    execute_recipe,
)
from aivia.core.router import Router
from aivia.core.run_dirs import RunDirectory
from aivia.core.run_state import RunStateMachine
from aivia.core.stages import Stage, StageContext
from aivia.providers.protocols.storage import StorageProvider


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def machine(engine: Engine) -> RunStateMachine:
    return RunStateMachine(RunRepository(engine))


class _WriteStage(Stage):
    name = "write"
    artifacts = ("script.md",)

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    async def run(self, context: StageContext) -> StageResult:
        self.events.append(self.name)
        (context.run_dir / "script.md").write_text("script", encoding="utf-8")
        return StageResult(
            run_id=context.run.id if context.run is not None else "",
            stage=self.name,
            artifacts=["script.md"],
            cost_usd=0.25,
        )


class _RenderStage(Stage):
    name = "render"
    artifacts = ("clip.mp4",)

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    async def run(self, context: StageContext) -> StageResult:
        self.events.append(self.name)
        (context.run_dir / "clip.mp4").write_text("clip", encoding="utf-8")
        return StageResult(
            run_id=context.run.id if context.run is not None else "",
            stage=self.name,
            artifacts=["clip.mp4"],
            cost_usd=0.75,
        )


class _ProviderDown(AiviaError):
    code = "provider_exhausted"


class _BoomStage(Stage):
    name = "boom"

    def __init__(self, events: list[str], error: BaseException | None = None) -> None:
        super().__init__()
        self.events = events
        self.error = error or _ProviderDown("provider down")

    async def run(self, context: StageContext) -> StageResult:
        self.events.append(self.name)
        raise self.error


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
    run: Run,
    stages: list[Stage],
    run_dir: Path,
    **options: Any,
) -> Orchestrator:
    return Orchestrator(
        stages=stages,
        run=run,
        repository=StageRunRepository(engine),
        run_dir=run_dir,
        **options,
    )


def test_stages_run_in_order_and_every_result_is_persisted(
    engine: Engine,
    machine: RunStateMachine,
    tmp_path: Path,
) -> None:
    run = machine.create(Run())
    events: list[str] = []
    cost = cast(CostTracker, object())
    storage = cast(StorageProvider, object())
    orchestrator = _orchestrator(
        engine,
        run,
        [_WriteStage(events), _RenderStage(events)],
        tmp_path,
        config={"format": "sample"},
        cost=cost,
        storage=storage,
        router=Router(),
    )

    result = asyncio.run(orchestrator.execute())

    assert events == ["write", "render"]
    assert result.ok is True
    assert result.failed is False
    assert [item.stage for item in result.stages] == ["write", "render"]
    assert [item.status for item in result.stages] == [
        StageState.SUCCEEDED,
        StageState.SUCCEEDED,
    ]
    assert [item.index for item in result.stages] == [0, 1]
    assert result.run.state is RunState.SUCCEEDED
    assert result.run.started_at is not None
    assert result.run.finished_at is not None
    assert result.run.cost_usd == pytest.approx(1.0)
    assert result.run.id == run.id

    repository = StageRunRepository(engine)
    stored_write = repository.get(run.id, "write")
    stored_render = repository.get(run.id, "render")
    assert stored_write is not None and stored_write.succeeded
    assert stored_render is not None and stored_render.succeeded
    assert stored_write.artifacts == ["script.md"]
    assert stored_render.artifacts == ["clip.mp4"]

    directory = RunDirectory(tmp_path)
    assert [record.path for record in directory.artifacts("write")] == ["script.md"]
    assert [record.path for record in directory.artifacts("render")] == ["clip.mp4"]


def test_the_context_receives_the_shared_collaborators_and_run(
    engine: Engine,
    machine: RunStateMachine,
    tmp_path: Path,
) -> None:
    run = machine.create(Run())
    cost = cast(CostTracker, object())
    storage = cast(StorageProvider, object())
    router = Router()
    seen: list[StageContext] = []

    class Inspect(Stage):
        name = "inspect"

        async def run(self, context: StageContext) -> StageResult:
            seen.append(context)
            return StageResult(stage=self.name)

    asyncio.run(
        _orchestrator(
            engine,
            run,
            [Inspect()],
            tmp_path,
            config={"language": "en"},
            cost=cost,
            storage=storage,
            router=router,
        ).execute()
    )

    assert len(seen) == 1
    context = seen[0]
    assert context.run_dir == tmp_path
    assert context.config == {"language": "en"}
    assert context.cost is cost
    assert context.router is router
    assert context.storage is storage
    assert context.run is not None and context.run.id == run.id
    assert context.logger.name == "aivia.orchestrator.inspect"


def test_a_raising_stage_stops_the_recipe_and_records_stage_and_code(
    engine: Engine,
    machine: RunStateMachine,
    tmp_path: Path,
) -> None:
    run = machine.create(Run())
    events: list[str] = []
    orchestrator = _orchestrator(
        engine,
        run,
        [_WriteStage(events), _BoomStage(events), _AfterStage(events)],
        tmp_path,
    )

    with pytest.raises(StageFailure) as caught:
        asyncio.run(orchestrator.execute())

    failure = caught.value
    assert failure.stage == "boom"
    assert failure.stage_name == "boom"
    assert failure.error_code == "provider_exhausted"
    assert isinstance(failure.cause, _ProviderDown)
    assert failure.result is not None
    assert failure.result.failed_stage == "boom"
    assert failure.result.error_code == "provider_exhausted"
    assert events == ["write", "boom"]

    repository = StageRunRepository(engine)
    assert repository.get(run.id, "write") is not None
    stored_boom = repository.get(run.id, "boom")
    assert stored_boom is not None
    assert stored_boom.status is StageState.FAILED
    assert stored_boom.error_code == "provider_exhausted"
    assert stored_boom.error_message == "provider down"
    assert stored_boom.started_at is not None
    assert stored_boom.finished_at is not None
    assert repository.get(run.id, "after") is None

    stored_run = RunRepository(engine).get(run.id)
    assert stored_run is not None
    assert stored_run.state is RunState.FAILED
    assert stored_run.error_code == "provider_exhausted"
    assert "boom" in stored_run.error_message
    assert stored_run.finished_at is not None


def test_a_failed_stage_result_stops_the_recipe_without_raising_when_safe(
    engine: Engine,
    machine: RunStateMachine,
    tmp_path: Path,
) -> None:
    run = machine.create(Run())
    events: list[str] = []

    class Reject(Stage):
        name = "review"

        async def run(self, context: StageContext) -> StageResult:
            events.append(self.name)
            return StageResult(
                stage=self.name,
                status=StageState.FAILED,
                error_code="validation",
                error_message="script rejected",
            )

    result = asyncio.run(
        _orchestrator(
            engine,
            run,
            [_WriteStage(events), Reject(), _AfterStage(events)],
            tmp_path,
            raise_on_failure=False,
        ).execute()
    )

    assert events == ["write", "review"]
    assert result.failed is True
    assert result.failed_stage == "review"
    assert result.error_code == "validation"
    assert result.error_message == "script rejected"
    assert [item.stage for item in result.stages] == ["write", "review"]
    assert result.run.state is RunState.FAILED
    assert result.failure is not None
    assert result.failure.stage == "review"
    assert result.failure.error_code == "validation"

    stored = StageRunRepository(engine).get(run.id, "review")
    assert stored is not None
    assert stored.status is StageState.FAILED
    assert stored.error_code == "validation"


def test_a_generic_exception_gets_a_stable_snake_case_code(
    engine: Engine,
    machine: RunStateMachine,
    tmp_path: Path,
) -> None:
    run = machine.create(Run())
    events: list[str] = []

    with pytest.raises(StageFailure) as caught:
        asyncio.run(
            _orchestrator(
                engine,
                run,
                [_BoomStage(events, RuntimeError("disk full"))],
                tmp_path,
            ).execute()
        )

    assert caught.value.error_code == "runtime_error"
    assert caught.value.cause is not None
    stored = StageRunRepository(engine).get(run.id, "boom")
    assert stored is not None
    assert stored.error_code == "runtime_error"
    assert stored.error_message == "disk full"


def test_a_missing_declared_artifact_fails_the_stage(
    engine: Engine,
    machine: RunStateMachine,
    tmp_path: Path,
) -> None:
    run = machine.create(Run())

    class Missing(Stage):
        name = "render"
        artifacts = ("missing.mp4",)

        async def run(self, context: StageContext) -> StageResult:
            return StageResult(stage=self.name)

    with pytest.raises(StageFailure) as caught:
        asyncio.run(_orchestrator(engine, run, [Missing()], tmp_path).execute())

    assert caught.value.stage == "render"
    assert caught.value.error_code == "file_not_found_error"
    stored = StageRunRepository(engine).get(run.id, "render")
    assert stored is not None and stored.status is StageState.FAILED


def test_duplicate_stage_names_are_rejected_before_the_run_starts(
    engine: Engine,
) -> None:
    run = Run()
    orchestrator = Orchestrator(
        stages=[Stage("write"), Stage("write")],
        run=run,
        repository=StageRunRepository(engine),
    )

    with pytest.raises(ConfigError, match="duplicate stage"):
        asyncio.run(orchestrator.execute())

    assert run.state is RunState.QUEUED
    assert RunRepository(engine).get(run.id) is None


def test_execute_recipe_function_returns_the_orchestration_result(
    engine: Engine,
    machine: RunStateMachine,
    tmp_path: Path,
) -> None:
    run = machine.create(Run())
    events: list[str] = []

    result = asyncio.run(
        execute_recipe(
            run,
            [_WriteStage(events)],
            repository=StageRunRepository(engine),
            run_dir=tmp_path,
        )
    )

    assert isinstance(result, OrchestrationResult)
    assert result.ok is True
    assert events == ["write"]
    assert result.run.state is RunState.SUCCEEDED
