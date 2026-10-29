"""Resume from failure by reusing succeeded stages with unchanged inputs."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine

from aivia.core.db import RunRepository, StageRunRepository, create_database_engine
from aivia.core.db.migrate import upgrade_database
from aivia.core.errors import AiviaError
from aivia.core.models.run import Run, RunState, StageResult, StageState
from aivia.core.orchestrator import Orchestrator, resume_recipe
from aivia.core.resume import compute_input_hash
from aivia.core.run_state import RunStateMachine
from aivia.core.stages import Stage, StageContext


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


class _ProviderDown(AiviaError):
    code = "provider_exhausted"


class _ArtifactStage(Stage):
    def __init__(
        self,
        name: str,
        artifact: str,
        events: list[str],
        *,
        text: str = "content",
        cost: float = 0.25,
        options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name, options=options, artifacts=(artifact,))
        self.artifact = artifact
        self.events = events
        self.text = text
        self.cost = cost

    async def run(self, context: StageContext) -> StageResult:
        self.events.append(self.name)
        content = f"{self.text}:{self.options.get('mode', '')}"
        (context.run_dir / self.artifact).write_text(content, encoding="utf-8")
        return StageResult(
            run_id=context.run.id if context.run is not None else "",
            stage=self.name,
            artifacts=[self.artifact],
            cost_usd=self.cost,
        )


class _FlakyStage(Stage):
    name = "render"
    artifacts = ("clip.mp4",)

    def __init__(self, events: list[str], *, failures: int = 1) -> None:
        super().__init__()
        self.events = events
        self.failures = failures

    async def run(self, context: StageContext) -> StageResult:
        self.events.append(self.name)
        if self.failures > 0:
            self.failures -= 1
            raise _ProviderDown("provider down")
        (context.run_dir / "clip.mp4").write_text("clip", encoding="utf-8")
        return StageResult(
            run_id=context.run.id if context.run is not None else "",
            stage=self.name,
            artifacts=["clip.mp4"],
            cost_usd=0.5,
        )


def _orchestrator(
    engine: Engine,
    run: Run | str,
    stages: list[Stage],
    run_dir: Path,
    **options: Any,
) -> Orchestrator:
    return Orchestrator(
        stages=stages,
        run=run,
        repository=StageRunRepository(engine),
        run_dir=run_dir,
        raise_on_failure=False,
        **options,
    )


def test_resume_skips_succeeded_stages_and_reruns_the_failed_one(
    engine: Engine,
    tmp_path: Path,
) -> None:
    machine = RunStateMachine(RunRepository(engine))
    run = machine.create(Run())
    events: list[str] = []

    first = asyncio.run(
        _orchestrator(
            engine,
            run,
            [
                _ArtifactStage("write", "script.md", events),
                _FlakyStage(events),
                _ArtifactStage("after", "after.md", events),
            ],
            tmp_path,
            config={"format": "sample"},
        ).execute()
    )

    assert first.failed is True
    assert first.run.state is RunState.FAILED
    assert events == ["write", "render"]

    stored_write = StageRunRepository(engine).get(run.id, "write")
    assert stored_write is not None and stored_write.succeeded
    assert stored_write.input_hash
    assert stored_write.output_hash

    second = asyncio.run(
        _orchestrator(
            engine,
            run.id,
            [
                _ArtifactStage("write", "script.md", events),
                _FlakyStage(events, failures=0),
                _ArtifactStage("after", "after.md", events),
            ],
            tmp_path,
            config={"format": "sample"},
        ).execute()
    )

    assert events == ["write", "render", "render", "after"]
    assert second.ok is True
    assert [result.status for result in second.stages] == [
        StageState.SKIPPED,
        StageState.SUCCEEDED,
        StageState.SUCCEEDED,
    ]
    assert [result.stage for result in second.skipped_stages] == ["write"]
    assert second.stages[0].input_hash == stored_write.input_hash
    assert second.run.state is RunState.SUCCEEDED
    assert second.run.error_code == ""
    assert second.run.resumed_by == "orchestrator"
    assert second.run.resumed_at is not None
    assert second.run.cost_usd == pytest.approx(1.0)

    stored = StageRunRepository(engine).get(run.id, "write")
    assert stored is not None
    assert stored.status is StageState.SUCCEEDED


def test_changed_stage_options_rerun_the_stage_and_its_dependents(
    engine: Engine,
    tmp_path: Path,
) -> None:
    machine = RunStateMachine(RunRepository(engine))
    run = machine.create(Run())
    events: list[str] = []

    asyncio.run(
        _orchestrator(
            engine,
            run,
            [
                _ArtifactStage("write", "script.md", events, options={"mode": "facts"}),
                _ArtifactStage("compose", "clip.mp4", events),
                _FlakyStage(events),
            ],
            tmp_path,
        ).execute()
    )
    assert events == ["write", "compose", "render"]

    events.clear()
    result = asyncio.run(
        _orchestrator(
            engine,
            run.id,
            [
                _ArtifactStage("write", "script.md", events, options={"mode": "quote"}),
                _ArtifactStage("compose", "clip.mp4", events),
                _FlakyStage(events, failures=0),
            ],
            tmp_path,
        ).execute()
    )

    assert events == ["write", "compose", "render"]
    assert result.ok is True
    assert result.skipped_stages == ()
    assert all(item.status is StageState.SUCCEEDED for item in result.stages)


def test_changed_config_invalidates_every_stage_input(
    engine: Engine,
    tmp_path: Path,
) -> None:
    machine = RunStateMachine(RunRepository(engine))
    run = machine.create(Run())
    events: list[str] = []

    asyncio.run(
        _orchestrator(
            engine,
            run,
            [
                _ArtifactStage("write", "script.md", events),
                _ArtifactStage("compose", "clip.mp4", events),
                _FlakyStage(events),
            ],
            tmp_path,
            config={"language": "en"},
        ).execute()
    )
    events.clear()

    result = asyncio.run(
        _orchestrator(
            engine,
            run.id,
            [
                _ArtifactStage("write", "script.md", events),
                _ArtifactStage("compose", "clip.mp4", events),
                _FlakyStage(events, failures=0),
            ],
            tmp_path,
            config={"language": "de"},
        ).execute()
    )

    assert events == ["write", "compose", "render"]
    assert result.skipped_stages == ()


def test_a_second_resume_keeps_skipping_the_unchanged_prefix(
    engine: Engine,
    tmp_path: Path,
) -> None:
    machine = RunStateMachine(RunRepository(engine))
    run = machine.create(Run())
    events: list[str] = []

    stages = [
        _ArtifactStage("write", "script.md", events),
        _ArtifactStage("compose", "clip.mp4", events),
        _FlakyStage(events, failures=2),
    ]
    asyncio.run(_orchestrator(engine, run, stages, tmp_path).execute())
    events.clear()

    second = asyncio.run(
        _orchestrator(
            engine,
            run.id,
            [
                _ArtifactStage("write", "script.md", events),
                _ArtifactStage("compose", "clip.mp4", events),
                _FlakyStage(events, failures=1),
            ],
            tmp_path,
        ).execute()
    )
    assert second.failed is True
    assert events == ["render"]
    assert [item.status for item in second.skipped_stages] == [
        StageState.SKIPPED,
        StageState.SKIPPED,
    ]

    events.clear()
    third = asyncio.run(
        _orchestrator(
            engine,
            run.id,
            [
                _ArtifactStage("write", "script.md", events),
                _ArtifactStage("compose", "clip.mp4", events),
                _FlakyStage(events, failures=0),
            ],
            tmp_path,
        ).execute()
    )

    assert third.ok is True
    assert events == ["render"]
    assert [item.stage for item in third.skipped_stages] == ["write", "compose"]


def test_resuming_a_succeeded_run_skips_every_stage(
    engine: Engine,
    tmp_path: Path,
) -> None:
    machine = RunStateMachine(RunRepository(engine))
    run = machine.create(Run())
    events: list[str] = []

    stages = [
        _ArtifactStage("write", "script.md", events),
        _ArtifactStage("compose", "clip.mp4", events),
        _FlakyStage(events, failures=0),
    ]
    first = asyncio.run(_orchestrator(engine, run, stages, tmp_path).execute())
    assert first.ok is True
    events.clear()

    second = asyncio.run(
        _orchestrator(
            engine,
            run.id,
            [
                _ArtifactStage("write", "script.md", events),
                _ArtifactStage("compose", "clip.mp4", events),
                _FlakyStage(events, failures=0),
            ],
            tmp_path,
        ).execute()
    )

    assert second.ok is True
    assert events == []
    assert len(second.skipped_stages) == 3
    assert second.run.state is RunState.SUCCEEDED


def test_resume_recipe_function_resumes_the_failed_run(
    engine: Engine,
    tmp_path: Path,
) -> None:
    machine = RunStateMachine(RunRepository(engine))
    run = machine.create(Run())
    events: list[str] = []

    asyncio.run(
        _orchestrator(
            engine,
            run,
            [
                _ArtifactStage("write", "script.md", events),
                _FlakyStage(events),
            ],
            tmp_path,
        ).execute()
    )
    events.clear()

    result = asyncio.run(
        resume_recipe(
            run.id,
            [
                _ArtifactStage("write", "script.md", events),
                _FlakyStage(events, failures=0),
            ],
            repository=StageRunRepository(engine),
            run_dir=tmp_path,
        )
    )

    assert result.ok is True
    assert events == ["render"]
    assert result.skipped_stages[0].stage == "write"


def test_a_failed_stage_is_never_skipped_even_when_its_input_matches(
    engine: Engine,
    tmp_path: Path,
) -> None:
    machine = RunStateMachine(RunRepository(engine))
    run = machine.create(Run())
    events: list[str] = []

    asyncio.run(
        _orchestrator(
            engine,
            run,
            [_FlakyStage(events, failures=2)],
            tmp_path,
        ).execute()
    )
    failed = StageRunRepository(engine).get(run.id, "render")
    assert failed is not None and failed.status is StageState.FAILED
    assert failed.input_hash

    events.clear()
    result = asyncio.run(
        _orchestrator(
            engine,
            run.id,
            [_FlakyStage(events, failures=0)],
            tmp_path,
        ).execute()
    )

    assert events == ["render"]
    assert result.ok is True
    assert result.stages[0].status is StageState.SUCCEEDED


def test_the_input_hash_includes_the_stage_implementation_version() -> None:
    stage = Stage("write")
    original = compute_input_hash(stage)

    stage.version = "2"

    assert compute_input_hash(stage) != original
    assert compute_input_hash(Stage("write", version="2")) != original
    assert compute_input_hash(Stage("write", implementation_version="2")) != original


def test_a_changed_stage_version_reruns_that_stage(
    engine: Engine,
    tmp_path: Path,
) -> None:
    machine = RunStateMachine(RunRepository(engine))
    run = machine.create(Run())
    events: list[str] = []

    first = asyncio.run(
        _orchestrator(
            engine,
            run,
            [
                _ArtifactStage("write", "script.md", events),
                _ArtifactStage("compose", "clip.mp4", events),
                _FlakyStage(events, failures=0),
            ],
            tmp_path,
        ).execute()
    )
    assert first.ok is True
    assert events == ["write", "compose", "render"]

    stored_write = StageRunRepository(engine).get(run.id, "write")
    assert stored_write is not None

    events.clear()
    changed = _ArtifactStage("write", "script.md", events)
    changed.version = 2
    second = asyncio.run(
        _orchestrator(
            engine,
            run.id,
            [
                changed,
                _ArtifactStage("compose", "clip.mp4", events),
                _FlakyStage(events, failures=0),
            ],
            tmp_path,
        ).execute()
    )

    assert second.ok is True
    assert events == ["write"]
    assert [item.stage for item in second.skipped_stages] == ["compose", "render"]
    assert second.stages[0].input_hash != stored_write.input_hash
