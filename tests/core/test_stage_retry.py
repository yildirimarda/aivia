"""Per-stage timeout and retry policy applied by the orchestrator."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine

from aivia.core.db import RunRepository, StageRunRepository, create_database_engine
from aivia.core.db.migrate import upgrade_database
from aivia.core.errors import AiviaError, PermanentError, ProviderExhaustedError
from aivia.core.models.run import Run, RunState, StageResult, StageState
from aivia.core.orchestrator import Orchestrator, StageFailure
from aivia.core.run_state import RunStateMachine
from aivia.core.stage_retry import (
    StageRetryExhausted,
    StageRetryPolicy,
    execute_stage_with_retry,
)
from aivia.core.stages import Stage, StageContext


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


class _FlakyStage(Stage):
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        failures: int,
        error: BaseException | None = None,
        **options: Any,
    ) -> None:
        super().__init__(name, **options)
        self.events = events
        self.failures = failures
        self.error = error or ProviderExhaustedError("provider down")

    async def run(self, context: StageContext) -> StageResult:
        self.events.append(self.name)
        if self.failures > 0:
            self.failures -= 1
            raise self.error
        return StageResult(stage=self.name)


class _TimeoutStage(Stage):
    def __init__(self, name: str, events: list[str], *, timeouts: int, **options: Any):
        super().__init__(name, **options)
        self.events = events
        self.timeouts = timeouts

    async def run(self, context: StageContext) -> StageResult:
        self.events.append(self.name)
        if self.timeouts > 0:
            self.timeouts -= 1
            await asyncio.Event().wait()
        return StageResult(stage=self.name)


def _orchestrator(
    engine: Engine,
    run: Run | str,
    stages: list[Stage],
    run_dir: Path,
    **options: Any,
) -> Orchestrator:
    options.setdefault("raise_on_failure", False)
    return Orchestrator(
        stages=stages,
        run=run,
        repository=StageRunRepository(engine),
        run_dir=run_dir,
        **options,
    )


def test_policy_defaults_to_one_attempt_without_a_deadline() -> None:
    policy = StageRetryPolicy()

    assert policy.timeout_seconds is None
    assert policy.max_attempts == 1
    assert policy.attempts == 1
    assert policy.retries == 0
    assert policy.retry_delay_seconds == 0.0
    assert policy.should_retry(ProviderExhaustedError("down"), 1) is False


def test_policy_computes_exponential_delay_with_a_cap() -> None:
    policy = StageRetryPolicy(
        max_attempts=4,
        retry_delay_seconds=0.5,
        backoff_multiplier=2.0,
        max_retry_delay_seconds=1.5,
    )

    assert policy.delay_for(1) == pytest.approx(0.5)
    assert policy.delay_for(2) == pytest.approx(1.0)
    assert policy.delay_for(3) == pytest.approx(1.5)
    assert policy.delay_for(4) == pytest.approx(1.5)


def test_policy_retries_only_retryable_and_opted_in_codes() -> None:
    policy = StageRetryPolicy(max_attempts=3, retry_codes=("content_policy",))

    assert policy.should_retry(ProviderExhaustedError("down"), 1) is True
    assert policy.should_retry(PermanentError("no"), 1) is False
    assert policy.should_retry(TimeoutError("late"), 1) is True
    assert policy.should_retry(AiviaError("policy"), 1) is False

    class _ContentPolicy(PermanentError):
        code = "content_policy"

    assert policy.should_retry(_ContentPolicy("policy"), 1) is True
    assert policy.should_retry(ProviderExhaustedError("down"), 3) is False


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"timeout_seconds": float("inf")}, "timeout_seconds"),
        ({"max_attempts": 0}, "max_attempts"),
        ({"max_attempts": True}, "max_attempts"),
        ({"retry_delay_seconds": -1}, "retry_delay_seconds"),
        ({"backoff_multiplier": 0.5}, "backoff_multiplier"),
        ({"max_retry_delay_seconds": -1}, "max_retry_delay_seconds"),
        ({"retry_on_timeout": "yes"}, "retry_on_timeout"),
    ],
)
def test_policy_rejects_invalid_values(
    values: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        StageRetryPolicy(**values)


def test_stage_class_attributes_and_constructor_override_the_policy() -> None:
    class Fast(Stage):
        name = "render"
        timeout_seconds = 5.0
        max_attempts = 4
        retry_delay_seconds = 0.25

    declared = Fast()
    assert declared.retry_policy == StageRetryPolicy(
        timeout_seconds=5.0,
        max_attempts=4,
        retry_delay_seconds=0.25,
    )

    overridden = Fast(max_attempts=2, timeout_seconds=1.5)
    assert overridden.retry_policy.max_attempts == 2
    assert overridden.retry_policy.timeout_seconds == 1.5
    assert overridden.retry_policy.retry_delay_seconds == 0.25

    explicit = Stage("render", retry_policy={"max_attempts": 6})
    assert explicit.retry_policy.max_attempts == 6


def test_a_stage_can_declare_a_policy_object() -> None:
    class Declared(Stage):
        name = "render"
        retry_policy = StageRetryPolicy(max_attempts=5, timeout_seconds=2.0)

    assert Declared().retry_policy.max_attempts == 5
    assert Declared().retry_policy.timeout_seconds == 2.0


def test_retryable_failure_retries_until_the_stage_succeeds(
    engine: Engine,
    tmp_path: Path,
) -> None:
    machine = RunStateMachine(RunRepository(engine))
    run = machine.create(Run())
    events: list[str] = []
    stage = _FlakyStage("render", events, failures=2, max_attempts=3)

    result = asyncio.run(_orchestrator(engine, run, [stage], tmp_path).execute())

    assert events == ["render", "render", "render"]
    assert result.ok is True
    assert result.run.state is RunState.SUCCEEDED
    completed = result.stages[0]
    assert completed.status is StageState.SUCCEEDED
    assert completed.metadata["attempts"] == 3
    assert completed.metadata["attempt_count"] == 3
    assert completed.metadata["max_attempts"] == 3
    assert completed.metadata["retries"] == 2
    assert completed.metadata["exhausted"] is False
    assert [item["succeeded"] for item in completed.metadata["attempt_history"]] == [
        False,
        False,
        True,
    ]
    assert [
        item["error_code"]
        for item in completed.metadata["attempt_history"]
        if not item["succeeded"]
    ] == ["provider_exhausted", "provider_exhausted"]

    stored = StageRunRepository(engine).get(run.id, "render")
    assert stored is not None
    assert stored.metadata["attempts"] == 3
    assert stored.metadata["exhausted"] is False


def test_a_permanent_failure_is_not_retried_and_still_records_the_attempt(
    engine: Engine,
    tmp_path: Path,
) -> None:
    machine = RunStateMachine(RunRepository(engine))
    run = machine.create(Run())
    events: list[str] = []
    error = PermanentError("invalid prompt")
    stage = _FlakyStage(
        "render",
        events,
        failures=1,
        error=error,
        max_attempts=3,
    )

    result = asyncio.run(_orchestrator(engine, run, [stage], tmp_path).execute())

    assert events == ["render"]
    assert result.failed is True
    assert result.error_code == "permanent"
    assert result.run.state is RunState.FAILED
    failed = result.stages[0]
    assert failed.status is StageState.FAILED
    assert failed.metadata["attempts"] == 1
    assert failed.metadata["exhausted"] is False
    assert failed.metadata["attempt_history"][0]["error_code"] == "permanent"


def test_exhausting_every_attempt_fails_the_run_with_the_last_error(
    engine: Engine,
    tmp_path: Path,
) -> None:
    machine = RunStateMachine(RunRepository(engine))
    run = machine.create(Run())
    events: list[str] = []
    stage = _FlakyStage("render", events, failures=5, max_attempts=3)

    with pytest.raises(StageFailure) as caught:
        asyncio.run(
            _orchestrator(
                engine,
                run,
                [stage],
                tmp_path,
                raise_on_failure=True,
            ).execute()
        )

    failure = caught.value
    assert events == ["render", "render", "render"]
    assert failure.error_code == "provider_exhausted"
    assert isinstance(failure.cause, ProviderExhaustedError)
    assert failure.result is not None
    assert failure.result.run.state is RunState.FAILED
    failed = failure.result.stages[0]
    assert failed.status is StageState.FAILED
    assert failed.error_code == "provider_exhausted"
    assert failed.metadata["attempts"] == 3
    assert failed.metadata["max_attempts"] == 3
    assert failed.metadata["exhausted"] is True
    assert len(failed.metadata["attempt_history"]) == 3

    stored = StageRunRepository(engine).get(run.id, "render")
    assert stored is not None
    assert stored.metadata["exhausted"] is True
    stored_run = RunRepository(engine).get(run.id)
    assert stored_run is not None and stored_run.state is RunState.FAILED


def test_a_timeout_retries_the_attempt_and_then_succeeds(
    engine: Engine,
    tmp_path: Path,
) -> None:
    machine = RunStateMachine(RunRepository(engine))
    run = machine.create(Run())
    events: list[str] = []
    stage = _TimeoutStage(
        "render",
        events,
        timeouts=1,
        timeout_seconds=0.01,
        max_attempts=2,
    )

    result = asyncio.run(_orchestrator(engine, run, [stage], tmp_path).execute())

    assert events == ["render", "render"]
    assert result.ok is True
    completed = result.stages[0]
    assert completed.metadata["attempts"] == 2
    assert completed.metadata["exhausted"] is False
    history = completed.metadata["attempt_history"]
    assert history[0]["timed_out"] is True
    assert history[0]["error_code"] == "timeout"
    assert history[1]["succeeded"] is True


def test_timeout_exhaustion_reports_timeout_and_every_attempt(
    engine: Engine,
    tmp_path: Path,
) -> None:
    machine = RunStateMachine(RunRepository(engine))
    run = machine.create(Run())
    events: list[str] = []
    stage = _TimeoutStage(
        "render",
        events,
        timeouts=5,
        timeout_seconds=0.01,
        max_attempts=2,
    )

    result = asyncio.run(_orchestrator(engine, run, [stage], tmp_path).execute())

    assert events == ["render", "render"]
    assert result.failed is True
    assert result.error_code == "timeout"
    assert result.run.state is RunState.FAILED
    failed = result.stages[0]
    assert failed.status is StageState.FAILED
    assert failed.metadata["attempts"] == 2
    assert failed.metadata["exhausted"] is True
    assert all(
        item["timed_out"] for item in failed.metadata["attempt_history"]
    )


def test_each_stage_uses_its_own_policy(
    engine: Engine,
    tmp_path: Path,
) -> None:
    machine = RunStateMachine(RunRepository(engine))
    run = machine.create(Run())
    events: list[str] = []
    first = _FlakyStage("write", events, failures=1, max_attempts=3)
    second = _FlakyStage(
        "render",
        events,
        failures=1,
        error=PermanentError("bad"),
        max_attempts=1,
    )

    result = asyncio.run(
        _orchestrator(engine, run, [first, second], tmp_path).execute()
    )

    assert events == ["write", "write", "render"]
    assert result.failed_stage == "render"
    assert result.stages[0].metadata["attempts"] == 2
    assert result.stages[1].metadata["attempts"] == 1


def test_backoff_waits_between_attempts_and_not_after_the_last(
    engine: Engine,
    tmp_path: Path,
) -> None:
    machine = RunStateMachine(RunRepository(engine))
    run = machine.create(Run())
    events: list[str] = []
    waits: list[float] = []

    async def fake_sleep(delay: float) -> None:
        waits.append(delay)

    stage = _FlakyStage(
        "render",
        events,
        failures=5,
        max_attempts=3,
        retry_delay_seconds=0.5,
        backoff_multiplier=2.0,
    )

    result = asyncio.run(
        _orchestrator(
            engine,
            run,
            [stage],
            tmp_path,
            sleep=fake_sleep,
        ).execute()
    )

    assert result.failed is True
    assert waits == pytest.approx([0.5, 1.0])
    assert events == ["render", "render", "render"]


def test_a_retryable_failed_result_is_retried_and_each_attempt_is_costed(
    engine: Engine,
    tmp_path: Path,
) -> None:
    machine = RunStateMachine(RunRepository(engine))
    run = machine.create(Run())
    events: list[str] = []

    class _Rejecting(Stage):
        def __init__(self) -> None:
            super().__init__("render", max_attempts=3)
            self.failures = 2

        async def run(self, context: StageContext) -> StageResult:
            events.append(self.name)
            if self.failures > 0:
                self.failures -= 1
                return StageResult(
                    stage=self.name,
                    status=StageState.FAILED,
                    error_code="timeout",
                    error_message="provider timed out",
                    cost_usd=0.25,
                )
            return StageResult(stage=self.name, cost_usd=0.5)

    result = asyncio.run(
        _orchestrator(engine, run, [_Rejecting()], tmp_path).execute()
    )

    assert events == ["render", "render", "render"]
    assert result.ok is True
    assert result.run.cost_usd == pytest.approx(1.0)
    completed = result.stages[0]
    assert completed.metadata["attempts"] == 3
    assert completed.metadata["exhausted"] is False
    assert [
        item["error_code"]
        for item in completed.metadata["attempt_history"]
        if not item["succeeded"]
    ] == ["timeout", "timeout"]


def test_the_helper_raises_exhaustion_with_the_full_history(tmp_path: Path) -> None:
    events: list[str] = []
    stage = _FlakyStage("render", events, failures=5, max_attempts=2)

    async def run() -> None:
        await execute_stage_with_retry(
            stage,
            StageContext(run_dir=tmp_path),
            policy=stage.retry_policy,
            sleep=_no_wait,
        )

    with pytest.raises(StageRetryExhausted) as caught:
        asyncio.run(run())

    exhausted = caught.value
    assert exhausted.stage == "render"
    assert exhausted.error_code == "provider_exhausted"
    assert len(exhausted.attempts) == 2
    assert isinstance(exhausted.last_error, ProviderExhaustedError)
    assert exhausted.details["exhausted"] is True
    assert exhausted.details["max_attempts"] == 2


async def _no_wait(delay: float) -> None:
    return None
