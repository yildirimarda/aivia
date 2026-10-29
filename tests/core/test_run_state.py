"""The run state machine: legal transitions only, persisted on every change."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Engine

from aivia.core.db import RunRepository, create_database_engine, upgrade_database
from aivia.core.errors import AiviaError
from aivia.core.models import Run, RunState
from aivia.core.run_state import (
    ALLOWED_RUN_TRANSITIONS,
    LEGAL_RUN_TRANSITIONS,
    RUN_STATE_TRANSITIONS,
    IllegalRunTransition,
    IllegalTransition,
    IllegalTransitionError,
    InvalidRunTransition,
    InvalidRunTransitionError,
    InvalidTransition,
    InvalidTransitionError,
    RunNotFoundError,
    RunStateMachine,
    RunStateStore,
    RunTransitionError,
    can_transition,
    is_legal_transition,
    legal_transitions,
    transition_run,
)

START = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)
PAUSE = START + timedelta(minutes=5)
RESUME = PAUSE + timedelta(minutes=5)
FINISH = RESUME + timedelta(minutes=5)

ALL_STATES = tuple(RunState)


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


def stored(engine: Engine, run_id: str) -> Run:
    run = RunRepository(engine).get(run_id)
    assert run is not None
    return run


@pytest.mark.parametrize("source", ALL_STATES)
@pytest.mark.parametrize("target", ALL_STATES)
def test_only_the_declared_transitions_are_legal(
    source: RunState, target: RunState
) -> None:
    expected = target in LEGAL_RUN_TRANSITIONS[source]

    assert can_transition(source, target) is expected
    assert is_legal_transition(source, target) is expected
    assert (target in legal_transitions(source)) is expected


def test_terminal_states_never_leave_themselves() -> None:
    for state in (RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED):
        assert legal_transitions(state) == frozenset()


def test_an_illegal_transition_raises_and_does_not_mutate_the_run() -> None:
    run = Run(state=RunState.QUEUED)

    with pytest.raises(IllegalRunTransition, match="queued to succeeded"):
        transition_run(run, RunState.SUCCEEDED)

    assert run.state is RunState.QUEUED
    assert run.started_at is None
    assert run.finished_at is None


def test_the_transition_error_is_also_a_value_error_and_an_aivia_error() -> None:
    with pytest.raises(ValueError):
        transition_run(Run(state=RunState.SUCCEEDED), RunState.RUNNING)
    with pytest.raises(AiviaError):
        transition_run(Run(state=RunState.SUCCEEDED), RunState.RUNNING)


def test_error_aliases_point_at_the_same_class() -> None:
    assert InvalidRunTransition is IllegalRunTransition
    assert InvalidRunTransitionError is IllegalRunTransition
    assert RunTransitionError is IllegalRunTransition
    assert InvalidTransition is IllegalRunTransition
    assert InvalidTransitionError is IllegalRunTransition
    assert IllegalTransition is IllegalRunTransition
    assert IllegalTransitionError is IllegalRunTransition
    assert RunStateStore is RunStateMachine
    assert ALLOWED_RUN_TRANSITIONS is LEGAL_RUN_TRANSITIONS
    assert RUN_STATE_TRANSITIONS is LEGAL_RUN_TRANSITIONS


def test_transition_run_stamps_start_pause_resume_and_finish() -> None:
    run = Run(state=RunState.QUEUED)

    started = transition_run(run, RunState.RUNNING, now=START)
    paused = transition_run(
        started, RunState.PAUSED_BUDGET, now=PAUSE, pause_reason=" budget "
    )
    resumed = transition_run(
        paused,
        RunState.RUNNING,
        now=RESUME,
        resumed_by=" alice ",
        resume_reason=" cap raised ",
    )
    finished = transition_run(resumed, RunState.SUCCEEDED, now=FINISH)

    assert started.state is RunState.RUNNING
    assert started.started_at == START
    assert paused.state is RunState.PAUSED_BUDGET
    assert paused.paused_at == PAUSE
    assert paused.pause_reason == "budget"
    assert resumed.state is RunState.RUNNING
    assert resumed.started_at == START
    assert resumed.resumed_at == RESUME
    assert resumed.resumed_by == "alice"
    assert resumed.resume_reason == "cap raised"
    assert finished.state is RunState.SUCCEEDED
    assert finished.finished_at == FINISH
    assert run.state is RunState.QUEUED
    assert run.started_at is None


def test_a_pause_without_a_reason_records_the_budget_default() -> None:
    paused = transition_run(
        Run(state=RunState.RUNNING), RunState.PAUSED_BUDGET, now=PAUSE
    )

    assert paused.pause_reason == "budget"


def test_a_failure_records_the_error_and_finish_time() -> None:
    failed = transition_run(
        Run(state=RunState.RUNNING),
        RunState.FAILED,
        now=FINISH,
        error_code=" provider_exhausted ",
        error_message=" no provider left ",
    )

    assert failed.state is RunState.FAILED
    assert failed.finished_at == FINISH
    assert failed.error_code == "provider_exhausted"
    assert failed.error_message == "no provider left"


def test_the_machine_persists_every_accepted_change(
    engine: Engine, machine: RunStateMachine
) -> None:
    run = machine.create(Run(profile="daily"))

    running = machine.start(run, now=START)
    assert running.state is RunState.RUNNING
    assert stored(engine, run.id).state is RunState.RUNNING
    assert stored(engine, run.id).started_at == START

    paused = machine.pause(run.id, reason="cap", now=PAUSE)
    assert paused.state is RunState.PAUSED_BUDGET
    persisted = stored(engine, run.id)
    assert persisted.state is RunState.PAUSED_BUDGET
    assert persisted.paused_at == PAUSE
    assert persisted.pause_reason == "cap"

    resumed = machine.resume(run.id, resumed_by="alice", reason="raised", now=RESUME)
    assert resumed.state is RunState.RUNNING
    persisted = stored(engine, run.id)
    assert persisted.state is RunState.RUNNING
    assert persisted.resumed_at == RESUME
    assert persisted.resumed_by == "alice"
    assert persisted.resume_reason == "raised"
    assert persisted.started_at == START

    finished = machine.succeed(run.id, now=FINISH)
    assert finished.state is RunState.SUCCEEDED
    persisted = stored(engine, run.id)
    assert persisted.state is RunState.SUCCEEDED
    assert persisted.finished_at == FINISH


@pytest.mark.parametrize(
    ("source", "method"),
    [
        (RunState.QUEUED, "cancel"),
        (RunState.RUNNING, "cancel"),
        (RunState.PAUSED_BUDGET, "cancel"),
        (RunState.QUEUED, "fail"),
        (RunState.RUNNING, "fail"),
        (RunState.PAUSED_BUDGET, "fail"),
    ],
)
def test_fail_and_cancel_are_persisted_from_every_non_terminal_state(
    engine: Engine,
    machine: RunStateMachine,
    source: RunState,
    method: str,
) -> None:
    run = machine.create(Run(state=source))

    result = getattr(machine, method)(
        run.id,
        error_code="stopped",
        error_message="stopped by test",
        now=FINISH,
    )

    expected = RunState.CANCELLED if method == "cancel" else RunState.FAILED
    assert result.state is expected
    persisted = stored(engine, run.id)
    assert persisted.state is result.state
    assert persisted.finished_at == FINISH
    assert persisted.error_code == "stopped"
    assert persisted.error_message == "stopped by test"


def test_an_illegal_machine_transition_persists_nothing(
    engine: Engine, machine: RunStateMachine
) -> None:
    run = machine.create(Run(state=RunState.PAUSED_BUDGET))
    before = stored(engine, run.id)

    with pytest.raises(IllegalRunTransition, match="paused_budget to succeeded"):
        machine.succeed(run.id, now=FINISH)

    after = stored(engine, run.id)
    assert after == before
    assert after.state is RunState.PAUSED_BUDGET
    assert after.finished_at is None


def test_the_machine_reads_the_latest_state_not_a_stale_record(
    engine: Engine, machine: RunStateMachine
) -> None:
    stale = machine.create(Run())
    machine.start(stale, now=START)

    assert stale.state is RunState.QUEUED
    finished = machine.succeed(stale, now=FINISH)

    assert finished.state is RunState.SUCCEEDED
    assert stored(engine, stale.id).state is RunState.SUCCEEDED


def test_the_machine_reports_a_missing_run(machine: RunStateMachine) -> None:
    assert machine.get("run_missing") is None
    with pytest.raises(RunNotFoundError, match="run not found"):
        machine.start("run_missing")
    with pytest.raises(RunNotFoundError, match="run not found"):
        machine.transition(Run(id="run_missing"), RunState.RUNNING)


def test_the_machine_accepts_an_engine_directly(engine: Engine) -> None:
    machine = RunStateMachine(engine)
    run = machine.create(Run())

    assert machine.start(run).state is RunState.RUNNING


def test_machine_helpers_report_legal_transitions(machine: RunStateMachine) -> None:
    run = Run(state=RunState.PAUSED_BUDGET)

    assert machine.legal_transitions(run) == legal_transitions(RunState.PAUSED_BUDGET)
    assert machine.can_transition(run, RunState.RUNNING) is True
    assert machine.can_transition(run, RunState.SUCCEEDED) is False
    assert machine.transitions is LEGAL_RUN_TRANSITIONS
