"""The legal lifecycle of a run, persisted on every change.

A run starts ``queued``. It may start running, fail during pre-flight checks
or be cancelled. A running run may finish successfully, fail, be cancelled or
pause because a budget cap was reached. A paused run may resume, fail or be
cancelled. Terminal states never leave themselves.

The mapping in :data:`LEGAL_RUN_TRANSITIONS` is the single source of truth.
:func:`transition_run` applies one transition to a record without touching the
database, while :class:`RunStateMachine` reads the current record from a
:class:`~aivia.core.db.repos.RunRepository`, validates the transition and
writes the result back. An illegal transition raises
:class:`IllegalRunTransition` and persists nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Any

from sqlalchemy import Engine

from aivia.core.db.repos import RunRepository
from aivia.core.errors import AiviaError
from aivia.core.models import Run, RunState, utcnow

__all__ = [
    "ALLOWED_RUN_TRANSITIONS",
    "LEGAL_RUN_TRANSITIONS",
    "RUN_STATE_TRANSITIONS",
    "IllegalRunTransition",
    "IllegalTransition",
    "IllegalTransitionError",
    "InvalidRunTransition",
    "InvalidRunTransitionError",
    "InvalidTransition",
    "InvalidTransitionError",
    "RunNotFoundError",
    "RunStateError",
    "RunStateMachine",
    "RunStateStore",
    "RunTransitionError",
    "can_transition",
    "is_legal_transition",
    "legal_transitions",
    "transition_run",
]

LEGAL_RUN_TRANSITIONS: Mapping[RunState, frozenset[RunState]] = MappingProxyType(
    {
        RunState.QUEUED: frozenset(
            {RunState.RUNNING, RunState.FAILED, RunState.CANCELLED}
        ),
        RunState.RUNNING: frozenset(
            {
                RunState.PAUSED_BUDGET,
                RunState.SUCCEEDED,
                RunState.FAILED,
                RunState.CANCELLED,
            }
        ),
        RunState.PAUSED_BUDGET: frozenset(
            {RunState.RUNNING, RunState.FAILED, RunState.CANCELLED}
        ),
        RunState.SUCCEEDED: frozenset(),
        RunState.FAILED: frozenset(),
        RunState.CANCELLED: frozenset(),
    }
)
"""Every run state mapped to the states it may move to."""

_PERSISTED_FIELDS = (
    "state",
    "started_at",
    "finished_at",
    "paused_at",
    "pause_reason",
    "resumed_at",
    "resumed_by",
    "resume_reason",
    "error_code",
    "error_message",
    "updated_at",
)

_TERMINAL_STATES = frozenset({RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED})


class RunStateError(AiviaError):
    """Base class for run lifecycle errors."""

    code = "run_state"


class RunNotFoundError(RunStateError):
    """The requested run does not exist in the repository."""

    code = "run_not_found"


class IllegalRunTransition(RunStateError, ValueError):
    """A transition the state machine does not allow.

    The error is also a :class:`ValueError` so callers that validate a
    transition before applying it can use the usual exception handling.
    """

    code = "illegal_run_transition"

    def __init__(self, current: RunState, target: RunState) -> None:
        super().__init__(
            f"cannot transition a run from {current.value} to {target.value}",
            details={
                "current": current.value,
                "target": target.value,
                "allowed": sorted(
                    state.value for state in LEGAL_RUN_TRANSITIONS[current]
                ),
            },
        )
        self.current = current
        self.target = target


InvalidRunTransition = IllegalRunTransition
"""Alias for :class:`IllegalRunTransition`."""

InvalidRunTransitionError = IllegalRunTransition
"""Alias for :class:`IllegalRunTransition`."""

RunTransitionError = IllegalRunTransition
"""Alias for :class:`IllegalRunTransition`."""

InvalidTransition = IllegalRunTransition
"""Alias for :class:`IllegalRunTransition`."""

InvalidTransitionError = IllegalRunTransition
"""Alias for :class:`IllegalRunTransition`."""

IllegalTransition = IllegalRunTransition
"""Alias for :class:`IllegalRunTransition`."""

IllegalTransitionError = IllegalRunTransition
"""Alias for :class:`IllegalRunTransition`."""

ALLOWED_RUN_TRANSITIONS = LEGAL_RUN_TRANSITIONS
"""Alias for :data:`LEGAL_RUN_TRANSITIONS`."""

RUN_STATE_TRANSITIONS = LEGAL_RUN_TRANSITIONS
"""Alias for :data:`LEGAL_RUN_TRANSITIONS`."""


def _state(value: RunState | str) -> RunState:
    return value if isinstance(value, RunState) else RunState(value)


def legal_transitions(state: RunState | str) -> frozenset[RunState]:
    """Return the states a run in *state* may move to."""
    return LEGAL_RUN_TRANSITIONS[_state(state)]


def can_transition(source: RunState | str, target: RunState | str) -> bool:
    """Whether a run may move from *source* to *target*."""
    return _state(target) in LEGAL_RUN_TRANSITIONS[_state(source)]


is_legal_transition = can_transition
"""Alias for :func:`can_transition`."""


def _moment(value: datetime | None) -> datetime:
    return value if value is not None else utcnow()


def transition_run(
    run: Run,
    target: RunState | str,
    *,
    now: datetime | None = None,
    pause_reason: str = "",
    resumed_by: str = "",
    resume_reason: str = "",
    error_code: str = "",
    error_message: str = "",
) -> Run:
    """Return *run* moved to *target*, or raise when the move is illegal.

    ``started_at`` is stamped when a queued run starts and ``finished_at`` when
    it reaches a terminal state. Pausing stamps ``paused_at`` and
    ``pause_reason``; resuming stamps ``resumed_at`` plus the optional audit
    fields. The original record is never mutated.
    """
    current = run.state
    destination = _state(target)
    if destination not in LEGAL_RUN_TRANSITIONS[current]:
        raise IllegalRunTransition(current, destination)

    moment = _moment(now)
    changes: dict[str, Any] = {"state": destination, "updated_at": moment}

    if destination is RunState.RUNNING:
        if current is RunState.QUEUED and run.started_at is None:
            changes["started_at"] = moment
        if current is RunState.PAUSED_BUDGET:
            changes["resumed_at"] = moment
            changes["resumed_by"] = resumed_by.strip()
            changes["resume_reason"] = resume_reason.strip()
    elif destination is RunState.PAUSED_BUDGET:
        changes["paused_at"] = moment
        changes["pause_reason"] = pause_reason.strip() or "budget"
    elif destination in _TERMINAL_STATES:
        changes["finished_at"] = moment
        changes["error_code"] = error_code.strip()
        changes["error_message"] = error_message.strip()

    return run.model_copy(update=changes)


class RunStateMachine:
    """Move runs through legal states and persist every accepted change.

    *repository* may be a :class:`RunRepository` or an SQLAlchemy engine. The
    machine reads the latest record before each transition, so a stale
    :class:`Run` held by a caller cannot be used to skip a state. ``clock``
    supplies the timestamp for the transition and defaults to
    :func:`~aivia.core.models.base.utcnow`.
    """

    def __init__(
        self,
        repository: RunRepository | Engine,
        *,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        if isinstance(repository, Engine):
            repository = RunRepository(repository)
        self.repository = repository
        self.clock = clock

    @property
    def transitions(self) -> Mapping[RunState, frozenset[RunState]]:
        """The legal transition mapping used by this machine."""
        return LEGAL_RUN_TRANSITIONS

    def legal_transitions(self, run: Run | RunState | str) -> frozenset[RunState]:
        """Return the states the given run or state may move to."""
        state = run.state if isinstance(run, Run) else _state(run)
        return LEGAL_RUN_TRANSITIONS[state]

    def can_transition(self, run: Run | RunState | str, target: RunState | str) -> bool:
        """Whether *run* may move to *target*."""
        state = run.state if isinstance(run, Run) else _state(run)
        return _state(target) in LEGAL_RUN_TRANSITIONS[state]

    def create(self, run: Run | None = None, **values: Any) -> Run:
        """Persist a new run and return the stored record."""
        return self.repository.create(run, **values)

    def get(self, run_id: str) -> Run | None:
        """Return the stored run with *run_id*, if any."""
        return self.repository.get(run_id)

    def transition(
        self,
        run: Run | str,
        target: RunState | str,
        *,
        now: datetime | None = None,
        pause_reason: str = "",
        resumed_by: str = "",
        resume_reason: str = "",
        error_code: str = "",
        error_message: str = "",
    ) -> Run:
        """Persist one legal transition and return the updated record."""
        current = self._current(run)
        updated = transition_run(
            current,
            target,
            now=now if now is not None else self.clock(),
            pause_reason=pause_reason,
            resumed_by=resumed_by,
            resume_reason=resume_reason,
            error_code=error_code,
            error_message=error_message,
        )
        changes = {
            field: getattr(updated, field)
            for field in _PERSISTED_FIELDS
            if getattr(updated, field) != getattr(current, field)
        }
        persisted = self.repository.update(current.id, **changes)
        if persisted is None:
            raise RunNotFoundError(
                f"run not found: {current.id}",
                details={"run_id": current.id},
            )
        return persisted

    def start(self, run: Run | str, *, now: datetime | None = None) -> Run:
        """Move a queued run to ``running``."""
        return self.transition(run, RunState.RUNNING, now=now)

    def pause(
        self,
        run: Run | str,
        *,
        reason: str = "",
        now: datetime | None = None,
    ) -> Run:
        """Pause a running run on a budget cap."""
        return self.transition(
            run,
            RunState.PAUSED_BUDGET,
            pause_reason=reason,
            now=now,
        )

    def resume(
        self,
        run: Run | str,
        *,
        resumed_by: str = "",
        reason: str = "",
        now: datetime | None = None,
    ) -> Run:
        """Resume a paused run, recording who resumed it and why."""
        return self.transition(
            run,
            RunState.RUNNING,
            resumed_by=resumed_by,
            resume_reason=reason,
            now=now,
        )

    def succeed(self, run: Run | str, *, now: datetime | None = None) -> Run:
        """Move a running run to ``succeeded``."""
        return self.transition(run, RunState.SUCCEEDED, now=now)

    def fail(
        self,
        run: Run | str,
        *,
        error_code: str = "",
        error_message: str = "",
        now: datetime | None = None,
    ) -> Run:
        """Move a queued, running or paused run to ``failed``."""
        return self.transition(
            run,
            RunState.FAILED,
            error_code=error_code,
            error_message=error_message,
            now=now,
        )

    def cancel(
        self,
        run: Run | str,
        *,
        error_code: str = "",
        error_message: str = "",
        now: datetime | None = None,
    ) -> Run:
        """Move a queued, running or paused run to ``cancelled``."""
        return self.transition(
            run,
            RunState.CANCELLED,
            error_code=error_code,
            error_message=error_message,
            now=now,
        )

    def _current(self, run: Run | str) -> Run:
        run_id = run.id if isinstance(run, Run) else run
        current = self.repository.get(run_id)
        if current is None:
            if isinstance(run, Run):
                return run
            raise RunNotFoundError(
                f"run not found: {run_id}",
                details={"run_id": run_id},
            )
        return current


RunStateStore = RunStateMachine
"""Alias for :class:`RunStateMachine`."""
