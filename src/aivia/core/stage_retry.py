"""Per-stage timeout and retry policy for the orchestrator.

A stage is one idempotent step, so running it again after a transient failure
is safe. This module owns that policy in one place: how long a single attempt
may run, how many attempts are allowed, the delay between them and which
errors are worth retrying. The orchestrator applies the policy and records the
attempts on the stage result; a stage itself never implements a retry loop.

Only errors the shared taxonomy marks as retryable are retried, plus a timeout
raised by the per-attempt deadline and any extra codes a stage opts into.
Permanent errors stop immediately because a second attempt cannot change the
outcome. When the last allowed attempt still fails, the failure is reported as
:class:`StageRetryExhausted` carrying every attempt for the run record.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from aivia.core.errors import AiviaError
from aivia.core.errors import TimeoutError as AiviaTimeoutError
from aivia.core.models.base import utcnow

if TYPE_CHECKING:
    from aivia.core.models.run import StageResult
    from aivia.core.stages import Stage, StageContext

__all__ = [
    "StageAttempt",
    "StageExhaustedError",
    "StageRetryExhausted",
    "StageRetryOutcome",
    "StageRetryPolicy",
    "execute_stage_with_retry",
    "retry_policy_for",
    "run_stage_with_retry",
]

_FLOOR = 0.0
_FLOOR_COUNT = 1
_FLOOR_MULTIPLIER = 1.0
_ZERO_DELAY = 0.0

_RETRYABLE_CODES = frozenset(
    {
        "provider_exhausted",
        "rate_limit",
        "retryable",
        "timeout",
    }
)


def _snake_case(name: str) -> str:
    result: list[str] = []
    for index, character in enumerate(name):
        if character.isupper() and index and not name[index - 1].isupper():
            result.append("_")
        result.append(character.lower())
    return "".join(result)


def _finite(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _outcome_value(outcome: object, key: str, default: Any = None) -> Any:
    if isinstance(outcome, Mapping):
        return outcome.get(key, default)
    return getattr(outcome, key, default)


def _outcome_failure_code(outcome: object) -> str:
    status = _outcome_value(outcome, "status")
    value = getattr(status, "value", status)
    if value not in {"failed", "cancelled", "paused_budget"}:
        return ""
    code = _outcome_value(outcome, "error_code", "")
    return code.strip() if isinstance(code, str) else ""


def _outcome_message(outcome: object, fallback: str) -> str:
    message = _outcome_value(outcome, "error_message", "")
    if isinstance(message, str) and message.strip():
        return message.strip()
    return fallback


def _outcome_cost(outcome: object) -> float:
    cost = _outcome_value(outcome, "cost_usd", 0.0)
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return 0.0
    value = float(cost)
    return value if math.isfinite(value) and value > _ZERO_DELAY else _ZERO_DELAY


@dataclass(frozen=True, slots=True)
class StageRetryPolicy:
    """How often and how long one stage may run.

    ``timeout_seconds`` bounds a single attempt; ``None`` disables the
    deadline. ``max_attempts`` counts the first try, so the default of one
    means no retries. ``retry_delay_seconds`` is the wait after a failed
    attempt, multiplied by ``backoff_multiplier`` on each later attempt and
    capped by ``max_retry_delay_seconds``. ``retry_codes`` adds stable error
    codes beyond the retryable taxonomy, and ``retry_on_timeout`` controls the
    per-attempt deadline.
    """

    timeout_seconds: float | None = None
    max_attempts: int = 1
    retry_delay_seconds: float = 0.0
    backoff_multiplier: float = 1.0
    max_retry_delay_seconds: float | None = None
    retry_on_timeout: bool = True
    retry_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None:
            timeout = _finite(self.timeout_seconds, name="timeout_seconds")
            if timeout <= _FLOOR:
                raise ValueError("timeout_seconds must be greater than zero")
            object.__setattr__(self, "timeout_seconds", timeout)
        if isinstance(self.max_attempts, bool) or not isinstance(
            self.max_attempts, int
        ):
            raise TypeError("max_attempts must be an integer")
        if self.max_attempts < _FLOOR_COUNT:
            raise ValueError("max_attempts must be at least one")
        delay = _finite(self.retry_delay_seconds, name="retry_delay_seconds")
        if delay < _ZERO_DELAY:
            raise ValueError("retry_delay_seconds must not be negative")
        object.__setattr__(self, "retry_delay_seconds", delay)
        multiplier = _finite(self.backoff_multiplier, name="backoff_multiplier")
        if multiplier < _FLOOR_MULTIPLIER:
            raise ValueError("backoff_multiplier must be at least one")
        object.__setattr__(self, "backoff_multiplier", multiplier)
        if self.max_retry_delay_seconds is not None:
            maximum = _finite(
                self.max_retry_delay_seconds,
                name="max_retry_delay_seconds",
            )
            if maximum < _ZERO_DELAY:
                raise ValueError("max_retry_delay_seconds must not be negative")
            object.__setattr__(self, "max_retry_delay_seconds", maximum)
        if not isinstance(self.retry_on_timeout, bool):
            raise TypeError("retry_on_timeout must be a boolean")
        codes: list[str] = []
        for code in self.retry_codes:
            if not isinstance(code, str):
                raise TypeError("retry_codes must be strings")
            value = code.strip()
            if value and value not in codes:
                codes.append(value)
        object.__setattr__(self, "retry_codes", tuple(codes))

    @property
    def attempts(self) -> int:
        """Alias for :attr:`max_attempts`."""
        return self.max_attempts

    @property
    def retries(self) -> int:
        """How many retries follow the first attempt."""
        return self.max_attempts - 1

    def delay_for(self, attempt: int) -> float:
        """Return the wait after *attempt* before the next attempt starts."""
        if attempt < _FLOOR_COUNT:
            raise ValueError("attempt must be at least one")
        delay = self.retry_delay_seconds * (self.backoff_multiplier ** (attempt - 1))
        if self.max_retry_delay_seconds is not None:
            delay = min(delay, self.max_retry_delay_seconds)
        return max(_ZERO_DELAY, delay)

    def is_retryable_code(self, code: str) -> bool:
        """Whether a stable error *code* is one this policy retries."""
        value = code.strip()
        return value in self.retry_codes or value in _RETRYABLE_CODES

    def is_retryable(self, error: BaseException) -> bool:
        """Whether *error* is the kind this policy retries, ignoring the limit."""
        code = getattr(error, "code", "")
        if isinstance(code, str) and code.strip() in self.retry_codes:
            return True
        if isinstance(error, (asyncio.TimeoutError, AiviaTimeoutError)):
            return self.retry_on_timeout
        if isinstance(error, AiviaError):
            if error.retryable:
                return True
            return error.code in self.retry_codes or error.code in _RETRYABLE_CODES
        return False

    def should_retry(self, error: BaseException, attempt: int) -> bool:
        """Whether *error* on *attempt* leaves another attempt worth trying."""
        return attempt < self.max_attempts and self.is_retryable(error)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for a stage result."""
        return {
            "timeout_seconds": self.timeout_seconds,
            "max_attempts": self.max_attempts,
            "retry_delay_seconds": self.retry_delay_seconds,
            "backoff_multiplier": self.backoff_multiplier,
            "max_retry_delay_seconds": self.max_retry_delay_seconds,
            "retry_on_timeout": self.retry_on_timeout,
            "retry_codes": list(self.retry_codes),
        }


@dataclass(frozen=True, slots=True)
class StageAttempt:
    """One attempt to run a stage, successful or not.

    ``number`` is one-based. The timestamps bound that single attempt rather
    than the whole stage, so a retried stage still shows where time went.
    """

    number: int
    started_at: datetime
    finished_at: datetime
    succeeded: bool = False
    timed_out: bool = False
    retryable: bool = False
    error_code: str = ""
    error_message: str = ""
    error_type: str = ""

    @property
    def duration_seconds(self) -> float:
        """Elapsed seconds for this attempt."""
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def ok(self) -> bool:
        """Whether this attempt succeeded."""
        return self.succeeded

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for a stage result."""
        payload: dict[str, Any] = {
            "number": self.number,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "succeeded": self.succeeded,
            "timed_out": self.timed_out,
        }
        if not self.succeeded:
            payload.update(
                {
                    "retryable": self.retryable,
                    "error_code": self.error_code,
                    "error_message": self.error_message,
                    "error_type": self.error_type,
                }
            )
        return payload


@dataclass(frozen=True, slots=True)
class StageRetryOutcome:
    """The final stage outcome together with every attempt it took.

    ``prior_cost_usd`` sums the cost reported by failed attempts that were
    retried, so a paid failure is not lost when the next attempt succeeds.
    ``exhausted`` is true when a retryable failure used the last attempt.
    """

    outcome: StageResult | Mapping[str, Any] | None
    attempts: tuple[StageAttempt, ...] = field(default_factory=tuple)
    exhausted: bool = False
    prior_cost_usd: float = 0.0

    @property
    def attempt_count(self) -> int:
        """How many attempts ran."""
        return len(self.attempts)

    @property
    def retries(self) -> int:
        """How many attempts followed the first one."""
        return max(0, len(self.attempts) - 1)


class StageRetryExhausted(AiviaError):
    """Every allowed attempt failed and no retry remains.

    ``attempts`` holds the full history, ``last_error`` is the final failure
    and ``error_code`` mirrors that error so the orchestrator records the
    stable cause rather than a generic wrapper code.
    """

    code = "stage_exhausted"
    retryable = False

    def __init__(
        self,
        stage: str,
        attempts: tuple[StageAttempt, ...],
        last_error: BaseException,
        policy: StageRetryPolicy,
        *,
        prior_cost_usd: float = 0.0,
    ) -> None:
        self.stage = stage
        self.stage_name = stage
        self.attempts = attempts
        self.last_error = last_error
        self.policy = policy
        self.prior_cost_usd = prior_cost_usd
        self.error_code = getattr(last_error, "code", "") or _snake_case(
            type(last_error).__name__
        )
        self.error_message = str(last_error).strip() or type(last_error).__name__
        self.timed_out = isinstance(
            last_error, (asyncio.TimeoutError, AiviaTimeoutError)
        )
        super().__init__(
            f"stage {stage!r} exhausted {len(attempts)} attempt(s): "
            f"{self.error_message}",
            details={
                "stage": stage,
                "attempts": [attempt.to_dict() for attempt in attempts],
                "max_attempts": policy.max_attempts,
                "exhausted": True,
                "error_code": self.error_code,
                "prior_cost_usd": prior_cost_usd,
            },
        )


StageExhaustedError = StageRetryExhausted
"""Alias for :class:`StageRetryExhausted`."""


def retry_policy_for(stage: Stage) -> StageRetryPolicy:
    """Return the policy a stage declares, or the no-retry default."""
    policy = getattr(stage, "retry_policy", None)
    if isinstance(policy, StageRetryPolicy):
        return policy
    return StageRetryPolicy()


async def execute_stage_with_retry(
    stage: Stage,
    context: StageContext,
    *,
    policy: StageRetryPolicy | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    clock: Callable[[], datetime] = utcnow,
) -> StageRetryOutcome:
    """Run *stage* under its timeout and retry policy.

    A successful attempt returns a :class:`StageRetryOutcome` with the stage's
    own return value. A permanent failure is re-raised unchanged so the caller
    keeps the original exception and traceback, with ``attempts`` attached for
    the record. A retryable failure that survives the final attempt raises
    :class:`StageRetryExhausted`.
    """
    selected = policy or retry_policy_for(stage)
    pause = sleep or asyncio.sleep
    attempts: list[StageAttempt] = []
    prior_cost = _ZERO_DELAY

    for number in range(1, selected.max_attempts + 1):
        started = clock()
        try:
            if selected.timeout_seconds is None:
                outcome = await stage.run(context)
            else:
                outcome = await asyncio.wait_for(
                    stage.run(context),
                    timeout=selected.timeout_seconds,
                )
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            finished = clock()
            timed_out = isinstance(error, (asyncio.TimeoutError, AiviaTimeoutError))
            if timed_out and not isinstance(error, AiviaError):
                error = AiviaTimeoutError(
                    f"stage {stage.name!r} exceeded {selected.timeout_seconds:g}s",
                    details={
                        "stage": stage.name,
                        "timeout_seconds": selected.timeout_seconds,
                        "attempt": number,
                    },
                )
            retryable = selected.is_retryable(error)
            attempt = StageAttempt(
                number=number,
                started_at=started,
                finished_at=finished,
                timed_out=timed_out,
                retryable=retryable,
                error_code=getattr(error, "code", "")
                or _snake_case(type(error).__name__),
                error_message=str(error).strip() or type(error).__name__,
                error_type=type(error).__name__,
            )
            attempts.append(attempt)
            if not retryable:
                for attribute, value in (
                    ("attempts", tuple(attempts)),
                    ("attempt_history", tuple(attempts)),
                    ("prior_cost_usd", prior_cost),
                ):
                    setattr(error, attribute, value)
                raise
            if number >= selected.max_attempts:
                raise StageRetryExhausted(
                    stage.name,
                    tuple(attempts),
                    error,
                    selected,
                    prior_cost_usd=prior_cost,
                ) from error
            await pause(selected.delay_for(number))
            continue

        failure_code = _outcome_failure_code(outcome)
        if failure_code:
            retryable = selected.is_retryable_code(failure_code)
            attempts.append(
                StageAttempt(
                    number=number,
                    started_at=started,
                    finished_at=clock(),
                    retryable=retryable,
                    error_code=failure_code,
                    error_message=_outcome_message(outcome, failure_code),
                    error_type=type(outcome).__name__,
                )
            )
            if retryable and number < selected.max_attempts:
                prior_cost += _outcome_cost(outcome)
                await pause(selected.delay_for(number))
                continue
            return StageRetryOutcome(
                outcome=outcome,
                attempts=tuple(attempts),
                exhausted=retryable,
                prior_cost_usd=prior_cost,
            )

        attempts.append(
            StageAttempt(
                number=number,
                started_at=started,
                finished_at=clock(),
                succeeded=True,
            )
        )
        return StageRetryOutcome(
            outcome=outcome,
            attempts=tuple(attempts),
            prior_cost_usd=prior_cost,
        )

    raise AssertionError("retry loop ended without an outcome")


run_stage_with_retry = execute_stage_with_retry
"""Alias for :func:`execute_stage_with_retry`."""
