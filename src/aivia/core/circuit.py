"""Per-provider circuit breaker used by the router.

A provider that keeps failing is temporarily removed from selection instead of
being retried on every request. The breaker is closed while calls succeed,
opens after ``failure_threshold`` consecutive failures, and stays open for
``cooldown_seconds``. Once the cooldown elapses it half-opens: one probe request
is allowed through, and its outcome either closes the breaker again or reopens
it for another cooldown.

The router owns one breaker per configured provider name, so a failure on one
provider never affects another provider or capability. Time is read through an
injectable ``clock`` so tests can advance the cooldown without sleeping.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum

from aivia.core.config.circuit import CircuitBreakerConfig
from aivia.core.errors import ConfigError

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "ProviderCircuitBreaker",
]


class CircuitState(StrEnum):
    """The lifecycle state of a provider circuit."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Track consecutive failures for one provider and gate requests.

    ``failure_threshold`` is the number of consecutive failures that opens the
    circuit. ``cooldown_seconds`` is how long it stays open before allowing one
    probe. Both must be valid; invalid values raise
    :class:`~aivia.core.errors.ConfigError` rather than silently disabling the
    breaker.
    """

    def __init__(
        self,
        *,
        failure_threshold: int | None = None,
        cooldown_seconds: float | None = None,
        clock: Callable[[], float] | None = None,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        if config is not None and not isinstance(config, CircuitBreakerConfig):
            raise TypeError("config must be a CircuitBreakerConfig")
        values = config.model_dump() if config is not None else {}
        if failure_threshold is not None:
            values["failure_threshold"] = failure_threshold
        if cooldown_seconds is not None:
            values["cooldown_seconds"] = cooldown_seconds
        settings = CircuitBreakerConfig.from_mapping(values)
        if clock is not None and not callable(clock):
            raise ConfigError("circuit clock must be callable")

        self._failure_threshold = settings.failure_threshold
        self._cooldown_seconds = settings.cooldown_seconds
        self._clock = clock or time.monotonic
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    @property
    def failure_threshold(self) -> int:
        """Consecutive failures needed to open the circuit."""
        return self._failure_threshold

    @property
    def cooldown_seconds(self) -> float:
        """Seconds the circuit stays open before a half-open probe."""
        return self._cooldown_seconds

    @property
    def failures(self) -> int:
        """The current number of consecutive failures."""
        return self._failures

    @property
    def failure_count(self) -> int:
        """Alias for :attr:`failures`."""
        return self._failures

    @property
    def opened_at(self) -> float | None:
        """The clock reading when the circuit last opened, if it is open."""
        return self._opened_at

    @property
    def state(self) -> CircuitState:
        """The current state, cooling an open circuit down when due."""
        self._refresh()
        return self._state

    @property
    def is_closed(self) -> bool:
        """Whether requests are flowing normally."""
        return self.state is CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Whether the circuit is open and rejecting requests."""
        return self.state is CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Whether the circuit is cooling down and ready to probe."""
        return self.state is CircuitState.HALF_OPEN

    def cooldown_remaining(self) -> float:
        """Return the seconds left before an open circuit half-opens."""
        state = self.state
        if state is not CircuitState.OPEN or self._opened_at is None:
            return 0.0
        elapsed = self._clock() - self._opened_at
        return max(0.0, self._cooldown_seconds - elapsed)

    def available(self) -> bool:
        """Return whether a request may be attempted now.

        An open circuit rejects every request. A half-open circuit allows one
        probe at a time, so it also rejects while a probe is in flight.
        """
        state = self.state
        if state is CircuitState.OPEN:
            return False
        return not (state is CircuitState.HALF_OPEN and self._probe_in_flight)

    def allow_request(self) -> bool:
        """Alias for :meth:`available`."""
        return self.available()

    def acquire(self) -> bool:
        """Reserve permission for one request.

        Closed circuits always grant permission. A half-open circuit grants
        exactly one probe until :meth:`record_success`, :meth:`record_failure`
        or :meth:`release` is called. Open circuits reject.
        """
        if not self.available():
            return False
        if self._state is CircuitState.HALF_OPEN:
            self._probe_in_flight = True
        return True

    def release(self) -> None:
        """Release a half-open probe without changing the state.

        This is used when a request fails for a reason that says nothing about
        provider health, such as an invalid request, so the next caller can
        still probe the recovering provider.
        """
        self._probe_in_flight = False

    def record_success(self) -> None:
        """Record a successful call and close the circuit."""
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = None
        self._probe_in_flight = False

    def record_failure(self) -> None:
        """Record a failed call, opening the circuit when the limit is reached.

        A failure while half-open reopens the circuit immediately because the
        probe proved the provider is still unhealthy. Failures while already
        open are ignored so a late in-flight result cannot extend the cooldown.
        """
        state = self.state
        if state is CircuitState.OPEN:
            return
        if state is CircuitState.HALF_OPEN:
            self._open()
            return
        self._failures += 1
        if self._failures >= self._failure_threshold:
            self._open()

    def reset(self) -> None:
        """Return the circuit to a clean closed state."""
        self.record_success()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot for logs and errors."""
        state = self.state
        payload: dict[str, object] = {
            "state": state.value,
            "failures": self._failures,
            "failure_threshold": self._failure_threshold,
            "cooldown_seconds": self._cooldown_seconds,
        }
        if state is CircuitState.OPEN:
            payload["cooldown_remaining"] = self.cooldown_remaining()
        return payload

    def _open(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = self._clock()
        self._probe_in_flight = False

    def _refresh(self) -> None:
        if self._state is not CircuitState.OPEN or self._opened_at is None:
            return
        if self._clock() - self._opened_at >= self._cooldown_seconds:
            self._state = CircuitState.HALF_OPEN
            self._probe_in_flight = False


ProviderCircuitBreaker = CircuitBreaker
