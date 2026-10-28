"""Per-provider token bucket rate limiting for router calls.

A provider declares a sustained request rate and an optional burst allowance in
its :class:`~aivia.core.models.provider.RateLimit` metadata. This module turns
that declaration into a token bucket: tokens refill continuously at the
sustained rate, the bucket never holds more than the burst allowance, and each
provider call consumes one token.

The router owns one bucket per configured provider, so a busy provider never
drains another provider's allowance. A caller can either wait for the next
token or fail fast with :class:`~aivia.core.errors.RateLimitError`. Both modes
are safe under concurrency: an :class:`asyncio.Lock` serialises token accounting,
and waiting callers reserve their future token so a burst of coroutines cannot
overdraw the bucket. Time and sleep are injectable so tests never wait in real
time.

A provider can also declare ``max_concurrency``, the number of simultaneous
calls it accepts. :class:`ConcurrencyLimiter` enforces that cap with the same
wait-or-fail policy as the bucket, and every :class:`TokenBucket` owns one when
its metadata declares a positive cap. Callers acquire a token and a slot
together and must call :meth:`TokenBucket.release` when the call finishes.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
from collections.abc import Awaitable, Callable

from aivia.core.config.ratelimit import (
    OnLimit,
    OnLimitBehavior,
    OnLimitBehaviour,
    OnRateLimit,
    RateLimitBehavior,
    RateLimitBehaviour,
    RateLimiterConfig,
)
from aivia.core.errors import ConfigError, RateLimitError
from aivia.core.models.provider import RateLimit

__all__ = [
    "ConcurrencyLimiter",
    "OnLimit",
    "OnLimitBehavior",
    "OnLimitBehaviour",
    "OnRateLimit",
    "ProviderRateLimiter",
    "RateLimitBehavior",
    "RateLimitBehaviour",
    "RateLimiter",
    "RateLimiterConfig",
    "RateLimiterSettings",
    "TokenBucket",
]

RateLimiterSettings = RateLimiterConfig


def _coerce_float(value: object, *, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"{label} must be a finite number",
            details={label: value},
        )
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ConfigError(
            f"{label} must be a finite number greater than or equal to {minimum}",
            details={label: value},
        )
    return number


def _coerce_on_limit(value: OnRateLimit | str) -> OnRateLimit:
    if isinstance(value, OnRateLimit):
        return value
    if isinstance(value, str):
        try:
            return OnRateLimit(value.strip())
        except ValueError:
            pass
    allowed = ", ".join(mode.value for mode in OnRateLimit)
    raise ConfigError(
        f"unknown rate limit behaviour: {value!r} (allowed: {allowed})",
        details={"on_limit": value, "allowed": [mode.value for mode in OnRateLimit]},
    )


def _coerce_limit(value: object, *, label: str = "max_concurrency") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"{label} must be a non-negative integer",
            details={label: value},
        )
    if value < 0:
        raise ConfigError(
            f"{label} must be a non-negative integer",
            details={label: value},
        )
    return value


class ConcurrencyLimiter:
    """A concurrency-safe cap on simultaneous provider calls.

    ``max_concurrency`` of ``0`` means the provider declares no concurrency
    limit. A positive cap admits at most that many callers at once; the rest
    either queue in arrival order or raise
    :class:`~aivia.core.errors.RateLimitError`, according to ``on_limit``.
    A per-call ``wait`` argument overrides the default policy.

    Every successful :meth:`acquire` must be paired with :meth:`release` once
    the call finishes. The limiter can also be used as an async context
    manager. ``clock`` is injectable so tests can measure waits without
    sleeping.
    """

    def __init__(
        self,
        max_concurrency: int = 0,
        *,
        limit: int | None = None,
        config: RateLimiterConfig | None = None,
        on_limit: OnRateLimit | str | None = None,
        wait: bool | None = None,
        name: str | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if config is not None and not isinstance(config, RateLimiterConfig):
            raise TypeError("config must be a RateLimiterConfig")
        if name is not None and not isinstance(name, str):
            raise TypeError("name must be a string")
        if clock is not None and not callable(clock):
            raise ConfigError("concurrency limit clock must be callable")

        settings = config or RateLimiterConfig()
        if limit is not None:
            max_concurrency = limit
        self._max_concurrency = _coerce_limit(max_concurrency)

        if wait is not None:
            if not isinstance(wait, bool):
                raise TypeError("wait must be a boolean")
            selected = OnRateLimit.WAIT if wait else OnRateLimit.RAISE
        elif on_limit is not None:
            selected = _coerce_on_limit(on_limit)
        else:
            selected = settings.on_limit

        self._on_limit = selected
        self._name = name
        self._clock = clock or time.monotonic
        self._active = 0
        self._waiters: list[asyncio.Future[None]] = []
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str | None:
        """The configured provider name, used in error details."""
        return self._name

    @property
    def max_concurrency(self) -> int:
        """The largest number of simultaneous calls, where ``0`` is unlimited."""
        return self._max_concurrency

    @property
    def limit(self) -> int:
        """Alias for :attr:`max_concurrency`."""
        return self._max_concurrency

    @property
    def on_limit(self) -> OnRateLimit:
        """The default behaviour when the cap is reached."""
        return self._on_limit

    @property
    def unlimited(self) -> bool:
        """Whether the provider declared no concurrency limit."""
        return self._max_concurrency == 0

    @property
    def active(self) -> int:
        """The number of calls currently holding a slot."""
        return self._active

    @property
    def in_flight(self) -> int:
        """Alias for :attr:`active`."""
        return self._active

    def available(self) -> bool:
        """Return whether one slot can be taken immediately."""
        return self.unlimited or self._active < self._max_concurrency

    async def acquire(self, *, wait: bool | None = None) -> float:
        """Take a slot, waiting or raising according to the configured policy.

        Returns the number of seconds the caller waited, which is ``0`` when a
        slot was available immediately. Waiting callers queue in arrival order.
        """
        should_wait = self._on_limit is OnRateLimit.WAIT if wait is None else wait
        if not isinstance(should_wait, bool):
            raise TypeError("wait must be a boolean")
        if self.unlimited:
            return 0.0

        async with self._lock:
            if self._active < self._max_concurrency:
                self._active += 1
                return 0.0
            if not should_wait:
                raise self._limited_error()
            loop = asyncio.get_running_loop()
            waiter: asyncio.Future[None] = loop.create_future()
            self._waiters.append(waiter)
            started = self._clock()

        try:
            await waiter
        except asyncio.CancelledError:
            if waiter.done() and not waiter.cancelled():
                self.release()
            else:
                with contextlib.suppress(ValueError):
                    self._waiters.remove(waiter)
            raise
        return self._clock() - started

    async def acquire_nowait(self) -> float:
        """Take a slot or raise immediately when the cap is reached."""
        return await self.acquire(wait=False)

    async def wait_for_slot(self) -> float:
        """Wait for a slot, then take it."""
        return await self.acquire(wait=True)

    def release(self) -> None:
        """Release a slot, handing it directly to the next waiter if any."""
        if self.unlimited:
            return
        while self._waiters:
            waiter = self._waiters.pop(0)
            if not waiter.done():
                waiter.set_result(None)
                return
        if self._active <= 0:
            raise RuntimeError("concurrency slot released without being acquired")
        self._active -= 1

    def reset(self) -> None:
        """Drop every waiter and reset the active count to zero."""
        for waiter in self._waiters:
            if not waiter.done():
                waiter.cancel()
        self._waiters.clear()
        self._active = 0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot for logs and errors."""
        payload: dict[str, object] = {
            "max_concurrency": self._max_concurrency,
            "active": self._active,
            "on_limit": self._on_limit.value,
            "unlimited": self.unlimited,
        }
        if self._name is not None:
            payload["provider"] = self._name
        return payload

    async def __aenter__(self) -> ConcurrencyLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.release()

    def _limited_error(self) -> RateLimitError:
        provider = f" for provider {self._name!r}" if self._name else ""
        details: dict[str, object] = {
            "max_concurrency": self._max_concurrency,
            "active": self._active,
        }
        if self._name is not None:
            details["provider"] = self._name
        return RateLimitError(
            f"concurrency limit reached{provider}; "
            f"{self._active}/{self._max_concurrency} calls in flight",
            retry_after=None,
            details=details,
        )


class TokenBucket:
    """A concurrency-safe token bucket for one provider.

    ``requests_per_minute`` is the sustained refill rate and ``burst`` (or
    ``capacity``) is the largest allowance. Passing ``rate_limit`` copies each
    value the provider explicitly declared; any value it left unset comes from
    ``config`` instead. A sustained rate of ``0`` means the provider declares
    no limit, so every call is allowed immediately.

    ``on_limit`` selects the default behaviour when no token is available:
    :attr:`~aivia.core.config.ratelimit.OnRateLimit.WAIT` sleeps until the next
    token, while ``RAISE`` raises :class:`~aivia.core.errors.RateLimitError`
    with the delay as ``retry_after``. A per-call ``wait`` argument overrides
    the default.

    ``max_concurrency`` caps simultaneous calls; ``0`` means unlimited. It is
    copied from ``rate_limit`` when given and enforced by an owned
    :class:`ConcurrencyLimiter` using the same wait-or-raise policy. A caller
    that acquires a bucket must call :meth:`release` when the call finishes so
    the concurrency slot is returned.

    ``clock`` and ``sleep`` are injectable for tests. The bucket starts full so
    a provider can serve one burst immediately.
    """

    def __init__(
        self,
        requests_per_minute: float | None = None,
        burst: int | None = None,
        *,
        rate: float | None = None,
        capacity: float | None = None,
        rate_limit: RateLimit | None = None,
        max_concurrency: int | None = None,
        config: RateLimiterConfig | None = None,
        on_limit: OnRateLimit | str | None = None,
        wait: bool | None = None,
        name: str | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if config is not None and not isinstance(config, RateLimiterConfig):
            raise TypeError("config must be a RateLimiterConfig")
        if rate_limit is not None and not isinstance(rate_limit, RateLimit):
            raise TypeError("rate_limit must be a RateLimit")
        if name is not None and not isinstance(name, str):
            raise TypeError("name must be a string")
        if clock is not None and not callable(clock):
            raise ConfigError("rate limit clock must be callable")
        if sleep is not None and not callable(sleep):
            raise ConfigError("rate limit sleep must be callable")

        settings = config or RateLimiterConfig()
        declared = (
            rate_limit.model_fields_set if rate_limit is not None else frozenset()
        )
        if rate_limit is not None and "requests_per_minute" in declared:
            minute_rate: float = float(rate_limit.requests_per_minute)
        elif rate_limit is not None:
            minute_rate = float(settings.requests_per_minute)
        else:
            minute_rate = (
                float(requests_per_minute)
                if requests_per_minute is not None
                else settings.requests_per_minute
            )

        if rate is not None and requests_per_minute is None and rate_limit is None:
            minute_rate = _coerce_float(rate, label="rate") * 60.0

        self._requests_per_minute = _coerce_float(
            minute_rate, label="requests_per_minute"
        )
        if burst is not None and burst < 0:
            raise ConfigError(
                "burst must be a non-negative integer",
                details={"burst": burst},
            )
        if capacity is not None:
            allowance = float(capacity)
        elif burst is not None:
            allowance = float(burst) if burst > 0 else minute_rate
        elif rate_limit is not None:
            if "burst" in declared:
                allowance = (
                    float(rate_limit.burst) if rate_limit.burst > 0 else minute_rate
                )
            elif settings.burst > 0:
                allowance = float(settings.burst)
            else:
                allowance = minute_rate
        elif requests_per_minute is not None or rate is not None:
            allowance = minute_rate
        else:
            allowance = float(settings.effective_burst)
        allowance = _coerce_float(allowance, label="capacity")
        if self._requests_per_minute > 0 and allowance <= 0:
            raise ConfigError(
                "capacity must be positive when requests_per_minute is positive",
                details={"capacity": allowance},
            )

        if wait is not None:
            if not isinstance(wait, bool):
                raise TypeError("wait must be a boolean")
            selected = OnRateLimit.WAIT if wait else OnRateLimit.RAISE
        elif on_limit is not None:
            selected = _coerce_on_limit(on_limit)
        else:
            selected = settings.on_limit

        self._capacity = allowance
        self._unlimited = self._requests_per_minute == 0
        self._on_limit = selected
        self._name = name
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._tokens = allowance
        self._updated_at = self._clock()
        self._lock = asyncio.Lock()

        if max_concurrency is None:
            concurrency_limit = (
                rate_limit.max_concurrency if rate_limit is not None else 0
            )
        else:
            concurrency_limit = _coerce_limit(max_concurrency)
        self._concurrency = ConcurrencyLimiter(
            concurrency_limit,
            on_limit=selected,
            name=name,
            clock=self._clock,
        )

    @property
    def name(self) -> str | None:
        """The configured provider name, used in error details."""
        return self._name

    @property
    def requests_per_minute(self) -> float:
        """The sustained request rate, where ``0`` means unlimited."""
        return self._requests_per_minute

    @property
    def rate_per_second(self) -> float:
        """The sustained refill rate expressed per second."""
        return self._requests_per_minute / 60.0

    @property
    def capacity(self) -> float:
        """The largest number of tokens the bucket can hold."""
        return self._capacity

    @property
    def burst(self) -> float:
        """Alias for :attr:`capacity`."""
        return self._capacity

    @property
    def effective_burst(self) -> float:
        """Alias for :attr:`capacity`."""
        return self._capacity

    @property
    def unlimited(self) -> bool:
        """Whether the provider declared no sustained request limit."""
        return self._unlimited

    @property
    def on_limit(self) -> OnRateLimit:
        """The default behaviour when no token is available."""
        return self._on_limit

    @property
    def max_concurrency(self) -> int:
        """The largest number of simultaneous calls, where ``0`` is unlimited."""
        return self._concurrency.max_concurrency

    @property
    def concurrency(self) -> ConcurrencyLimiter:
        """The concurrency gate enforcing :attr:`max_concurrency`."""
        return self._concurrency

    @property
    def active(self) -> int:
        """The number of calls currently holding a concurrency slot."""
        return self._concurrency.active

    @property
    def in_flight(self) -> int:
        """Alias for :attr:`active`."""
        return self._concurrency.active

    @property
    def tokens(self) -> float:
        """The tokens available now, never negative even while reserved."""
        self._refill()
        return max(0.0, self._tokens)

    @property
    def available_tokens(self) -> float:
        """Alias for :attr:`tokens`."""
        return self.tokens

    def available(self) -> bool:
        """Return whether a token and a concurrency slot can be taken now."""
        if not self._concurrency.available():
            return False
        if self._unlimited:
            return True
        self._refill()
        return self._tokens >= 1.0

    def time_until(self, tokens: float = 1.0) -> float:
        """Return the seconds until *tokens* can be taken without waiting."""
        cost = _coerce_float(tokens, label="tokens")
        if self._unlimited or cost == 0:
            return 0.0
        self._refill()
        if self._tokens >= cost:
            return 0.0
        return (cost - self._tokens) / self.rate_per_second

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Take *tokens* if available and return whether it succeeded."""
        cost = _coerce_float(tokens, label="tokens")
        if self._unlimited or cost == 0:
            return True
        self._refill()
        if self._tokens < cost:
            return False
        self._tokens -= cost
        return True

    async def acquire(
        self,
        tokens: float = 1.0,
        *,
        wait: bool | None = None,
    ) -> float:
        """Take *tokens* and a concurrency slot, waiting or raising per policy.

        Returns the number of seconds the caller waited, which is ``0`` when
        both the token and the slot were available immediately. A waiting
        caller reserves its token before sleeping, so concurrent waiters queue
        in token order instead of all waking and overdrawing the bucket. The
        caller must call :meth:`release` once the provider call finishes.
        """
        cost = _coerce_float(tokens, label="tokens")
        should_wait = self._on_limit is OnRateLimit.WAIT if wait is None else wait
        if not isinstance(should_wait, bool):
            raise TypeError("wait must be a boolean")

        waited_concurrency = await self._concurrency.acquire(wait=should_wait)
        try:
            waited_token = await self._acquire_token(cost, should_wait)
        except BaseException:
            self._concurrency.release()
            raise
        return waited_concurrency + waited_token

    async def _acquire_token(self, cost: float, should_wait: bool) -> float:
        if self._unlimited or cost == 0:
            return 0.0

        async with self._lock:
            self._refill()
            if self._tokens >= cost:
                self._tokens -= cost
                return 0.0
            delay = (cost - self._tokens) / self.rate_per_second
            if not should_wait:
                raise self._limited_error(cost, delay)
            self._tokens -= cost

        if delay > 0:
            await self._sleep(delay)
        return delay

    async def acquire_nowait(self, tokens: float = 1.0) -> float:
        """Take *tokens* or raise immediately when none is available."""
        return await self.acquire(tokens, wait=False)

    async def wait_for_token(self, tokens: float = 1.0) -> float:
        """Wait until *tokens* are available, then take them."""
        return await self.acquire(tokens, wait=True)

    def release(self) -> None:
        """Release the concurrency slot taken by :meth:`acquire`."""
        self._concurrency.release()

    def reset(self, tokens: float | None = None) -> None:
        """Refill the bucket to *tokens*, defaulting to full capacity."""
        if tokens is None:
            self._tokens = self._capacity
        else:
            self._tokens = _coerce_float(tokens, label="tokens")
        self._updated_at = self._clock()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot for logs and errors."""
        payload: dict[str, object] = {
            "requests_per_minute": self._requests_per_minute,
            "capacity": self._capacity,
            "tokens": self.tokens,
            "max_concurrency": self.max_concurrency,
            "active": self.active,
            "on_limit": self._on_limit.value,
            "unlimited": self._unlimited,
        }
        if self._name is not None:
            payload["provider"] = self._name
        return payload

    def _refill(self) -> None:
        if self._unlimited:
            return
        now = self._clock()
        elapsed = now - self._updated_at
        if elapsed <= 0:
            return
        self._tokens = min(
            self._capacity,
            self._tokens + elapsed * self.rate_per_second,
        )
        self._updated_at = now

    def _limited_error(self, cost: float, delay: float) -> RateLimitError:
        provider = f" for provider {self._name!r}" if self._name else ""
        details: dict[str, object] = {
            "requests_per_minute": self._requests_per_minute,
            "capacity": self._capacity,
            "tokens": max(0.0, self._tokens),
            "requested": cost,
        }
        if self._name is not None:
            details["provider"] = self._name
        return RateLimitError(
            f"rate limit reached{provider}; retry in {delay:.3f}s",
            retry_after=delay,
            details=details,
        )


RateLimiter = TokenBucket
"""Alias for :class:`TokenBucket`."""

ProviderRateLimiter = TokenBucket
"""Alias for :class:`TokenBucket` when a name mentions the provider scope."""
