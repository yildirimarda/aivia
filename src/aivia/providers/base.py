"""Base class shared by every provider implementation.

A provider is a thin adapter around one external service. It never decides
which provider to use, when to fall back or whether a call fits the budget:
those policies belong to the router. The base class only owns the parts every
adapter needs in the same way:

* validated :class:`~aivia.core.models.provider.ProviderMeta` describing the
  capabilities, prices, rate limits and licensing terms;
* a health report that is safe to call before and after shutdown;
* an idempotent asynchronous ``close`` so connections and clients are always
  released, including through ``async with``.

Concrete capabilities (LLM, image, video, speech, …) subclass :class:`Provider`
and add their own typed methods. The reusable contract test suite in
``tests/contracts`` checks the behaviour described here for every provider.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from types import TracebackType
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field

from aivia.core.config.http import HttpClientConfig
from aivia.core.http import HttpClient
from aivia.core.models.base import utcnow
from aivia.core.models.provider import ProviderMeta
from aivia.core.redact import redact, redact_data

__all__ = [
    "DEFAULT_HEALTH_TIMEOUT_SECONDS",
    "HealthStatus",
    "Provider",
    "ProviderHealth",
    "ProviderState",
]

DEFAULT_HEALTH_TIMEOUT_SECONDS: float = 5.0
"""Seconds a health probe may take before it is reported as unhealthy."""


class HealthStatus(StrEnum):
    """The outcome of a provider health check.

    ``DEGRADED`` means the provider can still serve requests but should not be
    preferred over a healthy candidate. ``UNKNOWN`` means the adapter has not
    checked yet; the router treats it as usable but unverified.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ProviderState(StrEnum):
    """The lifecycle state of a provider instance."""

    READY = "ready"
    CLOSED = "closed"


class ProviderHealth(BaseModel):
    """A structured, log-safe health report.

    ``latency_ms`` is filled by :meth:`Provider.health` when the adapter does
    not report one itself. ``details`` must contain only data that is safe to
    log: never credentials or raw request bodies.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: HealthStatus = HealthStatus.UNKNOWN
    message: str = ""
    checked_at: datetime = Field(default_factory=utcnow)
    latency_ms: float | None = Field(default=None, ge=0.0)
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        """Whether the provider reported the healthy status."""
        return self.status is HealthStatus.HEALTHY

    @property
    def ok(self) -> bool:
        """Alias for :attr:`healthy`."""
        return self.healthy

    @property
    def is_healthy(self) -> bool:
        """Alias for :attr:`healthy`."""
        return self.healthy

    @classmethod
    def unknown(cls, message: str = "not checked") -> ProviderHealth:
        """Return a report for a provider that has not been checked yet."""
        return cls(status=HealthStatus.UNKNOWN, message=message)

    @classmethod
    def closed(cls) -> ProviderHealth:
        """Return the report a closed provider always produces."""
        return cls(status=HealthStatus.UNHEALTHY, message="provider is closed")


def _elapsed_ms(started: float) -> float:
    return max(0.0, (time.perf_counter() - started) * 1000.0)


def _health_timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"health_timeout_seconds must be a number, got {type(value).__name__}"
        )
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(
            f"health_timeout_seconds must be a positive finite number, got {value!r}"
        )
    return timeout


class Provider:
    """Base class for every provider adapter.

    ``metadata`` may be a :class:`ProviderMeta`, a mapping validated into one,
    or ``None``. A subclass may set :attr:`default_metadata` so callers can
    construct it without repeating static metadata. ``name`` overrides the
    metadata name when given, which lets the same adapter class serve several
    configured instances.

    The lifecycle is deliberately small: :meth:`health` may be called at any
    time, :meth:`close` releases resources and is idempotent, and both are
    async because health checks and client shutdown usually touch the network.
    Every health probe is bounded by :attr:`health_timeout_seconds` so an
    unresponsive adapter cannot stall a status command.
    """

    default_metadata: ClassVar[ProviderMeta | Mapping[str, Any] | None] = None
    """Static metadata a subclass declares when callers do not pass any."""

    health_timeout_seconds: float = DEFAULT_HEALTH_TIMEOUT_SECONDS
    """Seconds a health probe may take before it is reported as unhealthy.

    Subclasses may override this class attribute, and callers may pass a
    different value to the constructor for one instance.
    """

    def __init__(
        self,
        metadata: ProviderMeta | Mapping[str, Any] | None = None,
        *,
        name: str | None = None,
        health_timeout_seconds: float | None = None,
        http_config: HttpClientConfig | None = None,
    ) -> None:
        if metadata is None:
            metadata = self.default_metadata
        self._metadata = _coerce_metadata(metadata, name=name)
        self._state = ProviderState.READY
        self._close_lock = asyncio.Lock()
        if health_timeout_seconds is None:
            health_timeout_seconds = self.health_timeout_seconds
        self.health_timeout_seconds = _health_timeout(health_timeout_seconds)
        if http_config is not None and not isinstance(http_config, HttpClientConfig):
            raise TypeError("http_config must be an HttpClientConfig")
        self._http_config = http_config or HttpClientConfig()
        self._http: HttpClient | None = None

    @property
    def http_config(self) -> HttpClientConfig:
        """The retry and timeout policy passed into this provider."""
        return self._http_config

    @property
    def http(self) -> HttpClient:
        """The shared HTTP client built from :attr:`http_config`.

        The client is created lazily and reused so a provider does not open a
        new connection pool for every call.
        """
        if self._http is None:
            self._http = HttpClient(self._http_config)
        return self._http

    @property
    def http_client(self) -> HttpClient:
        """Alias for :attr:`http`."""
        return self.http

    def create_http_client(self, **kwargs: Any) -> HttpClient:
        """Build a new HTTP client with this provider's retry and timeout policy."""
        return HttpClient(self._http_config, **kwargs)

    @property
    def metadata(self) -> ProviderMeta:
        """The validated metadata for this provider instance."""
        return self._metadata

    @property
    def meta(self) -> ProviderMeta:
        """Alias for :attr:`metadata`."""
        return self._metadata

    @property
    def name(self) -> str:
        """The provider name from its metadata."""
        return self._metadata.name

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Every capability this provider declares, in declaration order."""
        return tuple(self._metadata.capabilities)

    @property
    def state(self) -> ProviderState:
        """The current lifecycle state."""
        return self._state

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has completed."""
        return self._state is ProviderState.CLOSED

    def supports(self, capability: str) -> bool:
        """Return whether this provider declares *capability*."""
        return self._metadata.supports(capability)

    async def health(self) -> ProviderHealth:
        """Return the provider's current health.

        A closed provider is always unhealthy. Errors raised by
        :meth:`check_health` are caught, redacted and reported as an unhealthy
        result so a health probe never crashes the caller. A probe that runs
        longer than :attr:`health_timeout_seconds` is cancelled and reported as
        unhealthy instead of blocking the caller.
        """
        if self.closed:
            return ProviderHealth.closed()

        started = time.perf_counter()
        try:
            report = await asyncio.wait_for(
                self.check_health(),
                timeout=self.health_timeout_seconds,
            )
        except TimeoutError:
            return ProviderHealth(
                status=HealthStatus.UNHEALTHY,
                message=(
                    f"health check timed out after {self.health_timeout_seconds:g}s"
                ),
                latency_ms=_elapsed_ms(started),
                details={"error": "TimeoutError"},
            )
        except Exception as exc:
            message = redact(str(exc)) or type(exc).__name__
            return ProviderHealth(
                status=HealthStatus.UNHEALTHY,
                message=message,
                latency_ms=_elapsed_ms(started),
                details={"error": type(exc).__name__},
            )

        if not isinstance(report, ProviderHealth):
            raise TypeError(
                f"check_health must return ProviderHealth, got {type(report).__name__}"
            )
        updates: dict[str, Any] = {}
        if report.latency_ms is None:
            updates["latency_ms"] = _elapsed_ms(started)
        safe_message = redact(report.message)
        if safe_message != report.message:
            updates["message"] = safe_message
        safe_details = redact_data(dict(report.details))
        if safe_details != report.details:
            updates["details"] = safe_details
        if updates:
            report = report.model_copy(update=updates)
        return report

    async def check_health(self) -> ProviderHealth:
        """Adapter hook returning a health report; the default is healthy.

        Subclasses that can probe their service override this method. The
        public :meth:`health` wraps it with closed-state handling, latency
        measurement and error redaction.
        """
        return ProviderHealth(status=HealthStatus.HEALTHY, message="ok")

    async def close(self) -> None:
        """Release resources held by this provider; safe to call repeatedly.

        Concurrent callers share one shutdown: the close hook runs once and
        every caller returns after it has completed. If the hook raises, the
        provider stays open so a later call can retry.
        """
        if self.closed:
            return
        async with self._close_lock:
            if self.closed:
                return
            await self._close()
            http = self._http
            self._http = None
            if http is not None:
                await http.aclose()
            self._state = ProviderState.CLOSED

    async def _close(self) -> None:
        """Adapter hook for releasing clients; the default is a no-op."""
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    def __repr__(self) -> str:
        capabilities = ", ".join(self.capabilities)
        state = self.state.value
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"capabilities=[{capabilities}], state={state!r})"
        )


def _coerce_metadata(
    metadata: ProviderMeta | Mapping[str, Any] | None,
    *,
    name: str | None,
) -> ProviderMeta:
    if metadata is None:
        values: dict[str, Any] = {}
    elif isinstance(metadata, ProviderMeta):
        if name is None:
            return metadata
        values = metadata.model_dump()
    elif isinstance(metadata, Mapping):
        values = dict(metadata)
    else:
        raise TypeError(
            "metadata must be a ProviderMeta, a mapping or None, "
            f"got {type(metadata).__name__}"
        )

    if name is not None:
        values["name"] = name
    return ProviderMeta.model_validate(values)
