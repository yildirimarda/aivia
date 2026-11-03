"""The collection stage: run selected topic sources concurrently.

Collection is the first research stage after source selection. It asks every
selected source for candidates at the same time, bounds each call with its own
timeout and keeps the answers in source order so a run is deterministic even
when a fast source finishes first.

A failed or timed out source never discards the work of the others. Instead it
is recorded in a :class:`FailureReport`, which the caller can inspect, log or
show to the user. Passing ``continue_on_error=False`` turns the first failure
back into an exception for callers that prefer fail-fast behaviour.

The stage does not deduplicate, score or select: those policies belong to later
research stages. It only collects the raw candidate lists and the failure
report.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import ConfigError
from aivia.core.errors import TimeoutError as AiviaTimeoutError
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.source import SourceRequest, SourceResponse
from aivia.providers.sources import SourceRegistry, source_registry
from aivia.research.sources import ResolvedSource, SourceSelection

__all__ = [
    "DEFAULT_COLLECTION_CONCURRENCY",
    "DEFAULT_COLLECT_CONCURRENCY",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_SOURCE_TIMEOUT",
    "MAX_COLLECTION_CONCURRENCY",
    "MAX_COLLECT_CONCURRENCY",
    "MAX_CONCURRENCY",
    "Collect",
    "CollectedSource",
    "Collection",
    "CollectionResult",
    "CollectionStage",
    "FailureReport",
    "SourceFailure",
    "collect",
    "collect_sources",
]

DEFAULT_SOURCE_TIMEOUT: float = float(
    HttpClientConfig.model_fields["timeout_seconds"].default
)
"""Seconds a single source may take before it is reported as timed out.

The value follows the configured HTTP attempt deadline so the collection stage
does not duplicate a tuning literal that belongs in configuration.
"""

DEFAULT_CONCURRENCY = 8
"""The number of source calls allowed to run at the same time."""

DEFAULT_COLLECT_CONCURRENCY = DEFAULT_CONCURRENCY
"""Alias for :data:`DEFAULT_CONCURRENCY`."""

DEFAULT_COLLECTION_CONCURRENCY = DEFAULT_CONCURRENCY
"""Alias for :data:`DEFAULT_CONCURRENCY`."""

MAX_CONCURRENCY = 64
"""The largest source concurrency a caller may configure."""

MAX_COLLECT_CONCURRENCY = MAX_CONCURRENCY
"""Alias for :data:`MAX_CONCURRENCY`."""

MAX_COLLECTION_CONCURRENCY = MAX_CONCURRENCY
"""Alias for :data:`MAX_CONCURRENCY`."""


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ConfigError(f"{label} must be a positive finite number")
    return number


def _validate_timeout(value: Any, label: str = "timeout") -> float | None:
    if value is None:
        return None
    return _positive_number(value, label)


def _validate_concurrency(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("concurrency must be an integer")
    if not 1 <= value <= MAX_CONCURRENCY:
        raise ConfigError(f"concurrency must be between 1 and {MAX_CONCURRENCY}")
    return value


def _validate_timeouts(values: Mapping[str, Any] | None) -> dict[str, float]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ConfigError("source timeouts must be a mapping")
    timeouts: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key.strip():
            raise ConfigError("source timeout names must be non-empty strings")
        timeouts[key.strip()] = _positive_number(
            value,
            f"timeout for source {key!r}",
        )
    return timeouts


def _merge_timeouts(
    timeouts: Mapping[str, Any] | None,
    source_timeouts: Mapping[str, Any] | None,
) -> dict[str, float]:
    merged = _validate_timeouts(timeouts)
    for key, value in _validate_timeouts(source_timeouts).items():
        if key in merged and merged[key] != value:
            raise ConfigError(f"conflicting timeout for source {key!r}")
        merged[key] = value
    return merged


@dataclass(frozen=True, slots=True)
class SourceFailure:
    """One source that failed or exceeded its timeout during collection."""

    source: str
    error: str
    error_type: str
    timeout: bool = False
    duration_seconds: float | None = None

    @property
    def message(self) -> str:
        """Alias for :attr:`error`."""
        return self.error

    @property
    def type(self) -> str:
        """Alias for :attr:`error_type`."""
        return self.error_type

    @property
    def is_timeout(self) -> bool:
        """Whether this failure was caused by a deadline."""
        return self.timeout

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable failure record."""
        payload: dict[str, Any] = {
            "source": self.source,
            "error": self.error,
            "type": self.error_type,
            "timeout": self.timeout,
        }
        if self.duration_seconds is not None:
            payload["duration_seconds"] = round(self.duration_seconds, 6)
        return payload


@dataclass(frozen=True, slots=True)
class FailureReport:
    """The failures from one collection run and the counts around them."""

    failures: tuple[SourceFailure, ...] = ()
    attempted: int = 0
    succeeded: int = 0

    @property
    def count(self) -> int:
        """The number of failed sources."""
        return len(self.failures)

    @property
    def failed_sources(self) -> list[str]:
        """The names of the failed sources in collection order."""
        return [failure.source for failure in self.failures]

    @property
    def complete(self) -> bool:
        """Whether every source succeeded."""
        return not self.failures

    @property
    def ok(self) -> bool:
        """Alias for :attr:`complete`."""
        return self.complete

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable failure report."""
        return {
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "failed": len(self.failures),
            "failures": [failure.to_dict() for failure in self.failures],
        }

    def __bool__(self) -> bool:
        return bool(self.failures)

    def __iter__(self) -> Iterator[SourceFailure]:
        return iter(self.failures)

    def __len__(self) -> int:
        return len(self.failures)


@dataclass(frozen=True, slots=True)
class CollectedSource:
    """One source's successful response and how long it took."""

    source: str
    response: SourceResponse
    duration_seconds: float | None = None

    @property
    def candidates(self) -> list[TopicCandidate]:
        """The candidates this source returned."""
        return list(self.response.candidates)

    @property
    def provider(self) -> str:
        """The provider name reported by the source."""
        return self.response.provider

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable success record."""
        payload: dict[str, Any] = {
            "source": self.source,
            "provider": self.response.provider,
            "candidates": len(self.response.candidates),
        }
        if self.duration_seconds is not None:
            payload["duration_seconds"] = round(self.duration_seconds, 6)
        return payload


@dataclass(frozen=True, slots=True)
class CollectionResult:
    """Everything one collection run produced: candidates and failures."""

    request: SourceRequest
    results: tuple[CollectedSource, ...] = ()
    failures: tuple[SourceFailure, ...] = ()
    attempted: int = 0
    concurrency: int = DEFAULT_CONCURRENCY
    timeout: float | None = DEFAULT_SOURCE_TIMEOUT
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def responses(self) -> list[SourceResponse]:
        """The successful responses in source order."""
        return [result.response for result in self.results]

    @property
    def candidates(self) -> list[TopicCandidate]:
        """Every candidate in source order, without deduplication."""
        return [
            candidate
            for result in self.results
            for candidate in result.response.candidates
        ]

    @property
    def failure_report(self) -> FailureReport:
        """The failures and success counts for this run."""
        return FailureReport(
            failures=self.failures,
            attempted=self.attempted,
            succeeded=len(self.results),
        )

    @property
    def errors(self) -> list[SourceFailure]:
        """Alias for :attr:`failures` as a list."""
        return list(self.failures)

    @property
    def succeeded_sources(self) -> list[str]:
        """The names of the sources that returned a response."""
        return [result.source for result in self.results]

    @property
    def failed_sources(self) -> list[str]:
        """The names of the sources that failed."""
        return [failure.source for failure in self.failures]

    @property
    def complete(self) -> bool:
        """Whether every attempted source succeeded."""
        return not self.failures

    @property
    def ok(self) -> bool:
        """Alias for :attr:`complete`."""
        return self.complete

    @property
    def count(self) -> int:
        """The number of collected candidates."""
        return len(self.candidates)

    def to_response(self, *, provider: str = "collection") -> SourceResponse:
        """Merge the successful responses into one :class:`SourceResponse`."""
        request_id: str | None = None
        metadata: dict[str, Any] = {}
        for response in self.responses:
            if request_id is None:
                request_id = response.request_id
            for key, value in response.metadata.items():
                metadata.setdefault(key, value)
        metadata.update(self.metadata)
        return SourceResponse(
            provider=provider,
            query=self.request.query,
            candidates=self.candidates,
            request_id=request_id,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of this run."""
        return {
            "request": {
                "query": self.request.query,
                "language": self.request.language,
                "limit": self.request.limit,
                "mode": self.request.mode,
            },
            "candidates": [
                candidate.model_dump(mode="json") for candidate in self.candidates
            ],
            "failure_report": self.failure_report.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class _SourceEntry:
    name: str
    provider: Any


def _provider_name(provider: Any, fallback: str) -> str:
    for attribute in ("name", "mock_name"):
        value = getattr(provider, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _instantiate(provider: Any, name: str) -> Any:
    if isinstance(provider, type):
        try:
            provider = provider()
        except Exception as error:
            raise ConfigError(
                f"cannot create source {name!r}: {error}",
                details={"source": name},
            ) from error
    if not callable(getattr(provider, "discover", None)):
        raise ConfigError(
            f"source {name!r} must define discover()",
            details={"source": name},
        )
    return provider


def _entry_for(item: Any, registry: SourceRegistry) -> _SourceEntry:
    if isinstance(item, ResolvedSource):
        return _SourceEntry(item.name, _instantiate(item.provider, item.name))
    if isinstance(item, str):
        name = item.strip()
        if not name:
            raise ConfigError("source name must not be empty")
        provider = registry.resolve_source(name)
        return _SourceEntry(name, _instantiate(provider, name))
    if isinstance(item, type):
        name = _provider_name(item, item.__name__)
        return _SourceEntry(name, _instantiate(item, name))
    if callable(getattr(item, "discover", None)):
        name = _provider_name(item, type(item).__name__)
        return _SourceEntry(name, _instantiate(item, name))
    raise ConfigError(f"invalid source: {item!r}")


def _source_items(sources: Any) -> list[Any]:
    if sources is None:
        return []
    if isinstance(sources, (ResolvedSource, str, type)):
        return [sources]
    if isinstance(sources, SourceSelection):
        return list(sources)
    if isinstance(sources, Sequence) and not isinstance(sources, (bytes, bytearray)):
        return list(sources)
    if isinstance(sources, Iterable) and not isinstance(sources, (bytes, bytearray)):
        return list(sources)
    return [sources]


def _entries(
    sources: Any,
    registry: SourceRegistry,
) -> list[_SourceEntry]:
    if isinstance(sources, Mapping):
        entries: list[_SourceEntry] = []
        for name, provider in sources.items():
            if not isinstance(name, str) or not name.strip():
                raise ConfigError("source names must be non-empty strings")
            cleaned = name.strip()
            if provider is None:
                provider = registry.resolve_source(cleaned)
            elif isinstance(provider, ResolvedSource):
                provider = provider.provider
            entries.append(_SourceEntry(cleaned, _instantiate(provider, cleaned)))
        return entries
    return [_entry_for(item, registry) for item in _source_items(sources)]


def _failure(
    source: str,
    error: BaseException,
    *,
    started: float,
    timeout: bool,
) -> SourceFailure:
    message = str(error).strip()
    if not message and timeout:
        message = f"source {source!r} timed out"
    return SourceFailure(
        source=source,
        error=message or type(error).__name__,
        error_type=type(error).__name__,
        timeout=timeout,
        duration_seconds=time.perf_counter() - started,
    )


async def _collect_one(
    entry: _SourceEntry,
    request: SourceRequest,
    timeout: float | None,
    semaphore: asyncio.Semaphore,
    capture: bool,
) -> CollectedSource | SourceFailure:
    async with semaphore:
        started = time.perf_counter()
        try:
            if timeout is None:
                response = await entry.provider.discover(request)
            else:
                async with asyncio.timeout(timeout):
                    response = await entry.provider.discover(request)
            if not isinstance(response, SourceResponse):
                raise ConfigError(
                    f"source {entry.name!r} must return a SourceResponse",
                    details={"source": entry.name},
                )
            return CollectedSource(
                source=entry.name,
                response=response,
                duration_seconds=time.perf_counter() - started,
            )
        except TimeoutError as error:
            if not capture:
                raise
            return _failure(entry.name, error, started=started, timeout=True)
        except Exception as error:
            if not capture:
                raise
            return _failure(
                entry.name,
                error,
                started=started,
                timeout=isinstance(error, AiviaTimeoutError),
            )


def _metadata(
    entries: Sequence[_SourceEntry],
    results: Sequence[CollectedSource],
    failures: Sequence[SourceFailure],
    *,
    concurrency: int,
    timeout: float | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "sources": [entry.name for entry in entries],
        "succeeded_sources": [result.source for result in results],
        "failed_sources": [failure.source for failure in failures],
        "attempted": len(entries),
        "succeeded": len(results),
        "failed": len(failures),
        "candidate_count": sum(len(result.response.candidates) for result in results),
        "concurrency": concurrency,
        "timeout_seconds": timeout,
    }
    if failures:
        metadata["errors"] = [failure.to_dict() for failure in failures]
    return metadata


async def collect_sources(
    sources: Any,
    request: SourceRequest | Mapping[str, Any] | None = None,
    *,
    timeout: float | None = DEFAULT_SOURCE_TIMEOUT,
    timeouts: Mapping[str, float] | None = None,
    source_timeouts: Mapping[str, float] | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    continue_on_error: bool = True,
    registry: SourceRegistry | None = None,
) -> CollectionResult:
    """Collect candidates from *sources* concurrently.

    ``sources`` may be a :class:`~aivia.research.sources.SourceSelection`, a
    sequence of resolved sources, provider classes, provider instances or
    registered source names, or a mapping of name to provider. ``timeouts``
    maps a source name to its own deadline; ``timeout`` is the deadline for
    every source without an entry. A ``None`` timeout disables the outer
    deadline.

    When ``continue_on_error`` is true (the default), a failing or slow source
    is recorded in the result's failure report and the other sources still
    contribute candidates. When it is false, the first failure is raised.
    """
    if request is None:
        active_request = SourceRequest()
    elif isinstance(request, SourceRequest):
        active_request = request
    elif isinstance(request, Mapping):
        active_request = SourceRequest.model_validate(dict(request))
    else:
        raise TypeError("request must be a SourceRequest or a mapping")

    active_concurrency = _validate_concurrency(concurrency)
    default_timeout = _validate_timeout(timeout)
    per_source = _merge_timeouts(timeouts, source_timeouts)
    if not isinstance(continue_on_error, bool):
        raise ConfigError("continue_on_error must be a boolean")

    active_registry = registry if registry is not None else source_registry
    entries = _entries(sources, active_registry)
    names = {entry.name for entry in entries}
    unknown = sorted(set(per_source) - names)
    if unknown:
        raise ConfigError(
            f"timeout configured for unknown source: {unknown[0]}",
            details={"sources": sorted(names), "unknown": unknown},
        )

    if not entries:
        results: list[CollectedSource] = []
        failures: list[SourceFailure] = []
        return CollectionResult(
            request=active_request,
            attempted=0,
            concurrency=active_concurrency,
            timeout=default_timeout,
            metadata=_metadata(
                entries,
                results,
                failures,
                concurrency=active_concurrency,
                timeout=default_timeout,
            ),
        )

    semaphore = asyncio.Semaphore(active_concurrency)
    tasks = [
        _collect_one(
            entry,
            active_request,
            per_source.get(entry.name, default_timeout),
            semaphore,
            continue_on_error,
        )
        for entry in entries
    ]
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    failures = []
    for entry, outcome in zip(entries, raw, strict=True):
        if isinstance(outcome, BaseException):
            if isinstance(
                outcome,
                (
                    asyncio.CancelledError,
                    GeneratorExit,
                    KeyboardInterrupt,
                    SystemExit,
                ),
            ):
                raise outcome
            if not continue_on_error:
                raise outcome
            failures.append(
                _failure(
                    entry.name,
                    outcome,
                    started=time.perf_counter(),
                    timeout=isinstance(
                        outcome,
                        (TimeoutError, AiviaTimeoutError),
                    ),
                )
            )
            continue
        if isinstance(outcome, SourceFailure):
            failures.append(outcome)
            continue
        results.append(outcome)

    return CollectionResult(
        request=active_request,
        results=tuple(results),
        failures=tuple(failures),
        attempted=len(entries),
        concurrency=active_concurrency,
        timeout=default_timeout,
        metadata=_metadata(
            entries,
            results,
            failures,
            concurrency=active_concurrency,
            timeout=default_timeout,
        ),
    )


collect = collect_sources
"""Alias for :func:`collect_sources`."""


class CollectionStage:
    """A reusable collection stage bound to one timeout and concurrency."""

    def __init__(
        self,
        *,
        timeout: float | None = DEFAULT_SOURCE_TIMEOUT,
        timeouts: Mapping[str, float] | None = None,
        source_timeouts: Mapping[str, float] | None = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        continue_on_error: bool = True,
        registry: SourceRegistry | None = None,
    ) -> None:
        self._timeout = _validate_timeout(timeout)
        self._timeouts = _merge_timeouts(timeouts, source_timeouts)
        self._concurrency = _validate_concurrency(concurrency)
        if not isinstance(continue_on_error, bool):
            raise ConfigError("continue_on_error must be a boolean")
        self._continue_on_error = continue_on_error
        self._registry = registry

    @property
    def timeout(self) -> float | None:
        """The default deadline applied to a source without its own."""
        return self._timeout

    @property
    def timeouts(self) -> dict[str, float]:
        """The per-source deadlines configured on this stage."""
        return dict(self._timeouts)

    @property
    def concurrency(self) -> int:
        """The number of source calls this stage runs at once."""
        return self._concurrency

    @property
    def continue_on_error(self) -> bool:
        """Whether failures are reported instead of raised."""
        return self._continue_on_error

    async def collect(
        self,
        sources: Any,
        request: SourceRequest | Mapping[str, Any] | None = None,
    ) -> CollectionResult:
        """Collect from *sources* using this stage's configuration."""
        return await collect_sources(
            sources,
            request,
            timeout=self._timeout,
            timeouts=self._timeouts,
            concurrency=self._concurrency,
            continue_on_error=self._continue_on_error,
            registry=self._registry,
        )

    run = collect

    __call__ = collect


Collect = CollectionStage
"""Alias for :class:`CollectionStage`."""

Collection = CollectionStage
"""Alias for :class:`CollectionStage`."""
