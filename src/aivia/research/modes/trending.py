"""The trending topic mode: merge sources and order candidates by signal.

A trending profile is not steered by a niche brief. It asks every configured
source for what is popular right now and merges the answers into one ranked
list. Each source already attaches its own strength to a candidate, such as
pageviews, points or search traffic; this mode orders candidates by that
``signal`` so the strongest topic comes first.

A candidate without a finite signal is kept but sorted after every candidate
that has one, because a missing signal is not evidence of popularity. A
``min_signal`` threshold may drop weak candidates, and ``require_signal`` may
drop candidates that carry no signal at all. Duplicate titles are reported
once, keeping the copy with the strongest signal.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from aivia.core.errors import ConfigError
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.source import SourceRequest, SourceResponse
from aivia.providers.sources import WikipediaPageviewsSourceProvider
from aivia.research.modes.base import register_mode
from aivia.research.modes.niche import NicheMode
from aivia.research.modes.otd import normalize_title

__all__ = [
    "DEFAULT_DESCENDING",
    "DEFAULT_MIN_SIGNAL",
    "DEFAULT_ORDER",
    "DEFAULT_REQUIRE_SIGNAL",
    "DEFAULT_SIGNAL_ORDER",
    "DEFAULT_SORT_BY",
    "DEFAULT_TRENDING_SOURCE",
    "Trending",
    "TrendingMode",
    "TrendingModeProvider",
    "TrendingSourceMode",
    "TrendingTopicMode",
    "Trends",
    "TrendsMode",
    "WikipediaTrendingMode",
    "order_by_signal",
    "rank_by_signal",
    "signal_order",
    "sort_by_signal",
]

DEFAULT_SORT_BY = "signal"
"""The default ordering: strongest source signal first."""

DEFAULT_ORDER = DEFAULT_SORT_BY
"""Alias for :data:`DEFAULT_SORT_BY`."""

DEFAULT_DESCENDING = True
"""Whether a signal sort puts the strongest candidate first."""

DEFAULT_SIGNAL_ORDER = "descending"
"""The default direction as a word rather than a boolean."""

DEFAULT_MIN_SIGNAL: float | None = None
"""No minimum signal is applied unless the caller sets one."""

DEFAULT_REQUIRE_SIGNAL = False
"""Whether candidates without a signal are dropped by default."""

DEFAULT_TRENDING_SOURCE = WikipediaPageviewsSourceProvider
"""The source used when a trending mode is created without one."""

_SOURCE_ORDERS = frozenset(
    {"source", "sources", "discovery", "found", "none", "unsorted", "insertion"}
)
_SIGNAL_ORDERS = frozenset({"signal", "signals", "trending", "trend", "popularity"})


def _validate_signal(
    value: Any,
    *,
    label: str,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{label} must be finite")
    return number


def _validate_order(value: Any) -> str:
    if value is None:
        return DEFAULT_SORT_BY
    if not isinstance(value, str):
        raise ConfigError("trending order must be a string")
    text = value.strip().lower().replace("_", "-")
    if not text:
        return DEFAULT_SORT_BY
    if text in _SOURCE_ORDERS:
        return "source"
    if text in _SIGNAL_ORDERS:
        return "signal"
    raise ConfigError(f"unknown trending order: {value!r} (allowed: signal, source)")


def _signal_key(candidate: TopicCandidate) -> float | None:
    signal = candidate.signal
    if signal is None or not math.isfinite(float(signal)):
        return None
    return float(signal)


def order_by_signal(
    candidates: Iterable[TopicCandidate],
    *,
    descending: bool = DEFAULT_DESCENDING,
) -> list[TopicCandidate]:
    """Return *candidates* ordered by signal, with missing signals last.

    The sort is stable, so candidates with the same signal keep the order in
    which the sources reported them.
    """

    items = list(candidates)
    if descending:

        def key(candidate: TopicCandidate) -> tuple[bool, float]:
            signal = _signal_key(candidate)
            return (signal is None, -(signal if signal is not None else 0.0))

    else:

        def key(candidate: TopicCandidate) -> tuple[bool, float]:
            signal = _signal_key(candidate)
            return (signal is None, signal if signal is not None else 0.0)

    return sorted(items, key=key)


sort_by_signal = order_by_signal
"""Alias for :func:`order_by_signal`."""

rank_by_signal = order_by_signal
"""Alias for :func:`order_by_signal`."""

signal_order = order_by_signal
"""Alias for :func:`order_by_signal`."""


def _profile_sources(profile: Any) -> list[Any] | None:
    if profile is None:
        return None
    if isinstance(profile, Mapping):
        value = profile.get(
            "sources",
            profile.get("source", profile.get("topic_sources")),
        )
    else:
        value = getattr(profile, "sources", None)
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        value = list(value)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return None
    names: list[Any] = []
    for entry in value:
        if isinstance(entry, Mapping):
            name = entry.get("name", entry.get("source", entry.get("type")))
            if name is not None:
                names.append(name)
        else:
            names.append(entry)
    return names or None


class TrendingMode(NicheMode):
    """Discover popular topics and order them by their source signal.

    ``source`` accepts the same forms as the niche mode: a provider instance,
    a class, a factory, a registered source name or a list of any of those.
    When none is supplied the profile's sources are used, falling back to the
    pageviews source.

    ``sort_by`` may be ``signal`` (the default) or ``source`` to preserve the
    order the sources reported. ``descending`` controls the signal direction,
    ``min_signal`` drops weak candidates and ``require_signal`` drops
    candidates that carry no signal at all.
    """

    name = "trending"

    def __init__(
        self,
        source: Any = None,
        *,
        provider: Any = None,
        source_factory: Any = None,
        sources: Any = None,
        profile: Any = None,
        queries: Any = None,
        search_queries: Any = None,
        query: str | None = None,
        min_signal: float | None = None,
        max_signal: float | None = None,
        require_signal: bool = DEFAULT_REQUIRE_SIGNAL,
        sort_by: str | None = None,
        order_by: str | None = None,
        ordering: str | None = None,
        order: str | None = None,
        descending: bool | None = None,
        reverse: bool | None = None,
        ascending: bool | None = None,
        min_relevance: float | None = None,
        name: str | None = None,
        **options: Any,
    ) -> None:
        supplied_order = [
            value for value in (sort_by, order_by, ordering, order) if value is not None
        ]
        if len(supplied_order) > 1:
            raise ConfigError("use only one of sort_by, order_by, ordering or order")
        effective_order = supplied_order[0] if supplied_order else DEFAULT_SORT_BY

        directions = [
            value for value in (descending, reverse, ascending) if value is not None
        ]
        if len(directions) > 1:
            raise ConfigError("use only one of descending, reverse or ascending")
        if descending is not None:
            effective_descending = bool(descending)
        elif reverse is not None:
            effective_descending = not bool(reverse)
        elif ascending is not None:
            effective_descending = not bool(ascending)
        else:
            effective_descending = DEFAULT_DESCENDING

        if search_queries is not None:
            if queries is not None:
                raise ConfigError("use either queries or search_queries, not both")
            queries = search_queries
        if query is not None:
            if queries is not None:
                raise ConfigError("use either queries or query, not both")
            queries = [query]
        if queries is None:
            queries = []

        if (
            source is None
            and provider is None
            and source_factory is None
            and sources is None
        ):
            profile_sources = _profile_sources(profile)
            source = profile_sources or [DEFAULT_TRENDING_SOURCE]

        if not isinstance(require_signal, bool):
            raise ConfigError("require_signal must be a boolean")

        minimum = _validate_signal(min_signal, label="min signal")
        maximum = _validate_signal(max_signal, label="max signal")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ConfigError("min signal must not exceed max signal")

        self._min_signal = minimum
        self._max_signal = maximum
        self._require_signal = require_signal
        self._sort_by = _validate_order(effective_order)
        self._descending = effective_descending

        super().__init__(
            source,
            provider=provider,
            source_factory=source_factory,
            sources=sources,
            queries=queries,
            min_relevance=min_relevance,
            name=name,
            **options,
        )

    @property
    def min_signal(self) -> float | None:
        """The lowest signal a candidate may carry."""
        return self._min_signal

    @property
    def max_signal(self) -> float | None:
        """The highest signal a candidate may carry."""
        return self._max_signal

    @property
    def require_signal(self) -> bool:
        """Whether candidates without a signal are dropped."""
        return self._require_signal

    @property
    def sort_by(self) -> str:
        """The active ordering, either ``signal`` or ``source``."""
        return self._sort_by

    @property
    def order_by(self) -> str:
        """Alias for :attr:`sort_by`."""
        return self._sort_by

    @property
    def descending(self) -> bool:
        """Whether the signal sort is strongest first."""
        return self._descending

    @property
    def signal_ordered(self) -> bool:
        """Whether this mode orders candidates by signal."""
        return self._sort_by == "signal"

    def _signal_allowed(self, signal: float | None) -> bool:
        if self._require_signal and signal is None:
            return False
        if signal is None:
            return self._min_signal is None
        if self._min_signal is not None and signal < self._min_signal:
            return False
        return not (self._max_signal is not None and signal > self._max_signal)

    def _select(
        self,
        candidates: Iterable[TopicCandidate],
    ) -> tuple[list[TopicCandidate], int, int, int]:
        kept: list[TopicCandidate] = []
        positions: dict[str, int] = {}
        skipped_duplicates = 0
        skipped_signal = 0
        skipped_missing_signal = 0

        for candidate in candidates:
            signal = _signal_key(candidate)
            if signal is None and self._require_signal:
                skipped_missing_signal += 1
                continue
            if not self._signal_allowed(signal):
                skipped_signal += 1
                continue

            key = normalize_title(candidate.title)
            position = positions.get(key)
            if position is not None:
                skipped_duplicates += 1
                current = _signal_key(kept[position])
                if signal is not None and (current is None or signal > current):
                    kept[position] = candidate.model_copy(
                        update={"mode": self.name, "rank": None}
                    )
                continue

            positions[key] = len(kept)
            kept.append(candidate.model_copy(update={"mode": self.name, "rank": None}))

        if self._sort_by == "signal":
            kept = order_by_signal(kept, descending=self._descending)

        for rank, candidate in enumerate(kept):
            kept[rank] = candidate.model_copy(update={"rank": rank})

        return kept, skipped_duplicates, skipped_signal, skipped_missing_signal

    async def discover(self, request: SourceRequest) -> SourceResponse:
        """Discover popular candidates and order them by signal."""
        if not isinstance(request, SourceRequest):
            raise TypeError("request must be a SourceRequest")

        queries = self._queries_for(request)
        sources = self.sources
        semaphore = asyncio.Semaphore(self.concurrency)
        tasks = [
            self._discover_one(source, request, query, semaphore)
            for source in sources
            for query in queries
        ]
        results: list[Any]
        if getattr(self, "_continue_on_error", False):
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = list(await asyncio.gather(*tasks))

        labels = [
            (index // len(queries), queries[index % len(queries)])
            for index in range(len(tasks))
        ]
        responses: list[SourceResponse] = []
        errors: list[dict[str, Any]] = []
        for (source_index, query), result in zip(labels, results, strict=True):
            if isinstance(result, BaseException):
                errors.append(
                    {
                        "query": query,
                        "source": source_index,
                        "error": str(result),
                        "type": type(result).__name__,
                    }
                )
                continue
            responses.append(result)

        metadata: dict[str, Any] = {}
        providers: list[str] = []
        request_id: str | None = None
        found: list[TopicCandidate] = []
        for response in responses:
            for key, value in response.metadata.items():
                metadata.setdefault(key, value)
            providers.append(response.provider)
            if request_id is None:
                request_id = response.request_id
            found.extend(response.candidates)

        selected, skipped_duplicates, skipped_signal, skipped_missing = self._select(
            found
        )
        served = selected[: request.limit]
        for rank, candidate in enumerate(served):
            served[rank] = candidate.model_copy(update={"rank": rank})

        metadata.update(
            {
                "mode": self.name,
                "source_provider": self._provider_label(providers),
                "queries": list(queries),
                "expanded_queries": list(queries),
                "query_count": len(queries),
                "source_count": len(sources),
                "concurrency": self.concurrency,
                "signal_ordered": self.signal_ordered,
                "sort_by": self._sort_by,
                "order_by": self._sort_by,
                "descending": self._descending,
                "min_signal": self._min_signal,
                "max_signal": self._max_signal,
                "require_signal": self._require_signal,
                "skipped_duplicates": skipped_duplicates,
                "skipped_signal": skipped_signal,
                "skipped_low_signal": skipped_signal,
                "skipped_missing_signal": skipped_missing,
                "found_count": len(found),
                "served_count": len(served),
            }
        )
        if errors:
            metadata["errors"] = errors
            metadata["failed_queries"] = len(errors)

        return SourceResponse(
            provider=self.name,
            query=request.query or (queries[0] if queries else ""),
            candidates=served,
            request_id=request_id,
            metadata=metadata,
        )


Trending = TrendingMode
"""Alias for :class:`TrendingMode`."""

TrendingTopicMode = TrendingMode
"""Alias for :class:`TrendingMode`."""

TrendingSourceMode = TrendingMode
"""Alias for :class:`TrendingMode`."""

TrendingModeProvider = TrendingMode
"""Alias for :class:`TrendingMode`."""

TrendsMode = TrendingMode
"""Alias for :class:`TrendingMode`."""

Trends = TrendingMode
"""Alias for :class:`TrendingMode`."""

WikipediaTrendingMode = TrendingMode
"""Alias for :class:`TrendingMode`."""

register_mode("trending", TrendingMode)
