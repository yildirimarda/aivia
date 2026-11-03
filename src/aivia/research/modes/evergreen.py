"""The evergreen ideas topic mode: a maintained backlog of timeless topics.

An evergreen profile does not want a topic tied to today's news or a date. It
wants a steady supply of ideas that stay relevant, so this mode keeps a backlog
of discovered candidates and serves the next ideas from it. When the backlog
falls to a low-water mark, or when the next run needs more ideas than are
queued, the mode replenishes from its source before serving.

Replenishment reuses the niche mode's query expansion, concurrent discovery and
relevance gate. The backlog is then deduplicated against itself and against the
titles this mode has already returned inside the configured dedupe window. The
history surface is the same small ``seen_within`` / ``record`` contract used by
the On this day mode, so a caller can preload it from a persistent store.

The backlog is in memory by default. Passing a preloaded backlog, a history and
an injectable clock makes a run deterministic: the same inputs always yield the
same next ideas.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from aivia.core.errors import ConfigError
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.source import SourceRequest, SourceResponse
from aivia.research.modes.base import register_mode
from aivia.research.modes.niche import NicheMode
from aivia.research.modes.otd import (
    DEFAULT_DEDUPE_WINDOW,
    TopicHistory,
    _resolve_now,
    _validate_window,
    normalize_title,
)

__all__ = [
    "DEFAULT_BACKLOG_SIZE",
    "DEFAULT_BACKLOG_TARGET",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DEDUPE_WINDOW",
    "DEFAULT_MIN_BACKLOG",
    "DEFAULT_REPLENISH_AT",
    "DEFAULT_REPLENISH_BATCH",
    "DEFAULT_REPLENISH_THRESHOLD",
    "Evergreen",
    "EvergreenIdeas",
    "EvergreenIdeasMode",
    "EvergreenIdeasTopicMode",
    "EvergreenMode",
    "EvergreenModeProvider",
    "IdeasMode",
    "WikipediaEvergreenMode",
]

DEFAULT_BACKLOG_SIZE = 20
"""The number of ideas the mode tries to keep queued."""

DEFAULT_BACKLOG_TARGET = DEFAULT_BACKLOG_SIZE
"""Alias for :data:`DEFAULT_BACKLOG_SIZE`."""

DEFAULT_MIN_BACKLOG = 5
"""Replenish once the queued ideas fall to this low-water mark."""

DEFAULT_REPLENISH_THRESHOLD = DEFAULT_MIN_BACKLOG
"""Alias for :data:`DEFAULT_MIN_BACKLOG`."""

DEFAULT_REPLENISH_AT = DEFAULT_MIN_BACKLOG
"""Alias for :data:`DEFAULT_MIN_BACKLOG`."""

DEFAULT_REPLENISH_BATCH = 20
"""The most ideas one replenishment asks the source for."""

DEFAULT_BATCH_SIZE = DEFAULT_REPLENISH_BATCH
"""Alias for :data:`DEFAULT_REPLENISH_BATCH`."""

_MAX_SOURCE_RESULTS = 200
"""The source protocol's largest accepted per-request result count."""


def _validate_count(
    value: Any,
    *,
    label: str,
    default: int,
    minimum: int = 1,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    if value < minimum:
        raise ConfigError(f"{label} must be at least {minimum}")
    return value


def _coerce_candidates(value: Any) -> list[TopicCandidate]:
    if value is None:
        return []
    if isinstance(value, TopicCandidate):
        return [value]
    if isinstance(value, str):
        return [TopicCandidate(title=value)]
    if isinstance(value, Mapping):
        return [TopicCandidate.model_validate(value)]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        candidates: list[TopicCandidate] = []
        for item in value:
            if isinstance(item, TopicCandidate):
                candidates.append(item)
            elif isinstance(item, str):
                candidates.append(TopicCandidate(title=item))
            elif isinstance(item, Mapping):
                candidates.append(TopicCandidate.model_validate(item))
            else:
                raise ConfigError(
                    "backlog entries must be titles, candidates or mappings"
                )
        return candidates
    raise ConfigError("backlog must be a candidate or an iterable of candidates")


def _coerce_history(
    history: Any,
    now: datetime | Callable[[], datetime] | None,
) -> Any:
    if history is None:
        return TopicHistory(now=now)
    if isinstance(history, TopicHistory):
        return history
    if isinstance(history, Iterable) and not isinstance(history, (str, bytes, Mapping)):
        return TopicHistory(history, now=now)
    if hasattr(history, "seen_within") and hasattr(history, "record"):
        return history
    raise ConfigError(
        "history must be a TopicHistory, an iterable of titles or a store "
        "with seen_within() and record()"
    )


@dataclass(slots=True)
class _Replenishment:
    response: SourceResponse
    added: list[TopicCandidate]
    skipped_duplicates: int
    skipped_past: int


class EvergreenMode(NicheMode):
    """Serve timeless ideas from a maintained, deduplicated backlog.

    ``source`` and the query-expansion options are inherited from
    :class:`~aivia.research.modes.niche.NicheMode`. ``backlog`` may preload
    queued candidates as titles, :class:`TopicCandidate` records or mappings.
    ``history`` may be a :class:`TopicHistory`, an iterable of titles or any
    store with ``seen_within`` and ``record`` methods.

    A run replenishes when the backlog is at or below ``replenish_threshold``
    or when it cannot satisfy the request's ``limit``. Replenishment stops at
    ``backlog_size`` queued ideas. ``dedupe_window`` measures how long a title
    that was already served stays suppressed; a zero window disables history
    deduplication.
    """

    name = "evergreen_ideas"

    def __init__(
        self,
        source: Any = None,
        *,
        backlog: Any = None,
        history: Any = None,
        backlog_size: int | None = None,
        target_backlog: int | None = None,
        replenish_threshold: int | None = None,
        min_backlog: int | None = None,
        replenish_at: int | None = None,
        replenish_batch: int | None = None,
        batch_size: int | None = None,
        dedupe_window: timedelta | int | float | None = DEFAULT_DEDUPE_WINDOW,
        now: datetime | Callable[[], datetime] | None = None,
        name: str | None = None,
        **options: Any,
    ) -> None:
        if target_backlog is not None:
            if backlog_size is not None:
                raise ConfigError("use either backlog_size or target_backlog, not both")
            backlog_size = target_backlog
        if min_backlog is not None:
            if replenish_threshold is not None:
                raise ConfigError(
                    "use either replenish_threshold or min_backlog, not both"
                )
            replenish_threshold = min_backlog
        if replenish_at is not None:
            if replenish_threshold is not None:
                raise ConfigError(
                    "use either replenish_threshold or replenish_at, not both"
                )
            replenish_threshold = replenish_at
        if batch_size is not None:
            if replenish_batch is not None:
                raise ConfigError("use either replenish_batch or batch_size, not both")
            replenish_batch = batch_size

        target = _validate_count(
            backlog_size,
            label="backlog size",
            default=DEFAULT_BACKLOG_SIZE,
        )
        threshold = _validate_count(
            replenish_threshold,
            label="replenish threshold",
            default=DEFAULT_REPLENISH_THRESHOLD,
            minimum=0,
        )
        if threshold > target:
            raise ConfigError("replenish threshold must not exceed backlog size")
        batch = _validate_count(
            replenish_batch,
            label="replenish batch",
            default=DEFAULT_REPLENISH_BATCH,
        )

        self._now = now
        self._history = _coerce_history(history, now)
        self._dedupe_window = _validate_window(dedupe_window)
        self._target = target
        self._threshold = threshold
        self._batch = batch
        self._backlog: list[TopicCandidate] = []
        self._last_provider = ""

        super().__init__(source, name=name, **options)
        self.add(backlog)

    @property
    def backlog(self) -> tuple[TopicCandidate, ...]:
        """The queued candidates in the order they will be served."""
        return tuple(self._backlog)

    @property
    def backlog_count(self) -> int:
        """The number of queued candidates."""
        return len(self._backlog)

    @property
    def pending(self) -> int:
        """Alias for :attr:`backlog_count`."""
        return len(self._backlog)

    @property
    def queued(self) -> int:
        """Alias for :attr:`backlog_count`."""
        return len(self._backlog)

    @property
    def backlog_titles(self) -> tuple[str, ...]:
        """The queued titles in serving order."""
        return tuple(candidate.title for candidate in self._backlog)

    @property
    def backlog_size(self) -> int:
        """The number of ideas the mode tries to keep queued."""
        return self._target

    @property
    def target_backlog(self) -> int:
        """Alias for :attr:`backlog_size`."""
        return self._target

    @property
    def backlog_target(self) -> int:
        """Alias for :attr:`backlog_size`."""
        return self._target

    @property
    def replenish_threshold(self) -> int:
        """The low-water mark that triggers replenishment."""
        return self._threshold

    @property
    def min_backlog(self) -> int:
        """Alias for :attr:`replenish_threshold`."""
        return self._threshold

    @property
    def replenish_at(self) -> int:
        """Alias for :attr:`replenish_threshold`."""
        return self._threshold

    @property
    def replenish_batch(self) -> int:
        """The most ideas one replenishment asks the source for."""
        return self._batch

    @property
    def batch_size(self) -> int:
        """Alias for :attr:`replenish_batch`."""
        return self._batch

    @property
    def history(self) -> Any:
        """The dedupe history this mode reads and writes."""
        return self._history

    @property
    def dedupe_window(self) -> timedelta:
        """How long an already served title stays suppressed."""
        return self._dedupe_window

    @property
    def needs_replenishment(self) -> bool:
        """Whether the backlog is at or below its low-water mark."""
        return len(self._backlog) <= self._threshold

    def add(self, candidates: Any) -> list[TopicCandidate]:
        """Queue *candidates*, skipping duplicates and past topics."""
        now = _resolve_now(self._now)
        existing = {normalize_title(candidate.title) for candidate in self._backlog}
        added: list[TopicCandidate] = []
        for candidate in _coerce_candidates(candidates):
            key = normalize_title(candidate.title)
            if key in existing or self._is_past(candidate, now):
                continue
            if len(self._backlog) + len(added) >= self._target:
                break
            queued = candidate.model_copy(update={"mode": self.name, "rank": None})
            added.append(queued)
            existing.add(key)
        self._backlog.extend(added)
        return added

    def clear_backlog(self) -> None:
        """Forget every queued candidate without touching the history."""
        self._backlog.clear()

    def _is_past(self, candidate: TopicCandidate, now: datetime) -> bool:
        return bool(
            self._history.seen_within(
                candidate.title,
                self._dedupe_window,
                now=now,
            )
        )

    def _needs_replenishment(self, limit: int) -> bool:
        wanted = min(limit, self._target)
        return len(self._backlog) < wanted or len(self._backlog) <= self._threshold

    def _replenish_request(self, request: SourceRequest) -> SourceRequest:
        needed = max(self._target - len(self._backlog), 0)
        limit = max(self._batch, needed, 1)
        return request.model_copy(
            update={
                "limit": min(limit, _MAX_SOURCE_RESULTS),
                "mode": self.name,
            }
        )

    def _absorb(
        self,
        candidates: Iterable[TopicCandidate],
        now: datetime,
    ) -> tuple[list[TopicCandidate], int, int]:
        existing = {normalize_title(candidate.title) for candidate in self._backlog}
        added: list[TopicCandidate] = []
        skipped_duplicates = 0
        skipped_past = 0
        for candidate in candidates:
            key = normalize_title(candidate.title)
            if key in existing:
                skipped_duplicates += 1
                continue
            if self._is_past(candidate, now):
                skipped_past += 1
                continue
            if len(self._backlog) + len(added) >= self._target:
                break
            queued = candidate.model_copy(update={"mode": self.name, "rank": None})
            added.append(queued)
            existing.add(key)
        self._backlog.extend(added)
        return added, skipped_duplicates, skipped_past

    async def _replenish(self, request: SourceRequest) -> _Replenishment:
        response = await super().discover(self._replenish_request(request))
        if not isinstance(response, SourceResponse):
            raise ConfigError("evergreen source must return a SourceResponse")
        now = _resolve_now(self._now)
        added, skipped_duplicates, skipped_past = self._absorb(
            response.candidates,
            now,
        )
        provider = response.metadata.get("source_provider") or response.provider
        if provider:
            self._last_provider = str(provider)
        return _Replenishment(
            response=response,
            added=added,
            skipped_duplicates=skipped_duplicates,
            skipped_past=skipped_past,
        )

    async def replenish(
        self,
        request: SourceRequest | None = None,
    ) -> list[TopicCandidate]:
        """Force a replenishment and return the candidates added to the backlog."""
        if request is None:
            request = SourceRequest()
        if not isinstance(request, SourceRequest):
            raise TypeError("request must be a SourceRequest")
        result = await self._replenish(request)
        return list(result.added)

    refill = replenish
    """Alias for :meth:`replenish`."""

    async def discover(self, request: SourceRequest) -> SourceResponse:
        """Serve the next evergreen ideas, replenishing the backlog when needed."""
        if not isinstance(request, SourceRequest):
            raise TypeError("request must be a SourceRequest")

        now = _resolve_now(self._now)
        backlog_before = len(self._backlog)
        replenishment: _Replenishment | None = None
        if self._needs_replenishment(request.limit):
            replenishment = await self._replenish(request)

        served: list[TopicCandidate] = []
        skipped_past = 0
        while self._backlog and len(served) < request.limit:
            candidate = self._backlog.pop(0)
            if self._is_past(candidate, now):
                skipped_past += 1
                continue
            served.append(
                candidate.model_copy(update={"mode": self.name, "rank": len(served)})
            )

        self._history.record(served, at=now, mode=self.name)

        metadata: dict[str, Any] = {}
        request_id: str | None = None
        skipped_duplicates = 0
        replenished_count = 0
        if replenishment is not None:
            metadata.update(replenishment.response.metadata)
            request_id = replenishment.response.request_id
            skipped_duplicates = (
                int(metadata.get("skipped_duplicates", 0) or 0)
                + replenishment.skipped_duplicates
            )
            skipped_past += replenishment.skipped_past
            replenished_count = len(replenishment.added)

        metadata.update(
            {
                "mode": self.name,
                "source_provider": self._last_provider or self.name,
                "backlog_before": backlog_before,
                "backlog_after": len(self._backlog),
                "backlog_size": len(self._backlog),
                "backlog_target": self._target,
                "replenish_threshold": self._threshold,
                "replenished": replenishment is not None,
                "replenished_count": replenished_count,
                "skipped_duplicates": skipped_duplicates,
                "skipped_past": skipped_past,
                "dedupe_window_seconds": self._dedupe_window.total_seconds(),
                "history_size": len(self._history),
            }
        )

        return SourceResponse(
            provider=self.name,
            query=request.query,
            candidates=served,
            request_id=request_id,
            metadata=metadata,
        )


EvergreenIdeasMode = EvergreenMode
"""Alias for :class:`EvergreenMode`."""

EvergreenIdeasTopicMode = EvergreenMode
"""Alias for :class:`EvergreenMode`."""

EvergreenIdeas = EvergreenMode
"""Alias for :class:`EvergreenMode`."""

Evergreen = EvergreenMode
"""Alias for :class:`EvergreenMode`."""

IdeasMode = EvergreenMode
"""Alias for :class:`EvergreenMode`."""

EvergreenModeProvider = EvergreenMode
"""Alias for :class:`EvergreenMode`."""

WikipediaEvergreenMode = EvergreenMode
"""Alias for :class:`EvergreenMode`."""

register_mode("evergreen_ideas", EvergreenMode)
