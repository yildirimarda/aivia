"""The On this day topic mode: important dated entries without repeats.

The mode wraps an On this day source, keeps only entries whose category
importance clears a threshold, and drops topics that already appeared inside a
dedupe window. It is deliberately deterministic: importance is a configured
category weight (or an injected callable) and the window is measured against
an injectable clock, so the same request, history and time always produce the
same candidates.

History is kept in memory by default. A caller that needs the window to
survive a restart passes a :class:`TopicHistory` preloaded from its own store;
the mode only reads and writes through the small ``seen_within`` / ``record``
surface.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from aivia.core.errors import ConfigError
from aivia.core.models.base import utcnow
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.source import SourceRequest, SourceResponse
from aivia.providers.sources import WikipediaOnThisDaySourceProvider
from aivia.research.modes.base import TopicModeBase, register_mode

__all__ = [
    "DEFAULT_DEDUPE_WINDOW",
    "DEFAULT_IMPORTANCE_WEIGHTS",
    "DEFAULT_MIN_IMPORTANCE",
    "OTDMode",
    "OnThisDay",
    "OnThisDayMode",
    "OnThisDayTopicMode",
    "SeenTopic",
    "TopicHistory",
    "WikipediaOnThisDayMode",
    "candidate_importance",
    "normalize_title",
]

DEFAULT_DEDUPE_WINDOW = timedelta(days=365)
"""A full year, so an annually recurring On this day entry is not repeated."""

DEFAULT_MIN_IMPORTANCE = 0.5
"""The category importance a candidate must reach by default."""

DEFAULT_IMPORTANCE_WEIGHTS: dict[str, float] = {
    "selected": 1.0,
    "events": 0.8,
    "births": 0.6,
    "deaths": 0.6,
    "holidays": 0.4,
}
"""Category weights used when the caller does not supply its own."""

DEFAULT_IMPORTANCE = 0.5
"""Importance for an entry whose category is unknown."""


def normalize_title(title: str) -> str:
    """Return the case- and whitespace-insensitive key used for dedupe."""
    if not isinstance(title, str):
        raise ConfigError("topic title must be a string")
    text = unicodedata.normalize("NFKC", title).replace("_", " ")
    return " ".join(text.split()).casefold()


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _resolve_now(now: datetime | Callable[[], datetime] | None) -> datetime:
    if now is None:
        return utcnow()
    if callable(now):
        return _ensure_utc(now())
    return _ensure_utc(now)


def _validate_window(value: timedelta | int | float | None) -> timedelta:
    if value is None:
        return DEFAULT_DEDUPE_WINDOW
    if isinstance(value, timedelta):
        window = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        window = timedelta(days=float(value))
    else:
        raise ConfigError("dedupe window must be a timedelta or a number of days")
    if window < timedelta(0):
        raise ConfigError("dedupe window must not be negative")
    return window


def _validate_threshold(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("min importance must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError("min importance must be finite")
    if not 0.0 <= number <= 1.0:
        raise ConfigError("min importance must be between zero and one")
    return number


def _normalize_importance_weights(
    weights: Mapping[str, float] | None,
) -> dict[str, float]:
    if weights is None:
        return dict(DEFAULT_IMPORTANCE_WEIGHTS)
    normalised: dict[str, float] = {}
    for name, weight in weights.items():
        if not isinstance(name, str):
            raise ConfigError("importance category names must be strings")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ConfigError(
                f"importance weight for {name!r} must be a number",
                details={"category": name},
            )
        number = float(weight)
        if not math.isfinite(number) or number < 0.0:
            raise ConfigError(
                f"importance weight for {name!r} must be finite and non-negative",
                details={"category": name},
            )
        normalised[name.strip().lower()] = number
    return normalised


def _clamp_importance(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("candidate importance must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError("candidate importance must be finite")
    return min(1.0, max(0.0, number))


def candidate_importance(
    candidate: TopicCandidate,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Return the importance of *candidate* from its category weight.

    An explicit ``importance`` value on the candidate wins over the category,
    which lets a source or a test attach its own judgement without changing
    the mode. Unknown categories fall back to :data:`DEFAULT_IMPORTANCE`.
    """
    explicit = getattr(candidate, "importance", None)
    if explicit is not None:
        return _clamp_importance(explicit)
    table = DEFAULT_IMPORTANCE_WEIGHTS if weights is None else weights
    category = candidate.category.strip().lower()
    if category in table:
        return _clamp_importance(table[category])
    return _clamp_importance(DEFAULT_IMPORTANCE)


@dataclass(frozen=True, slots=True)
class SeenTopic:
    """One title remembered by a :class:`TopicHistory` and when it was seen."""

    title: str
    seen_at: datetime
    mode: str = ""
    key: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "seen_at", _ensure_utc(self.seen_at))
        if not self.key:
            object.__setattr__(self, "key", normalize_title(self.title))


class TopicHistory:
    """An in-memory record of titles seen inside a dedupe window.

    The store keeps the latest sighting of each normalized title. A caller may
    preload it with titles, :class:`SeenTopic` records or mappings so a mode
    run can honour history collected before the process started.
    """

    def __init__(
        self,
        entries: Iterable[SeenTopic | str | Mapping[str, Any]] | None = None,
        *,
        now: datetime | Callable[[], datetime] | None = None,
    ) -> None:
        self._seen: dict[str, SeenTopic] = {}
        if entries is not None:
            for entry in entries:
                self.add(entry, seen_at=_resolve_now(now))

    @property
    def entries(self) -> tuple[SeenTopic, ...]:
        """Every remembered sighting in insertion order."""
        return tuple(self._seen.values())

    @property
    def titles(self) -> tuple[str, ...]:
        """Every remembered title in insertion order."""
        return tuple(entry.title for entry in self._seen.values())

    def add(
        self,
        entry: SeenTopic | str | Mapping[str, Any],
        *,
        seen_at: datetime | None = None,
        mode: str = "",
    ) -> SeenTopic:
        """Remember *entry*, keeping the newest timestamp for its title."""
        if isinstance(entry, SeenTopic):
            record = entry
        elif isinstance(entry, str):
            record = SeenTopic(
                title=entry,
                seen_at=seen_at or utcnow(),
                mode=mode,
            )
        elif isinstance(entry, Mapping):
            title = entry.get("title") or entry.get("name")
            if not isinstance(title, str) or not title.strip():
                raise ConfigError("history entry needs a title")
            raw_at = entry.get("seen_at") or entry.get("at")
            timestamp = raw_at if isinstance(raw_at, datetime) else seen_at
            record = SeenTopic(
                title=title,
                seen_at=timestamp or utcnow(),
                mode=str(entry.get("mode", mode)),
            )
        else:
            raise ConfigError("history entries must be titles, records or mappings")

        existing = self._seen.get(record.key)
        if existing is None or record.seen_at >= existing.seen_at:
            self._seen[record.key] = record
        return self._seen[record.key]

    def record(
        self,
        candidates: Iterable[TopicCandidate | str],
        *,
        at: datetime | None = None,
        mode: str = "",
    ) -> list[SeenTopic]:
        """Remember every candidate title as seen at *at*."""
        timestamp = _ensure_utc(at) if at is not None else utcnow()
        records: list[SeenTopic] = []
        for candidate in candidates:
            title = (
                candidate.title if isinstance(candidate, TopicCandidate) else candidate
            )
            records.append(self.add(title, seen_at=timestamp, mode=mode))
        return records

    def seen_within(
        self,
        title: str,
        window: timedelta | int | float | None,
        *,
        now: datetime | Callable[[], datetime] | None = None,
    ) -> bool:
        """Whether *title* was seen inside *window* ending at *now*."""
        span = _validate_window(window)
        if span <= timedelta(0):
            return False
        entry = self._seen.get(normalize_title(title))
        if entry is None:
            return False
        elapsed = _resolve_now(now) - entry.seen_at
        return elapsed < span

    def is_duplicate(
        self,
        title: str,
        window: timedelta | int | float | None,
        *,
        now: datetime | Callable[[], datetime] | None = None,
    ) -> bool:
        """Alias for :meth:`seen_within`."""
        return self.seen_within(title, window, now=now)

    def prune(
        self,
        window: timedelta | int | float | None,
        *,
        now: datetime | Callable[[], datetime] | None = None,
    ) -> int:
        """Forget entries outside *window* and return how many were removed."""
        span = _validate_window(window)
        current = _resolve_now(now)
        expired = [
            key for key, entry in self._seen.items() if current - entry.seen_at >= span
        ]
        for key in expired:
            del self._seen[key]
        return len(expired)

    def clear(self) -> None:
        """Forget every remembered title."""
        self._seen.clear()

    def __contains__(self, title: object) -> bool:
        if not isinstance(title, str):
            return False
        return normalize_title(title) in self._seen

    def __len__(self) -> int:
        return len(self._seen)

    def __iter__(self) -> Iterator[SeenTopic]:
        return iter(self._seen.values())


class OnThisDayMode(TopicModeBase):
    """Discover On this day entries, filter importance and suppress repeats.

    ``source`` may be a provider instance, a provider class or a zero-argument
    factory. It is created lazily, so constructing the mode never touches the
    network. ``importance`` replaces the category-weight judgement with a
    callable, while ``min_importance=None`` disables the filter. ``history``
    accepts a :class:`TopicHistory` or any object with the same
    ``seen_within`` and ``record`` methods.
    """

    name = "on_this_day"

    def __init__(
        self,
        source: Any = None,
        *,
        provider: Any = None,
        source_factory: Callable[[], Any] | None = None,
        min_importance: float | None = DEFAULT_MIN_IMPORTANCE,
        importance: Callable[[TopicCandidate], float] | None = None,
        importance_weights: Mapping[str, float] | None = None,
        dedupe_window: timedelta | int | float | None = DEFAULT_DEDUPE_WINDOW,
        history: Any = None,
        now: datetime | Callable[[], datetime] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        if provider is not None:
            if source is not None:
                raise ConfigError("use either source or provider, not both")
            source = provider
        if source_factory is not None:
            if source is not None:
                raise ConfigError("use either source or source_factory, not both")
            source = source_factory
        self._source = source
        self._source_instance: Any = None
        self._min_importance = _validate_threshold(min_importance)
        self._importance = importance
        self._importance_weights = _normalize_importance_weights(importance_weights)
        self._dedupe_window = _validate_window(dedupe_window)
        self._history = self._coerce_history(history, now)
        self._now = now

    @staticmethod
    def _coerce_history(
        history: Any,
        now: datetime | Callable[[], datetime] | None,
    ) -> Any:
        if history is None:
            return TopicHistory(now=now)
        if isinstance(history, TopicHistory):
            return history
        if isinstance(history, Iterable) and not isinstance(
            history, (str, bytes, Mapping)
        ):
            return TopicHistory(history, now=now)
        if hasattr(history, "seen_within") and hasattr(history, "record"):
            return history
        raise ConfigError(
            "history must be a TopicHistory, an iterable of titles or a store "
            "with seen_within() and record()"
        )

    @property
    def source(self) -> Any:
        """The source instance, creating the lazy default on first use."""
        if self._source_instance is None:
            source = self._source
            if source is None:
                source = WikipediaOnThisDaySourceProvider()
            elif isinstance(source, type) or (
                callable(source) and not hasattr(source, "discover")
            ):
                source = source()
            if not hasattr(source, "discover"):
                raise ConfigError("on this day source must define discover()")
            self._source_instance = source
        return self._source_instance

    @property
    def history(self) -> Any:
        """The dedupe history this mode reads and writes."""
        return self._history

    @property
    def min_importance(self) -> float | None:
        """The importance threshold, or ``None`` when the filter is disabled."""
        return self._min_importance

    @property
    def importance_weights(self) -> dict[str, float]:
        """The category importance weights used by the default judgement."""
        return dict(self._importance_weights)

    @property
    def dedupe_window(self) -> timedelta:
        """How long a previously returned title stays suppressed."""
        return self._dedupe_window

    def importance_of(self, candidate: TopicCandidate) -> float:
        """Return the importance the mode uses to filter *candidate*."""
        if self._importance is not None:
            return _clamp_importance(self._importance(candidate))
        return candidate_importance(candidate, self._importance_weights)

    def _passes_importance(self, candidate: TopicCandidate) -> bool:
        if self._min_importance is None:
            return True
        return self.importance_of(candidate) >= self._min_importance

    def _is_repeat(self, candidate: TopicCandidate, now: datetime) -> bool:
        return bool(
            self._history.seen_within(
                candidate.title,
                self._dedupe_window,
                now=now,
            )
        )

    async def discover(self, request: SourceRequest) -> SourceResponse:
        """Discover and filter On this day entries for *request*."""
        if not isinstance(request, SourceRequest):
            raise TypeError("request must be a SourceRequest")

        response = await self.source.discover(request)
        now = _resolve_now(self._now)
        candidates: list[TopicCandidate] = []
        seen_in_batch: set[str] = set()
        skipped_importance = 0
        skipped_duplicates = 0

        for candidate in response.candidates:
            if not self._passes_importance(candidate):
                skipped_importance += 1
                continue
            key = normalize_title(candidate.title)
            if key in seen_in_batch or self._is_repeat(candidate, now):
                skipped_duplicates += 1
                continue
            seen_in_batch.add(key)
            candidates.append(
                candidate.model_copy(
                    update={"mode": self.name, "rank": len(candidates)}
                )
            )

        self._history.record(candidates, at=now, mode=self.name)

        metadata = dict(response.metadata)
        metadata.update(
            {
                "mode": self.name,
                "source_provider": response.provider,
                "min_importance": self._min_importance,
                "skipped_importance": skipped_importance,
                "skipped_duplicates": skipped_duplicates,
                "dedupe_window_seconds": self._dedupe_window.total_seconds(),
            }
        )
        return SourceResponse(
            provider=self.name,
            query=response.query,
            candidates=candidates,
            request_id=response.request_id,
            metadata=metadata,
        )


OTDMode = OnThisDayMode
"""Alias for :class:`OnThisDayMode`."""

OnThisDayTopicMode = OnThisDayMode
"""Alias for :class:`OnThisDayMode`."""

WikipediaOnThisDayMode = OnThisDayMode
"""Alias for :class:`OnThisDayMode`."""

OnThisDay = OnThisDayMode
"""Alias for :class:`OnThisDayMode`."""

register_mode("on_this_day", OnThisDayMode)
