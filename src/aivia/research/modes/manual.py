"""The manual topic mode: serve an explicit, ordered queue of topics.

A manual profile is driven by the editor rather than by a discovery source.
Topics are queued in the order they should be produced, and each run serves
the next items from the front of the queue. Nothing is scored, filtered by
relevance or fetched from the network.

The queue lives in memory by default. It may be preloaded with titles,
candidate records or mappings, and callers can append, remove and clear items
between runs. Duplicate titles are reported once unless deduplication is
disabled, using the same title normalisation as the other modes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

from aivia.core.errors import ConfigError
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.source import SourceRequest, SourceResponse
from aivia.research.modes.base import TopicModeBase, register_mode
from aivia.research.modes.otd import normalize_title

__all__ = [
    "DEFAULT_DEDUPE",
    "DEFAULT_REQUEUE",
    "Manual",
    "ManualMode",
    "ManualModeProvider",
    "ManualQueue",
    "ManualSourceMode",
    "ManualTopicMode",
    "ManualTopicQueue",
    "Queue",
    "QueueMode",
    "TopicQueue",
]

DEFAULT_DEDUPE = True
"""Whether a title already in the queue is skipped when it is added again."""

DEFAULT_REQUEUE = False
"""Whether served topics are put back at the end of the queue."""

_PROFILE_KEYS = (
    "manual_topics",
    "manual_queue",
    "manual_candidates",
    "topic_queue",
    "queued_topics",
    "queue",
)


def _coerce_candidates(value: Any) -> list[TopicCandidate]:
    if value is None:
        return []
    if isinstance(value, TopicCandidate):
        return [value.model_copy()]
    if isinstance(value, str):
        return [TopicCandidate(title=value)]
    if isinstance(value, Mapping):
        return [TopicCandidate.model_validate(value)]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        candidates: list[TopicCandidate] = []
        for item in value:
            if isinstance(item, TopicCandidate):
                candidates.append(item.model_copy())
            elif isinstance(item, str):
                candidates.append(TopicCandidate(title=item))
            elif isinstance(item, Mapping):
                candidates.append(TopicCandidate.model_validate(item))
            else:
                raise ConfigError(
                    "manual topics must be titles, candidates or mappings"
                )
        return candidates
    raise ConfigError("manual topics must be a candidate or an iterable of candidates")


def _profile_topics(profile: Any) -> Any:
    if profile is None:
        return None
    for key in _PROFILE_KEYS:
        if isinstance(profile, Mapping):
            value = profile.get(key)
        else:
            value = getattr(profile, key, None)
        if value is not None:
            return value
    if isinstance(profile, Mapping):
        value = profile.get("topics")
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return value
    return None


class ManualMode(TopicModeBase):
    """Serve an explicit queue of topics in insertion order.

    ``topics`` accepts a single title, candidate or mapping, or an iterable of
    them. ``queue``, ``candidates``, ``items`` and ``titles`` are aliases.
    When no topics are supplied the profile's ``manual_topics`` field is used.

    ``dedupe`` skips a title already present in the queue. ``requeue`` puts
    served topics back at the end of the queue so a later run can serve them
    again.
    """

    name = "manual"

    def __init__(
        self,
        topics: Any = None,
        *,
        queue: Any = None,
        candidates: Any = None,
        items: Any = None,
        titles: Any = None,
        manual_topics: Any = None,
        profile: Any = None,
        dedupe: bool = DEFAULT_DEDUPE,
        requeue: bool = DEFAULT_REQUEUE,
        name: str | None = None,
        **options: Any,
    ) -> None:
        supplied = [
            ("topics", topics),
            ("queue", queue),
            ("candidates", candidates),
            ("items", items),
            ("titles", titles),
            ("manual_topics", manual_topics),
        ]
        provided = [(label, value) for label, value in supplied if value is not None]
        if len(provided) > 1:
            labels = ", ".join(label for label, _ in provided)
            raise ConfigError(
                f"use only one of topics, queue, candidates, items, titles "
                f"or manual_topics (got {labels})"
            )
        initial = provided[0][1] if provided else _profile_topics(profile)

        if not isinstance(dedupe, bool):
            raise ConfigError("dedupe must be a boolean")
        if not isinstance(requeue, bool):
            raise ConfigError("requeue must be a boolean")

        super().__init__(name)
        self._dedupe = dedupe
        self._requeue = requeue
        self._queue: list[TopicCandidate] = []
        self._skipped_duplicates = 0
        self._options = dict(options)
        self.add(initial)

    @property
    def queue(self) -> tuple[TopicCandidate, ...]:
        """The queued candidates in serving order."""
        return tuple(self._queue)

    @property
    def topics(self) -> tuple[TopicCandidate, ...]:
        """Alias for :attr:`queue`."""
        return self.queue

    @property
    def candidates(self) -> tuple[TopicCandidate, ...]:
        """Alias for :attr:`queue`."""
        return self.queue

    @property
    def titles(self) -> tuple[str, ...]:
        """The queued titles in serving order."""
        return tuple(candidate.title for candidate in self._queue)

    @property
    def pending(self) -> int:
        """The number of queued candidates."""
        return len(self._queue)

    @property
    def count(self) -> int:
        """Alias for :attr:`pending`."""
        return len(self._queue)

    @property
    def size(self) -> int:
        """Alias for :attr:`pending`."""
        return len(self._queue)

    @property
    def empty(self) -> bool:
        """Whether the queue has no candidates left."""
        return not self._queue

    @property
    def is_empty(self) -> bool:
        """Alias for :attr:`empty`."""
        return self.empty

    @property
    def dedupe(self) -> bool:
        """Whether duplicate titles are skipped when added."""
        return self._dedupe

    @property
    def requeue(self) -> bool:
        """Whether served topics return to the end of the queue."""
        return self._requeue

    @property
    def options(self) -> dict[str, Any]:
        """Additional options accepted for forward compatibility."""
        return dict(self._options)

    @property
    def skipped_duplicates(self) -> int:
        """How many duplicate titles have been rejected while adding."""
        return self._skipped_duplicates

    def add(self, topics: Any) -> list[TopicCandidate]:
        """Append *topics* to the queue and return the candidates added."""
        existing = {normalize_title(candidate.title) for candidate in self._queue}
        added: list[TopicCandidate] = []
        for candidate in _coerce_candidates(topics):
            key = normalize_title(candidate.title)
            if self._dedupe and key in existing:
                self._skipped_duplicates += 1
                continue
            queued = candidate.model_copy(update={"mode": self.name, "rank": None})
            self._queue.append(queued)
            added.append(queued)
            existing.add(key)
        return added

    def enqueue(self, topics: Any) -> list[TopicCandidate]:
        """Alias for :meth:`add`."""
        return self.add(topics)

    def extend(self, topics: Any) -> list[TopicCandidate]:
        """Alias for :meth:`add`."""
        return self.add(topics)

    def push(self, topics: Any) -> list[TopicCandidate]:
        """Alias for :meth:`add`."""
        return self.add(topics)

    def put(self, topics: Any) -> list[TopicCandidate]:
        """Alias for :meth:`add`."""
        return self.add(topics)

    def append(self, topic: Any) -> TopicCandidate:
        """Append one topic and return it, rejecting a duplicate."""
        added = self.add(topic)
        if not added:
            raise ConfigError("topic is already queued")
        return added[0]

    def peek(self) -> TopicCandidate | None:
        """Return the next candidate without removing it."""
        return self._queue[0] if self._queue else None

    def dequeue(self) -> TopicCandidate | None:
        """Remove and return the next candidate, or ``None`` when empty."""
        if not self._queue:
            return None
        return self._queue.pop(0)

    def pop(self) -> TopicCandidate | None:
        """Alias for :meth:`dequeue`."""
        return self.dequeue()

    def discard(self, topic: Any) -> bool:
        """Remove *topic* from the queue, returning whether it was found."""
        key = self._key_for(topic)
        for index, candidate in enumerate(self._queue):
            if normalize_title(candidate.title) == key:
                del self._queue[index]
                return True
        return False

    def remove(self, topic: Any) -> TopicCandidate:
        """Remove and return *topic*, raising when it is not queued."""
        key = self._key_for(topic)
        for index, candidate in enumerate(self._queue):
            if normalize_title(candidate.title) == key:
                return self._queue.pop(index)
        raise ConfigError(f"topic is not queued: {topic!r}")

    def clear(self) -> None:
        """Remove every queued topic."""
        self._queue.clear()

    def _key_for(self, topic: Any) -> str:
        if isinstance(topic, TopicCandidate):
            return normalize_title(topic.title)
        if isinstance(topic, str):
            return normalize_title(topic)
        if isinstance(topic, Mapping):
            title = topic.get("title", topic.get("name", topic.get("topic")))
            if isinstance(title, str):
                return normalize_title(title)
        raise ConfigError("topic must be a title, candidate or mapping")

    async def discover(self, request: SourceRequest) -> SourceResponse:
        """Serve the next queued topics in insertion order."""
        if not isinstance(request, SourceRequest):
            raise TypeError("request must be a SourceRequest")

        before = len(self._queue)
        served: list[TopicCandidate] = []
        while self._queue and len(served) < request.limit:
            candidate = self._queue.pop(0)
            served.append(
                candidate.model_copy(update={"mode": self.name, "rank": len(served)})
            )

        if self._requeue and served:
            for candidate in served:
                self._queue.append(
                    candidate.model_copy(update={"mode": self.name, "rank": None})
                )

        metadata: dict[str, Any] = {
            "mode": self.name,
            "source_provider": self.name,
            "queue_before": before,
            "queue_after": len(self._queue),
            "queue_size": len(self._queue),
            "served_count": len(served),
            "requested_limit": request.limit,
            "requeued": self._requeue,
            "dedupe": self._dedupe,
            "skipped_duplicates": self._skipped_duplicates,
        }

        return SourceResponse(
            provider=self.name,
            query=request.query,
            candidates=served,
            metadata=metadata,
        )

    def __len__(self) -> int:
        return len(self._queue)

    def __iter__(self) -> Iterator[TopicCandidate]:
        return iter(self._queue)

    def __bool__(self) -> bool:
        return bool(self._queue)

    def __contains__(self, topic: object) -> bool:
        try:
            key = self._key_for(topic)
        except ConfigError:
            return False
        return any(normalize_title(candidate.title) == key for candidate in self._queue)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, pending={len(self._queue)})"


Manual = ManualMode
"""Alias for :class:`ManualMode`."""

ManualTopicMode = ManualMode
"""Alias for :class:`ManualMode`."""

ManualSourceMode = ManualMode
"""Alias for :class:`ManualMode`."""

ManualModeProvider = ManualMode
"""Alias for :class:`ManualMode`."""

ManualQueue = ManualMode
"""Alias for :class:`ManualMode`."""

ManualTopicQueue = ManualMode
"""Alias for :class:`ManualMode`."""

TopicQueue = ManualMode
"""Alias for :class:`ManualMode`."""

QueueMode = ManualMode
"""Alias for :class:`ManualMode`."""

Queue = ManualMode
"""Alias for :class:`ManualMode`."""

register_mode("manual", ManualMode)
