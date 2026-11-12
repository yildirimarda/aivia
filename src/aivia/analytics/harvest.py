"""Harvest comments from read-capable accounts into manual queue topic suggestions.

Idea harvesting is opt in through the ``analytics`` config section
(``analytics.idea_harvest.enabled``). Comments are read through a read-capable
API (a :class:`~aivia.publish.account_reference.ReferenceAccountSource`); reading
requires the ``read`` scope and a missing scope fails with
:class:`~aivia.publish.account_reference.MissingScopeError` instead of silently
skipping the account.

Harvested comments are deduplicated by normalised title and clustered into topic
suggestions, then appended to a :class:`~aivia.research.modes.manual.ManualMode`
queue.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Any

from aivia.core.config.base import ConfigModel
from aivia.core.config.registry import SectionRegistry, registry
from aivia.core.errors import ConfigError
from aivia.core.models.base import canonical_json
from aivia.core.models.topic import TopicCandidate
from aivia.publish.account_reference import (
    CAPABILITY_READ,
    AccountReference,
    MissingScopeError,
    ReferenceAccount,
    ReferenceAccountSource,
    TopPost,
)
from aivia.publish.accounts import AccountMap
from aivia.research.modes.manual import ManualMode
from aivia.research.modes.otd import normalize_title

__all__ = [
    "ANALYTICS_SECTION_KEY",
    "IdeaHarvest",
    "IdeaHarvestConfig",
    "register_analytics_section",
]

ANALYTICS_SECTION_KEY = "analytics"
"""The top-level config key owned by the analytics module."""


class IdeaHarvestConfig(ConfigModel):
    """Configuration for idea harvesting: opt in and its limits."""

    enabled: bool = False
    """Whether comment harvesting into the manual queue is opted in."""

    max_suggestions: int = 10
    """The most topic suggestions a single harvest may append."""

    limit_per_account: int = 10
    """The most comments read from each account."""

    cluster_threshold: float = 0.5
    """The Jaccard similarity at or above which comments form one cluster."""


_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> frozenset[str]:
    """Return the distinct lowercased words of *text*."""
    return frozenset(_WORD.findall(text.casefold()))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Return the Jaccard similarity of two word sets in ``[0, 1]``."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _cluster(comments: Sequence[str], threshold: float) -> list[list[str]]:
    """Greedily cluster *comments* by Jaccard similarity of their words.

    Clusters are deterministic: comments are visited in order and join the first
    cluster whose representative is similar enough, otherwise they start a new
    cluster whose representative is the comment itself.
    """
    clusters: list[list[str]] = []
    representatives: list[frozenset[str]] = []
    for comment in comments:
        words = _words(comment)
        for index, cluster in enumerate(clusters):
            if _jaccard(words, representatives[index]) >= threshold:
                cluster.append(comment)
                break
        else:
            clusters.append([comment])
            representatives.append(words)
    return clusters


def _suggest_title(cluster: Sequence[str]) -> str:
    """Return the representative title of a cluster: its first comment."""
    return cluster[0].strip()


class IdeaHarvest:
    """Read comments from read-capable accounts and queue topic suggestions.

    ``source`` is the read-capable API that fetches an account's posts. ``queue``
    is the manual queue suggestions are appended to (a new empty queue is used
    when none is supplied). ``config`` drives the opt-in flag and limits.
    """

    def __init__(
        self,
        accounts: AccountMap,
        source: ReferenceAccountSource,
        *,
        config: IdeaHarvestConfig | None = None,
        queue: ManualMode | None = None,
        now: Any = None,
    ) -> None:
        self._accounts = accounts
        self._source = source
        self._config = config or IdeaHarvestConfig()
        self._queue = queue if queue is not None else ManualMode()
        self._now = now or time.monotonic
        self._cache: dict[str, tuple[float, list[TopPost]]] = {}
        self._reference = AccountReference(accounts, source)

    @property
    def config(self) -> IdeaHarvestConfig:
        """The harvest configuration."""
        return self._config

    @property
    def enabled(self) -> bool:
        """Whether harvesting is opted in."""
        return self._config.enabled

    @property
    def queue(self) -> ManualMode:
        """The manual queue suggestions are appended to."""
        return self._queue

    @property
    def source(self) -> ReferenceAccountSource:
        """The read-capable API used to fetch comments."""
        return self._source

    def _cache_key(self, platform: str, language: str, limit: int) -> str:
        return canonical_json([platform.strip(), language.strip(), int(limit)])

    def _cached(self, key: str) -> list[TopPost] | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        return list(entry[1])

    def _store(self, key: str, posts: list[TopPost]) -> None:
        self._cache[key] = (self._now(), list(posts))

    async def _resolve_account(self, platform: str, language: str) -> ReferenceAccount:
        return await self._reference.account(platform, language)

    async def _top_posts(self, account: ReferenceAccount, limit: int) -> list[TopPost]:
        key = self._cache_key(account.platform, account.language, limit)
        posts = self._cached(key)
        if posts is None:
            posts = await self._source.top_posts(account, limit=limit)
            self._store(key, posts)
        return posts

    def _suggest(self, posts: Sequence[TopPost]) -> list[TopicCandidate]:
        """Turn *posts* into deduped, clustered topic suggestions."""
        seen: set[str] = set()
        comments: list[tuple[str, str | None, float]] = []
        for post in posts:
            title = (post.text or "").strip()
            if not title:
                continue
            key = normalize_title(title)
            if key in seen:
                continue
            seen.add(key)
            comments.append((title, post.url, post.score))

        clusters = _cluster(
            [text for text, _, _ in comments], self._config.cluster_threshold
        )

        suggestions: list[TopicCandidate] = []
        for cluster in clusters[: self._config.max_suggestions]:
            title = _suggest_title(cluster)
            source_url: str | None = None
            signal: float = 0.0
            for text, url, score in comments:
                if text.strip() == title:
                    source_url = url
                    signal = score
                    break
            candidate = TopicCandidate(
                title=title,
                mode="idea_harvest",
                source_urls=[source_url] if source_url else [],
                signal=signal,
            )
            suggestions.append(candidate)
        return suggestions

    async def harvest(
        self,
        platform: str,
        language: str,
        *,
        limit: int | None = None,
    ) -> list[TopicCandidate]:
        """Harvest *platform*/*language* comments into the manual queue.

        Returns the topic suggestions appended to the queue. When harvesting is
        not opted in this returns an empty list and appends nothing. Raises
        :class:`MissingScopeError` when the account does not hold the ``read``
        scope.
        """
        if not self._config.enabled:
            return []
        resolved_limit = limit if limit is not None else self._config.limit_per_account
        account = await self._resolve_account(platform, language)
        if not account.has_scope(CAPABILITY_READ):
            raise MissingScopeError(
                f"account {account.key} is not authorised to read comments",
                details={
                    "platform": platform.strip(),
                    "language": language.strip(),
                    "account": account.key,
                    "required_scope": CAPABILITY_READ,
                    "scopes": list(account.scopes),
                },
            )
        posts = await self._top_posts(account, resolved_limit)
        suggestions = self._suggest(posts)
        added = self._queue.add(suggestions)
        return added

    async def harvest_all(self) -> dict[str, list[TopicCandidate]]:
        """Harvest every configured account into the manual queue.

        Returns a mapping of ``platform/language`` to the suggestions appended.
        Accounts without the ``read`` scope raise :class:`MissingScopeError`.
        """
        results: dict[str, list[TopicCandidate]] = {}
        for key in self._accounts.names():
            platform, separator, language = key.partition("/")
            if not separator:
                continue
            results[key] = await self.harvest(platform, language)
        return results


def register_analytics_section(
    sections: SectionRegistry | None = None,
) -> SectionRegistry:
    """Register the ``analytics`` section on *sections* and return the registry.

    The call is idempotent for :class:`IdeaHarvestConfig`. A key already owned by
    another model still fails, so two modules cannot silently claim ``analytics``.
    """
    target = registry if sections is None else sections
    existing = (
        target.get(ANALYTICS_SECTION_KEY) if ANALYTICS_SECTION_KEY in target else None
    )
    if existing is not None:
        if existing.model is IdeaHarvestConfig:
            return target
        raise ConfigError(
            f"config section already registered: {ANALYTICS_SECTION_KEY}",
            details={
                "key": ANALYTICS_SECTION_KEY,
                "existing": existing.model.__name__,
                "new": IdeaHarvestConfig.__name__,
            },
        )
    target.register(
        ANALYTICS_SECTION_KEY,
        IdeaHarvestConfig,
        module="analytics",
    )
    return target
