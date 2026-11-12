"""Reference accounts used as style sources.

A configured publish account can be read as a style reference: its top posts
are fetched through a read-capable API and merged into a single style spec.
Reading an account requires the ``read`` scope. When an account does not hold
that scope a :class:`MissingScopeError` is raised with a clear message instead
of silently skipping the account. API responses are cached so repeating the
same lookup does not refetch.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from aivia.core.errors import ConfigError
from aivia.core.models.base import canonical_json
from aivia.core.models.provider import ProviderMeta
from aivia.providers.base import Provider
from aivia.publish.accounts import AccountCredentials, AccountMap
from aivia.reference.merge import merge_references
from aivia.reference.schema import ReferenceAnalysis, ReferenceKind, StyleSpec

__all__ = [
    "CAPABILITY_READ",
    "MissingScopeError",
    "ReferenceAccount",
    "ReferenceAccountSource",
    "TopPost",
    "account_reference",
]

CAPABILITY_READ = "read"
"""The scope an account must hold to be read as a style source."""


class MissingScopeError(ConfigError):
    """A reference account does not hold the scope required to read its posts."""

    code = "account_reference_scope"


@dataclass(frozen=True, slots=True)
class TopPost:
    """One post fetched from a reference account."""

    post_id: str
    text: str
    url: str | None = None
    created_at: datetime | None = None
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class ReferenceAccount:
    """A publish account cleared to be read as a style source."""

    platform: str
    language: str
    credential_refs: tuple[str, ...]
    scopes: tuple[str, ...]

    @property
    def key(self) -> str:
        """The ``platform/language`` account key."""
        return f"{self.platform}/{self.language}"

    def has_scope(self, scope: str) -> bool:
        """Whether the account holds *scope*."""
        return scope.strip() in self.scopes


class ReferenceAccountSource(Provider):
    """A read-capable API returning the top posts of a reference account.

    Concrete adapters implement :meth:`top_posts`. The class declares the
    ``read`` capability so callers can tell a read-capable provider apart from
    one that only publishes.
    """

    default_metadata = ProviderMeta(
        name="reference-account",
        capabilities=[CAPABILITY_READ],
    )

    async def top_posts(
        self, account: ReferenceAccount, *, limit: int = 10
    ) -> list[TopPost]:
        """Return the *limit* most relevant posts of *account*."""
        raise NotImplementedError


class AccountReference:
    """Resolve a reference account, check its read scope, fetch top posts and
    merge them into a style spec.

    ``accounts`` resolves publish accounts. ``source`` is the read-capable API
    that fetches top posts. Responses are cached keyed by account and limit so
    a repeated lookup does not refetch the API.
    """

    def __init__(
        self,
        accounts: AccountMap,
        source: ReferenceAccountSource,
        *,
        cache: Mapping[str, tuple[float, list[TopPost]]] | None = None,
        ttl_seconds: float = 300.0,
        now: Callable[[], float] | None = None,
    ) -> None:
        self._accounts = accounts
        self._source = source
        self._cache: dict[str, tuple[float, list[TopPost]]] = dict(cache or {})
        self.ttl_seconds = ttl_seconds
        self._now = now or time.monotonic

    @property
    def source(self) -> ReferenceAccountSource:
        """The read-capable API used to fetch top posts."""
        return self._source

    @property
    def accounts(self) -> AccountMap:
        """The account map used to resolve publish accounts."""
        return self._accounts

    def _cache_key(self, platform: str, language: str, limit: int) -> str:
        return canonical_json([platform.strip(), language.strip(), int(limit)])

    def _cached(self, key: str) -> list[TopPost] | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        fetched_at, posts = entry
        if self._now() - fetched_at > self.ttl_seconds:
            return None
        return list(posts)

    def _store(self, key: str, posts: list[TopPost]) -> None:
        self._cache[key] = (self._now(), list(posts))

    @property
    def responses(self) -> tuple[tuple[str, tuple[TopPost, ...]], ...]:
        """The cached responses, most recent first, as ``(key, posts)`` pairs."""
        ordered = sorted(self._cache.items(), key=lambda item: item[1][0], reverse=True)
        return tuple((key, tuple(posts)) for key, (_, posts) in ordered)

    def clear(self) -> None:
        """Drop every cached response."""
        self._cache.clear()

    async def account(self, platform: str, language: str) -> ReferenceAccount:
        """Resolve *platform*/*language* as a reference account."""
        credentials = self._accounts.resolve(platform, language)
        return self._reference_account(credentials)

    def _reference_account(self, credentials: AccountCredentials) -> ReferenceAccount:
        account_config = self._accounts.accounts.get(credentials.key)
        return ReferenceAccount(
            platform=credentials.platform,
            language=credentials.language,
            credential_refs=credentials.credential_refs,
            scopes=self._scopes_for(account_config),
        )

    @staticmethod
    def _scopes_for(account_config: Any) -> tuple[str, ...]:
        if account_config is None:
            return ()
        raw = account_config.options.get("scopes", [])
        if isinstance(raw, str):
            raw = [raw]
        scopes: list[str] = []
        for scope in raw:
            text = str(scope).strip()
            if text and text not in scopes:
                scopes.append(text)
        return tuple(scopes)

    async def style_spec(
        self,
        platform: str,
        language: str,
        *,
        limit: int = 10,
        name: str = "",
    ) -> StyleSpec:
        """Return the merged style spec of *platform*/*language*'s top posts.

        Raises :class:`MissingScopeError` when the account does not hold the
        ``read`` scope.
        """
        account = await self.account(platform, language)
        if not account.has_scope(CAPABILITY_READ):
            raise MissingScopeError(
                f"account {account.key} is not authorised to read posts",
                details={
                    "platform": platform.strip(),
                    "language": language.strip(),
                    "account": account.key,
                    "required_scope": CAPABILITY_READ,
                    "scopes": list(account.scopes),
                },
            )
        key = self._cache_key(platform, language, limit)
        posts = self._cached(key)
        if posts is None:
            posts = await self._source.top_posts(account, limit=limit)
            self._store(key, posts)
        return self._merge(account, posts, name=name or f"{account.key} style")

    def _merge(
        self,
        account: ReferenceAccount,
        posts: Sequence[TopPost],
        name: str = "",
    ) -> StyleSpec:
        if not posts:
            return StyleSpec(name=name)
        references: list[ReferenceAnalysis] = []
        for index, post in enumerate(posts, start=1):
            references.append(
                ReferenceAnalysis(
                    reference_id=f"{account.key}:{post.post_id}",
                    kind=ReferenceKind.TEXT,
                    provider=self._source.name,
                    detected_text=post.text,
                    prompt_suffix=post.text,
                    confidence=_post_confidence(post.score, index),
                    metadata={
                        "post_id": post.post_id,
                        "url": post.url,
                        "score": post.score,
                    },
                )
            )
        return merge_references(references, name=name).spec


def _post_confidence(score: float, index: int) -> float:
    position = max(0.0, min(1.0, 1.0 - (index - 1) / 20.0))
    value = max(0.0, min(1.0, score / 100.0)) * 0.5 + position * 0.5
    return round(value, 4)


account_reference = AccountReference
"""Alias for :class:`AccountReference`."""
