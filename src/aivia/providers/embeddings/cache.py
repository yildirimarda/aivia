"""Per-input cache for embeddings.

Embedding the same text twice is wasted spend. :class:`EmbeddingCache` stores
one vector per provider, model, dimension, input type and normalization flag,
and :class:`CachedEmbeddingProvider` only sends the inputs that are still
missing to the wrapped provider. A partially cached batch keeps the caller's
input order and reports how many vectors came from the cache.

The cache is bounded and can be backed by SQLite. Without an engine it is
in-memory only, which is safe for a worker or a test and never touches the
database. With an engine, vectors are written through to the ``embedding_cache``
table and a fresh cache instance can serve them after a restart. It never
decides whether a call is affordable.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import cast

from sqlalchemy import Engine
from sqlalchemy.exc import IntegrityError

from aivia.core.db.repos import EmbeddingCacheRepository
from aivia.core.models.base import canonical_json
from aivia.core.models.embedding import EmbeddingCacheEntry
from aivia.providers.base import Provider, ProviderHealth
from aivia.providers.protocols.embeddings import (
    Embedding,
    EmbeddingInputType,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)

__all__ = [
    "DEFAULT_CACHE_SIZE",
    "CachedEmbeddingProvider",
    "EmbeddingCache",
    "EmbeddingCacheEntry",
    "EmbeddingCacheKey",
    "EmbeddingCacheStats",
]

DEFAULT_CACHE_SIZE = 10_000
"""The default number of vectors kept in memory."""


def _text(value: str) -> str:
    return value.strip()


@dataclass(frozen=True, slots=True)
class EmbeddingCacheKey:
    """Everything that makes two embedded inputs the same cached vector."""

    provider: str
    model: str
    text: str
    dimensions: int | None = None
    input_type: str = ""
    normalize: bool = False
    truncate: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _text(self.provider))
        object.__setattr__(self, "model", _text(self.model))
        object.__setattr__(self, "input_type", _text(self.input_type))

    @property
    def digest(self) -> str:
        """The stable SHA-256 digest used as the stored cache key."""
        payload = {
            "provider": self.provider,
            "model": self.model,
            "text": self.text,
            "dimensions": self.dimensions,
            "input_type": self.input_type,
            "normalize": self.normalize,
            "truncate": self.truncate,
        }
        encoded = canonical_json(payload).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def cache_key(self) -> str:
        """Alias for :attr:`digest`."""
        return self.digest


@dataclass(frozen=True, slots=True)
class EmbeddingCacheStats:
    """Counters describing how the cache has been used."""

    hits: int = 0
    misses: int = 0
    entries: int = 0

    @property
    def lookups(self) -> int:
        """The total number of lookups."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """The share of lookups served from the cache, or zero without lookups."""
        return self.hits / self.lookups if self.lookups else 0.0


class EmbeddingCache:
    """A bounded least-recently-used store of embedding vectors.

    Pass an SQLAlchemy :class:`~sqlalchemy.Engine` (positionally or as
    ``engine=``) or a ready :class:`~aivia.core.db.repos.EmbeddingCacheRepository`
    to persist vectors in the ``embedding_cache`` table. A cache built on the
    same database later serves those vectors without calling a provider.
    """

    def __init__(
        self,
        max_entries: int | Engine = DEFAULT_CACHE_SIZE,
        *,
        no_cache: bool = False,
        engine: Engine | None = None,
        repository: EmbeddingCacheRepository | None = None,
    ) -> None:
        if isinstance(max_entries, Engine):
            if engine is not None:
                raise TypeError("engine was given twice")
            engine = max_entries
            max_entries = DEFAULT_CACHE_SIZE
        if isinstance(max_entries, bool) or not isinstance(max_entries, int):
            raise TypeError("max_entries must be an integer")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if engine is not None and repository is not None:
            raise TypeError("pass either an engine or a repository, not both")
        if repository is not None and not isinstance(
            repository, EmbeddingCacheRepository
        ):
            raise TypeError("repository must be an EmbeddingCacheRepository")
        self._max_entries = max_entries
        self.no_cache = bool(no_cache)
        self._entries: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._repository: EmbeddingCacheRepository | None
        if repository is not None:
            self._repository = repository
        elif engine is not None:
            self._repository = EmbeddingCacheRepository(engine)
        else:
            self._repository = None

    @property
    def max_entries(self) -> int:
        """The largest number of vectors the cache keeps."""
        return self._max_entries

    @property
    def repository(self) -> EmbeddingCacheRepository | None:
        """The persistent store, or ``None`` for an in-memory cache."""
        return self._repository

    @property
    def persistent(self) -> bool:
        """Whether vectors are written through to SQLite."""
        return self._repository is not None

    @property
    def hits(self) -> int:
        """The number of lookups served from the cache."""
        return self._hits

    @property
    def misses(self) -> int:
        """The number of lookups that found no vector."""
        return self._misses

    def __len__(self) -> int:
        with self._lock:
            return self._entry_count()

    def stats(self) -> EmbeddingCacheStats:
        """Return a snapshot of the cache counters."""
        with self._lock:
            return EmbeddingCacheStats(
                hits=self._hits,
                misses=self._misses,
                entries=self._entry_count(),
            )

    def get(self, key: EmbeddingCacheKey) -> list[float] | None:
        """Return a copy of the cached vector for *key*, or ``None``."""
        if not isinstance(key, EmbeddingCacheKey):
            raise TypeError("key must be an EmbeddingCacheKey")
        with self._lock:
            if self.no_cache:
                self._misses += 1
                return None
            digest = key.digest
            vector = self._entries.get(digest)
            if vector is not None:
                self._entries.move_to_end(digest)
                self._hits += 1
                return list(vector)
            if self._repository is not None:
                entry = self._repository.get(digest)
                if entry is not None and entry.vector:
                    stored = list(entry.vector)
                    self._remember(digest, stored)
                    self._hits += 1
                    return list(stored)
            self._misses += 1
            return None

    def put(self, key: EmbeddingCacheKey, vector: list[float]) -> None:
        """Store a copy of *vector* under *key*, evicting the oldest entry."""
        if not isinstance(key, EmbeddingCacheKey):
            raise TypeError("key must be an EmbeddingCacheKey")
        if not vector:
            raise ValueError("vector must not be empty")
        with self._lock:
            if self.no_cache:
                return
            self._remember(key.digest, vector)
            if self._repository is not None:
                self._persist(key, vector)

    def clear(self) -> None:
        """Remove every vector without resetting the counters."""
        with self._lock:
            self._entries.clear()
            if self._repository is not None:
                self._repository.clear()

    def _entry_count(self) -> int:
        if self.no_cache:
            return 0
        if self._repository is not None:
            return self._repository.count()
        return len(self._entries)

    def _remember(self, digest: str, vector: list[float]) -> None:
        self._entries[digest] = list(vector)
        self._entries.move_to_end(digest)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def _persist(self, key: EmbeddingCacheKey, vector: list[float]) -> None:
        assert self._repository is not None
        entry = EmbeddingCacheEntry(
            cache_key=key.digest,
            provider=key.provider,
            model=key.model,
            text=key.text,
            dimensions=key.dimensions,
            input_type=key.input_type,
            normalize=key.normalize,
            truncate=key.truncate,
            vector=list(vector),
        )
        try:
            self._repository.create(entry)
        except IntegrityError:
            self._repository.update(key.digest, vector=list(vector))
        self._repository.prune(self._max_entries)


def _default_model(provider: Provider) -> str:
    value = getattr(provider, "default_model", None)
    return value.strip() if isinstance(value, str) else ""


def _default_dimensions(provider: Provider) -> int | None:
    for attribute in ("dimensions", "default_dimensions"):
        value = getattr(provider, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


class CachedEmbeddingProvider(Provider):
    """An embeddings provider that serves already-embedded inputs from a cache."""

    def __init__(
        self,
        provider: Provider,
        cache: EmbeddingCache | None = None,
        *,
        no_cache: bool = False,
    ) -> None:
        if not callable(getattr(provider, "embed", None)):
            raise TypeError("provider must expose an async embed method")
        if cache is not None and not isinstance(cache, EmbeddingCache):
            raise TypeError("cache must be an EmbeddingCache")
        super().__init__(provider.metadata)
        self._provider = provider
        self._cache = cache if cache is not None else EmbeddingCache(no_cache=no_cache)
        self.no_cache = bool(no_cache) or self._cache.no_cache

    @property
    def provider(self) -> Provider:
        """The wrapped provider."""
        return self._provider

    @property
    def cache(self) -> EmbeddingCache:
        """The cache used for lookups and stores."""
        return self._cache

    @property
    def default_model(self) -> str:
        """The wrapped provider's default model id, when it has one."""
        return _default_model(self._provider)

    @property
    def default_dimensions(self) -> int | None:
        """The wrapped provider's default vector dimension, when it has one."""
        return _default_dimensions(self._provider)

    def key_for(self, request: EmbeddingRequest, text: str) -> EmbeddingCacheKey:
        """Return the cache key for one input of *request*."""
        model = request.model or self.default_model
        dimensions = request.dimensions or self.default_dimensions
        input_type = (
            request.input_type.value
            if isinstance(request.input_type, EmbeddingInputType)
            else ""
        )
        return EmbeddingCacheKey(
            provider=self._provider.name,
            model=model,
            text=text,
            dimensions=dimensions,
            input_type=input_type,
            normalize=request.normalize,
            truncate=request.truncate,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Return cached vectors and send only the missing inputs to the provider."""
        if not isinstance(request, EmbeddingRequest):
            raise TypeError("request must be an EmbeddingRequest")
        if self.no_cache:
            return cast(
                EmbeddingResponse,
                await self._provider.embed(request),  # type: ignore[attr-defined]
            )

        vectors: dict[int, list[float]] = {}
        missing: list[tuple[int, str, EmbeddingCacheKey]] = []
        for index, text in enumerate(request.inputs):
            key = self.key_for(request, text)
            hit = self._cache.get(key)
            if hit is None:
                missing.append((index, text, key))
            else:
                vectors[index] = hit

        cache_hits = len(vectors)
        if not missing:
            return EmbeddingResponse(
                embeddings=[
                    Embedding(index=index, vector=vector)
                    for index, vector in sorted(vectors.items())
                ],
                model=request.model or self.default_model,
                provider=self.name,
                usage=EmbeddingUsage(),
                metadata={
                    "cache": "hit",
                    "cache_hits": cache_hits,
                    "cache_misses": 0,
                },
            )

        batch = request.model_copy(update={"inputs": [text for _, text, _ in missing]})
        response = cast(
            EmbeddingResponse,
            await self._provider.embed(batch),  # type: ignore[attr-defined]
        )
        if not isinstance(response, EmbeddingResponse):
            raise TypeError("provider embed must return an EmbeddingResponse")
        if len(response.embeddings) != len(missing):
            raise ValueError(
                "provider returned the wrong number of vectors for the missing inputs"
            )
        by_index = {embedding.index: embedding for embedding in response.embeddings}
        for position, (original_index, _text_value, key) in enumerate(missing):
            embedding = by_index.get(position)
            if embedding is None:
                raise ValueError("provider did not return a vector for every input")
            self._cache.put(key, embedding.vector)
            vectors[original_index] = list(embedding.vector)

        metadata = dict(response.metadata)
        metadata.update(
            {
                "cache": "partial" if cache_hits else "miss",
                "cache_hits": cache_hits,
                "cache_misses": len(missing),
            }
        )
        return EmbeddingResponse(
            embeddings=[
                Embedding(index=index, vector=vector)
                for index, vector in sorted(vectors.items())
            ],
            model=response.model or request.model or self.default_model,
            provider=self.name,
            usage=response.usage,
            request_id=response.request_id,
            metadata=metadata,
        )

    async def check_health(self) -> ProviderHealth:
        """Report the wrapped provider's health."""
        return await self._provider.health()

    async def _close(self) -> None:
        await self._provider.close()
