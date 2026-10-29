"""Per-input cache for embeddings.

Embedding the same text twice is wasted spend. :class:`EmbeddingCache` stores
one vector per provider, model, dimension, input type and normalization flag,
and :class:`CachedEmbeddingProvider` only sends the inputs that are still
missing to the wrapped provider. A partially cached batch keeps the caller's
input order and reports how many vectors came from the cache.

The cache is deliberately in-memory and bounded: it is safe for a worker or a
test, never touches the database and never decides whether a call is affordable.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import cast

from aivia.core.models.base import canonical_json
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
    """A bounded least-recently-used store of embedding vectors."""

    def __init__(
        self,
        max_entries: int = DEFAULT_CACHE_SIZE,
        *,
        no_cache: bool = False,
    ) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int):
            raise TypeError("max_entries must be an integer")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self.no_cache = bool(no_cache)
        self._entries: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0

    @property
    def max_entries(self) -> int:
        """The largest number of vectors the cache keeps."""
        return self._max_entries

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
            return len(self._entries)

    def stats(self) -> EmbeddingCacheStats:
        """Return a snapshot of the cache counters."""
        with self._lock:
            return EmbeddingCacheStats(
                hits=self._hits,
                misses=self._misses,
                entries=len(self._entries),
            )

    def get(self, key: EmbeddingCacheKey) -> list[float] | None:
        """Return a copy of the cached vector for *key*, or ``None``."""
        if not isinstance(key, EmbeddingCacheKey):
            raise TypeError("key must be an EmbeddingCacheKey")
        with self._lock:
            if self.no_cache:
                self._misses += 1
                return None
            vector = self._entries.get(key.digest)
            if vector is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key.digest)
            self._hits += 1
            return list(vector)

    def put(self, key: EmbeddingCacheKey, vector: list[float]) -> None:
        """Store a copy of *vector* under *key*, evicting the oldest entry."""
        if not isinstance(key, EmbeddingCacheKey):
            raise TypeError("key must be an EmbeddingCacheKey")
        if not vector:
            raise ValueError("vector must not be empty")
        with self._lock:
            if self.no_cache:
                return
            self._entries[key.digest] = list(vector)
            self._entries.move_to_end(key.digest)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Remove every vector without resetting the counters."""
        with self._lock:
            self._entries.clear()


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
