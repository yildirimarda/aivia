"""Dependency-free local embeddings provider.

The adapter produces deterministic vectors without a network call, so it works
offline, in tests and as a fallback when no hosted embeddings service is
configured. It uses feature hashing over word unigrams and bigrams: identical
text always yields the same vector, and texts sharing vocabulary land close
together under cosine similarity.

Callers that have a sentence-transformers model can pass its ``encode``
function as ``encoder``; the adapter then uses those vectors instead. The
default hasher needs no optional dependency.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from itertools import pairwise
from typing import Any

from aivia.core.errors import ValidationError
from aivia.core.models.provider import ProviderMeta
from aivia.providers.embeddings.batch import (
    BatchedEmbeddingProvider,
    normalize_vector,
)
from aivia.providers.protocols.embeddings import (
    CAPABILITY_EMBEDDINGS,
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from aivia.providers.registry import register_provider

__all__ = [
    "DEFAULT_DIMENSIONS",
    "DEFAULT_MODEL",
    "LocalEmbeddingProvider",
    "LocalEmbeddings",
    "LocalEmbeddingsProvider",
]

DEFAULT_DIMENSIONS = 384
"""The default vector dimension of the built-in hashing encoder."""

DEFAULT_MODEL = "local-hash"
"""The model id reported by the built-in hashing encoder."""

_TOKEN = re.compile(r"[a-z0-9]+")
"""Word characters kept by the hashing tokenizer."""

Encoder = Callable[[list[str]], Sequence[Sequence[float]]]
"""A callable that turns a list of texts into one vector per text."""

_DEFAULT_METADATA = ProviderMeta(
    name="local-embeddings",
    capabilities=[CAPABILITY_EMBEDDINGS],
)


def _features(text: str) -> list[str]:
    tokens = _TOKEN.findall(text.casefold())
    features = list(tokens)
    features.extend(f"{left} {right}" for left, right in pairwise(tokens))
    return features


def _hash_vector(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    for feature in _features(text):
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    return normalize_vector(vector)


class LocalEmbeddingProvider(BatchedEmbeddingProvider):
    """A local embeddings provider using an injectable encoder or hashing.

    ``encoder`` may be any callable accepting a list of texts and returning one
    vector per text, such as a sentence-transformers model's ``encode`` method.
    Without one, the built-in deterministic hasher is used.
    """

    default_metadata = _DEFAULT_METADATA
    default_dimensions = DEFAULT_DIMENSIONS

    def __init__(
        self,
        *,
        encoder: Encoder | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
        default_model: str = DEFAULT_MODEL,
        name: str | None = None,
        health_timeout_seconds: float | None = None,
        close_timeout_seconds: float | None = None,
        **values: Any,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            dimensions=dimensions,
            name=name,
            health_timeout_seconds=health_timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            **values,
        )
        if encoder is not None and not callable(encoder):
            raise TypeError("encoder must be callable or None")
        if not isinstance(default_model, str):
            raise TypeError("default_model must be a string")
        model = default_model.strip()
        if not model:
            raise ValueError("default_model must be non-empty")
        self._encoder = encoder
        self._default_model = model

    @property
    def encoder(self) -> Encoder | None:
        """The injected encoder, or ``None`` for the built-in hasher."""
        return self._encoder

    @property
    def default_model(self) -> str:
        """The model id reported when a request does not name one."""
        return self._default_model

    def _vectors(self, texts: list[str], dimensions: int) -> list[list[float]]:
        if self._encoder is not None:
            raw = self._encoder(texts)
            vectors = [[float(component) for component in vector] for vector in raw]
        else:
            vectors = [_hash_vector(text, dimensions) for text in texts]
        if len(vectors) != len(texts):
            raise ValidationError(
                "local encoder returned the wrong number of vectors",
                details={"provider": self.name, "expected": len(texts)},
            )
        for vector in vectors:
            if len(vector) != dimensions:
                raise ValidationError(
                    "local encoder returned the wrong dimension",
                    details={
                        "provider": self.name,
                        "expected": dimensions,
                        "received": len(vector),
                    },
                )
        return vectors

    async def _embed_batch(self, request: EmbeddingRequest) -> EmbeddingResponse:
        dimensions = request.dimensions or self.dimensions or DEFAULT_DIMENSIONS
        vectors = self._vectors(list(request.inputs), dimensions)
        embeddings = [
            Embedding(index=index, vector=vector)
            for index, vector in enumerate(vectors)
        ]
        tokens = sum(max(1, len(_features(text))) for text in request.inputs)
        return EmbeddingResponse(
            embeddings=embeddings,
            model=request.model or self._default_model,
            provider=self.name,
            usage=EmbeddingUsage(input_tokens=tokens),
        )


LocalEmbeddings = LocalEmbeddingProvider
"""Alias for :class:`LocalEmbeddingProvider`."""

LocalEmbeddingsProvider = LocalEmbeddingProvider
"""Alias for :class:`LocalEmbeddingProvider`."""

register_provider("local-embeddings", LocalEmbeddingProvider, module=__name__)
