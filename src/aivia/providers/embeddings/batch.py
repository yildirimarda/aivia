"""Batching and normalization shared by every embeddings adapter.

An embeddings API accepts a limited number of inputs per request. This base
class turns one logical :class:`~aivia.providers.protocols.embeddings.EmbeddingRequest`
into as many provider calls as needed, keeps the caller's input order, sums the
reported token usage and applies optional unit-length normalization once after
the batch is complete. Adapters only implement ``_embed_batch`` for one
provider-sized request.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from aivia.core.errors import ValidationError
from aivia.providers.base import Provider
from aivia.providers.protocols.embeddings import (
    Embedding,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "BatchedEmbeddingProvider",
    "batch_inputs",
    "normalize_vector",
]

DEFAULT_BATCH_SIZE = 64
"""The default number of inputs sent in one provider request."""


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def batch_inputs(inputs: list[str], batch_size: int) -> list[list[str]]:
    """Split *inputs* into consecutive chunks of at most *batch_size* entries."""
    size = _positive_int(batch_size, "batch_size")
    if not inputs:
        raise ValueError("inputs must not be empty")
    return [inputs[index : index + size] for index in range(0, len(inputs), size)]


def normalize_vector(vector: list[float]) -> list[float]:
    """Return *vector* scaled to unit length, leaving a zero vector unchanged."""
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return list(vector)
    return [component / norm for component in vector]


class BatchedEmbeddingProvider(Provider):
    """An embeddings provider that batches inputs and normalizes on request."""

    default_batch_size: ClassVar[int] = DEFAULT_BATCH_SIZE
    """The number of inputs an adapter sends per request unless configured."""

    default_dimensions: int | None = None
    """The vector dimension used when a request does not ask for one."""

    def __init__(
        self,
        *,
        batch_size: int | None = None,
        dimensions: int | None = None,
        **values: Any,
    ) -> None:
        super().__init__(**values)
        self._batch_size = _positive_int(
            self.default_batch_size if batch_size is None else batch_size,
            "batch_size",
        )
        if dimensions is None:
            dimensions = self.default_dimensions
        elif isinstance(dimensions, bool) or not isinstance(dimensions, int):
            raise TypeError("dimensions must be an integer or None")
        elif dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    @property
    def batch_size(self) -> int:
        """The maximum number of inputs sent in one provider request."""
        return self._batch_size

    @property
    def dimensions(self) -> int | None:
        """The default vector dimension, or ``None`` when the provider decides."""
        return self._dimensions

    def requested_dimensions(self, request: EmbeddingRequest) -> int | None:
        """Return the request dimension, falling back to the adapter default."""
        if request.dimensions is not None:
            return request.dimensions
        return self._dimensions

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Embed every input, splitting the request into provider-sized batches."""
        if not isinstance(request, EmbeddingRequest):
            raise TypeError("request must be an EmbeddingRequest")

        requested = self.requested_dimensions(request)
        embeddings: list[Embedding] = []
        input_tokens = 0
        model = ""
        request_id: str | None = None
        batch_count = 0

        for offset, inputs in enumerate(batch_inputs(request.inputs, self._batch_size)):
            batch = request.model_copy(update={"inputs": inputs})
            response = await self._embed_batch(batch)
            if not isinstance(response, EmbeddingResponse):
                raise TypeError(
                    "_embed_batch must return an EmbeddingResponse, "
                    f"got {type(response).__name__}"
                )
            batch_count += 1
            if len(response.embeddings) != len(inputs):
                raise ValidationError(
                    "embeddings provider returned the wrong number of vectors",
                    details={
                        "provider": self.name,
                        "expected": len(inputs),
                        "received": len(response.embeddings),
                    },
                )
            for embedding in response.embeddings:
                if embedding.index >= len(inputs):
                    raise ValidationError(
                        "embeddings provider returned an out-of-range index",
                        details={
                            "provider": self.name,
                            "index": embedding.index,
                            "batch_size": len(inputs),
                        },
                    )
                vector = (
                    normalize_vector(embedding.vector)
                    if request.normalize
                    else list(embedding.vector)
                )
                embeddings.append(
                    Embedding(
                        index=offset * self._batch_size + embedding.index,
                        vector=vector,
                    )
                )
            input_tokens += response.usage.input_tokens
            model = model or response.model
            request_id = response.request_id or request_id

        response = EmbeddingResponse(
            embeddings=embeddings,
            model=model or request.model or "",
            provider=self.name,
            usage=EmbeddingUsage(input_tokens=input_tokens),
            request_id=request_id,
            metadata={"batches": batch_count},
        )
        if requested is not None and response.dimensions not in {None, requested}:
            raise ValidationError(
                "embeddings provider returned the wrong dimension",
                details={
                    "provider": self.name,
                    "expected": requested,
                    "received": response.dimensions,
                },
            )
        return response

    async def _embed_batch(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Embed one batch; adapters implement this hook."""
        raise NotImplementedError(f"{type(self).__name__} must implement _embed_batch")
