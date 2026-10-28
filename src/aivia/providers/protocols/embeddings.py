"""Typed embeddings protocol, request and response models.

An embeddings provider turns a batch of texts into one vector per input. The
request owns the batch, the requested dimension and whether the texts are
documents or queries, so an adapter only maps those fields onto its vendor
payload. Vectors are validated for finite values and a consistent dimension.

The protocol says nothing about vendors, model ids, prices or thresholds: those
belong to adapter modules, ``models.yaml`` and configuration.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import Field, field_validator, model_validator

from aivia.providers.protocols.base import CapabilityProvider, ProtocolModel

__all__ = [
    "CAPABILITY_EMBEDDINGS",
    "Embedding",
    "EmbeddingInputType",
    "EmbeddingProvider",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingUsage",
]

CAPABILITY_EMBEDDINGS = "embeddings"
"""The capability name an embeddings provider declares."""


class EmbeddingInputType(StrEnum):
    """Whether the embedded text is stored content or a search query."""

    DOCUMENT = "document"
    QUERY = "query"


class Embedding(ProtocolModel):
    """One vector and the position of its input in the request batch."""

    index: int = Field(ge=0)
    vector: list[float] = Field(min_length=1)

    @field_validator("vector")
    @classmethod
    def _validate_vector(cls, value: list[float]) -> list[float]:
        if not all(math.isfinite(component) for component in value):
            raise ValueError("embedding values must be finite")
        return value

    @property
    def dimensions(self) -> int:
        """The number of components in the vector."""
        return len(self.vector)


class EmbeddingUsage(ProtocolModel):
    """Token usage reported by an embeddings provider."""

    input_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        """Embeddings only bill input tokens."""
        return self.input_tokens

    @property
    def prompt_tokens(self) -> int:
        """Alias for :attr:`input_tokens`."""
        return self.input_tokens


class EmbeddingRequest(ProtocolModel):
    """A validated batch of texts to embed with one model.

    ``dimensions`` asks a provider that supports dimension reduction for a
    smaller vector. ``normalize`` requests unit-length vectors when the adapter
    can produce them; similarity code must still handle unnormalized vectors.
    """

    inputs: list[str] = Field(min_length=1)
    model: str | None = None
    dimensions: int | None = Field(default=None, ge=1)
    input_type: EmbeddingInputType | None = None
    normalize: bool = False
    truncate: bool = True
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("inputs", mode="before")
    @classmethod
    def _coerce_inputs(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("inputs")
    @classmethod
    def _validate_inputs(cls, value: list[str]) -> list[str]:
        texts: list[str] = []
        for entry in value:
            if not isinstance(entry, str):
                raise ValueError("embedding inputs must be strings")
            text = entry.strip()
            if not text:
                raise ValueError("embedding inputs must be non-empty")
            texts.append(text)
        return texts

    @field_validator("model", mode="before")
    @classmethod
    def _strip_model(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value


class EmbeddingResponse(ProtocolModel):
    """A validated batch of embeddings from one provider."""

    embeddings: list[Embedding] = Field(default_factory=list)
    model: str = ""
    provider: str = ""
    usage: EmbeddingUsage = Field(default_factory=EmbeddingUsage)
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model", "provider", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("request_id", mode="before")
    @classmethod
    def _strip_request_id(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value

    @model_validator(mode="after")
    def _validate_embeddings(self) -> Self:
        indices = [embedding.index for embedding in self.embeddings]
        if len(indices) != len(set(indices)):
            raise ValueError("embedding indices must be unique")
        dimensions = {embedding.dimensions for embedding in self.embeddings}
        if len(dimensions) > 1:
            raise ValueError("embedding vectors must all have the same dimension")
        return self

    @property
    def vectors(self) -> list[list[float]]:
        """Every vector in response order."""
        return [embedding.vector for embedding in self.embeddings]

    @property
    def dimensions(self) -> int | None:
        """The vector dimension, or ``None`` when the batch is empty."""
        if not self.embeddings:
            return None
        return self.embeddings[0].dimensions


@runtime_checkable
class EmbeddingProvider(CapabilityProvider, Protocol):
    """A provider that serves the ``embeddings`` capability."""

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Embed every input in *request* and return validated vectors."""
