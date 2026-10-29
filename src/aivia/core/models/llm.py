"""Persisted LLM response cache records.

An LLM response is reusable only when the provider, model, prompt version,
conversation and call parameters are all identical. The cache stores the
canonical request digest next to the serialized response so a later call can be
served without contacting a provider. ``hit_count`` counts the calls that were
answered from this entry.

The model only describes data. Hashing requests and deciding whether a hit is
allowed belongs to :mod:`aivia.llm.cache`.
"""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, Field, field_validator

from aivia.core.models.base import ModelBase

__all__ = [
    "LLMCacheEntry",
    "LLMCacheRecord",
    "LLMResponseCacheEntry",
]


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


class LLMCacheEntry(ModelBase):
    """One request digest pointing at the response that satisfied it.

    ``messages`` and ``parameters`` are kept for audit and for rebuilding the
    key without the original request. ``response`` is the serialized
    :class:`~aivia.providers.protocols.llm.LLMResponse`.
    """

    cache_key: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "cache_key",
            "key",
            "digest",
            "llm_cache_key",
        ),
    )
    provider: str = Field(
        default="",
        validation_alias=AliasChoices("provider", "vendor"),
    )
    model: str = Field(
        default="",
        validation_alias=AliasChoices("model", "model_id"),
    )
    prompt_version: str = Field(
        default="",
        validation_alias=AliasChoices(
            "prompt_version",
            "prompt_revision",
            "version",
        ),
    )
    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        validation_alias=AliasChoices("messages", "conversation"),
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("parameters", "params", "options"),
    )
    response: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("response", "payload", "result"),
    )
    hit_count: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices("hit_count", "hits"),
    )

    @field_validator(
        "cache_key",
        "provider",
        "model",
        "prompt_version",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("messages", mode="before")
    @classmethod
    def _coerce_messages(cls, value: Any) -> Any:
        return [] if value is None else value

    @field_validator("parameters", "response", mode="before")
    @classmethod
    def _coerce_mapping(cls, value: Any) -> Any:
        return {} if value is None else value

    @property
    def llm_cache_key(self) -> str:
        """Alias for :attr:`cache_key`."""
        return self.cache_key

    @property
    def hits(self) -> int:
        """Alias for :attr:`hit_count`."""
        return self.hit_count

    @property
    def prompt_version_name(self) -> str:
        """Alias for :attr:`prompt_version`."""
        return self.prompt_version


LLMCacheRecord = LLMCacheEntry
"""Alias for callers that prefer the record wording."""

LLMResponseCacheEntry = LLMCacheEntry
"""Alias for callers that name the cached response explicitly."""
