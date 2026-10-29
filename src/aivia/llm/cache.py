"""Cache LLM responses by the request that produced them.

Two calls may share one response only when the provider, model, prompt version,
conversation and call parameters are all identical. :class:`LLMCacheKey`
combines those values into one stable SHA-256 digest and :class:`LLMCache`
stores that digest in the ``llm_cache`` table next to the serialized response.
A lookup returns the response without calling a provider and records a
zero-amount cost event, so a cache hit is visible in the ledger while never
counting as spend.

With ``no_cache`` set, every lookup misses and nothing is written or recorded.
The cache never decides whether a call is safe or affordable; it only answers
whether an identical request already produced a response.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, cast

from sqlalchemy import Connection, Engine
from sqlalchemy.exc import IntegrityError

from aivia.core.cost import CostTracker
from aivia.core.db.repos import LLMCacheRepository
from aivia.core.models.base import canonical_json
from aivia.core.models.cost import CostEvent
from aivia.core.models.llm import LLMCacheEntry
from aivia.providers.llm.cost import LLM_OPERATION
from aivia.providers.protocols.llm import (
    ChatMessage,
    LLMRequest,
    LLMResponse,
    MessageRole,
)

__all__ = [
    "CACHE_HIT_OPERATION",
    "CachedLLMProvider",
    "LLMCache",
    "LLMCacheKey",
    "LLMResponseCache",
    "LLMResponseCacheEntry",
    "LLMResponseCacheKey",
    "llm_cache_key",
    "llm_response_cache_key",
    "parameters_for_request",
    "provider_name",
]

CACHE_HIT_OPERATION = LLM_OPERATION
"""The default operation recorded for an LLM cache hit."""


@contextmanager
def _write_transaction(engine: Engine) -> Iterator[Connection]:
    """Open a transaction that takes SQLite's write lock before reading.

    This mirrors the asset cache so two concurrent stores of the same request
    cannot both insert the same primary key.
    """
    with engine.begin() as connection:
        if engine.dialect.name == "sqlite":
            connection.exec_driver_sql("BEGIN IMMEDIATE")
        yield connection


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _jsonable(value: Any) -> Any:
    """Return a JSON-compatible copy of *value*, preserving key order."""
    return json.loads(canonical_json(value))


def _message_data(message: Any) -> dict[str, Any]:
    if isinstance(message, ChatMessage):
        return message.model_dump(mode="json", exclude_none=True)
    if isinstance(message, str):
        return ChatMessage(
            role=MessageRole.USER,
            content=message,
        ).model_dump(mode="json", exclude_none=True)
    if isinstance(message, Mapping):
        try:
            parsed = ChatMessage.model_validate(dict(message))
        except (TypeError, ValueError):
            return cast("dict[str, Any]", _jsonable(dict(message)))
        return parsed.model_dump(mode="json", exclude_none=True)
    dump = getattr(message, "model_dump", None)
    if callable(dump):
        return cast("dict[str, Any]", _jsonable(dump(mode="json", exclude_none=True)))
    raise TypeError("messages must contain ChatMessage, mapping or string values")


def _messages(
    values: Sequence[Any] | Mapping[str, Any] | Any,
) -> tuple[dict[str, Any], ...]:
    if isinstance(values, (str, Mapping, ChatMessage)):
        values = [values]
    if not isinstance(values, Sequence):
        raise TypeError("messages must be a sequence of chat messages")
    return tuple(_message_data(message) for message in values)


@dataclass(frozen=True, slots=True)
class LLMCacheKey:
    """Everything that makes two LLM requests the same request.

    Provider, model and prompt version whitespace is ignored. Message and
    parameter key order does not matter, so the digest is stable across
    equivalent inputs. Names stay case sensitive because model identifiers may
    be case sensitive.
    """

    provider: str
    model: str
    prompt_version: str = ""
    messages: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _text(self.provider))
        object.__setattr__(self, "model", _text(self.model))
        object.__setattr__(self, "prompt_version", _text(self.prompt_version))
        object.__setattr__(self, "messages", _messages(self.messages))
        object.__setattr__(self, "parameters", dict(self.parameters or {}))

    @property
    def digest(self) -> str:
        """The stable SHA-256 digest used as the stored cache key."""
        payload = {
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "messages": [dict(message) for message in self.messages],
            "parameters": dict(self.parameters),
        }
        encoded = canonical_json(payload).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def cache_key(self) -> str:
        """Alias for :attr:`digest`."""
        return self.digest

    def as_mapping(self) -> dict[str, Any]:
        """Return the key as a JSON-compatible mapping for sidecars and logs."""
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "messages": [dict(message) for message in self.messages],
            "parameters": dict(self.parameters),
            "cache_key": self.digest,
        }


def llm_cache_key(
    provider: str,
    model: str,
    prompt_version: str = "",
    messages: Sequence[Any] | Mapping[str, Any] | Any = (),
    *,
    parameters: Mapping[str, Any] | None = None,
) -> LLMCacheKey:
    """Build an :class:`LLMCacheKey` from the request values.

    For convenience the third positional argument may be the messages rather
    than the prompt version; the two are told apart by type.
    """
    if not isinstance(prompt_version, str):
        if messages:
            raise TypeError("messages were given twice")
        messages, prompt_version = prompt_version, ""
    return LLMCacheKey(
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        messages=_messages(messages),
        parameters=parameters or {},
    )


llm_response_cache_key = llm_cache_key


def provider_name(provider: Any) -> str:
    """Return the configured name of *provider*, or an empty string."""
    name = getattr(provider, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    metadata = getattr(provider, "metadata", None)
    name = getattr(metadata, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return ""


def parameters_for_request(request: LLMRequest) -> dict[str, Any]:
    """Return the call parameters that affect an LLM response.

    Messages, role, model, prompt name, prompt version and free-form metadata
    are excluded because they are keyed separately or are not sent to the
    provider.
    """
    if not isinstance(request, LLMRequest):
        raise TypeError("request must be an LLMRequest")
    data = request.model_dump(mode="json", exclude_none=True)
    for name in (
        "messages",
        "role",
        "model",
        "metadata",
        "prompt_name",
        "prompt_version",
    ):
        data.pop(name, None)
    return data


class LLMCache:
    """Look up and store LLM responses by their full request digest.

    Pass an SQLAlchemy :class:`~sqlalchemy.Engine` or a ready cache repository
    and cost tracker. The engine form builds the
    :class:`~aivia.core.cost.CostTracker` used to record zero cost hits. With
    ``no_cache`` set, every lookup misses and no hit is recorded, which lets a
    caller honour a ``--no-cache`` flag without changing its flow.
    """

    def __init__(
        self,
        engine: Engine | None = None,
        *,
        repository: LLMCacheRepository | None = None,
        cost: CostTracker | None = None,
        no_cache: bool = False,
    ) -> None:
        if engine is not None:
            if repository is not None:
                raise TypeError("pass either an engine or a repository, not both")
            self._cache = LLMCacheRepository(engine)
            self._cost = cost if cost is not None else CostTracker(engine)
        else:
            if repository is None:
                raise TypeError("LLMCache needs an engine or a cache repository")
            if cost is None:
                raise TypeError("LLMCache needs a cost tracker when given a repository")
            self._cache = repository
            self._cost = cost
        self.no_cache = bool(no_cache)

    @property
    def repository(self) -> LLMCacheRepository:
        """The repository holding request digests and their responses."""
        return self._cache

    @property
    def cost(self) -> CostTracker:
        """The tracker that records zero cost hits."""
        return self._cost

    def key(
        self,
        provider: str,
        model: str,
        prompt_version: str = "",
        messages: Sequence[Any] | Mapping[str, Any] | Any = (),
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> LLMCacheKey:
        """Build the cache key for one request."""
        return llm_cache_key(
            provider,
            model,
            prompt_version,
            messages,
            parameters=parameters,
        )

    def key_for_request(
        self,
        provider: Any,
        request: LLMRequest,
        *,
        prompt_version: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> LLMCacheKey:
        """Build the cache key for a typed request."""
        if not isinstance(request, LLMRequest):
            raise TypeError("request must be an LLMRequest")
        model = request.model or getattr(provider, "default_model", None) or ""
        values = parameters_for_request(request)
        if parameters is not None:
            values.update(parameters)
        version = request.prompt_version if prompt_version is None else prompt_version
        return self.key(
            provider_name(provider) if not isinstance(provider, str) else provider,
            model,
            version,
            request.messages,
            parameters=values,
        )

    def entry(self, key: LLMCacheKey) -> LLMCacheEntry | None:
        """Return the cache entry for *key*, or ``None`` on a miss."""
        if self.no_cache:
            return None
        return self._cache.get(key.digest)

    def get(self, key: LLMCacheKey) -> LLMResponse | None:
        """Return the cached response for *key*, or ``None`` on a miss."""
        entry = self.entry(key)
        if entry is None:
            return None
        return LLMResponse.model_validate(entry.response)

    def lookup(
        self,
        key: LLMCacheKey,
        *,
        run_id: str = "",
        operation: str = CACHE_HIT_OPERATION,
        record_hit: bool = True,
    ) -> LLMResponse | None:
        """Return the cached response and record a zero cost hit when found."""
        entry = self.entry(key)
        if entry is None:
            return None
        response = LLMResponse.model_validate(entry.response)
        if record_hit and not self.no_cache:
            self.record_hit(key, response, run_id=run_id, operation=operation)
        return response

    def record_hit(
        self,
        key: LLMCacheKey,
        response: LLMResponse,
        *,
        run_id: str = "",
        operation: str = CACHE_HIT_OPERATION,
    ) -> CostEvent:
        """Record a cache hit and increment the entry's hit counter.

        The cost event is an actual event with a zero amount, so reports show
        the hit without treating it as spend.
        """
        self._cache.increment_hit_count(key.digest)
        return self._cost.record(
            provider=key.provider or response.provider,
            model=key.model or response.model,
            operation=operation,
            units=0.0,
            unit_kind="call",
            amount_usd=0.0,
            estimated=False,
            run_id=run_id,
            metadata={
                "cache": "hit",
                "cache_key": key.digest,
                "prompt_version": key.prompt_version,
            },
        )

    def store(
        self,
        response: LLMResponse | Mapping[str, Any],
        key: LLMCacheKey | None = None,
        *,
        request: LLMRequest | None = None,
        provider: Any = None,
        prompt_version: str | None = None,
    ) -> LLMCacheEntry:
        """Store *response* under *key*, returning an existing entry on a hit.

        When *key* is omitted a *request* and *provider* must be supplied so
        the key can be derived. An existing entry with the same digest wins, so
        storing an equivalent response twice does not create a duplicate.
        """
        if isinstance(response, LLMCacheKey):
            if not isinstance(key, LLMResponse) and not isinstance(key, Mapping):
                raise TypeError("store expects a response and a key")
            response, key = key, response
        if key is None:
            if request is None:
                raise TypeError("store needs a key or a request and provider")
            key = self.key_for_request(
                provider,
                request,
                prompt_version=prompt_version,
            )
        if not isinstance(key, LLMCacheKey):
            raise TypeError("key must be an LLMCacheKey")
        stored_response = (
            response
            if isinstance(response, LLMResponse)
            else LLMResponse.model_validate(response)
        )
        entry = LLMCacheEntry(
            cache_key=key.digest,
            provider=key.provider,
            model=key.model,
            prompt_version=key.prompt_version,
            messages=[dict(message) for message in key.messages],
            parameters=dict(key.parameters),
            response=stored_response.model_dump(mode="json"),
        )
        if self.no_cache:
            return entry
        try:
            with _write_transaction(self._cache.engine) as connection:
                existing = self._cache.get_in(connection, key.digest)
                if existing is not None:
                    return existing
                return self._cache.create_in(connection, entry)
        except IntegrityError:
            existing = self._cache.get(key.digest)
            if existing is None:
                raise
            return existing

    def put(
        self,
        response: LLMResponse | Mapping[str, Any],
        key: LLMCacheKey | None = None,
        **values: Any,
    ) -> LLMCacheEntry:
        """Alias for :meth:`store`."""
        return self.store(response, key, **values)

    def wrap(
        self,
        provider: Any,
        *,
        prompt_version: str = "",
        parameters: Mapping[str, Any] | None = None,
        run_id: str = "",
        operation: str = CACHE_HIT_OPERATION,
        no_cache: bool = False,
    ) -> CachedLLMProvider:
        """Return *provider* wrapped so every completion goes through the cache."""
        return CachedLLMProvider(
            provider,
            self,
            prompt_version=prompt_version,
            parameters=parameters,
            run_id=run_id,
            operation=operation,
            no_cache=no_cache,
        )


LLMResponseCache = LLMCache
"""Alias for callers that prefer the explicit response wording."""

LLMResponseCacheEntry = LLMCacheEntry
"""Alias for the persisted cache record."""

LLMResponseCacheKey = LLMCacheKey
"""Alias for the cache key."""


class CachedLLMProvider:
    """An LLM provider that serves identical requests from an :class:`LLMCache`.

    A hit returns the stored response without calling the wrapped provider and
    records a zero cost event. A miss calls the provider and stores the
    response under the request digest.
    """

    def __init__(
        self,
        provider: Any,
        cache: LLMCache,
        *,
        prompt_version: str = "",
        parameters: Mapping[str, Any] | None = None,
        run_id: str = "",
        operation: str = CACHE_HIT_OPERATION,
        no_cache: bool = False,
    ) -> None:
        if not callable(getattr(provider, "complete", None)):
            raise TypeError("provider must expose an async complete method")
        if not isinstance(cache, LLMCache):
            raise TypeError("cache must be an LLMCache")
        self._provider = provider
        self._cache = cache
        self._prompt_version = prompt_version
        self._parameters = dict(parameters or {})
        self._run_id = run_id
        self._operation = operation
        self.no_cache = bool(no_cache) or cache.no_cache

    @property
    def provider(self) -> Any:
        """The wrapped provider."""
        return self._provider

    @property
    def cache(self) -> LLMCache:
        """The cache used for lookups and stores."""
        return self._cache

    def key_for(self, request: LLMRequest) -> LLMCacheKey:
        """Return the cache key for *request*."""
        return self._cache.key_for_request(
            self._provider,
            request,
            prompt_version=self._prompt_version or None,
            parameters=self._parameters,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return a cached response or call the wrapped provider once."""
        if self.no_cache:
            response = await self._provider.complete(request)
            if not isinstance(response, LLMResponse):
                raise TypeError("provider complete must return an LLMResponse")
            return response
        key = self.key_for(request)
        hit = self._cache.lookup(
            key,
            run_id=self._run_id,
            operation=self._operation,
        )
        if hit is not None:
            return hit
        response = await self._provider.complete(request)
        if not isinstance(response, LLMResponse):
            raise TypeError("provider complete must return an LLMResponse")
        self._cache.store(response, key)
        return response
