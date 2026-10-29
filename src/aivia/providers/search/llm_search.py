"""LLM-native web search adapter.

Some LLM providers can run a web search themselves and return the pages they
consulted as citations. This adapter turns the vendor-neutral
:class:`~aivia.providers.protocols.search.SearchRequest` into an
:class:`~aivia.providers.protocols.llm.LLMRequest` that enables that built-in
search and maps the returned citations back into
:class:`~aivia.providers.protocols.search.SearchResult` values.

The adapter is deliberately vendor-neutral: it names the web search tool
``web_search`` and reads results from a tool call, from citation metadata or
from a JSON body. A wrapped provider that performs the search natively can
answer with any of those shapes, while a provider without native search can
still be driven through structured output. The adapter owns no routing, retry
or budget policy; it delegates the call to the configured LLM provider.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

from pydantic import ValidationError as PydanticValidationError

from aivia.core.errors import ConfigError, ValidationError
from aivia.core.models.provider import ProviderMeta
from aivia.providers.base import Provider
from aivia.providers.protocols.llm import (
    ChatMessage,
    LLMRequest,
    LLMResponse,
    MessageRole,
    ToolChoice,
    ToolDefinition,
)
from aivia.providers.protocols.search import (
    CAPABILITY_SEARCH,
    SafeSearch,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from aivia.providers.registry import register_provider

__all__ = [
    "DEFAULT_TOOL_NAME",
    "WEB_SEARCH_TOOL",
    "LLMNative",
    "LLMNativeProvider",
    "LLMNativeSearchProvider",
    "LLMSearch",
    "LLMSearchProvider",
    "LLMWebSearchProvider",
    "NativeSearch",
    "NativeSearchProvider",
]

DEFAULT_TOOL_NAME = "web_search"
"""The tool name a wrapped provider maps to its native web search."""

WEB_SEARCH_TOOL = ToolDefinition(
    name=DEFAULT_TOOL_NAME,
    description=(
        "Search the web for pages matching the query. Return the pages you "
        "found, each with a title, url and snippet."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "description": "The largest number of pages to return.",
            },
        },
        "required": ["query"],
    },
)
"""The tool definition that enables a provider's native web search."""

DEFAULT_SYSTEM_PROMPT = (
    "You are a web search assistant. Use your native web search capability to "
    "find current, relevant pages for the user's query. Return only a JSON "
    'object of the form {"results": [{"title": "...", "url": "...", '
    '"snippet": "..."}]}. Every result must have an absolute http(s) URL. '
    "Never invent a page or a URL."
)

_RESULT_KEYS = (
    "results",
    "web_results",
    "web_search_results",
    "citations",
    "sources",
)

_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

_DEFAULT_METADATA = ProviderMeta(
    name="llm_native",
    capabilities=[CAPABILITY_SEARCH],
)


class _LLMCompleter(Protocol):
    """The only provider surface a native search needs."""

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


def _coerce_llm(value: object, label: str) -> _LLMCompleter:
    complete = getattr(value, "complete", None)
    if not callable(complete):
        raise TypeError(f"{label} must provide a callable complete method")
    return value  # type: ignore[return-value]


def _user_prompt(request: SearchRequest, max_results: int) -> str:
    lines = [
        f"Search the web for: {request.query}",
        f"Return at most {max_results} results.",
    ]
    if request.include_domains:
        lines.append("Only use these domains: " + ", ".join(request.include_domains))
    if request.exclude_domains:
        lines.append("Exclude these domains: " + ", ".join(request.exclude_domains))
    if request.recency is not None:
        lines.append(f"Prefer pages from the last {request.recency.value}.")
    if request.start_date is not None:
        lines.append(f"Published on or after {request.start_date.isoformat()}.")
    if request.end_date is not None:
        lines.append(f"Published on or before {request.end_date.isoformat()}.")
    if request.language is not None:
        lines.append(f"Prefer {request.language} pages.")
    if request.country is not None:
        lines.append(f"Focus on the {request.country} region.")
    if request.safe_search is not SafeSearch.OFF:
        lines.append(f"Use {request.safe_search.value} safe search.")
    return "\n".join(lines)


def _json_payload(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    fenced = _FENCED_JSON.search(stripped)
    if fenced is not None:
        stripped = fenced.group(1).strip()
    try:
        return json.loads(stripped)
    except ValueError:
        return None


def _as_results(value: Any) -> list[Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    if isinstance(value, Mapping):
        for key in _RESULT_KEYS:
            nested = value.get(key)
            if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
                return list(nested)
    return None


def _metadata_results(response: LLMResponse) -> list[Any] | None:
    for key in _RESULT_KEYS:
        results = _as_results(response.metadata.get(key))
        if results is not None:
            return results
    return None


def _tool_results(response: LLMResponse, tool_name: str) -> list[Any] | None:
    for call in response.tool_calls:
        if call.name != tool_name:
            continue
        results = _as_results(call.arguments)
        if results is not None:
            return results
    return None


def _content_results(response: LLMResponse) -> list[Any] | None:
    payload = _json_payload(response.content)
    if payload is None:
        return None
    return _as_results(payload)


def _first_text(item: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _url_from(item: Mapping[str, Any]) -> str:
    url = _first_text(item, ("url", "link", "uri"))
    if url:
        return url
    web = item.get("web")
    if isinstance(web, Mapping):
        return _first_text(web, ("url", "link", "uri"))
    return ""


def _parse_date(item: Mapping[str, Any]) -> datetime | None:
    value = _first_text(item, ("published_at", "published_date", "date"))
    if not value:
        return None
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None


class LLMSearchProvider(Provider):
    """A search provider that uses a wrapped LLM's native web search.

    ``llm`` is any object exposing an async ``complete`` method, normally an
    :class:`~aivia.providers.protocols.llm.LLMProvider`. ``provider`` is an
    alias for the same argument so configuration can name it either way.
    ``model`` and ``role`` are forwarded to the LLM request, while
    ``system_prompt`` and ``tool_name`` let an operator adapt the prompt and the
    native tool marker.

    ``max_results`` caps the number of pages requested from the model and is
    itself capped by :attr:`SearchRequest.max_results`.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        llm: _LLMCompleter | None = None,
        *,
        provider: _LLMCompleter | None = None,
        model: str | None = None,
        role: str = "default",
        system_prompt: str | None = None,
        tool_name: str = DEFAULT_TOOL_NAME,
        max_results: int | None = None,
        name: str | None = None,
        health_timeout_seconds: float | None = None,
        close_timeout_seconds: float | None = None,
    ) -> None:
        super().__init__(
            name=name,
            health_timeout_seconds=health_timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
        )
        if llm is not None and provider is not None and llm is not provider:
            raise TypeError("pass either llm or provider, not both")
        wrapped = llm if llm is not None else provider
        if wrapped is None:
            raise ConfigError(
                "llm search provider requires an llm provider",
                details={"field": "llm"},
            )
        if not isinstance(role, str) or not role.strip():
            raise ConfigError("role must be a non-empty string")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ConfigError("tool_name must be a non-empty string")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string or None")
        if max_results is not None:
            if isinstance(max_results, bool) or not isinstance(max_results, int):
                raise TypeError("max_results must be an integer or None")
            if max_results < 1:
                raise ConfigError("max_results must be positive")

        self._llm = _coerce_llm(wrapped, "llm")
        self._model = model.strip() if isinstance(model, str) else model
        self._role = role.strip()
        self._system_prompt = (
            system_prompt.strip() if system_prompt is not None else None
        ) or DEFAULT_SYSTEM_PROMPT
        self._tool_name = tool_name.strip()
        self._max_results = max_results

    @property
    def llm(self) -> _LLMCompleter:
        """The wrapped LLM provider."""
        return self._llm

    @property
    def tool_name(self) -> str:
        """The tool name that enables the wrapped provider's web search."""
        return self._tool_name

    def _tool(self) -> ToolDefinition:
        if self._tool_name == DEFAULT_TOOL_NAME:
            return WEB_SEARCH_TOOL
        return WEB_SEARCH_TOOL.model_copy(update={"name": self._tool_name})

    def _build_request(self, request: SearchRequest) -> LLMRequest:
        limit = request.max_results
        if self._max_results is not None:
            limit = min(limit, self._max_results)
        return LLMRequest(
            messages=[
                ChatMessage(role=MessageRole.SYSTEM, content=self._system_prompt),
                ChatMessage(
                    role=MessageRole.USER,
                    content=_user_prompt(request, limit),
                ),
            ],
            role=self._role,
            model=self._model,
            tools=[self._tool()],
            tool_choice=ToolChoice.AUTO,
            timeout_seconds=request.timeout_seconds,
            metadata={"web_search": True, "max_results": limit},
        )

    def _extract(self, response: LLMResponse) -> list[Any] | None:
        results = _metadata_results(response)
        if results is not None:
            return results
        results = _tool_results(response, self._tool_name)
        if results is not None:
            return results
        return _content_results(response)

    def _to_results(self, items: Sequence[Any], limit: int) -> list[SearchResult]:
        results: list[SearchResult] = []
        for item in items:
            if len(results) >= limit:
                break
            result = self._to_result(item)
            if result is not None:
                results.append(result)
        return results

    def _to_result(self, item: Any) -> SearchResult | None:
        if isinstance(item, SearchResult):
            return item
        if isinstance(item, str):
            text = item.strip()
            if not text:
                return None
            return SearchResult(url=text, source=self.name)
        if not isinstance(item, Mapping):
            raise ValidationError(
                "llm search returned a malformed result",
                details={"provider": self.name},
            )
        url = _url_from(item)
        if not url:
            raise ValidationError(
                "llm search returned a result without a url",
                details={"provider": self.name},
            )
        web = item.get("web")
        nested = web if isinstance(web, Mapping) else {}
        title = _first_text(item, ("title", "name")) or _first_text(
            nested, ("title", "name")
        )
        snippet = _first_text(
            item,
            ("snippet", "content", "text", "cited_text", "description"),
        )
        score = item.get("score")
        try:
            return SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                content=_first_text(item, ("content", "text")),
                published_at=_parse_date(item),
                score=score if isinstance(score, int | float) else None,
                source=self.name,
                metadata={
                    key: value
                    for key, value in item.items()
                    if key
                    in {
                        "published_at",
                        "published_date",
                        "date",
                        "author",
                        "type",
                    }
                },
            )
        except PydanticValidationError as exc:
            raise ValidationError(
                "llm search returned a malformed result",
                details={"provider": self.name},
            ) from exc

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Search the web through the wrapped LLM provider."""
        if not isinstance(request, SearchRequest):
            raise TypeError("request must be a SearchRequest")

        llm_request = self._build_request(request)
        response = await self._llm.complete(llm_request)
        if not isinstance(response, LLMResponse):
            raise ValidationError(
                "llm provider returned an unexpected response",
                details={"provider": self.name},
            )

        items = self._extract(response)
        if items is None:
            if _json_payload(response.content) is not None:
                raise ValidationError(
                    "llm search returned no parseable results",
                    details={"provider": self.name},
                )
            items = []

        limit = llm_request.metadata.get("max_results", request.max_results)
        results = self._to_results(items, int(limit))
        return SearchResponse(
            query=request.query,
            results=results,
            provider=self.name,
            answer=None if items else response.content or None,
            total_results=len(results),
            request_id=response.request_id,
            metadata={
                "model": response.model,
                "finish_reason": response.finish_reason.value,
                "usage": response.usage.model_dump(),
            },
        )


LLMWebSearchProvider = LLMSearchProvider
"""Alias for :class:`LLMSearchProvider`."""

LLMNativeSearchProvider = LLMSearchProvider
"""Alias for :class:`LLMSearchProvider`."""

LLMNativeProvider = LLMSearchProvider
"""Alias for :class:`LLMSearchProvider`."""

LLMNative = LLMSearchProvider
"""Alias for :class:`LLMSearchProvider`."""

NativeSearchProvider = LLMSearchProvider
"""Alias for :class:`LLMSearchProvider`."""

LLMSearch = LLMSearchProvider
"""Alias for :class:`LLMSearchProvider`."""

NativeSearch = LLMSearchProvider
"""Alias for :class:`LLMSearchProvider`."""

register_provider("llm_native", LLMSearchProvider, module=__name__)
register_provider("llm_search", LLMSearchProvider, module=__name__)
