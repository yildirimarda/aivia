"""Tavily search API adapter.

The adapter translates the vendor-neutral
:class:`~aivia.providers.protocols.search.SearchRequest` into a Tavily search
request and maps the reply back into a
:class:`~aivia.providers.protocols.search.SearchResponse`. Tavily answers a
request with a list of pages, an optional answer and a response time; the
adapter keeps those fields typed so the research module never handles a vendor
payload directly.

The provider owns no routing, retry or budget policy. It uses the shared HTTP
client for transport and maps unsuccessful responses to the shared error
taxonomy. Tavily reports plan and pay-as-you-go quota exhaustion with its own
``432`` and ``433`` status codes; those are mapped to
:class:`~aivia.core.errors.RateLimitError` so the router can fall back to
another search provider instead of failing the whole run.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import (
    AiviaError,
    AuthError,
    ConfigError,
    InvalidRequestError,
    RateLimitError,
    RetryableError,
    TimeoutError,
    ValidationError,
)
from aivia.core.http import error_for_response
from aivia.core.models.provider import ProviderMeta
from aivia.core.secrets import SecretsLoader
from aivia.providers.base import Provider
from aivia.providers.protocols.search import (
    CAPABILITY_SEARCH,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from aivia.providers.registry import register_provider

__all__ = [
    "DEFAULT_BASE_URL",
    "Tavily",
    "TavilyProvider",
    "TavilySearchProvider",
]

DEFAULT_BASE_URL = "https://api.tavily.com"
"""The Tavily API root, without a trailing slash."""

MAX_RESULTS = 20
"""The largest result count Tavily accepts for one search request."""

_DEFAULT_METADATA = ProviderMeta(
    name="tavily",
    capabilities=[CAPABILITY_SEARCH],
)

_QUOTA_STATUS_CODES = frozenset({432, 433})
_QUOTA_MARKERS = (
    "quota",
    "plan limit",
    "usage limit",
    "credit limit",
    "paygo limit",
    "pay-as-you-go limit",
)


class _TavilyResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    url: str
    content: str = ""
    raw_content: str | None = None
    score: float | None = None
    published_date: str | None = None


class _TavilyResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = ""
    results: list[_TavilyResult] = Field(default_factory=list)
    answer: str | None = None
    response_time: float | None = None
    request_id: str | None = None


def _validate_base_url(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("base_url must be a string")
    text = value.strip().rstrip("/")
    parsed = urlsplit(text)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ConfigError(
            f"base_url must be an absolute http(s) URL: {value!r}",
            details={"field": "base_url"},
        )
    if parsed.username is not None or parsed.password is not None:
        raise ConfigError(
            "base_url must not contain credentials",
            details={"field": "base_url"},
        )
    return text


def _resolve_api_key(
    api_key: str | None,
    api_key_ref: str | None,
    secrets: SecretsLoader | None,
) -> str:
    if secrets is not None and not isinstance(secrets, SecretsLoader):
        raise TypeError("secrets must be a SecretsLoader")
    if api_key is not None:
        if not isinstance(api_key, str):
            raise TypeError("api_key must be a string")
        text = api_key.strip()
        if not text:
            raise ConfigError("tavily API key must be non-empty")
        return text
    if api_key_ref is None or not isinstance(api_key_ref, str):
        raise ConfigError(
            "tavily provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    reference = api_key_ref.strip()
    if not reference:
        raise ConfigError(
            "tavily provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    loader = secrets or SecretsLoader()
    return loader.get(reference)


def _parse_published_date(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    normalized = f"{text[:-1]}+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None


def _error_payload(
    response: httpx.Response,
) -> tuple[str | None, str | None]:
    try:
        body = response.json()
    except ValueError:
        return None, None
    if not isinstance(body, Mapping):
        return None, None

    message: str | None = None
    code: str | None = None
    for key in ("detail", "error", "message"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            message = value.strip()
            break
        if not isinstance(value, Mapping):
            continue
        for nested in ("error", "message", "detail", "code"):
            raw = value.get(nested)
            if isinstance(raw, str) and raw.strip():
                if nested == "code":
                    code = raw.strip()
                elif message is None:
                    message = raw.strip()
    return message, code


def _is_quota_error(
    status: int,
    message: str | None,
    code: str | None,
) -> bool:
    if status in _QUOTA_STATUS_CODES:
        return True
    text = " ".join(part for part in (message, code) if part).lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


def _error_class(status: int) -> type[AiviaError] | None:
    if status in {401, 403, 407}:
        return AuthError
    if status == 429:
        return RateLimitError
    if status in {408, 504}:
        return TimeoutError
    if status >= 500:
        return RetryableError
    if 400 <= status < 500:
        return InvalidRequestError
    return None


class TavilySearchProvider(Provider):
    """A search provider backed by the Tavily Search API.

    ``api_key`` accepts an already resolved credential while ``api_key_ref``
    names a secret resolved through ``secrets`` (the environment, secrets file
    and keyring loader by default). ``base_url`` may point at a local test
    server; ``transport`` injects an ``httpx`` mock transport so the offline
    suite never touches the network.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_ref: str | None = None,
        secrets: SecretsLoader | None = None,
        base_url: str = DEFAULT_BASE_URL,
        include_answer: bool = False,
        include_raw_content: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
        name: str | None = None,
        health_timeout_seconds: float | None = None,
        close_timeout_seconds: float | None = None,
        http_config: HttpClientConfig | None = None,
    ) -> None:
        super().__init__(
            name=name,
            health_timeout_seconds=health_timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            http_config=http_config,
        )
        if not isinstance(include_answer, bool):
            raise TypeError("include_answer must be a boolean")
        if not isinstance(include_raw_content, bool):
            raise TypeError("include_raw_content must be a boolean")
        if transport is not None and not isinstance(
            transport, httpx.AsyncBaseTransport
        ):
            raise TypeError("transport must be an httpx.AsyncBaseTransport")

        self._api_key = _resolve_api_key(api_key, api_key_ref, secrets)
        self._base_url = _validate_base_url(base_url)
        self._include_answer = include_answer
        self._include_raw_content = include_raw_content
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def base_url(self) -> str:
        """The configured API root, without a trailing slash."""
        return self._base_url

    @property
    def endpoint(self) -> str:
        """The search endpoint derived from :attr:`base_url`."""
        return f"{self._base_url}/search"

    def _headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
            "accept": "application/json",
        }

    def _build_payload(self, request: SearchRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "query": request.query,
            "max_results": min(request.max_results, MAX_RESULTS),
            "search_depth": request.search_depth.value,
        }
        if request.include_domains:
            body["include_domains"] = list(request.include_domains)
        if request.exclude_domains:
            body["exclude_domains"] = list(request.exclude_domains)
        if request.recency is not None:
            body["time_range"] = request.recency.value
        if request.start_date is not None:
            body["start_date"] = request.start_date.isoformat()
        if request.end_date is not None:
            body["end_date"] = request.end_date.isoformat()
        if request.country is not None:
            body["country"] = request.country
        if self._include_answer:
            body["include_answer"] = True
        if self._include_raw_content:
            body["include_raw_content"] = True
        return body

    def _raise_for_response(self, response: httpx.Response) -> None:
        fallback = error_for_response(response)
        if fallback is None:
            return

        message, code = _error_payload(response)
        details: dict[str, Any] = {
            "provider": self.name,
            "status": response.status_code,
        }
        if code is not None:
            details["code"] = code
        request_id = response.headers.get("x-request-id")
        if request_id:
            details["request_id"] = request_id

        text = fallback.message
        if message is not None:
            text = f"{text}: {message}"

        if _is_quota_error(response.status_code, message, code):
            retry_after = getattr(fallback, "retry_after", None)
            raise RateLimitError(
                text,
                retry_after=retry_after,
                details=details,
            )

        error_class = _error_class(response.status_code)
        if error_class is RateLimitError:
            retry_after = getattr(fallback, "retry_after", None)
            raise RateLimitError(
                text,
                retry_after=retry_after,
                details=details,
            )
        if error_class is TimeoutError:
            raise TimeoutError(text, details=details)
        if error_class is not None:
            raise error_class(text, details=details)
        raise fallback

    def _to_response(
        self,
        parsed: _TavilyResponse,
        request: SearchRequest,
        response: httpx.Response,
    ) -> SearchResponse:
        try:
            results = [
                SearchResult(
                    title=item.title,
                    url=item.url,
                    snippet=item.content,
                    content=item.raw_content or "",
                    published_at=_parse_published_date(item.published_date),
                    score=item.score,
                    source=self.name,
                )
                for item in parsed.results
            ]
        except PydanticValidationError as exc:
            raise ValidationError(
                "tavily returned a malformed result",
                details={"provider": self.name},
            ) from exc

        request_id = parsed.request_id or response.headers.get("x-request-id")
        return SearchResponse(
            query=parsed.query or request.query,
            results=results,
            provider=self.name,
            answer=parsed.answer,
            total_results=len(results),
            took_ms=(
                parsed.response_time * 1000.0
                if parsed.response_time is not None
                else None
            ),
            request_id=request_id,
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute *request* against the Tavily Search API."""
        if not isinstance(request, SearchRequest):
            raise TypeError("request must be a SearchRequest")

        response = await self.http.post(
            self.endpoint,
            headers=self._headers(),
            json=self._build_payload(request),
            timeout=request.timeout_seconds,
        )
        self._raise_for_response(response)

        try:
            body = response.json()
        except ValueError as exc:
            raise ValidationError(
                "tavily returned a non-JSON response",
                details={"provider": self.name},
            ) from exc
        if not isinstance(body, Mapping):
            raise ValidationError(
                "tavily returned a malformed response",
                details={"provider": self.name},
            )
        try:
            parsed = _TavilyResponse.model_validate(body)
        except PydanticValidationError as exc:
            raise ValidationError(
                "tavily returned a malformed response",
                details={"provider": self.name},
            ) from exc
        return self._to_response(parsed, request, response)


TavilyProvider = TavilySearchProvider
"""Alias for :class:`TavilySearchProvider`."""

Tavily = TavilySearchProvider
"""Alias for :class:`TavilySearchProvider`."""

register_provider("tavily", TavilySearchProvider, module=__name__)
