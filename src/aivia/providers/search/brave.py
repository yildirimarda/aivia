"""Brave Search API adapter.

The adapter translates the vendor-neutral
:class:`~aivia.providers.protocols.search.SearchRequest` into a Brave Web
Search request and maps the reply back into a
:class:`~aivia.providers.protocols.search.SearchResponse`. Brave answers with a
``web`` object holding the page results; the adapter flattens that envelope so
the research module never handles a vendor payload directly.

The provider owns no routing, retry or budget policy. It uses the shared HTTP
client for transport and maps unsuccessful responses to the shared error
taxonomy. Brave reports an exhausted plan with a quota message (usually a
``429``); that is mapped to
:class:`~aivia.core.errors.RateLimitError` so the router can fall back to
another search provider.

Brave has no full-page text field. ``description`` becomes the snippet, while
the optional ``extra_snippets`` excerpts are joined into ``content`` when the
caller enables them.
"""

from __future__ import annotations

import time
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
    BudgetError,
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
    SearchRecency,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from aivia.providers.registry import register_provider

__all__ = [
    "BRAVE_DEFAULT_BASE_URL",
    "DEFAULT_BASE_URL",
    "Brave",
    "BraveProvider",
    "BraveSearchProvider",
]

DEFAULT_BASE_URL = "https://api.search.brave.com/res/v1"
"""The Brave Search API root, without a trailing slash."""

BRAVE_DEFAULT_BASE_URL = DEFAULT_BASE_URL
"""Alias for :data:`DEFAULT_BASE_URL`, distinct from other search defaults."""

MAX_RESULTS = 20
"""The largest result count Brave accepts for one web search request."""

_FRESHNESS = {
    SearchRecency.DAY: "pd",
    SearchRecency.WEEK: "pw",
    SearchRecency.MONTH: "pm",
    SearchRecency.YEAR: "py",
}

_QUOTA_MARKERS = (
    "quota",
    "plan limit",
    "usage limit",
    "credit limit",
    "subscription limit",
    "rate limit",
    "too many requests",
)

_DEFAULT_METADATA = ProviderMeta(
    name="brave",
    capabilities=[CAPABILITY_SEARCH],
)


class _BraveQuery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    original: str = ""
    more_results_available: bool | None = None


class _BraveProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    long_name: str | None = None
    img: str | None = None


class _BraveMetaUrl(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hostname: str | None = None
    favicon: str | None = None


class _BraveResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    url: str
    description: str = ""
    page_age: str | None = None
    age: str | None = None
    language: str | None = None
    profile: _BraveProfile | None = None
    meta_url: _BraveMetaUrl | None = None
    extra_snippets: list[str] | None = None


class _BraveWeb(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[_BraveResult] = Field(default_factory=list)


class _BraveResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str | None = None
    query: _BraveQuery = Field(default_factory=_BraveQuery)
    web: _BraveWeb | None = None


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
            raise ConfigError("brave API key must be non-empty")
        return text
    if api_key_ref is None or not isinstance(api_key_ref, str):
        raise ConfigError(
            "brave provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    reference = api_key_ref.strip()
    if not reference:
        raise ConfigError(
            "brave provider requires an API key or api_key_ref",
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


def _freshness(request: SearchRequest) -> str | None:
    if request.start_date is not None and request.end_date is not None:
        return f"{request.start_date.isoformat()}to{request.end_date.isoformat()}"
    if request.start_date is not None:
        return f"{request.start_date.isoformat()}to"
    if request.end_date is not None:
        return f"to{request.end_date.isoformat()}"
    if request.recency is not None:
        return _FRESHNESS[request.recency]
    return None


def _query_with_domains(request: SearchRequest) -> str:
    parts = [request.query]
    for domain in request.include_domains:
        parts.append(f"site:{domain}")
    for domain in request.exclude_domains:
        parts.append(f"-site:{domain}")
    return " ".join(parts)


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
    error = body.get("error")
    if isinstance(error, Mapping):
        detail = error.get("detail")
        if isinstance(detail, str) and detail.strip():
            message = detail.strip()
        raw_code = error.get("code")
        if isinstance(raw_code, str) and raw_code.strip():
            code = raw_code.strip()
    for key in ("detail", "message", "error"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            if message is None:
                message = value.strip()
            break
    return message, code


def _is_quota_error(status: int, message: str | None, code: str | None) -> bool:
    text = " ".join(part for part in (message, code) if part).lower()
    if any(marker in text for marker in _QUOTA_MARKERS):
        return True
    return status in {429, 432, 433}


def _error_class(status: int) -> type[AiviaError] | None:
    if status in {401, 403, 407}:
        return AuthError
    if status == 402:
        return BudgetError
    if status == 429:
        return RateLimitError
    if status in {408, 504}:
        return TimeoutError
    if status >= 500:
        return RetryableError
    if 400 <= status < 500:
        return InvalidRequestError
    return None


def _now() -> float:
    return time.time()


def _retry_after(
    response: httpx.Response,
    fallback: float | None,
) -> float | None:
    if fallback is not None:
        return fallback
    raw = response.headers.get("x-ratelimit-reset")
    if raw is None:
        return None
    try:
        reset = float(raw)
    except ValueError:
        return None
    return max(0.0, reset - _now())


class BraveSearchProvider(Provider):
    """A search provider backed by the Brave Web Search API.

    ``api_key`` accepts an already resolved credential while ``api_key_ref``
    names a secret resolved through ``secrets`` (the environment, secrets file
    and keyring loader by default). ``base_url`` may point at a local test
    server; ``transport`` injects an ``httpx`` mock transport so the offline
    suite never touches the network.

    ``extra_snippets`` asks Brave for up to five additional excerpts per
    result; it is off by default because it enlarges the response.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_ref: str | None = None,
        secrets: SecretsLoader | None = None,
        base_url: str = DEFAULT_BASE_URL,
        extra_snippets: bool = False,
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
        if not isinstance(extra_snippets, bool):
            raise TypeError("extra_snippets must be a boolean")
        if transport is not None and not isinstance(
            transport, httpx.AsyncBaseTransport
        ):
            raise TypeError("transport must be an httpx.AsyncBaseTransport")

        self._api_key = _resolve_api_key(api_key, api_key_ref, secrets)
        self._base_url = _validate_base_url(base_url)
        self._extra_snippets = extra_snippets
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def base_url(self) -> str:
        """The configured API root, without a trailing slash."""
        return self._base_url

    @property
    def endpoint(self) -> str:
        """The web search endpoint derived from :attr:`base_url`."""
        return f"{self._base_url}/web/search"

    def _headers(self) -> dict[str, str]:
        return {
            "x-subscription-token": self._api_key,
            "accept": "application/json",
            "accept-encoding": "gzip",
        }

    def _build_params(self, request: SearchRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "q": _query_with_domains(request),
            "count": min(request.max_results, MAX_RESULTS),
            "safesearch": request.safe_search.value,
        }
        if request.country is not None:
            params["country"] = request.country
        if request.language is not None:
            params["search_lang"] = request.language
        freshness = _freshness(request)
        if freshness is not None:
            params["freshness"] = freshness
        if self._extra_snippets:
            params["extra_snippets"] = "true"
        return params

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
            retry_after = _retry_after(
                response,
                getattr(fallback, "retry_after", None),
            )
            raise RateLimitError(
                text,
                retry_after=retry_after,
                details=details,
            )

        error_class = _error_class(response.status_code)
        if error_class is None:
            raise fallback
        if error_class is RateLimitError:
            retry_after = _retry_after(
                response,
                getattr(fallback, "retry_after", None),
            )
            raise RateLimitError(
                text,
                retry_after=retry_after,
                details=details,
            )
        if error_class is TimeoutError:
            raise TimeoutError(text, details=details)
        raise error_class(text, details=details)

    def _to_response(
        self,
        parsed: _BraveResponse,
        request: SearchRequest,
        response: httpx.Response,
    ) -> SearchResponse:
        raw_results = parsed.web.results if parsed.web is not None else []
        try:
            results = [self._to_result(item) for item in raw_results]
        except PydanticValidationError as exc:
            raise ValidationError(
                "brave returned a malformed result",
                details={"provider": self.name},
            ) from exc

        metadata: dict[str, Any] = {}
        if parsed.query.more_results_available is not None:
            metadata["more_results_available"] = parsed.query.more_results_available
        return SearchResponse(
            query=parsed.query.original or request.query,
            results=results,
            provider=self.name,
            total_results=len(results),
            request_id=response.headers.get("x-request-id"),
            metadata=metadata,
        )

    def _to_result(self, item: _BraveResult) -> SearchResult:
        extra = [
            snippet.strip()
            for snippet in (item.extra_snippets or [])
            if snippet.strip()
        ]

        metadata: dict[str, Any] = {}
        if item.age:
            metadata["age"] = item.age
        if item.language:
            metadata["language"] = item.language
        if extra:
            metadata["extra_snippets"] = extra
        if item.profile is not None:
            profile: dict[str, Any] = {}
            if item.profile.name:
                profile["name"] = item.profile.name
            if item.profile.long_name:
                profile["long_name"] = item.profile.long_name
            if item.profile.img:
                profile["img"] = item.profile.img
            if profile:
                metadata["profile"] = profile
        if item.meta_url is not None:
            meta_url: dict[str, Any] = {}
            if item.meta_url.hostname:
                meta_url["hostname"] = item.meta_url.hostname
            if item.meta_url.favicon:
                meta_url["favicon"] = item.meta_url.favicon
            if meta_url:
                metadata["meta_url"] = meta_url

        return SearchResult(
            title=item.title,
            url=item.url,
            snippet=item.description,
            content="\n\n".join(extra),
            published_at=_parse_published_date(item.page_age),
            source=self.name,
            metadata=metadata,
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute *request* against the Brave Web Search API."""
        if not isinstance(request, SearchRequest):
            raise TypeError("request must be a SearchRequest")

        response = await self.http.get(
            self.endpoint,
            headers=self._headers(),
            params=self._build_params(request),
            timeout=request.timeout_seconds,
        )
        self._raise_for_response(response)

        try:
            body = response.json()
        except ValueError as exc:
            raise ValidationError(
                "brave returned a non-JSON response",
                details={"provider": self.name},
            ) from exc
        if not isinstance(body, Mapping):
            raise ValidationError(
                "brave returned a malformed response",
                details={"provider": self.name},
            )
        try:
            parsed = _BraveResponse.model_validate(body)
        except PydanticValidationError as exc:
            raise ValidationError(
                "brave returned a malformed response",
                details={"provider": self.name},
            ) from exc
        return self._to_response(parsed, request, response)


BraveProvider = BraveSearchProvider
"""Alias for :class:`BraveSearchProvider`."""

Brave = BraveSearchProvider
"""Alias for :class:`BraveSearchProvider`."""

register_provider("brave", BraveSearchProvider, module=__name__)
