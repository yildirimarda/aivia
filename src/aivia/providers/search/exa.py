"""Exa search API adapter.

The adapter translates the vendor-neutral
:class:`~aivia.providers.protocols.search.SearchRequest` into an Exa search
request and maps the reply back into a
:class:`~aivia.providers.protocols.search.SearchResponse`. Exa answers with a
list of pages plus an optional request id and server-side search time; the
adapter keeps those fields typed so the research module never handles a vendor
payload directly.

The provider owns no routing, retry or budget policy. It uses the shared HTTP
client for transport and maps unsuccessful responses to the shared error
taxonomy. Exa reports an exhausted credit balance with ``402``; that is mapped
to :class:`~aivia.core.errors.BudgetError` so the router can fall back to
another search provider instead of failing the whole run.

Exa only returns page text, highlights or a summary when ``contents`` asks for
them. The constructor enables those options explicitly so a plain search stays
cheap; highlights and summaries are still mapped into the normalized result
when a caller requests them.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
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
    SafeSearch,
    SearchDepth,
    SearchRecency,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from aivia.providers.registry import register_provider

__all__ = [
    "DEFAULT_BASE_URL",
    "EXA_DEFAULT_BASE_URL",
    "Exa",
    "ExaProvider",
    "ExaSearchProvider",
]

DEFAULT_BASE_URL = "https://api.exa.ai"
"""The Exa API root, without a trailing slash."""

EXA_DEFAULT_BASE_URL = DEFAULT_BASE_URL
"""Alias for :data:`DEFAULT_BASE_URL`, distinct from the Tavily default."""

MAX_RESULTS = 100
"""The largest result count Exa accepts for one search request."""

_SEARCH_TYPES = {
    SearchDepth.BASIC: "auto",
    SearchDepth.ADVANCED: "deep",
}
_RECENCY_DAYS = {
    SearchRecency.DAY: 1,
    SearchRecency.WEEK: 7,
    SearchRecency.MONTH: 30,
    SearchRecency.YEAR: 365,
}

_DEFAULT_METADATA = ProviderMeta(
    name="exa",
    capabilities=[CAPABILITY_SEARCH],
)


class _ExaResult(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str | None = None
    title: str = ""
    url: str
    published_date: str | None = Field(default=None, alias="publishedDate")
    author: str | None = None
    score: float | None = None
    text: str | None = None
    highlights: list[str] | None = None
    highlight_scores: list[float] | None = Field(
        default=None,
        alias="highlightScores",
    )
    summary: str | None = None
    image: str | None = None
    favicon: str | None = None


class _ExaResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    request_id: str | None = Field(default=None, alias="requestId")
    resolved_search_type: str | None = Field(
        default=None,
        alias="resolvedSearchType",
    )
    results: list[_ExaResult] = Field(default_factory=list)
    search_time: float | None = Field(default=None, alias="searchTime")
    cost_dollars: dict[str, Any] | None = Field(default=None, alias="costDollars")
    warnings: list[Any] | None = None


def _now() -> datetime:
    return datetime.now(UTC)


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
            raise ConfigError("exa API key must be non-empty")
        return text
    if api_key_ref is None or not isinstance(api_key_ref, str):
        raise ConfigError(
            "exa provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    reference = api_key_ref.strip()
    if not reference:
        raise ConfigError(
            "exa provider requires an API key or api_key_ref",
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


def _recency_start(value: SearchRecency) -> str:
    days = _RECENCY_DAYS[value]
    moment = _now()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return (moment.astimezone(UTC) - timedelta(days=days)).date().isoformat()


def _error_payload(
    response: httpx.Response,
) -> tuple[str | None, str | None, str | None]:
    try:
        body = response.json()
    except ValueError:
        return None, None, None
    if not isinstance(body, Mapping):
        return None, None, None

    message: str | None = None
    code: str | None = None
    for key in ("error", "detail", "message"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            message = value.strip()
            break
        if not isinstance(value, Mapping):
            continue
        for nested in ("error", "message", "detail", "tag", "code"):
            raw = value.get(nested)
            if isinstance(raw, str) and raw.strip():
                if nested in {"tag", "code"}:
                    code = raw.strip()
                elif message is None:
                    message = raw.strip()

    if code is None:
        raw_tag = body.get("tag")
        if isinstance(raw_tag, str) and raw_tag.strip():
            code = raw_tag.strip()
    raw_request_id = body.get("requestId")
    request_id = (
        raw_request_id.strip()
        if isinstance(raw_request_id, str) and raw_request_id.strip()
        else None
    )
    return message, code, request_id


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


class ExaSearchProvider(Provider):
    """A search provider backed by the Exa Search API.

    ``api_key`` accepts an already resolved credential while ``api_key_ref``
    names a secret resolved through ``secrets`` (the environment, secrets file
    and keyring loader by default). ``base_url`` may point at a local test
    server; ``transport`` injects an ``httpx`` mock transport so the offline
    suite never touches the network.

    ``search_type`` overrides the Exa mode derived from
    :attr:`SearchRequest.search_depth`. The content flags add Exa's ``contents``
    options, which are off by default because they cost extra.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_ref: str | None = None,
        secrets: SecretsLoader | None = None,
        base_url: str = DEFAULT_BASE_URL,
        search_type: str | None = None,
        include_text: bool = False,
        include_highlights: bool = False,
        include_summary: bool = False,
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
        if not isinstance(include_text, bool):
            raise TypeError("include_text must be a boolean")
        if not isinstance(include_highlights, bool):
            raise TypeError("include_highlights must be a boolean")
        if not isinstance(include_summary, bool):
            raise TypeError("include_summary must be a boolean")
        if transport is not None and not isinstance(
            transport, httpx.AsyncBaseTransport
        ):
            raise TypeError("transport must be an httpx.AsyncBaseTransport")
        if search_type is not None:
            if not isinstance(search_type, str):
                raise TypeError("search_type must be a string or None")
            search_type = search_type.strip()
            if not search_type:
                raise ConfigError("search_type must be non-empty")

        self._api_key = _resolve_api_key(api_key, api_key_ref, secrets)
        self._base_url = _validate_base_url(base_url)
        self._search_type = search_type
        self._include_text = include_text
        self._include_highlights = include_highlights
        self._include_summary = include_summary
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
            "x-api-key": self._api_key,
            "content-type": "application/json",
            "accept": "application/json",
        }

    def _build_payload(self, request: SearchRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "query": request.query,
            "numResults": min(request.max_results, MAX_RESULTS),
            "type": self._search_type or _SEARCH_TYPES[request.search_depth],
        }
        if request.include_domains:
            body["includeDomains"] = list(request.include_domains)
        if request.exclude_domains:
            body["excludeDomains"] = list(request.exclude_domains)
        if request.recency is not None:
            body["startPublishedDate"] = _recency_start(request.recency)
        if request.start_date is not None:
            body["startPublishedDate"] = request.start_date.isoformat()
        if request.end_date is not None:
            body["endPublishedDate"] = request.end_date.isoformat()
        if request.country is not None:
            body["userLocation"] = request.country
        if request.safe_search is not SafeSearch.OFF:
            body["moderation"] = True

        contents: dict[str, bool] = {}
        if self._include_text:
            contents["text"] = True
        if self._include_highlights:
            contents["highlights"] = True
        if self._include_summary:
            contents["summary"] = True
        if contents:
            body["contents"] = contents
        return body

    def _raise_for_response(self, response: httpx.Response) -> None:
        fallback = error_for_response(response)
        if fallback is None:
            return

        message, code, body_request_id = _error_payload(response)
        details: dict[str, Any] = {
            "provider": self.name,
            "status": response.status_code,
        }
        if code is not None:
            details["code"] = code
        request_id = body_request_id or response.headers.get("x-request-id")
        if request_id:
            details["request_id"] = request_id

        text = fallback.message
        if message is not None:
            text = f"{text}: {message}"

        error_class = _error_class(response.status_code)
        if error_class is None:
            raise fallback
        if error_class is RateLimitError:
            retry_after = getattr(fallback, "retry_after", None)
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
        parsed: _ExaResponse,
        request: SearchRequest,
        response: httpx.Response,
    ) -> SearchResponse:
        try:
            results = [self._to_result(item) for item in parsed.results]
        except PydanticValidationError as exc:
            raise ValidationError(
                "exa returned a malformed result",
                details={"provider": self.name},
            ) from exc

        request_id = parsed.request_id or response.headers.get("x-request-id")
        metadata: dict[str, Any] = {}
        if parsed.resolved_search_type:
            metadata["resolved_search_type"] = parsed.resolved_search_type
        if parsed.warnings:
            metadata["warnings"] = list(parsed.warnings)
        return SearchResponse(
            query=request.query,
            results=results,
            provider=self.name,
            total_results=len(results),
            took_ms=parsed.search_time,
            request_id=request_id,
            metadata=metadata,
        )

    def _to_result(self, item: _ExaResult) -> SearchResult:
        highlights = [
            highlight.strip()
            for highlight in (item.highlights or [])
            if highlight.strip()
        ]
        snippet = "\n\n".join(highlights)
        if not snippet and item.summary:
            snippet = item.summary.strip()

        metadata: dict[str, Any] = {}
        if item.id:
            metadata["id"] = item.id
        if item.author:
            metadata["author"] = item.author
        if item.summary:
            metadata["summary"] = item.summary
        if item.highlight_scores:
            metadata["highlight_scores"] = list(item.highlight_scores)
        if item.image:
            metadata["image"] = item.image
        if item.favicon:
            metadata["favicon"] = item.favicon

        return SearchResult(
            title=item.title,
            url=item.url,
            snippet=snippet,
            content=item.text or "",
            published_at=_parse_published_date(item.published_date),
            score=item.score,
            source=self.name,
            metadata=metadata,
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute *request* against the Exa Search API."""
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
                "exa returned a non-JSON response",
                details={"provider": self.name},
            ) from exc
        if not isinstance(body, Mapping):
            raise ValidationError(
                "exa returned a malformed response",
                details={"provider": self.name},
            )
        try:
            parsed = _ExaResponse.model_validate(body)
        except PydanticValidationError as exc:
            raise ValidationError(
                "exa returned a malformed response",
                details={"provider": self.name},
            ) from exc
        return self._to_response(parsed, request, response)


ExaProvider = ExaSearchProvider
"""Alias for :class:`ExaSearchProvider`."""

Exa = ExaSearchProvider
"""Alias for :class:`ExaSearchProvider`."""

register_provider("exa", ExaSearchProvider, module=__name__)
