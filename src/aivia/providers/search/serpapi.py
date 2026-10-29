"""SerpAPI search adapter.

The adapter translates the vendor-neutral
:class:`~aivia.providers.protocols.search.SearchRequest` into a SerpAPI Google
search request and maps the reply back into a
:class:`~aivia.providers.protocols.search.SearchResponse`. SerpAPI answers a
successful search with ``organic_results`` plus optional answer and knowledge
panels; the adapter keeps those fields typed so the research module never
handles a vendor payload directly.

SerpAPI reports a rejected request in the body with HTTP ``200`` for some
errors, so the adapter inspects both the status and the payload before parsing.
The provider owns no routing, retry or budget policy: it uses the shared HTTP
client for transport and maps failures to the shared error taxonomy.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
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
    SearchRecency,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from aivia.providers.registry import register_provider

__all__ = [
    "DEFAULT_BASE_URL",
    "SERPAPI_DEFAULT_BASE_URL",
    "SerpAPI",
    "SerpAPIProvider",
    "SerpAPISearchProvider",
]

DEFAULT_BASE_URL = "https://serpapi.com"
"""The SerpAPI API root, without a trailing slash."""

SERPAPI_DEFAULT_BASE_URL = DEFAULT_BASE_URL
"""Alias for :data:`DEFAULT_BASE_URL`, distinct from other search defaults."""

MAX_RESULTS = 100
"""The largest result count SerpAPI accepts for one Google search request."""

_DEFAULT_ENGINE = "google"

_RECENCY_FILTERS = {
    SearchRecency.DAY: "qdr:d",
    SearchRecency.WEEK: "qdr:w",
    SearchRecency.MONTH: "qdr:m",
    SearchRecency.YEAR: "qdr:y",
}

_QUOTA_MARKERS = (
    "quota",
    "plan limit",
    "usage limit",
    "credit limit",
    "run out",
    "out of searches",
    "rate limit",
    "too many requests",
    "exceeded your",
)

_DEFAULT_METADATA = ProviderMeta(
    name="serpapi",
    capabilities=[CAPABILITY_SEARCH],
)


class _SerpAPIResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    position: int | None = None
    title: str = ""
    link: str
    displayed_link: str | None = None
    snippet: str = ""
    date: str | None = None
    source: str | None = None
    favicon: str | None = None


class _SerpAPIAnswerBox(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    answer: str | None = None
    snippet: str | None = None
    link: str | None = None


class _SerpAPIKnowledgeGraph(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    description: str | None = None


class _SerpAPISearchInformation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query_displayed: str | None = None
    total_results: int | None = None
    organic_results_state: str | None = None


class _SerpAPISearchMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    status: str | None = None
    total_time_taken: float | None = None


class _SerpAPIResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    search_metadata: _SerpAPISearchMetadata = Field(
        default_factory=_SerpAPISearchMetadata
    )
    search_information: _SerpAPISearchInformation = Field(
        default_factory=_SerpAPISearchInformation
    )
    organic_results: list[_SerpAPIResult] = Field(default_factory=list)
    answer_box: _SerpAPIAnswerBox | None = None
    knowledge_graph: _SerpAPIKnowledgeGraph | None = None
    error: str | None = None


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
            raise ConfigError("serpapi API key must be non-empty")
        return text
    if api_key_ref is None or not isinstance(api_key_ref, str):
        raise ConfigError(
            "serpapi provider requires an API key or api_key_ref",
            details={"field": "api_key_ref"},
        )
    reference = api_key_ref.strip()
    if not reference:
        raise ConfigError(
            "serpapi provider requires an API key or api_key_ref",
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
        pass
    for pattern in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    return None


def _query_with_domains(request: SearchRequest) -> str:
    parts = [request.query]
    for domain in request.include_domains:
        parts.append(f"site:{domain}")
    for domain in request.exclude_domains:
        parts.append(f"-site:{domain}")
    return " ".join(parts)


def _date_range(start: date | None, end: date | None) -> str | None:
    if start is None and end is None:
        return None
    parts = ["cdr:1"]
    if start is not None:
        parts.append(f"cd_min:{start.month}/{start.day}/{start.year}")
    if end is not None:
        parts.append(f"cd_max:{end.month}/{end.day}/{end.year}")
    return ",".join(parts)


def _tbs(request: SearchRequest) -> str | None:
    date_range = _date_range(request.start_date, request.end_date)
    if date_range is not None:
        return date_range
    if request.recency is not None:
        return _RECENCY_FILTERS[request.recency]
    return None


def _error_payload(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, Mapping):
        return None
    for key in ("error", "message", "detail"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _is_quota_error(message: str | None) -> bool:
    text = (message or "").lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


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


def _error_for_message(message: str) -> AiviaError:
    text = message.lower()
    if _is_quota_error(message):
        return RateLimitError(message)
    if "api key" in text or "unauthorized" in text or "invalid key" in text:
        return AuthError(message)
    return InvalidRequestError(message)


class SerpAPISearchProvider(Provider):
    """A search provider backed by the SerpAPI Google search engine.

    ``api_key`` accepts an already resolved credential while ``api_key_ref``
    names a secret resolved through ``secrets`` (the environment, secrets file
    and keyring loader by default). ``base_url`` may point at a local test
    server; ``transport`` injects an ``httpx`` mock transport so the offline
    suite never touches the network.

    ``engine`` selects the SerpAPI engine and defaults to Google. It is exposed
    so an operator can route through another supported engine without a new
    adapter.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_ref: str | None = None,
        secrets: SecretsLoader | None = None,
        base_url: str = DEFAULT_BASE_URL,
        engine: str = _DEFAULT_ENGINE,
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
        if not isinstance(engine, str):
            raise TypeError("engine must be a string")
        if not engine.strip():
            raise ConfigError("engine must be non-empty")
        if transport is not None and not isinstance(
            transport, httpx.AsyncBaseTransport
        ):
            raise TypeError("transport must be an httpx.AsyncBaseTransport")

        self._api_key = _resolve_api_key(api_key, api_key_ref, secrets)
        self._base_url = _validate_base_url(base_url)
        self._engine = engine.strip()
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def base_url(self) -> str:
        """The configured API root, without a trailing slash."""
        return self._base_url

    @property
    def endpoint(self) -> str:
        """The search endpoint derived from :attr:`base_url`."""
        return f"{self._base_url}/search.json"

    @property
    def engine(self) -> str:
        """The SerpAPI engine used for every request."""
        return self._engine

    def _headers(self) -> dict[str, str]:
        return {"accept": "application/json"}

    def _build_params(self, request: SearchRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "engine": self._engine,
            "q": _query_with_domains(request),
            "api_key": self._api_key,
            "num": min(request.max_results, MAX_RESULTS),
        }
        if request.language is not None:
            params["hl"] = request.language
        if request.country is not None:
            params["gl"] = request.country
        params["safe"] = "off" if request.safe_search is SafeSearch.OFF else "active"
        tbs = _tbs(request)
        if tbs is not None:
            params["tbs"] = tbs
        return params

    def _raise_for_response(self, response: httpx.Response) -> None:
        fallback = error_for_response(response)
        message = _error_payload(response)
        if fallback is None and message is None:
            return

        details: dict[str, Any] = {
            "provider": self.name,
            "status": response.status_code,
        }
        request_id = response.headers.get("x-request-id")
        if request_id:
            details["request_id"] = request_id

        if fallback is not None:
            text = fallback.message
            if message is not None:
                text = f"{text}: {message}"
            error_class = _error_class(response.status_code)
            if _is_quota_error(message):
                raise RateLimitError(
                    text,
                    retry_after=getattr(fallback, "retry_after", None),
                    details=details,
                )
            if error_class is RateLimitError:
                raise RateLimitError(
                    text,
                    retry_after=getattr(fallback, "retry_after", None),
                    details=details,
                )
            if error_class is TimeoutError:
                raise TimeoutError(text, details=details)
            if error_class is not None:
                raise error_class(text, details=details)
            raise fallback

        assert message is not None
        error = _error_for_message(message)
        if isinstance(error, RateLimitError):
            raise RateLimitError(message, details=details)
        raise error

    def _to_response(
        self,
        parsed: _SerpAPIResponse,
        request: SearchRequest,
        response: httpx.Response,
    ) -> SearchResponse:
        try:
            results = [self._to_result(item) for item in parsed.organic_results]
        except PydanticValidationError as exc:
            raise ValidationError(
                "serpapi returned a malformed result",
                details={"provider": self.name},
            ) from exc

        metadata: dict[str, Any] = {"engine": self._engine}
        if parsed.search_information.organic_results_state:
            metadata["organic_results_state"] = (
                parsed.search_information.organic_results_state
            )
        if parsed.search_metadata.status:
            metadata["status"] = parsed.search_metadata.status

        request_id = parsed.search_metadata.id or response.headers.get("x-request-id")
        total = parsed.search_information.total_results
        return SearchResponse(
            query=parsed.search_information.query_displayed or request.query,
            results=results,
            provider=self.name,
            answer=self._answer(parsed),
            total_results=total if total is not None else len(results),
            took_ms=(
                parsed.search_metadata.total_time_taken * 1000.0
                if parsed.search_metadata.total_time_taken is not None
                else None
            ),
            request_id=request_id,
            metadata=metadata,
        )

    def _answer(self, parsed: _SerpAPIResponse) -> str | None:
        box = parsed.answer_box
        if box is not None:
            for value in (box.answer, box.snippet):
                if value and value.strip():
                    return value.strip()
        graph = parsed.knowledge_graph
        if graph is not None and graph.description and graph.description.strip():
            return graph.description.strip()
        return None

    def _to_result(self, item: _SerpAPIResult) -> SearchResult:
        metadata: dict[str, Any] = {}
        if item.position is not None:
            metadata["position"] = item.position
        if item.displayed_link:
            metadata["displayed_link"] = item.displayed_link
        if item.source:
            metadata["source"] = item.source
        if item.favicon:
            metadata["favicon"] = item.favicon
        if item.date:
            metadata["date"] = item.date

        return SearchResult(
            title=item.title,
            url=item.link,
            snippet=item.snippet,
            content="",
            published_at=_parse_published_date(item.date),
            source=self.name,
            metadata=metadata,
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute *request* against the SerpAPI Google engine."""
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
                "serpapi returned a non-JSON response",
                details={"provider": self.name},
            ) from exc
        if not isinstance(body, Mapping):
            raise ValidationError(
                "serpapi returned a malformed response",
                details={"provider": self.name},
            )
        try:
            parsed = _SerpAPIResponse.model_validate(body)
        except PydanticValidationError as exc:
            raise ValidationError(
                "serpapi returned a malformed response",
                details={"provider": self.name},
            ) from exc
        return self._to_response(parsed, request, response)


SerpAPIProvider = SerpAPISearchProvider
"""Alias for :class:`SerpAPISearchProvider`."""

SerpAPI = SerpAPISearchProvider
"""Alias for :class:`SerpAPISearchProvider`."""

register_provider("serpapi", SerpAPISearchProvider, module=__name__)
