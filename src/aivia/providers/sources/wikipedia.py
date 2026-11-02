"""Wikipedia topic source for search extracts and article links.

The adapter asks one Wikipedia language edition for search results and maps
each page into a vendor-neutral
:class:`~aivia.providers.protocols.source.SourceResponse`. It uses the
MediaWiki Action API, so a request can ask for the page extract and its URL in
one round trip instead of fetching every article again.

Language editions live on their own subdomain, for example ``en`` or ``tr``.
A language that has no Wikipedia edition resolves to a missing host or an
HTTP 404. That is reported as a :class:`~aivia.core.errors.RetryableError`
instead of a permanent invalid request so the research collector can fall
back to another source rather than failing the run.

The provider owns no routing, retry or budget policy. Transport and retry
behaviour come from the shared HTTP client and failures map to the shared
error taxonomy.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
from pydantic import ValidationError as PydanticValidationError

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import (
    ConfigError,
    InvalidRequestError,
    RateLimitError,
    RetryableError,
    TimeoutError,
    ValidationError,
)
from aivia.core.http import error_for_response
from aivia.core.models.provider import ProviderMeta
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.source import (
    CAPABILITY_SOURCE,
    SourceRequest,
    SourceResponse,
)
from aivia.providers.registry import register_provider
from aivia.providers.sources.base import SourceProviderBase, register_source

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_LANGUAGE",
    "DEFAULT_USER_AGENT",
    "MAX_RESULTS",
    "Wikipedia",
    "WikipediaAPI",
    "WikipediaProvider",
    "WikipediaSource",
    "WikipediaSourceProvider",
    "edition_url",
    "normalize_language",
]

DEFAULT_LANGUAGE = "en"
"""The language edition used when neither the provider nor request sets one."""

DEFAULT_BASE_URL = "https://{language}.wikipedia.org/w/api.php"
"""The Action API template; ``{language}`` becomes the edition code."""

DEFAULT_USER_AGENT = "aivia/0.1 (https://github.com/yildirimarda/aivia)"
"""A descriptive User-Agent, as the Wikimedia API etiquette requests."""

MAX_RESULTS = 50
"""The largest number of search results requested from one call."""

_LANGUAGE_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_TAG_PATTERN = re.compile(r"<[^>]+>")
_DEFAULT_METADATA = ProviderMeta(
    name="wikipedia",
    capabilities=[CAPABILITY_SOURCE],
)
_EDITION_HINTS = (
    "edition",
    "language",
    "wiki",
    "not found",
    "missing",
    "does not exist",
)


def normalize_language(value: str | None) -> str:
    """Return a lower-case Wikipedia edition code or raise :class:`ConfigError`.

    An underscore is accepted as a separator because request aliases use it,
    but the host name Wikipedia expects uses hyphens.
    """
    if value is None:
        return DEFAULT_LANGUAGE
    if not isinstance(value, str):
        raise TypeError("language must be a string")
    text = value.strip().lower().replace("_", "-")
    if not text:
        return DEFAULT_LANGUAGE
    if len(text) > 30 or not _LANGUAGE_PATTERN.fullmatch(text):
        raise ConfigError(
            f"invalid Wikipedia language edition: {value!r}",
            details={"field": "language"},
        )
    return text


def _validate_base_url(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("base_url must be a string")
    text = value.strip()
    if not text:
        raise ConfigError(
            "base_url must be a non-empty URL",
            details={"field": "base_url"},
        )
    # Validate the concrete form; a template is filled before this check.
    parsed = urlsplit(text.replace("{language}", DEFAULT_LANGUAGE))
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


def _endpoint_for(language: str, base_url: str) -> str:
    url = base_url.replace("{language}", language)
    if "api.php" not in urlsplit(url).path:
        url = f"{url.rstrip('/')}/w/api.php"
    return url


def edition_url(language: str | None = None, base_url: str | None = None) -> str:
    """Return the Action API URL for one language edition."""
    code = normalize_language(language)
    template = DEFAULT_BASE_URL if base_url is None else _validate_base_url(base_url)
    return _endpoint_for(code, template)


def _strip_html(value: str) -> str:
    return html.unescape(_TAG_PATTERN.sub("", value)).strip()


def _article_url(language: str, title: str) -> str:
    return (
        f"https://{language}.wikipedia.org/wiki/"
        f"{quote(title.replace(' ', '_'), safe='')}"
    )


def _page_url(page: Mapping[str, Any], language: str, title: str) -> str:
    for key in ("fullurl", "canonicalurl", "url"):
        value = page.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _article_url(language, title)


def _extract(page: Mapping[str, Any]) -> str:
    for key in ("extract", "excerpt", "description"):
        value = page.get(key)
        if isinstance(value, str) and value.strip():
            return _strip_html(value)
    return ""


def _page_items(body: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    query = body.get("query")
    if isinstance(query, Mapping):
        pages = query.get("pages")
        if isinstance(pages, Mapping):
            return [page for page in pages.values() if isinstance(page, Mapping)]
        if isinstance(pages, Sequence) and not isinstance(pages, (str, bytes)):
            return [page for page in pages if isinstance(page, Mapping)]
        search = query.get("search")
        if isinstance(search, Sequence) and not isinstance(search, (str, bytes)):
            return [item for item in search if isinstance(item, Mapping)]
        if pages is None and search is None:
            return []
    pages = body.get("pages")
    if isinstance(pages, Mapping):
        return [page for page in pages.values() if isinstance(page, Mapping)]
    if isinstance(pages, Sequence) and not isinstance(pages, (str, bytes)):
        return [page for page in pages if isinstance(page, Mapping)]
    search = body.get("search")
    if isinstance(search, Sequence) and not isinstance(search, (str, bytes)):
        return [item for item in search if isinstance(item, Mapping)]
    return []


def _error_details(
    response: httpx.Response,
    language: str,
    provider: str,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "provider": provider,
        "language": language,
        "status": response.status_code,
    }
    request_id = response.headers.get("x-request-id")
    if request_id:
        details["request_id"] = request_id
    return details


def _missing_edition(
    language: str,
    response: httpx.Response,
    provider: str,
) -> RetryableError:
    return RetryableError(
        f"Wikipedia edition not found: {language}",
        details=_error_details(response, language, provider),
    )


def _raise_for_response(
    response: httpx.Response,
    language: str,
    provider: str,
) -> None:
    status = response.status_code
    if status == 404:
        raise _missing_edition(language, response, provider)
    if 300 <= status < 400:
        location = response.headers.get("location", "").lower()
        if "incubator" in location or "missing.php" in location:
            raise _missing_edition(language, response, provider)

    fallback = error_for_response(response)
    if fallback is None:
        return
    details = _error_details(response, language, provider)
    if isinstance(fallback, RateLimitError):
        raise RateLimitError(
            fallback.message,
            retry_after=fallback.retry_after,
            details=details,
        )
    if isinstance(fallback, TimeoutError):
        raise TimeoutError(fallback.message, details=details)
    raise type(fallback)(fallback.message, details=details)


def _raise_api_error(
    error: Mapping[str, Any],
    language: str,
    provider: str,
) -> None:
    code = str(error.get("code", "")).strip()
    info = str(error.get("info", "")).strip()
    text = info or code or "Wikipedia API error"
    lowered = text.lower()
    if any(hint in lowered for hint in _EDITION_HINTS):
        raise RetryableError(
            f"Wikipedia edition not found: {language}",
            details={"provider": provider, "language": language, "error": text},
        )
    raise InvalidRequestError(
        f"Wikipedia rejected the request: {text}",
        details={"provider": provider, "language": language, "code": code},
    )


class WikipediaSourceProvider(SourceProviderBase):
    """A topic source backed by one Wikipedia language edition.

    ``language`` is the default edition. A request may override it with its own
    ``language`` field. ``base_url`` may be a concrete Action API URL or a
    template containing ``{language}``; ``transport`` injects an ``httpx`` mock
    transport so the offline suite never touches the network.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        language: str | None = None,
        base_url: str | None = None,
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
        if transport is not None and not isinstance(
            transport, httpx.AsyncBaseTransport
        ):
            raise TypeError("transport must be an httpx.AsyncBaseTransport")

        self._language = normalize_language(language)
        self._base_url = _validate_base_url(
            DEFAULT_BASE_URL if base_url is None else base_url
        )
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def language(self) -> str:
        """The default Wikipedia edition code."""
        return self._language

    @property
    def base_url(self) -> str:
        """The configured Action API template or URL."""
        return self._base_url

    @property
    def endpoint(self) -> str:
        """The Action API URL for the default language edition."""
        return self.endpoint_for(self._language)

    @property
    def api_url(self) -> str:
        """Alias for :attr:`endpoint`."""
        return self.endpoint

    def endpoint_for(self, language: str | None = None) -> str:
        """Return the Action API URL for *language* or the default edition."""
        code = self._language if language is None else normalize_language(language)
        return _endpoint_for(code, self._base_url)

    def _headers(self) -> dict[str, str]:
        return {
            "accept": "application/json",
            "user-agent": DEFAULT_USER_AGENT,
        }

    def _build_params(self, request: SourceRequest) -> dict[str, Any]:
        return {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrsearch": request.query,
            "gsrlimit": min(request.limit, MAX_RESULTS),
            "gsrnamespace": "0",
            "prop": "extracts|info",
            "exintro": "1",
            "explaintext": "1",
            "exlimit": "max",
            "inprop": "url",
            "redirects": "1",
            "utf8": "1",
        }

    def _to_response(
        self,
        body: Mapping[str, Any],
        request: SourceRequest,
        language: str,
        response: httpx.Response,
    ) -> SourceResponse:
        error = body.get("error")
        if isinstance(error, Mapping):
            _raise_api_error(error, language, self.name)

        pages = _page_items(body)
        if (
            not pages
            and not isinstance(body.get("query"), Mapping)
            and not ("pages" in body or "search" in body or "batchcomplete" in body)
        ):
            raise ValidationError(
                "wikipedia returned a malformed response",
                details={"provider": self.name, "language": language},
            )

        candidates: list[TopicCandidate] = []
        for page in pages:
            if "missing" in page or "invalid" in page:
                continue
            title = page.get("title")
            if not isinstance(title, str) or not title.strip():
                raise ValidationError(
                    "wikipedia returned a page without a title",
                    details={"provider": self.name, "language": language},
                )
            title = title.strip()
            index = page.get("index")
            rank = (
                max(0, index - 1)
                if isinstance(index, int) and not isinstance(index, bool)
                else len(candidates)
            )
            try:
                candidate = TopicCandidate(
                    title=title,
                    summary=_extract(page),
                    source_urls=[_page_url(page, language, title)],
                    mode=request.mode,
                    rank=rank,
                )
            except PydanticValidationError as exc:
                raise ValidationError(
                    "wikipedia returned a malformed page",
                    details={"provider": self.name, "language": language},
                ) from exc
            candidates.append(candidate)

        metadata: dict[str, Any] = {
            "language": language,
            "total": len(candidates),
        }
        request_id = response.headers.get("x-request-id")
        if request_id:
            metadata["request_id"] = request_id
        return SourceResponse(
            provider=self.name,
            query=request.query,
            candidates=candidates,
            request_id=request_id,
            metadata=metadata,
        )

    async def discover(self, request: SourceRequest) -> SourceResponse:
        """Discover Wikipedia articles matching *request*."""
        if not isinstance(request, SourceRequest):
            raise TypeError("request must be a SourceRequest")

        language = normalize_language(request.language or self._language)
        if not request.query:
            return SourceResponse(
                provider=self.name,
                query=request.query,
                metadata={"language": language, "total": 0},
            )

        response = await self.http.get(
            self.endpoint_for(language),
            headers=self._headers(),
            params=self._build_params(request),
            timeout=request.timeout_seconds,
        )
        _raise_for_response(response, language, self.name)

        try:
            body = response.json()
        except ValueError as exc:
            raise ValidationError(
                "wikipedia returned a non-JSON response",
                details={"provider": self.name, "language": language},
            ) from exc
        if not isinstance(body, Mapping):
            raise ValidationError(
                "wikipedia returned a malformed response",
                details={"provider": self.name, "language": language},
            )
        return self._to_response(body, request, language, response)


WikipediaSource = WikipediaSourceProvider
"""Alias for :class:`WikipediaSourceProvider`."""

WikipediaProvider = WikipediaSourceProvider
"""Alias for :class:`WikipediaSourceProvider`."""

WikipediaAPI = WikipediaSourceProvider
"""Alias for :class:`WikipediaSourceProvider`."""

Wikipedia = WikipediaSourceProvider
"""Alias for :class:`WikipediaSourceProvider`."""

register_provider("wikipedia", WikipediaSourceProvider, module=__name__)
register_source("wikipedia", WikipediaSourceProvider, module=__name__)
