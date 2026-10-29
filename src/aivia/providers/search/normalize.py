"""Normalize provider search payloads into one de-duplicated model.

Every search adapter answers with the same
:class:`~aivia.providers.protocols.search.SearchResponse`, so callers never
branch on a vendor shape. Providers are still allowed to
return the same page more than once, with a fragment, a different host casing
or a trailing slash. This module owns that policy: results are validated into
:class:`~aivia.providers.protocols.search.SearchResult` records and de-duplicated
by a canonical URL key, keeping the first occurrence in provider order.

De-duplication never rewrites the stored result URL. A result keeps the exact
URL the provider returned, while two spellings of the same page collapse to the
first one. Merging several responses uses the same rule, so a multi-provider
search cannot inflate its result count with the same page.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from aivia.providers.protocols.search import SearchResponse, SearchResult

__all__ = [
    "SearchNormalizer",
    "SearchResultNormalizer",
    "canonical_url",
    "dedupe_by_url",
    "dedupe_results",
    "dedupe_search_results",
    "merge_responses",
    "merge_search_responses",
    "normalize_response",
    "normalize_results",
    "normalize_search_response",
    "normalize_search_result",
    "normalize_search_results",
    "url_key",
]

_DEFAULT_PORTS: dict[str, int] = {"http": 80, "https": 443}


def _provider_name(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("provider must be a string or None")
    text = value.strip()
    return text or None


def canonical_url(url: str) -> str:
    """Return the key used to compare two result URLs.

    The scheme and host are lower-cased, a default port and the fragment are
    removed, and a trailing slash on a non-root path is dropped. Path case and
    the query string are preserved because they can identify different pages.
    A value that is not an absolute URL is returned stripped, which keeps the
    helper safe to call on data that has not been validated yet.
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")
    text = url.strip()
    if not text:
        return ""
    parts = urlsplit(text)
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if not scheme or not host:
        return text
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        return text
    if port is None or port == _DEFAULT_PORTS.get(scheme):
        netloc = host
    else:
        netloc = f"{host}:{port}"
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, parts.query, ""))


url_key = canonical_url
"""Alias for :func:`canonical_url`."""


def _as_result(
    value: SearchResult | Mapping[str, Any],
    *,
    provider: str | None = None,
) -> SearchResult:
    if isinstance(value, SearchResult):
        result = value
    elif isinstance(value, Mapping):
        result = SearchResult.model_validate(value)
    else:
        raise TypeError(
            "search results must be SearchResult instances or mappings, "
            f"not {type(value).__name__}"
        )
    if provider and result.source is None:
        return result.model_copy(update={"source": provider})
    return result


def normalize_search_result(
    result: SearchResult | Mapping[str, Any],
    *,
    provider: str | None = None,
) -> SearchResult:
    """Validate one result and fill its source from *provider* when missing."""
    return _as_result(result, provider=_provider_name(provider))


def dedupe_results(results: Iterable[SearchResult]) -> list[SearchResult]:
    """Return *results* without duplicate URLs, keeping the first occurrence."""
    unique: list[SearchResult] = []
    seen: set[str] = set()
    for result in results:
        if not isinstance(result, SearchResult):
            raise TypeError(
                "results must contain SearchResult instances, "
                f"not {type(result).__name__}"
            )
        key = canonical_url(result.url)
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


dedupe_search_results = dedupe_results
"""Alias for :func:`dedupe_results`."""

dedupe_by_url = dedupe_results
"""Alias for :func:`dedupe_results`."""


def normalize_search_results(
    results: Iterable[SearchResult | Mapping[str, Any]],
    *,
    provider: str | None = None,
    dedupe: bool = True,
) -> list[SearchResult]:
    """Validate every result, then de-duplicate by canonical URL."""
    name = _provider_name(provider)
    normalized = [_as_result(result, provider=name) for result in results]
    return dedupe_results(normalized) if dedupe else normalized


def normalize_search_response(
    response: SearchResponse | Mapping[str, Any],
    *,
    provider: str | None = None,
    dedupe: bool = True,
) -> SearchResponse:
    """Return *response* with validated, de-duplicated results.

    The provider's reported ``total_results`` is left untouched because it
    describes what the engine found, not what this response returned. It is
    filled from the result count only when the provider omitted it.
    """
    if isinstance(response, SearchResponse):
        value = response
    elif isinstance(response, Mapping):
        value = SearchResponse.model_validate(response)
    else:
        raise TypeError(
            "response must be a SearchResponse or mapping, "
            f"not {type(response).__name__}"
        )
    name = _provider_name(provider)
    results = normalize_search_results(
        value.results,
        provider=name or value.provider,
        dedupe=dedupe,
    )
    updates: dict[str, Any] = {"results": results}
    if name is not None:
        updates["provider"] = name
    if value.total_results is None:
        updates["total_results"] = len(results)
    return value.model_copy(update=updates)


def normalize_results(
    results: Iterable[SearchResult | Mapping[str, Any]],
    *,
    provider: str | None = None,
    dedupe: bool = True,
) -> list[SearchResult]:
    """Alias for :func:`normalize_search_results`."""
    return normalize_search_results(results, provider=provider, dedupe=dedupe)


def merge_search_responses(
    responses: Sequence[SearchResponse | Mapping[str, Any]],
    *,
    query: str | None = None,
    provider: str | None = None,
    dedupe: bool = True,
) -> SearchResponse:
    """Merge several provider responses into one de-duplicated response.

    The first response supplies the query unless *query* is given. Each
    provider's results are normalized before they are concatenated, so a page
    returned by two engines appears once. ``metadata["providers"]`` records the
    engines that contributed so the merge stays explainable.
    """
    values = list(responses)
    if not values:
        if query is None:
            raise ValueError("merging search responses needs at least one response")
        return SearchResponse(query=query, provider=_provider_name(provider) or "")

    normalized = [normalize_search_response(value, dedupe=dedupe) for value in values]
    chosen_query = query if query is not None else normalized[0].query
    results: list[SearchResult] = []
    providers: list[str] = []
    total: int | None = 0
    for value in normalized:
        results.extend(value.results)
        if value.provider and value.provider not in providers:
            providers.append(value.provider)
        if value.total_results is None:
            total = None
        elif total is not None:
            total += value.total_results

    name = _provider_name(provider)
    if name is None:
        name = providers[0] if len(providers) == 1 else ",".join(providers)
    metadata: dict[str, Any] = {}
    if providers:
        metadata["providers"] = providers
    merged = dedupe_results(results) if dedupe else results
    return SearchResponse(
        query=chosen_query,
        results=merged,
        provider=name,
        total_results=total,
        metadata=metadata,
    )


def normalize_response(
    response: SearchResponse | Mapping[str, Any],
    *,
    provider: str | None = None,
    dedupe: bool = True,
) -> SearchResponse:
    """Alias for :func:`normalize_search_response`."""
    return normalize_search_response(response, provider=provider, dedupe=dedupe)


def merge_responses(
    responses: Sequence[SearchResponse | Mapping[str, Any]],
    *,
    query: str | None = None,
    provider: str | None = None,
    dedupe: bool = True,
) -> SearchResponse:
    """Alias for :func:`merge_search_responses`."""
    return merge_search_responses(
        responses,
        query=query,
        provider=provider,
        dedupe=dedupe,
    )


class SearchNormalizer:
    """Reusable normalization policy for a search service.

    ``dedupe`` defaults to ``True``; a caller that needs the raw provider order
    can turn it off for one response without rebuilding the normalizer.
    """

    def __init__(self, *, dedupe: bool = True) -> None:
        self._dedupe = bool(dedupe)

    @property
    def dedupe(self) -> bool:
        """Whether responses are de-duplicated by default."""
        return self._dedupe

    def result(
        self,
        result: SearchResult | Mapping[str, Any],
        *,
        provider: str | None = None,
    ) -> SearchResult:
        """Normalize one result."""
        return normalize_search_result(result, provider=provider)

    def results(
        self,
        results: Iterable[SearchResult | Mapping[str, Any]],
        *,
        provider: str | None = None,
    ) -> list[SearchResult]:
        """Normalize and optionally de-duplicate a result sequence."""
        return normalize_search_results(
            results,
            provider=provider,
            dedupe=self._dedupe,
        )

    def response(
        self,
        response: SearchResponse | Mapping[str, Any],
        *,
        provider: str | None = None,
        dedupe: bool | None = None,
    ) -> SearchResponse:
        """Normalize a response using this normalizer's dedupe default."""
        enabled = self._dedupe if dedupe is None else bool(dedupe)
        return normalize_search_response(
            response,
            provider=provider,
            dedupe=enabled,
        )

    def merge(
        self,
        responses: Sequence[SearchResponse | Mapping[str, Any]],
        *,
        query: str | None = None,
        provider: str | None = None,
    ) -> SearchResponse:
        """Merge several responses using this normalizer's dedupe default."""
        return merge_search_responses(
            responses,
            query=query,
            provider=provider,
            dedupe=self._dedupe,
        )

    def dedupe_results(self, results: Iterable[SearchResult]) -> list[SearchResult]:
        """De-duplicate results by canonical URL."""
        return dedupe_results(results)


SearchResultNormalizer = SearchNormalizer
"""Alias for :class:`SearchNormalizer`."""
