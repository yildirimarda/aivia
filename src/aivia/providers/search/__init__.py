"""Search normalization, cost tracking and routed search execution.

Importing this package gives callers the normalizer, the cost calculator and
the service that executes a typed search request through the router. Vendor
adapters live in their own modules and are registered separately.
"""

from aivia.providers.search.cost import (
    SEARCH_OPERATION,
    SEARCH_UNIT,
    SearchCost,
    SearchCostCalculator,
    SearchPrice,
    search_cost_event,
)
from aivia.providers.search.exa import (
    EXA_DEFAULT_BASE_URL,
    Exa,
    ExaProvider,
    ExaSearchProvider,
)
from aivia.providers.search.normalize import (
    SearchNormalizer,
    SearchResultNormalizer,
    canonical_url,
    dedupe_by_url,
    dedupe_results,
    dedupe_search_results,
    merge_responses,
    merge_search_responses,
    normalize_response,
    normalize_results,
    normalize_search_response,
    normalize_search_result,
    normalize_search_results,
    url_key,
)
from aivia.providers.search.service import (
    RoutedSearch,
    SearchClient,
    SearchExecutor,
    SearchOutcome,
    SearchService,
    search,
)
from aivia.providers.search.tavily import (
    DEFAULT_BASE_URL,
    Tavily,
    TavilyProvider,
    TavilySearchProvider,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "EXA_DEFAULT_BASE_URL",
    "SEARCH_OPERATION",
    "SEARCH_UNIT",
    "Exa",
    "ExaProvider",
    "ExaSearchProvider",
    "RoutedSearch",
    "SearchClient",
    "SearchCost",
    "SearchCostCalculator",
    "SearchExecutor",
    "SearchNormalizer",
    "SearchOutcome",
    "SearchPrice",
    "SearchResultNormalizer",
    "SearchService",
    "Tavily",
    "TavilyProvider",
    "TavilySearchProvider",
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
    "search",
    "search_cost_event",
    "url_key",
]
