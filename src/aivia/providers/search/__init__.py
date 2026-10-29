"""Search normalization, cost tracking and routed search execution.

Importing this package gives callers the normalizer, the cost calculator and
the service that executes a typed search request through the router. Vendor
adapters live in their own modules and are registered separately.
"""

from aivia.providers.search.brave import (
    BRAVE_DEFAULT_BASE_URL,
    Brave,
    BraveProvider,
    BraveSearchProvider,
)
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
from aivia.providers.search.llm_search import (
    DEFAULT_TOOL_NAME,
    WEB_SEARCH_TOOL,
    LLMNative,
    LLMNativeProvider,
    LLMNativeSearchProvider,
    LLMSearch,
    LLMSearchProvider,
    LLMWebSearchProvider,
    NativeSearch,
    NativeSearchProvider,
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
from aivia.providers.search.serpapi import (
    SERPAPI_DEFAULT_BASE_URL,
    SerpAPI,
    SerpAPIProvider,
    SerpAPISearchProvider,
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
    "BRAVE_DEFAULT_BASE_URL",
    "DEFAULT_BASE_URL",
    "DEFAULT_TOOL_NAME",
    "EXA_DEFAULT_BASE_URL",
    "SEARCH_OPERATION",
    "SEARCH_UNIT",
    "SERPAPI_DEFAULT_BASE_URL",
    "WEB_SEARCH_TOOL",
    "Brave",
    "BraveProvider",
    "BraveSearchProvider",
    "Exa",
    "ExaProvider",
    "ExaSearchProvider",
    "LLMNative",
    "LLMNativeProvider",
    "LLMNativeSearchProvider",
    "LLMSearch",
    "LLMSearchProvider",
    "LLMWebSearchProvider",
    "NativeSearch",
    "NativeSearchProvider",
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
    "SerpAPI",
    "SerpAPIProvider",
    "SerpAPISearchProvider",
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
