"""Typed capability protocols shared by providers, the router and modules.

The protocols are structural interfaces. A real adapter and a test fake are
interchangeable because both expose the same metadata, lifecycle and capability
method. Requests and responses are immutable pydantic models so they can be
validated, cached and logged without a provider depending on another module.
"""

from aivia.providers.protocols.base import (
    CapabilityProvider,
    ProtocolModel,
    ensure_utc,
    normalize_domain,
    validate_http_url,
)
from aivia.providers.protocols.llm import (
    CAPABILITY_LLM,
    ChatMessage,
    ContentPart,
    FinishReason,
    ImageDetail,
    ImagePart,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    MessageRole,
    MessageRoleEnum,
    TextPart,
    ToolCall,
    ToolChoice,
    ToolDefinition,
)
from aivia.providers.protocols.page_fetch import (
    CAPABILITY_PAGE_FETCH,
    PageFetchProvider,
    PageFetchRequest,
    PageFetchResponse,
    PageKind,
)
from aivia.providers.protocols.search import (
    CAPABILITY_SEARCH,
    SafeSearch,
    SearchDepth,
    SearchProvider,
    SearchRecency,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

__all__ = [
    "CAPABILITY_LLM",
    "CAPABILITY_PAGE_FETCH",
    "CAPABILITY_SEARCH",
    "CapabilityProvider",
    "ChatMessage",
    "ContentPart",
    "FinishReason",
    "ImageDetail",
    "ImagePart",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LLMUsage",
    "MessageRole",
    "MessageRoleEnum",
    "PageFetchProvider",
    "PageFetchRequest",
    "PageFetchResponse",
    "PageKind",
    "ProtocolModel",
    "SafeSearch",
    "SearchDepth",
    "SearchProvider",
    "SearchRecency",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "TextPart",
    "ToolCall",
    "ToolChoice",
    "ToolDefinition",
    "ensure_utc",
    "normalize_domain",
    "validate_http_url",
]
