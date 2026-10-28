"""Typed capability protocols shared by providers, the router and modules.

The protocols are structural interfaces. A real adapter and a test fake are
interchangeable because both expose the same metadata, lifecycle and capability
method. Requests and responses are immutable pydantic models so they can be
validated, cached and logged without a provider depending on another module.
"""

from aivia.providers.protocols.audio import (
    AudioFormat,
    TranscriptSegment,
    WordTimestamp,
)
from aivia.providers.protocols.base import (
    CapabilityProvider,
    ProtocolModel,
    ensure_utc,
    normalize_domain,
    validate_http_url,
)
from aivia.providers.protocols.embeddings import (
    CAPABILITY_EMBEDDINGS,
    Embedding,
    EmbeddingInputType,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
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
from aivia.providers.protocols.stt import (
    CAPABILITY_STT,
    STTProvider,
    STTRequest,
    STTResponse,
)
from aivia.providers.protocols.tts import (
    CAPABILITY_TTS,
    TTSProvider,
    TTSRequest,
    TTSResponse,
)

__all__ = [
    "CAPABILITY_EMBEDDINGS",
    "CAPABILITY_LLM",
    "CAPABILITY_PAGE_FETCH",
    "CAPABILITY_SEARCH",
    "CAPABILITY_STT",
    "CAPABILITY_TTS",
    "AudioFormat",
    "CapabilityProvider",
    "ChatMessage",
    "ContentPart",
    "Embedding",
    "EmbeddingInputType",
    "EmbeddingProvider",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingUsage",
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
    "STTProvider",
    "STTRequest",
    "STTResponse",
    "SafeSearch",
    "SearchDepth",
    "SearchProvider",
    "SearchRecency",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "TTSProvider",
    "TTSRequest",
    "TTSResponse",
    "TextPart",
    "ToolCall",
    "ToolChoice",
    "ToolDefinition",
    "TranscriptSegment",
    "WordTimestamp",
    "ensure_utc",
    "normalize_domain",
    "validate_http_url",
]
