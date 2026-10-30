"""Provider adapters, the shared base class and capability protocols."""

from typing import TYPE_CHECKING, Any

from aivia.providers.base import (
    HealthStatus,
    Provider,
    ProviderHealth,
    ProviderState,
)
from aivia.providers.console import (
    ConsoleNotificationProvider,
    ConsoleNotifier,
    console_notifier,
    default_notifier,
)
from aivia.providers.http_spec import (
    DEFAULT_RESULT_FIELDS,
    DeclarativeHttpProvider,
    HttpAsyncSpec,
    HTTPProvider,
    HttpProviderSpec,
    HttpRequestSpec,
    HttpResponseSpec,
    HttpSpec,
    HttpSpecEngine,
    HttpSpecError,
    HttpSpecProvider,
    HttpSpecResult,
    HttpSpecTemplateError,
    ProviderSpec,
    build_http_provider,
    extract_path,
    http_provider_from_spec,
    load_http_spec,
    load_http_spec_file,
    load_http_specs,
    parse_http_spec,
)
from aivia.providers.llm import (
    DEFAULT_API_VERSION,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOKENS,
    LLM_OPERATION,
    TOKEN_UNIT,
    TOKENS_PER_MILLION,
    Anthropic,
    AnthropicLLMProvider,
    AnthropicProvider,
    Gemini,
    GeminiLLMProvider,
    GeminiProvider,
    LLMCostCalculator,
    LLMTokenCost,
    OpenAI,
    OpenAICompat,
    OpenAICompatible,
    OpenAICompatibleLLMProvider,
    OpenAICompatibleProvider,
    OpenAICompatLLMProvider,
    OpenAICompatProvider,
    OpenAILLMProvider,
    OpenAIProvider,
    TokenCostCalculator,
    cost_event_for_response,
    token_cost_for_usage,
)
from aivia.providers.media import (
    AsyncJob,
    AsyncJobBackend,
    AsyncJobHelper,
    AsyncJobState,
    AsyncJobStatus,
    AsyncJobStore,
    AsyncJobTimeoutError,
    AsyncMediaJob,
    AsyncMediaJobHelper,
    DatabaseAsyncJobStore,
    InMemoryAsyncJobStore,
    MediaJob,
    MediaJobHelper,
)
from aivia.providers.mock import (
    MOCK_EPOCH,
    MockAll,
    MockCall,
    MockCharacterTrainer,
    MockDelivery,
    MockDeliveryProvider,
    MockEmbedding,
    MockEmbeddingProvider,
    MockEmbeddings,
    MockImage,
    MockImageEdit,
    MockImageEditProvider,
    MockImageProvider,
    MockLipSync,
    MockLipSyncProvider,
    MockLLM,
    MockLLMProvider,
    MockMusic,
    MockMusicProvider,
    MockNotifier,
    MockNotify,
    MockPageFetch,
    MockPageFetchProvider,
    MockProvider,
    MockPublish,
    MockPublisher,
    MockSearch,
    MockSearchProvider,
    MockSFX,
    MockSFXProvider,
    MockStock,
    MockStockProvider,
    MockStorage,
    MockStorageProvider,
    MockSTT,
    MockSTTProvider,
    MockTTS,
    MockTTSProvider,
    MockVideo,
    MockVideoProvider,
    build_mock_providers,
    mock_provider_for,
    register_mock_providers,
)
from aivia.providers.protocols import (
    CAPABILITY_EMBEDDINGS,
    CAPABILITY_IMAGE_EDIT,
    CAPABILITY_IMAGE_GEN,
    CAPABILITY_IMAGE_INPUT,
    CAPABILITY_LLM,
    CAPABILITY_MUSIC_GEN,
    CAPABILITY_PAGE_FETCH,
    CAPABILITY_SEARCH,
    CAPABILITY_SFX,
    CAPABILITY_STOCK,
    CAPABILITY_STT,
    CAPABILITY_TTS,
    CAPABILITY_VIDEO_GEN,
    AudioFormat,
    CapabilityProvider,
    ChatMessage,
    ContentPart,
    Embedding,
    EmbeddingInputType,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    FinishReason,
    ImageDetail,
    ImageEditOperation,
    ImageEditProvider,
    ImageEditRequest,
    ImageEditResponse,
    ImageFormat,
    ImagePart,
    ImageProvider,
    ImageRequest,
    ImageResponse,
    ImageResult,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    MediaDimensions,
    MediaInput,
    MessageRole,
    MusicProvider,
    MusicRequest,
    MusicResponse,
    MusicTrack,
    PageFetchProvider,
    PageFetchRequest,
    PageFetchResponse,
    PageKind,
    ProtocolModel,
    SafeSearch,
    SearchDepth,
    SearchProvider,
    SearchRecency,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SFXClip,
    SFXProvider,
    SFXRequest,
    SFXResponse,
    StockAsset,
    StockMediaKind,
    StockOrientation,
    StockProvider,
    StockRequest,
    StockResponse,
    StockResult,
    STTProvider,
    STTRequest,
    STTResponse,
    TextPart,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    TranscriptSegment,
    TTSProvider,
    TTSRequest,
    TTSResponse,
    VideoFormat,
    VideoProvider,
    VideoRequest,
    VideoResponse,
    VideoResult,
    WordTimestamp,
)
from aivia.providers.provider_factory import (
    ProviderFactory,
    build_provider,
    build_provider_instances,
    build_providers,
    build_providers_from_config,
    create_provider,
    create_providers,
    provider_from_config,
    provider_instances_from_config,
    providers_from_config,
)
from aivia.providers.registry import (
    ProviderRegistration,
    ProviderRegistry,
    provider_registry,
    register_provider,
    registry,
    resolve_provider,
)
from aivia.providers.storage_factory import (
    build_storage_provider,
    create_storage_provider,
    storage_provider_from_config,
)
from aivia.providers.storage_local import LocalStorage, LocalStorageProvider
from aivia.providers.storage_s3 import S3Storage, S3StorageProvider

if TYPE_CHECKING:
    from aivia.providers.embeddings import (
        BatchedEmbeddingProvider,
        CachedEmbeddingProvider,
        EmbeddingCache,
        EmbeddingCacheEntry,
        EmbeddingCacheKey,
        EmbeddingCacheStats,
        EmbeddingCostCalculator,
        EmbeddingOutcome,
        EmbeddingService,
        EmbeddingTokenCost,
        LocalEmbeddingProvider,
        LocalEmbeddings,
        LocalEmbeddingsProvider,
        OpenAIEmbeddingProvider,
        OpenAIEmbeddings,
        OpenAIEmbeddingsProvider,
        Voyage,
        VoyageEmbeddingProvider,
        VoyageEmbeddings,
        embedding_service_from_config,
        service_from_config,
    )
    from aivia.providers.image.fal import (
        Fal,
        FalImageProvider,
        FalProvider,
    )
    from aivia.providers.image.google import (
        Google,
        GoogleImage,
        GoogleImagen,
        GoogleImagenProvider,
        GoogleImageProvider,
        GoogleImages,
        GoogleImagesProvider,
        GoogleProvider,
    )
    from aivia.providers.image.openai import (
        OpenAIImage,
        OpenAIImageProvider,
        OpenAIImages,
        OpenAIImagesProvider,
    )
    from aivia.providers.image.replicate import (
        Replicate,
        ReplicateImageProvider,
        ReplicateProvider,
    )
    from aivia.providers.image_edit.aggregator import (
        AGGREGATOR_DEFAULT_BASE_URL,
        SUPPORTED_OPERATIONS,
        Aggregator,
        AggregatorImageEdit,
        AggregatorImageEditProvider,
        AggregatorProvider,
    )
    from aivia.providers.image_edit.rembg import (
        DEFAULT_MODEL as REMBG_DEFAULT_MODEL,
    )
    from aivia.providers.image_edit.rembg import (
        DEFAULT_TOLERANCE,
        LocalBackgroundRemoval,
        LocalBackgroundRemovalProvider,
        LocalBackgroundRemover,
        Rembg,
        RembgImageEdit,
        RembgImageEditProvider,
        RembgProvider,
        Remover,
        remove_background,
    )
    from aivia.providers.image_edit.remote_ops import (
        DEFAULT_SCALES,
        DEFAULT_UPSCALE_FACTORS,
        REMOTE_OPS_DEFAULT_BASE_URL,
        SUPPORTED_UPSCALE_FACTORS,
        RemoteBackgroundRemovalAndUpscale,
        RemoteBackgroundRemovalAndUpscaleProvider,
        RemoteImageEdit,
        RemoteImageEditProvider,
        RemoteImageOps,
        RemoteImageOpsProvider,
        RemoteOps,
        RemoteOpsImageEditProvider,
        RemoteOpsProvider,
    )
    from aivia.providers.pagefetch import (
        HttpPageFetchProvider,
        PageFetch,
        PageFetcher,
        ReadableContent,
        ReadablePageFetchProvider,
        WebPageFetchProvider,
        extract_main_text,
        extract_readable,
        host_is_allowed,
        is_blocked_ip,
        readability_extract,
    )
    from aivia.providers.search.brave import (
        Brave,
        BraveProvider,
        BraveSearchProvider,
    )
    from aivia.providers.search.exa import (
        Exa,
        ExaProvider,
        ExaSearchProvider,
    )
    from aivia.providers.search.llm_search import (
        LLMNative,
        LLMNativeProvider,
        LLMNativeSearchProvider,
        LLMSearch,
        LLMSearchProvider,
        LLMWebSearchProvider,
        NativeSearch,
        NativeSearchProvider,
    )
    from aivia.providers.search.serpapi import (
        SerpAPI,
        SerpAPIProvider,
        SerpAPISearchProvider,
    )
    from aivia.providers.search.tavily import (
        Tavily,
        TavilyProvider,
        TavilySearchProvider,
    )

_LAZY_EMBEDDING_PROVIDERS = frozenset(
    {
        "BatchedEmbeddingProvider",
        "CachedEmbeddingProvider",
        "EmbeddingCache",
        "EmbeddingCacheEntry",
        "EmbeddingCacheKey",
        "EmbeddingCacheStats",
        "EmbeddingCostCalculator",
        "EmbeddingOutcome",
        "EmbeddingService",
        "EmbeddingTokenCost",
        "embedding_service_from_config",
        "service_from_config",
        "LocalEmbeddingProvider",
        "LocalEmbeddings",
        "LocalEmbeddingsProvider",
        "OpenAIEmbeddingProvider",
        "OpenAIEmbeddings",
        "OpenAIEmbeddingsProvider",
        "Voyage",
        "VoyageEmbeddingProvider",
        "VoyageEmbeddings",
        "batch_inputs",
        "embed",
        "embedding_cost_event",
        "normalize_vector",
    }
)
_LAZY_BRAVE_PROVIDERS = frozenset({"Brave", "BraveProvider", "BraveSearchProvider"})
_LAZY_EXA_PROVIDERS = frozenset({"Exa", "ExaProvider", "ExaSearchProvider"})
_LAZY_FAL_PROVIDERS = frozenset({"Fal", "FalImageProvider", "FalProvider"})
_LAZY_GOOGLE_IMAGE_PROVIDERS = frozenset(
    {
        "Google",
        "GoogleImage",
        "GoogleImageProvider",
        "GoogleImages",
        "GoogleImagesProvider",
        "GoogleImagen",
        "GoogleImagenProvider",
        "GoogleProvider",
    }
)
_LAZY_OPENAI_IMAGE_PROVIDERS = frozenset(
    {
        "OpenAIImage",
        "OpenAIImageProvider",
        "OpenAIImages",
        "OpenAIImagesProvider",
    }
)
_LAZY_REPLICATE_PROVIDERS = frozenset(
    {"Replicate", "ReplicateImageProvider", "ReplicateProvider"}
)
_LAZY_AGGREGATOR_PROVIDERS = frozenset(
    {
        "AGGREGATOR_DEFAULT_BASE_URL",
        "SUPPORTED_OPERATIONS",
        "Aggregator",
        "AggregatorImageEdit",
        "AggregatorImageEditProvider",
        "AggregatorProvider",
    }
)
_LAZY_REMBG_PROVIDERS = frozenset(
    {
        "DEFAULT_TOLERANCE",
        "LocalBackgroundRemoval",
        "LocalBackgroundRemovalProvider",
        "LocalBackgroundRemover",
        "REMBG_DEFAULT_MODEL",
        "Rembg",
        "RembgImageEdit",
        "RembgImageEditProvider",
        "RembgProvider",
        "Remover",
        "remove_background",
    }
)
_LAZY_REMOTE_OPS_PROVIDERS = frozenset(
    {
        "DEFAULT_SCALES",
        "DEFAULT_UPSCALE_FACTORS",
        "REMOTE_OPS_DEFAULT_BASE_URL",
        "SUPPORTED_UPSCALE_FACTORS",
        "RemoteBackgroundRemovalAndUpscale",
        "RemoteBackgroundRemovalAndUpscaleProvider",
        "RemoteImageEdit",
        "RemoteImageEditProvider",
        "RemoteImageOps",
        "RemoteImageOpsProvider",
        "RemoteOps",
        "RemoteOpsImageEditProvider",
        "RemoteOpsProvider",
    }
)
_LAZY_LLM_SEARCH_PROVIDERS = frozenset(
    {
        "LLMNative",
        "LLMNativeProvider",
        "LLMNativeSearchProvider",
        "LLMSearch",
        "LLMSearchProvider",
        "LLMWebSearchProvider",
        "NativeSearch",
        "NativeSearchProvider",
    }
)
_LAZY_SERPAPI_PROVIDERS = frozenset(
    {"SerpAPI", "SerpAPIProvider", "SerpAPISearchProvider"}
)
_LAZY_TAVILY_PROVIDERS = frozenset({"Tavily", "TavilyProvider", "TavilySearchProvider"})
_LAZY_PAGE_FETCH_PROVIDERS = frozenset(
    {
        "HttpPageFetchProvider",
        "PageFetch",
        "PageFetcher",
        "ReadableContent",
        "ReadablePageFetchProvider",
        "WebPageFetchProvider",
        "extract_main_text",
        "extract_readable",
        "host_is_allowed",
        "is_blocked_ip",
        "readability_extract",
    }
)


def __getattr__(name: str) -> Any:
    """Load an adapter only when it is explicitly requested."""
    if name in _LAZY_EMBEDDING_PROVIDERS:
        from aivia.providers import embeddings as embeddings_package

        value = getattr(embeddings_package, name)
        globals()[name] = value
        return value
    if name in _LAZY_PAGE_FETCH_PROVIDERS:
        from aivia.providers.pagefetch import (
            HttpPageFetchProvider,
            PageFetch,
            PageFetcher,
            ReadableContent,
            ReadablePageFetchProvider,
            WebPageFetchProvider,
            extract_main_text,
            extract_readable,
            host_is_allowed,
            is_blocked_ip,
            readability_extract,
        )

        page_fetch_values = {
            "HttpPageFetchProvider": HttpPageFetchProvider,
            "PageFetch": PageFetch,
            "PageFetcher": PageFetcher,
            "ReadableContent": ReadableContent,
            "ReadablePageFetchProvider": ReadablePageFetchProvider,
            "WebPageFetchProvider": WebPageFetchProvider,
            "extract_main_text": extract_main_text,
            "extract_readable": extract_readable,
            "host_is_allowed": host_is_allowed,
            "is_blocked_ip": is_blocked_ip,
            "readability_extract": readability_extract,
        }
        page_fetch_value = page_fetch_values[name]
        globals()[name] = page_fetch_value
        return page_fetch_value
    if name in _LAZY_BRAVE_PROVIDERS:
        from aivia.providers.search.brave import (
            Brave,
            BraveProvider,
            BraveSearchProvider,
        )

        brave_values = {
            "Brave": Brave,
            "BraveProvider": BraveProvider,
            "BraveSearchProvider": BraveSearchProvider,
        }
        brave_value = brave_values[name]
        globals()[name] = brave_value
        return brave_value
    if name in _LAZY_EXA_PROVIDERS:
        from aivia.providers.search.exa import (
            Exa,
            ExaProvider,
            ExaSearchProvider,
        )

        exa_values = {
            "Exa": Exa,
            "ExaProvider": ExaProvider,
            "ExaSearchProvider": ExaSearchProvider,
        }
        exa_value = exa_values[name]
        globals()[name] = exa_value
        return exa_value
    if name in _LAZY_FAL_PROVIDERS:
        from aivia.providers.image.fal import (
            Fal,
            FalImageProvider,
            FalProvider,
        )

        fal_values = {
            "Fal": Fal,
            "FalImageProvider": FalImageProvider,
            "FalProvider": FalProvider,
        }
        fal_value = fal_values[name]
        globals()[name] = fal_value
        return fal_value
    if name in _LAZY_GOOGLE_IMAGE_PROVIDERS:
        from aivia.providers.image.google import (
            Google,
            GoogleImage,
            GoogleImagen,
            GoogleImagenProvider,
            GoogleImageProvider,
            GoogleImages,
            GoogleImagesProvider,
            GoogleProvider,
        )

        google_image_values = {
            "Google": Google,
            "GoogleImage": GoogleImage,
            "GoogleImageProvider": GoogleImageProvider,
            "GoogleImages": GoogleImages,
            "GoogleImagesProvider": GoogleImagesProvider,
            "GoogleImagen": GoogleImagen,
            "GoogleImagenProvider": GoogleImagenProvider,
            "GoogleProvider": GoogleProvider,
        }
        google_image_value = google_image_values[name]
        globals()[name] = google_image_value
        return google_image_value
    if name in _LAZY_OPENAI_IMAGE_PROVIDERS:
        from aivia.providers.image.openai import (
            OpenAIImage,
            OpenAIImageProvider,
            OpenAIImages,
            OpenAIImagesProvider,
        )

        openai_image_values = {
            "OpenAIImage": OpenAIImage,
            "OpenAIImageProvider": OpenAIImageProvider,
            "OpenAIImages": OpenAIImages,
            "OpenAIImagesProvider": OpenAIImagesProvider,
        }
        openai_image_value = openai_image_values[name]
        globals()[name] = openai_image_value
        return openai_image_value
    if name in _LAZY_REPLICATE_PROVIDERS:
        from aivia.providers.image.replicate import (
            Replicate,
            ReplicateImageProvider,
            ReplicateProvider,
        )

        replicate_values = {
            "Replicate": Replicate,
            "ReplicateImageProvider": ReplicateImageProvider,
            "ReplicateProvider": ReplicateProvider,
        }
        replicate_value = replicate_values[name]
        globals()[name] = replicate_value
        return replicate_value
    if name in _LAZY_AGGREGATOR_PROVIDERS:
        from aivia.providers.image_edit.aggregator import (
            AGGREGATOR_DEFAULT_BASE_URL,
            SUPPORTED_OPERATIONS,
            Aggregator,
            AggregatorImageEdit,
            AggregatorImageEditProvider,
            AggregatorProvider,
        )

        aggregator_values = {
            "AGGREGATOR_DEFAULT_BASE_URL": AGGREGATOR_DEFAULT_BASE_URL,
            "SUPPORTED_OPERATIONS": SUPPORTED_OPERATIONS,
            "Aggregator": Aggregator,
            "AggregatorImageEdit": AggregatorImageEdit,
            "AggregatorImageEditProvider": AggregatorImageEditProvider,
            "AggregatorProvider": AggregatorProvider,
        }
        aggregator_value = aggregator_values[name]
        globals()[name] = aggregator_value
        return aggregator_value
    if name in _LAZY_REMBG_PROVIDERS:
        from aivia.providers.image_edit.rembg import (
            DEFAULT_MODEL,
            DEFAULT_TOLERANCE,
            LocalBackgroundRemoval,
            LocalBackgroundRemovalProvider,
            LocalBackgroundRemover,
            Rembg,
            RembgImageEdit,
            RembgImageEditProvider,
            RembgProvider,
            Remover,
            remove_background,
        )

        rembg_values = {
            "DEFAULT_TOLERANCE": DEFAULT_TOLERANCE,
            "LocalBackgroundRemoval": LocalBackgroundRemoval,
            "LocalBackgroundRemovalProvider": LocalBackgroundRemovalProvider,
            "LocalBackgroundRemover": LocalBackgroundRemover,
            "REMBG_DEFAULT_MODEL": DEFAULT_MODEL,
            "Rembg": Rembg,
            "RembgImageEdit": RembgImageEdit,
            "RembgImageEditProvider": RembgImageEditProvider,
            "RembgProvider": RembgProvider,
            "Remover": Remover,
            "remove_background": remove_background,
        }
        rembg_value = rembg_values[name]
        globals()[name] = rembg_value
        return rembg_value
    if name in _LAZY_REMOTE_OPS_PROVIDERS:
        from aivia.providers.image_edit.remote_ops import (
            DEFAULT_SCALES,
            DEFAULT_UPSCALE_FACTORS,
            REMOTE_OPS_DEFAULT_BASE_URL,
            SUPPORTED_UPSCALE_FACTORS,
            RemoteBackgroundRemovalAndUpscale,
            RemoteBackgroundRemovalAndUpscaleProvider,
            RemoteImageEdit,
            RemoteImageEditProvider,
            RemoteImageOps,
            RemoteImageOpsProvider,
            RemoteOps,
            RemoteOpsImageEditProvider,
            RemoteOpsProvider,
        )

        remote_ops_values = {
            "DEFAULT_SCALES": DEFAULT_SCALES,
            "DEFAULT_UPSCALE_FACTORS": DEFAULT_UPSCALE_FACTORS,
            "REMOTE_OPS_DEFAULT_BASE_URL": REMOTE_OPS_DEFAULT_BASE_URL,
            "SUPPORTED_UPSCALE_FACTORS": SUPPORTED_UPSCALE_FACTORS,
            "RemoteBackgroundRemovalAndUpscale": (
                RemoteBackgroundRemovalAndUpscale
            ),
            "RemoteBackgroundRemovalAndUpscaleProvider": (
                RemoteBackgroundRemovalAndUpscaleProvider
            ),
            "RemoteImageEdit": RemoteImageEdit,
            "RemoteImageEditProvider": RemoteImageEditProvider,
            "RemoteImageOps": RemoteImageOps,
            "RemoteImageOpsProvider": RemoteImageOpsProvider,
            "RemoteOps": RemoteOps,
            "RemoteOpsImageEditProvider": RemoteOpsImageEditProvider,
            "RemoteOpsProvider": RemoteOpsProvider,
        }
        remote_ops_value = remote_ops_values[name]
        globals()[name] = remote_ops_value
        return remote_ops_value
    if name in _LAZY_LLM_SEARCH_PROVIDERS:
        from aivia.providers.search.llm_search import (
            LLMNative,
            LLMNativeProvider,
            LLMNativeSearchProvider,
            LLMSearch,
            LLMSearchProvider,
            LLMWebSearchProvider,
            NativeSearch,
            NativeSearchProvider,
        )

        llm_search_values = {
            "LLMNative": LLMNative,
            "LLMNativeProvider": LLMNativeProvider,
            "LLMNativeSearchProvider": LLMNativeSearchProvider,
            "LLMSearch": LLMSearch,
            "LLMSearchProvider": LLMSearchProvider,
            "LLMWebSearchProvider": LLMWebSearchProvider,
            "NativeSearch": NativeSearch,
            "NativeSearchProvider": NativeSearchProvider,
        }
        llm_search_value = llm_search_values[name]
        globals()[name] = llm_search_value
        return llm_search_value
    if name in _LAZY_SERPAPI_PROVIDERS:
        from aivia.providers.search.serpapi import (
            SerpAPI,
            SerpAPIProvider,
            SerpAPISearchProvider,
        )

        serpapi_values = {
            "SerpAPI": SerpAPI,
            "SerpAPIProvider": SerpAPIProvider,
            "SerpAPISearchProvider": SerpAPISearchProvider,
        }
        serpapi_value = serpapi_values[name]
        globals()[name] = serpapi_value
        return serpapi_value
    if name in _LAZY_TAVILY_PROVIDERS:
        from aivia.providers.search.tavily import (
            Tavily,
            TavilyProvider,
            TavilySearchProvider,
        )

        tavily_values = {
            "Tavily": Tavily,
            "TavilyProvider": TavilyProvider,
            "TavilySearchProvider": TavilySearchProvider,
        }
        tavily_value = tavily_values[name]
        globals()[name] = tavily_value
        return tavily_value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AGGREGATOR_DEFAULT_BASE_URL",
    "CAPABILITY_EMBEDDINGS",
    "CAPABILITY_IMAGE_EDIT",
    "CAPABILITY_IMAGE_GEN",
    "CAPABILITY_IMAGE_INPUT",
    "CAPABILITY_LLM",
    "CAPABILITY_MUSIC_GEN",
    "CAPABILITY_PAGE_FETCH",
    "CAPABILITY_SEARCH",
    "CAPABILITY_SFX",
    "CAPABILITY_STOCK",
    "CAPABILITY_STT",
    "CAPABILITY_TTS",
    "CAPABILITY_VIDEO_GEN",
    "DEFAULT_API_VERSION",
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_RESULT_FIELDS",
    "DEFAULT_SCALES",
    "DEFAULT_TOLERANCE",
    "DEFAULT_UPSCALE_FACTORS",
    "EMBEDDING_DEFAULT_BATCH_SIZE",
    "EMBEDDING_DEFAULT_CACHE_SIZE",
    "EMBEDDING_DEFAULT_DIMENSIONS",
    "EMBEDDING_DEFAULT_MODEL",
    "EMBEDDING_OPERATION",
    "EMBEDDING_TOKENS_PER_MILLION",
    "EMBEDDING_TOKEN_UNIT",
    "LLM_OPERATION",
    "MOCK_EPOCH",
    "REMBG_DEFAULT_MODEL",
    "REMOTE_OPS_DEFAULT_BASE_URL",
    "SUPPORTED_OPERATIONS",
    "SUPPORTED_UPSCALE_FACTORS",
    "TOKENS_PER_MILLION",
    "TOKEN_UNIT",
    "Aggregator",
    "AggregatorImageEdit",
    "AggregatorImageEditProvider",
    "AggregatorProvider",
    "Anthropic",
    "AnthropicLLMProvider",
    "AnthropicProvider",
    "AsyncJob",
    "AsyncJobBackend",
    "AsyncJobHelper",
    "AsyncJobState",
    "AsyncJobStatus",
    "AsyncJobStore",
    "AsyncJobTimeoutError",
    "AsyncMediaJob",
    "AsyncMediaJobHelper",
    "AudioFormat",
    "BatchedEmbeddingProvider",
    "Brave",
    "BraveProvider",
    "BraveSearchProvider",
    "CachedEmbeddingProvider",
    "CapabilityProvider",
    "ChatMessage",
    "ConsoleNotificationProvider",
    "ConsoleNotifier",
    "ContentPart",
    "DatabaseAsyncJobStore",
    "DeclarativeHttpProvider",
    "Embedding",
    "EmbeddingCache",
    "EmbeddingCacheEntry",
    "EmbeddingCacheKey",
    "EmbeddingCacheStats",
    "EmbeddingCostCalculator",
    "EmbeddingInputType",
    "EmbeddingOutcome",
    "EmbeddingProvider",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingService",
    "EmbeddingTokenCost",
    "EmbeddingUsage",
    "Exa",
    "ExaProvider",
    "ExaSearchProvider",
    "Fal",
    "FalImageProvider",
    "FalProvider",
    "FinishReason",
    "Gemini",
    "GeminiLLMProvider",
    "GeminiProvider",
    "Google",
    "GoogleImage",
    "GoogleImageProvider",
    "GoogleImagen",
    "GoogleImagenProvider",
    "GoogleImages",
    "GoogleImagesProvider",
    "GoogleProvider",
    "HTTPProvider",
    "HealthStatus",
    "HttpAsyncSpec",
    "HttpPageFetchProvider",
    "HttpProviderSpec",
    "HttpRequestSpec",
    "HttpResponseSpec",
    "HttpSpec",
    "HttpSpecEngine",
    "HttpSpecError",
    "HttpSpecProvider",
    "HttpSpecResult",
    "HttpSpecTemplateError",
    "ImageDetail",
    "ImageEditOperation",
    "ImageEditProvider",
    "ImageEditRequest",
    "ImageEditResponse",
    "ImageFormat",
    "ImagePart",
    "ImageProvider",
    "ImageRequest",
    "ImageResponse",
    "ImageResult",
    "InMemoryAsyncJobStore",
    "LLMCostCalculator",
    "LLMNative",
    "LLMNativeProvider",
    "LLMNativeSearchProvider",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LLMSearch",
    "LLMSearchProvider",
    "LLMTokenCost",
    "LLMUsage",
    "LLMWebSearchProvider",
    "LocalBackgroundRemoval",
    "LocalBackgroundRemovalProvider",
    "LocalBackgroundRemover",
    "LocalEmbeddingProvider",
    "LocalEmbeddings",
    "LocalEmbeddingsProvider",
    "LocalStorage",
    "LocalStorageProvider",
    "MediaDimensions",
    "MediaInput",
    "MediaJob",
    "MediaJobHelper",
    "MessageRole",
    "MockAll",
    "MockCall",
    "MockCharacterTrainer",
    "MockDelivery",
    "MockDeliveryProvider",
    "MockEmbedding",
    "MockEmbeddingProvider",
    "MockEmbeddings",
    "MockImage",
    "MockImageEdit",
    "MockImageEditProvider",
    "MockImageProvider",
    "MockLLM",
    "MockLLMProvider",
    "MockLipSync",
    "MockLipSyncProvider",
    "MockMusic",
    "MockMusicProvider",
    "MockNotifier",
    "MockNotify",
    "MockPageFetch",
    "MockPageFetchProvider",
    "MockProvider",
    "MockPublish",
    "MockPublisher",
    "MockSFX",
    "MockSFXProvider",
    "MockSTT",
    "MockSTTProvider",
    "MockSearch",
    "MockSearchProvider",
    "MockStock",
    "MockStockProvider",
    "MockStorage",
    "MockStorageProvider",
    "MockTTS",
    "MockTTSProvider",
    "MockVideo",
    "MockVideoProvider",
    "MusicProvider",
    "MusicRequest",
    "MusicResponse",
    "MusicTrack",
    "NativeSearch",
    "NativeSearchProvider",
    "OpenAI",
    "OpenAICompat",
    "OpenAICompatLLMProvider",
    "OpenAICompatProvider",
    "OpenAICompatible",
    "OpenAICompatibleLLMProvider",
    "OpenAICompatibleProvider",
    "OpenAIEmbeddingProvider",
    "OpenAIEmbeddings",
    "OpenAIEmbeddingsProvider",
    "OpenAIImage",
    "OpenAIImageProvider",
    "OpenAIImages",
    "OpenAIImagesProvider",
    "OpenAILLMProvider",
    "OpenAIProvider",
    "PageFetch",
    "PageFetchProvider",
    "PageFetchRequest",
    "PageFetchResponse",
    "PageFetcher",
    "PageKind",
    "ProtocolModel",
    "Provider",
    "ProviderFactory",
    "ProviderHealth",
    "ProviderRegistration",
    "ProviderRegistry",
    "ProviderSpec",
    "ProviderState",
    "ReadableContent",
    "ReadablePageFetchProvider",
    "Rembg",
    "RembgImageEdit",
    "RembgImageEditProvider",
    "RembgProvider",
    "RemoteBackgroundRemovalAndUpscale",
    "RemoteBackgroundRemovalAndUpscaleProvider",
    "RemoteImageEdit",
    "RemoteImageEditProvider",
    "RemoteImageOps",
    "RemoteImageOpsProvider",
    "RemoteOps",
    "RemoteOpsImageEditProvider",
    "RemoteOpsProvider",
    "Remover",
    "Replicate",
    "ReplicateImageProvider",
    "ReplicateProvider",
    "S3Storage",
    "S3StorageProvider",
    "SFXClip",
    "SFXProvider",
    "SFXRequest",
    "SFXResponse",
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
    "SerpAPI",
    "SerpAPIProvider",
    "SerpAPISearchProvider",
    "StockAsset",
    "StockMediaKind",
    "StockOrientation",
    "StockProvider",
    "StockRequest",
    "StockResponse",
    "StockResult",
    "TTSProvider",
    "TTSRequest",
    "TTSResponse",
    "Tavily",
    "TavilyProvider",
    "TavilySearchProvider",
    "TextPart",
    "TokenCostCalculator",
    "ToolCall",
    "ToolChoice",
    "ToolDefinition",
    "TranscriptSegment",
    "VideoFormat",
    "VideoProvider",
    "VideoRequest",
    "VideoResponse",
    "VideoResult",
    "Voyage",
    "VoyageEmbeddingProvider",
    "VoyageEmbeddings",
    "WebPageFetchProvider",
    "WordTimestamp",
    "batch_inputs",
    "build_http_provider",
    "build_mock_providers",
    "build_provider",
    "build_provider_instances",
    "build_providers",
    "build_providers_from_config",
    "build_storage_provider",
    "console_notifier",
    "cost_event_for_response",
    "create_provider",
    "create_providers",
    "create_storage_provider",
    "default_notifier",
    "embed",
    "embedding_cost_event",
    "embedding_service_from_config",
    "extract_main_text",
    "extract_path",
    "extract_readable",
    "host_is_allowed",
    "http_provider_from_spec",
    "is_blocked_ip",
    "load_http_spec",
    "load_http_spec_file",
    "load_http_specs",
    "mock_provider_for",
    "normalize_vector",
    "parse_http_spec",
    "provider_from_config",
    "provider_instances_from_config",
    "provider_registry",
    "providers_from_config",
    "readability_extract",
    "register_mock_providers",
    "register_provider",
    "registry",
    "remove_background",
    "resolve_provider",
    "service_from_config",
    "storage_provider_from_config",
    "token_cost_for_usage",
]
