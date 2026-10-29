"""Embedding provider adapters, batching, cost tracking and caching.

Each adapter implements
:class:`~aivia.providers.protocols.embeddings.EmbeddingProvider` for one
backend and registers itself under a config provider type. Importing this
package makes every built-in adapter available to the provider registry.
"""

from aivia.providers.embeddings.batch import (
    DEFAULT_BATCH_SIZE,
    BatchedEmbeddingProvider,
    batch_inputs,
    normalize_vector,
)
from aivia.providers.embeddings.cache import (
    DEFAULT_CACHE_SIZE,
    CachedEmbeddingProvider,
    EmbeddingCache,
    EmbeddingCacheKey,
    EmbeddingCacheStats,
)
from aivia.providers.embeddings.cost import (
    EMBEDDING_OPERATION,
    TOKEN_UNIT,
    TOKENS_PER_MILLION,
    EmbeddingCostCalculator,
    EmbeddingTokenCost,
    embedding_cost_event,
)
from aivia.providers.embeddings.local import (
    DEFAULT_DIMENSIONS,
    DEFAULT_MODEL,
    LocalEmbeddingProvider,
    LocalEmbeddings,
    LocalEmbeddingsProvider,
)
from aivia.providers.embeddings.openai import (
    OpenAIEmbeddingProvider,
    OpenAIEmbeddings,
    OpenAIEmbeddingsProvider,
)
from aivia.providers.embeddings.service import (
    EmbeddingOutcome,
    EmbeddingService,
    embed,
)
from aivia.providers.embeddings.voyage import (
    Voyage,
    VoyageEmbeddingProvider,
    VoyageEmbeddings,
)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CACHE_SIZE",
    "DEFAULT_DIMENSIONS",
    "DEFAULT_MODEL",
    "EMBEDDING_OPERATION",
    "TOKENS_PER_MILLION",
    "TOKEN_UNIT",
    "BatchedEmbeddingProvider",
    "CachedEmbeddingProvider",
    "EmbeddingCache",
    "EmbeddingCacheKey",
    "EmbeddingCacheStats",
    "EmbeddingCostCalculator",
    "EmbeddingOutcome",
    "EmbeddingService",
    "EmbeddingTokenCost",
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
]
