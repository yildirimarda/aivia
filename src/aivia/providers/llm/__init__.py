"""LLM provider adapters.

Each adapter implements :class:`~aivia.providers.protocols.llm.LLMProvider` for
one vendor API and registers itself under a config provider type. Importing the
package makes every built-in adapter available to the provider registry.
"""

from aivia.providers.llm.anthropic import (
    DEFAULT_API_VERSION,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOKENS,
    Anthropic,
    AnthropicLLMProvider,
    AnthropicProvider,
)
from aivia.providers.llm.cost import (
    LLM_OPERATION,
    TOKEN_UNIT,
    TOKENS_PER_MILLION,
    LLMCostCalculator,
    LLMTokenCost,
    TokenCostCalculator,
    cost_event_for_response,
    token_cost_for_usage,
)
from aivia.providers.llm.gemini import (
    Gemini,
    GeminiLLMProvider,
    GeminiProvider,
)
from aivia.providers.llm.openai import (
    OpenAI,
    OpenAILLMProvider,
    OpenAIProvider,
)
from aivia.providers.llm.openai_compat import (
    OpenAICompat,
    OpenAICompatible,
    OpenAICompatibleLLMProvider,
    OpenAICompatibleProvider,
    OpenAICompatLLMProvider,
    OpenAICompatProvider,
)

__all__ = [
    "DEFAULT_API_VERSION",
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_TOKENS",
    "LLM_OPERATION",
    "TOKENS_PER_MILLION",
    "TOKEN_UNIT",
    "Anthropic",
    "AnthropicLLMProvider",
    "AnthropicProvider",
    "Gemini",
    "GeminiLLMProvider",
    "GeminiProvider",
    "LLMCostCalculator",
    "LLMTokenCost",
    "OpenAI",
    "OpenAICompat",
    "OpenAICompatLLMProvider",
    "OpenAICompatProvider",
    "OpenAICompatible",
    "OpenAICompatibleLLMProvider",
    "OpenAICompatibleProvider",
    "OpenAILLMProvider",
    "OpenAIProvider",
    "TokenCostCalculator",
    "cost_event_for_response",
    "token_cost_for_usage",
]
