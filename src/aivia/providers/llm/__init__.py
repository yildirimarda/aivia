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

__all__ = [
    "DEFAULT_API_VERSION",
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_TOKENS",
    "Anthropic",
    "AnthropicLLMProvider",
    "AnthropicProvider",
    "Gemini",
    "GeminiLLMProvider",
    "GeminiProvider",
    "OpenAI",
    "OpenAILLMProvider",
    "OpenAIProvider",
]
