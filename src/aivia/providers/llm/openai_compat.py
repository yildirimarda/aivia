"""Generic adapter for OpenAI-compatible Chat Completions endpoints.

Many vendors and local runtimes expose the OpenAI Chat Completions shape
without being OpenAI: Ollama, vLLM, OpenRouter and hosted gateways. This
adapter reuses the OpenAI request and response translation but requires an
explicit ``base_url`` so it never silently calls the public OpenAI API, and it
derives the chat endpoint from that URL.

``base_url`` may name the API origin (``http://127.0.0.1:11434``), a versioned
root (``https://openrouter.ai/api/v1``) or the full endpoint
(``https://example.test/v1/chat/completions``). A versioned root gets
``/chat/completions`` appended; an origin gets ``/v1/chat/completions``. The
key is resolved from ``api_key_ref`` through the shared secrets loader and the
model comes from the request or the configured default.

Like every adapter it owns no routing, retry or budget policy: transport,
error mapping and usage reporting are inherited from the OpenAI adapter.
"""

from __future__ import annotations

from urllib.parse import urlsplit

import httpx

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import ConfigError
from aivia.core.models.provider import ProviderMeta
from aivia.core.secrets import SecretsLoader
from aivia.providers.llm.openai import DEFAULT_MAX_TOKENS, OpenAILLMProvider
from aivia.providers.protocols.llm import CAPABILITY_LLM
from aivia.providers.registry import register_provider

__all__ = [
    "OpenAICompat",
    "OpenAICompatLLMProvider",
    "OpenAICompatProvider",
    "OpenAICompatible",
    "OpenAICompatibleLLMProvider",
    "OpenAICompatibleProvider",
]

_DEFAULT_METADATA = ProviderMeta(
    name="openai_compatible",
    capabilities=[CAPABILITY_LLM],
)


def _chat_endpoint(base_url: str) -> str:
    """Return the Chat Completions endpoint for an OpenAI-compatible root."""
    path = urlsplit(base_url).path.rstrip("/")
    if path.endswith("/chat/completions"):
        return base_url
    version = path.rsplit("/", 1)[-1] if path else ""
    if version[:1] == "v" and version[1:].isdigit():
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


class OpenAICompatibleLLMProvider(OpenAILLMProvider):
    """An LLM provider for any endpoint speaking OpenAI Chat Completions.

    ``base_url`` is required because a generic provider has no sensible
    default host. ``api_key_ref`` names the secret resolved through ``secrets``;
    a resolved ``api_key`` is accepted too. ``default_model`` supplies the
    model when a request does not name one. ``transport`` injects an ``httpx``
    transport for offline tests.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_ref: str | None = None,
        secrets: SecretsLoader | None = None,
        default_model: str | None = None,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
        transport: httpx.AsyncBaseTransport | None = None,
        name: str | None = None,
        health_timeout_seconds: float | None = None,
        close_timeout_seconds: float | None = None,
        http_config: HttpClientConfig | None = None,
    ) -> None:
        if base_url is None:
            raise ConfigError(
                "openai_compatible provider requires a base_url",
                details={"field": "base_url"},
            )
        super().__init__(
            api_key=api_key,
            api_key_ref=api_key_ref,
            secrets=secrets,
            base_url=base_url,
            default_model=default_model,
            default_max_tokens=default_max_tokens,
            transport=transport,
            name=name,
            health_timeout_seconds=health_timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            http_config=http_config,
        )

    @property
    def endpoint(self) -> str:
        """The Chat Completions endpoint derived from :attr:`base_url`."""
        return _chat_endpoint(self.base_url)


OpenAICompatibleProvider = OpenAICompatibleLLMProvider
"""Alias for :class:`OpenAICompatibleLLMProvider`."""

OpenAICompatible = OpenAICompatibleLLMProvider
"""Alias for :class:`OpenAICompatibleLLMProvider`."""

OpenAICompatLLMProvider = OpenAICompatibleLLMProvider
"""Alias for :class:`OpenAICompatibleLLMProvider`."""

OpenAICompatProvider = OpenAICompatibleLLMProvider
"""Alias for :class:`OpenAICompatibleLLMProvider`."""

OpenAICompat = OpenAICompatibleLLMProvider
"""Alias for :class:`OpenAICompatibleLLMProvider`."""

register_provider("openai_compatible", OpenAICompatibleLLMProvider, module=__name__)
