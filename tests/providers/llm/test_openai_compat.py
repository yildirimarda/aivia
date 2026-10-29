"""Generic OpenAI-compatible provider: configuration and a local round trip."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

import pytest

from aivia.core.errors import (
    AuthError,
    ConfigError,
    InvalidRequestError,
    RateLimitError,
)
from aivia.core.secrets import SecretsLoader
from aivia.providers import provider_registry
from aivia.providers.llm import (
    OpenAICompat,
    OpenAICompatible,
    OpenAICompatibleLLMProvider,
    OpenAICompatibleProvider,
    OpenAICompatLLMProvider,
    OpenAICompatProvider,
)
from aivia.providers.protocols import (
    CAPABILITY_LLM,
    ChatMessage,
    FinishReason,
    LLMRequest,
    MessageRole,
)

API_KEY = "local-key"
MODEL = "local-model"
KEY_REFERENCE = "LOCAL_LLM_API_KEY"


def _completion_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl_local",
        "object": "chat.completion",
        "created": 1700000000,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from local"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 4,
            "total_tokens": 13,
        },
    }
    body.update(overrides)
    return body


def _request(**overrides: Any) -> LLMRequest:
    values: dict[str, Any] = {
        "messages": [ChatMessage(role=MessageRole.USER, content="Hello")],
        "model": MODEL,
    }
    values.update(overrides)
    return LLMRequest(**values)


@dataclass(frozen=True, slots=True)
class _RecordedRequest:
    path: str
    headers: dict[str, str]
    body: dict[str, Any]


class _MockServer:
    """A loopback HTTP server speaking the Chat Completions shape."""

    def __init__(self) -> None:
        self.requests: list[_RecordedRequest] = []
        self.status = 200
        self.body: dict[str, Any] = _completion_body()
        self.response_headers: dict[str, str] = {}
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        port = self._httpd.server_address[1]
        return f"http://127.0.0.1:{port}"

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(length)
                try:
                    decoded = json.loads(raw) if raw else {}
                except ValueError:
                    decoded = {}
                body = decoded if isinstance(decoded, dict) else {}
                server.requests.append(
                    _RecordedRequest(
                        path=self.path,
                        headers=dict(self.headers.items()),
                        body=body,
                    )
                )
                payload = json.dumps(server.body).encode("utf-8")
                self.send_response(server.status)
                self.send_header("content-type", "application/json")
                for name, value in server.response_headers.items():
                    self.send_header(name, value)
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                return None

        return Handler

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def server() -> Iterator[_MockServer]:
    mock = _MockServer()
    try:
        yield mock
    finally:
        mock.close()


def _provider(server: _MockServer, **kwargs: Any) -> OpenAICompatibleLLMProvider:
    loader = SecretsLoader(environ={f"AIVA_{KEY_REFERENCE}": API_KEY})
    options: dict[str, Any] = {
        "base_url": server.base_url,
        "api_key_ref": KEY_REFERENCE,
        "secrets": loader,
        "default_model": MODEL,
    }
    options.update(kwargs)
    return OpenAICompatibleLLMProvider(**options)


def test_metadata_and_registration_describe_the_provider() -> None:
    provider = OpenAICompatibleLLMProvider(
        base_url="http://127.0.0.1:9999",
        api_key=API_KEY,
    )

    assert provider.name == "openai_compatible"
    assert provider.capabilities == (CAPABILITY_LLM,)
    assert provider.supports(CAPABILITY_LLM)
    assert provider_registry.resolve("openai_compatible") is OpenAICompatibleLLMProvider
    assert OpenAICompatibleProvider is OpenAICompatibleLLMProvider
    assert OpenAICompatible is OpenAICompatibleLLMProvider
    assert OpenAICompatProvider is OpenAICompatibleLLMProvider
    assert OpenAICompatLLMProvider is OpenAICompatibleLLMProvider
    assert OpenAICompat is OpenAICompatibleLLMProvider


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://127.0.0.1:9999", "http://127.0.0.1:9999/v1/chat/completions"),
        ("http://127.0.0.1:9999/", "http://127.0.0.1:9999/v1/chat/completions"),
        (
            "https://openrouter.ai/api/v1",
            "https://openrouter.ai/api/v1/chat/completions",
        ),
        (
            "http://127.0.0.1:11434/v1/",
            "http://127.0.0.1:11434/v1/chat/completions",
        ),
        (
            "https://example.test/custom/chat/completions",
            "https://example.test/custom/chat/completions",
        ),
    ],
)
def test_the_chat_endpoint_is_derived_from_the_base_url(
    base_url: str,
    expected: str,
) -> None:
    provider = OpenAICompatibleLLMProvider(base_url=base_url, api_key=API_KEY)

    assert provider.base_url == base_url.rstrip("/")
    assert provider.endpoint == expected


def test_requires_a_base_url() -> None:
    with pytest.raises(ConfigError) as raised:
        OpenAICompatibleLLMProvider(api_key=API_KEY)

    assert raised.value.details["field"] == "base_url"


def test_missing_api_key_is_a_config_error() -> None:
    with pytest.raises(ConfigError):
        OpenAICompatibleLLMProvider(base_url="http://127.0.0.1:9999")


def test_a_request_without_a_model_is_rejected_before_sending(
    server: _MockServer,
) -> None:
    provider = OpenAICompatibleLLMProvider(
        base_url=server.base_url,
        api_key=API_KEY,
    )

    with pytest.raises(InvalidRequestError):
        asyncio.run(
            provider.complete(
                LLMRequest(messages=[ChatMessage(role=MessageRole.USER, content="Hi")])
            )
        )

    assert server.requests == []


def test_completes_against_a_local_openai_compatible_server(
    server: _MockServer,
) -> None:
    provider = _provider(server)

    response = asyncio.run(provider.complete(_request()))

    assert len(server.requests) == 1
    sent = server.requests[0]
    assert sent.path == "/v1/chat/completions"
    assert sent.headers["authorization"] == f"Bearer {API_KEY}"
    assert sent.headers["content-type"] == "application/json"
    assert sent.body["model"] == MODEL
    assert sent.body["messages"] == [{"role": "user", "content": "Hello"}]

    assert response.content == "Hello from local"
    assert response.model == MODEL
    assert response.provider == "openai_compatible"
    assert response.request_id == "chatcmpl_local"
    assert response.finish_reason is FinishReason.STOP
    assert response.usage.input_tokens == 9
    assert response.usage.output_tokens == 4
    assert response.usage.total_tokens == 13


def test_a_versioned_base_url_round_trips_without_duplicating_the_version(
    server: _MockServer,
) -> None:
    provider = _provider(server, base_url=f"{server.base_url}/api/v1")

    asyncio.run(provider.complete(_request()))

    assert server.requests[0].path == "/api/v1/chat/completions"


def test_a_request_model_overrides_the_configured_default(
    server: _MockServer,
) -> None:
    provider = _provider(server)

    asyncio.run(provider.complete(_request(model="other-model")))

    assert server.requests[0].body["model"] == "other-model"


def test_maps_an_auth_error_from_the_local_server(server: _MockServer) -> None:
    server.status = 401
    server.body = {
        "error": {
            "message": "bad key",
            "type": "invalid_api_key",
            "code": "invalid_api_key",
        }
    }
    provider = _provider(server)

    with pytest.raises(AuthError) as raised:
        asyncio.run(provider.complete(_request()))

    error = raised.value
    assert error.details["provider"] == "openai_compatible"
    assert error.details["status"] == 401
    assert error.details["type"] == "invalid_api_key"


def test_keeps_the_retry_after_delay_from_the_local_server(
    server: _MockServer,
) -> None:
    server.status = 429
    server.body = {
        "error": {
            "message": "slow down",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
        }
    }
    server.response_headers = {"retry-after": "2.5"}
    provider = _provider(server)

    with pytest.raises(RateLimitError) as raised:
        asyncio.run(provider.complete(_request()))

    assert raised.value.retry_after == 2.5
