"""Shared provider contract and recorded golden fixtures for every LLM adapter.

The contract suite exercises lifecycle, metadata and health for each built-in
adapter. The golden fixtures under ``tests/fixtures/providers/<provider>/``
record the outgoing wire request and the vendor response so every adapter is
replayed offline through an ``httpx.MockTransport``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from aivia.core.models import ProviderMeta
from aivia.providers import Provider, provider_registry
from aivia.providers.llm import (
    AnthropicLLMProvider,
    GeminiLLMProvider,
    OpenAICompatibleLLMProvider,
    OpenAILLMProvider,
)
from aivia.providers.protocols import (
    CAPABILITY_LLM,
    LLMProvider,
    LLMRequest,
    LLMResponse,
)

from .provider_contract import ProviderContract

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "providers"
API_KEY = "test-key"

ProviderBuilder = Callable[[httpx.AsyncBaseTransport], LLMProvider]


@dataclass(frozen=True)
class AdapterSpec:
    """How to build one adapter offline and where its goldens live."""

    name: str
    model: str
    build: ProviderBuilder

    @property
    def fixture_dir(self) -> Path:
        return FIXTURES / self.name


def _anthropic(transport: httpx.AsyncBaseTransport) -> LLMProvider:
    return AnthropicLLMProvider(
        api_key=API_KEY,
        default_model="claude-test",
        transport=transport,
    )


def _openai(transport: httpx.AsyncBaseTransport) -> LLMProvider:
    return OpenAILLMProvider(
        api_key=API_KEY,
        default_model="gpt-test",
        transport=transport,
    )


def _gemini(transport: httpx.AsyncBaseTransport) -> LLMProvider:
    return GeminiLLMProvider(
        api_key=API_KEY,
        default_model="gemini-test",
        transport=transport,
    )


def _openai_compatible(transport: httpx.AsyncBaseTransport) -> LLMProvider:
    return OpenAICompatibleLLMProvider(
        base_url="https://llm.example.test",
        api_key=API_KEY,
        default_model="local-model",
        transport=transport,
    )


_ADAPTERS: dict[str, AdapterSpec] = {
    "anthropic": AdapterSpec("anthropic", "claude-test", _anthropic),
    "gemini": AdapterSpec("gemini", "gemini-test", _gemini),
    "openai": AdapterSpec("openai", "gpt-test", _openai),
    "openai_compatible": AdapterSpec(
        "openai_compatible",
        "local-model",
        _openai_compatible,
    ),
}


def _contract_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    return httpx.MockTransport(handler)


def _contract_provider(name: str) -> Provider:
    return cast(Provider, _ADAPTERS[name].build(_contract_transport()))


class TestAnthropicProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return _contract_provider("anthropic")


class TestGeminiProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return _contract_provider("gemini")


class TestOpenAIProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return _contract_provider("openai")


class TestOpenAICompatibleProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return _contract_provider("openai_compatible")


@dataclass(frozen=True)
class GoldenCase:
    """One recorded request and response for a single adapter."""

    provider_name: str
    path: Path
    data: dict[str, Any]

    @property
    def case_id(self) -> str:
        return f"{self.provider_name}-{self.path.stem}"


def _load_fixture(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _golden_cases() -> list[GoldenCase]:
    return [
        GoldenCase(name, path, _load_fixture(path))
        for name in sorted(_ADAPTERS)
        for path in sorted(_ADAPTERS[name].fixture_dir.glob("*.json"))
    ]


def test_every_registered_llm_provider_has_golden_fixtures() -> None:
    registered: set[str] = set()
    for registration in provider_registry.registrations():
        metadata = registration.provider.default_metadata
        if metadata is None:
            continue
        capabilities = (
            metadata.capabilities
            if isinstance(metadata, ProviderMeta)
            else metadata.get("capabilities", ())
        )
        if CAPABILITY_LLM in capabilities:
            registered.add(registration.name)

    assert registered == set(_ADAPTERS)
    for name in registered:
        assert list((FIXTURES / name).glob("*.json")), f"no goldens for {name}"


def test_every_fixture_records_a_request_response_and_capture_time() -> None:
    cases = _golden_cases()

    assert cases
    for case in cases:
        data = case.data
        assert set(data) == {
            "provider",
            "case",
            "captured_at",
            "input",
            "request",
            "response",
            "expected",
        }
        assert data["provider"] == case.provider_name
        assert datetime.fromisoformat(data["captured_at"]).tzinfo is not None
        assert data["request"]["method"] == "POST"
        assert data["request"]["url"].startswith("https://")
        assert isinstance(data["request"]["headers"], dict)
        assert isinstance(data["request"]["body"], dict)
        assert data["response"]["status"] == 200
        assert isinstance(data["response"]["headers"], dict)
        assert isinstance(data["response"]["body"], dict)
        LLMRequest.model_validate(data["input"])
        LLMResponse.model_validate(data["expected"])


@pytest.mark.parametrize("case", _golden_cases(), ids=lambda case: case.case_id)
def test_recorded_golden_fixture_replays_offline(case: GoldenCase) -> None:
    spec = _ADAPTERS[case.provider_name]
    data = case.data
    response_spec = data["response"]
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            response_spec["status"],
            headers=response_spec["headers"],
            json=response_spec["body"],
        )

    provider = spec.build(httpx.MockTransport(handler))
    request = LLMRequest.model_validate(data["input"])
    try:
        response = asyncio.run(provider.complete(request))
    finally:
        asyncio.run(provider.close())

    assert len(captured) == 1
    wire = captured[0]
    expected_request = data["request"]
    assert wire.method == expected_request["method"]
    assert str(wire.url) == expected_request["url"]
    for header, value in expected_request["headers"].items():
        assert wire.headers[header] == value
    assert json.loads(wire.content) == expected_request["body"]

    expected = LLMResponse.model_validate(data["expected"])
    assert response == expected
    assert response.total_tokens == (
        expected.usage.total_input_tokens + expected.usage.output_tokens
    )
