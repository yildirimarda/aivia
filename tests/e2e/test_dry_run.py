"""Dry runs swap every provider for a mock and never touch the network."""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from typing import cast

import pytest

from aivia.cli import main
from aivia.core.config.providers import ProvidersConfig
from aivia.core.dry_run import (
    DRY_RUN_PROVIDER_TYPE,
    apply_dry_run,
    dry_run_provider_config,
    dry_run_providers,
)
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.core.router import Router
from aivia.providers.base import Provider
from aivia.providers.mock import MockProvider
from aivia.providers.protocols.llm import (
    CAPABILITY_LLM,
    ChatMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    MessageRole,
)

_CONFIG = """
providers:
  text:
    type: missing.module:RealProvider
    options:
      api_key: real-key
    secret_refs:
      api_key: REAL_API_KEY
formats:
  sample:
    stages:
      - stage: write
    capabilities: [llm]
""".lstrip()


class NetworkLLMProvider(Provider):
    """A provider that would open a socket if it were ever called."""

    default_metadata = ProviderMeta(
        name="network-test",
        capabilities=[CAPABILITY_LLM],
        pricing={CAPABILITY_LLM: ProviderPrice(amount_usd=0.0)},
    )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)
        raise AssertionError("the network provider must not be reached")


def write_config(tmp_path: Path, text: str = _CONFIG) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def llm_request() -> LLMRequest:
    return LLMRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="hello")],
    )


def test_the_socket_guard_rejects_an_outbound_connection() -> None:
    with pytest.raises(AssertionError, match="outbound socket"):
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)


def test_a_real_provider_call_would_fail_the_socket_guard() -> None:
    provider = NetworkLLMProvider()

    with pytest.raises(AssertionError, match="outbound socket"):
        asyncio.run(provider.complete(llm_request()))


def test_dry_run_config_swaps_active_providers_without_mutating_the_original() -> None:
    section = ProvidersConfig.from_mapping(
        {
            "providers": {
                "text": {
                    "type": "missing.module:RealProvider",
                    "options": {"api_key": "real-key"},
                    "secret_refs": {"api_key": "REAL_API_KEY"},
                },
                "off": {
                    "type": "missing.module:DisabledProvider",
                    "enabled": False,
                },
            }
        }
    )

    swapped = dry_run_provider_config(section)

    assert swapped.type_for("text") == DRY_RUN_PROVIDER_TYPE
    assert swapped.options_for("text") == {}
    assert swapped.secret_refs_for("text") == {}
    assert swapped.providers["off"].type == "missing.module:DisabledProvider"
    assert section.type_for("text") == "missing.module:RealProvider"


def test_dry_run_uses_a_mock_when_no_provider_is_configured() -> None:
    swapped = dry_run_provider_config(ProvidersConfig())

    assert swapped.names() == ("mock",)
    assert swapped.type_for("mock") == DRY_RUN_PROVIDER_TYPE


def test_dry_run_providers_serve_router_calls_without_network() -> None:
    section = ProvidersConfig.from_mapping(
        {"providers": {"text": {"type": "missing.module:RealProvider"}}}
    )

    providers = dry_run_providers(section)

    text = providers["text"]
    assert isinstance(text, MockProvider)
    router = Router(providers=providers, chain=["text"])
    response = asyncio.run(
        router.execute(
            CAPABILITY_LLM,
            lambda provider: cast(LLMProvider, provider).complete(llm_request()),
        )
    )
    assert response.content == "mock: hello"
    assert response.provider == "text"


def test_apply_dry_run_keeps_a_config_without_a_provider_section_unchanged() -> None:
    config = object()

    assert apply_dry_run(config) is config


def test_dry_run_run_estimate_uses_mocks_instead_of_resolving_real_providers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path)

    assert (
        main(
            [
                "run",
                "--format",
                "sample",
                "--estimate",
                "--config",
                str(config),
            ]
        )
        == 1
    )
    rejected = capsys.readouterr()
    assert "cannot load provider type" in rejected.err

    assert (
        main(
            [
                "--dry-run",
                "run",
                "--format",
                "sample",
                "--estimate",
                "--config",
                str(config),
            ]
        )
        == 0
    )
    accepted = capsys.readouterr()
    assert accepted.err == ""
    assert "$0.000500" in accepted.out
