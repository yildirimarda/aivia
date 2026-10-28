"""Base provider behaviour and the reusable contract suite itself."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from aivia.core.models import ProviderMeta, ProviderPrice, RateLimit
from aivia.core.redact import register_secret, unregister_secret
from aivia.providers import (
    HealthStatus,
    Provider,
    ProviderHealth,
    ProviderState,
)

from .provider_contract import ProviderContract


class DummyProvider(Provider):
    """A minimal adapter used to exercise the base class and its contract."""

    def __init__(self) -> None:
        super().__init__(
            ProviderMeta(
                name="dummy",
                capabilities=["text_gen", "embedding"],
                pricing={"default": ProviderPrice(amount_usd=0.01)},
                rate_limit=RateLimit(requests_per_minute=30),
            )
        )
        self.close_calls = 0

    async def _close(self) -> None:
        self.close_calls += 1


class TestDummyProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return DummyProvider()


class StaticMetadataProvider(Provider):
    default_metadata = ProviderMeta(name="static", capabilities=["image_gen"])


def test_metadata_accepts_a_mapping_and_a_name_override() -> None:
    provider = Provider(
        metadata={"name": "ignored", "capabilities": ["tts"]},
        name="  named  ",
    )

    assert provider.name == "named"
    assert provider.capabilities == ("tts",)
    assert provider.supports("tts")
    assert provider.state is ProviderState.READY
    assert provider.closed is False


def test_metadata_accepts_a_model_and_uses_the_class_default() -> None:
    metadata = ProviderMeta(name="model", capabilities=["video_gen"])
    overridden = Provider(metadata=metadata, name="instance")
    static = StaticMetadataProvider()

    assert metadata.name == "model"
    assert overridden.name == "instance"
    assert overridden.capabilities == ("video_gen",)
    assert static.name == "static"
    assert static.capabilities == ("image_gen",)


def test_default_metadata_is_empty_and_valid() -> None:
    provider = Provider()

    assert provider.name == ""
    assert provider.capabilities == ()
    assert provider.metadata.pricing == {}


def test_metadata_rejects_an_unsupported_value() -> None:
    with pytest.raises(TypeError, match="metadata must be"):
        Provider(metadata=cast(Any, 42))


def test_default_health_is_healthy_and_measured() -> None:
    report = asyncio.run(Provider().health())

    assert report.status is HealthStatus.HEALTHY
    assert report.message == "ok"
    assert report.healthy is True
    assert report.latency_ms is not None
    assert report.latency_ms >= 0.0
    assert report.checked_at.tzinfo is not None


def test_check_health_can_report_degraded() -> None:
    class DegradedProvider(Provider):
        async def check_health(self) -> ProviderHealth:
            return ProviderHealth(
                status=HealthStatus.DEGRADED,
                message="slow upstream",
                latency_ms=12.5,
            )

    report = asyncio.run(DegradedProvider().health())

    assert report.status is HealthStatus.DEGRADED
    assert report.healthy is False
    assert report.message == "slow upstream"
    assert report.latency_ms == 12.5


def test_health_redacts_a_report_from_the_adapter() -> None:
    hidden = "supersecretvalue"

    class ChattyProvider(Provider):
        async def check_health(self) -> ProviderHealth:
            return ProviderHealth(
                status=HealthStatus.DEGRADED,
                message=f"slow for {hidden}",
                details={"note": hidden},
            )

    register_secret(hidden)
    try:
        report = asyncio.run(ChattyProvider().health())
    finally:
        unregister_secret(hidden)

    assert report.status is HealthStatus.DEGRADED
    assert hidden not in report.message
    assert report.details["note"] == "[REDACTED]"


def test_health_catches_and_redacts_a_probe_error() -> None:
    hidden = "supersecretvalue"

    class BrokenProvider(Provider):
        async def check_health(self) -> ProviderHealth:
            raise RuntimeError(f"upstream rejected {hidden}")

    register_secret(hidden)
    try:
        report = asyncio.run(BrokenProvider().health())
    finally:
        unregister_secret(hidden)

    assert report.status is HealthStatus.UNHEALTHY
    assert report.healthy is False
    assert hidden not in report.message
    assert "[REDACTED]" in report.message
    assert report.details == {"error": "RuntimeError"}


def test_health_rejects_a_non_health_result() -> None:
    class WrongProvider(Provider):
        async def check_health(self) -> ProviderHealth:
            return cast(ProviderHealth, "healthy")

    with pytest.raises(TypeError, match="check_health must return ProviderHealth"):
        asyncio.run(WrongProvider().health())


def test_close_runs_the_hook_once_and_marks_the_provider_closed() -> None:
    provider = DummyProvider()

    asyncio.run(provider.close())
    asyncio.run(provider.close())

    assert provider.closed is True
    assert provider.state is ProviderState.CLOSED
    assert provider.close_calls == 1
    assert "closed" in repr(provider)


def test_concurrent_close_runs_the_hook_once() -> None:
    provider = DummyProvider()

    async def close_twice() -> None:
        await asyncio.gather(provider.close(), provider.close())

    asyncio.run(close_twice())

    assert provider.closed is True
    assert provider.close_calls == 1


def test_health_after_close_skips_the_probe() -> None:
    calls = 0

    class CountingProvider(Provider):
        async def check_health(self) -> ProviderHealth:
            nonlocal calls
            calls += 1
            return ProviderHealth(status=HealthStatus.HEALTHY)

    provider = CountingProvider()
    asyncio.run(provider.close())

    report = asyncio.run(provider.health())

    assert calls == 0
    assert report.status is HealthStatus.UNHEALTHY
    assert report.healthy is False
    assert report.message == "provider is closed"


def test_a_failing_close_hook_leaves_the_provider_open() -> None:
    class FailingProvider(Provider):
        async def _close(self) -> None:
            raise RuntimeError("still connected")

    provider = FailingProvider()

    with pytest.raises(RuntimeError, match="still connected"):
        asyncio.run(provider.close())

    assert provider.closed is False


def test_context_manager_closes_even_when_the_body_raises() -> None:
    provider = DummyProvider()

    async def use() -> None:
        async with provider:
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        asyncio.run(use())

    assert provider.closed is True
    assert provider.close_calls == 1


def test_provider_health_helpers() -> None:
    unknown = ProviderHealth.unknown()
    closed = ProviderHealth.closed()

    assert unknown.status is HealthStatus.UNKNOWN
    assert unknown.healthy is False
    assert unknown.message == "not checked"
    assert closed.status is HealthStatus.UNHEALTHY
    assert closed.healthy is False
    assert closed.message == "provider is closed"
