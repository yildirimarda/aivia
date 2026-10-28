"""Shared contract every provider adapter must satisfy.

Concrete test modules subclass :class:`ProviderContract` and override the
``provider`` fixture::

    class TestMyProvider(ProviderContract):
        @pytest.fixture
        def provider(self) -> Provider:
            return MyProvider(...)

The class name must start with ``Test`` so pytest collects the inherited
tests. This module is intentionally not named ``test_*`` so the mixin itself is
never collected.
"""

from __future__ import annotations

import asyncio

import pytest

from aivia.core.models.provider import ProviderMeta
from aivia.providers import HealthStatus, Provider, ProviderHealth

__all__ = ["ProviderContract"]


class ProviderContract:
    """Reusable lifecycle, metadata and health assertions for a provider."""

    @pytest.fixture
    def provider(self) -> Provider:
        raise NotImplementedError("subclasses must override the provider fixture")

    def test_metadata_describes_the_provider(self, provider: Provider) -> None:
        metadata = provider.metadata

        assert isinstance(metadata, ProviderMeta)
        assert metadata is provider.meta
        assert provider.name == metadata.name
        assert provider.name.strip()
        assert provider.capabilities == tuple(metadata.capabilities)
        assert provider.capabilities

        for capability in provider.capabilities:
            assert provider.supports(capability)

    def test_supports_rejects_an_undeclared_capability(
        self, provider: Provider
    ) -> None:
        assert not provider.supports("aivia_contract_undeclared_capability")

    def test_health_returns_a_structured_report(self, provider: Provider) -> None:
        report = asyncio.run(provider.health())

        assert isinstance(report, ProviderHealth)
        assert isinstance(report.status, HealthStatus)
        assert report.checked_at.tzinfo is not None
        assert report.latency_ms is not None
        assert report.latency_ms >= 0.0
        assert isinstance(report.details, dict)
        assert report.healthy is (report.status is HealthStatus.HEALTHY)
        assert report.ok is report.healthy
        assert report.is_healthy is report.healthy
        assert not provider.closed

    def test_close_is_idempotent(self, provider: Provider) -> None:
        asyncio.run(provider.close())
        asyncio.run(provider.close())

        assert provider.closed

    def test_concurrent_close_is_safe(self, provider: Provider) -> None:
        async def close_twice() -> None:
            await asyncio.gather(provider.close(), provider.close())

        asyncio.run(close_twice())

        assert provider.closed

    def test_health_after_close_is_unhealthy(self, provider: Provider) -> None:
        asyncio.run(provider.close())

        report = asyncio.run(provider.health())

        assert provider.closed
        assert report.status is HealthStatus.UNHEALTHY
        assert report.healthy is False
        assert report.message

    def test_async_context_manager_closes_the_provider(
        self, provider: Provider
    ) -> None:
        async def use() -> None:
            async with provider as entered:
                assert entered is provider
                assert not provider.closed
            assert provider.closed

        asyncio.run(use())

    def test_repr_names_the_provider_and_state(self, provider: Provider) -> None:
        text = repr(provider)

        assert provider.name in text
        assert "ready" in text
