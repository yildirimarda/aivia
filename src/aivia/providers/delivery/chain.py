"""Delivery fallback chain: try providers in order, retrying transient failures."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from aivia.core.config.orders import DeliveryConfig
from aivia.core.errors import PermanentError, ProviderExhaustedError, RetryableError
from aivia.core.models.base import utcnow
from aivia.providers.protocols.delivery import (
    DeliveryProvider,
    DeliveryReceipt,
    DeliveryRequest,
)

__all__ = ["DeliveryChain"]


class DeliveryChain:
    """Deliver one request across a configured fallback chain of providers.

    Providers are tried in :attr:`DeliveryConfig.chain` order. A provider that
    fails with a retryable error is retried up to ``max_attempts`` times with
    ``retry_seconds`` between attempts; a permanent error moves straight to the
    next provider. When every provider fails, :class:`ProviderExhaustedError`
    is raised.
    """

    def __init__(
        self,
        config: DeliveryConfig,
        providers: Mapping[str, DeliveryProvider],
        *,
        sleep: Callable[[float], Any] = asyncio.sleep,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._config = config
        self._providers = dict(providers)
        self._sleep = sleep
        self._clock = clock

    @property
    def config(self) -> DeliveryConfig:
        """The delivery configuration this chain was built from."""
        return self._config

    @property
    def providers(self) -> dict[str, DeliveryProvider]:
        """The configured providers keyed by name."""
        return self._providers

    def _targets(self) -> list[str]:
        chain = list(self._config.chain)
        if not chain and self._config.default:
            chain = [self._config.default]
        return [name for name in chain if name in self._providers]

    async def _attempt(
        self, provider: DeliveryProvider, request: DeliveryRequest
    ) -> tuple[DeliveryReceipt | None, BaseException | None]:
        """Try *provider* once, returning ``(receipt, None)`` or ``(None, error)``."""
        max_attempts = max(1, self._config.max_attempts)
        delay = max(0.0, self._config.retry_seconds)
        last_error: BaseException | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await provider.deliver(request), None
            except RetryableError as exc:
                last_error = exc
                if attempt < max_attempts and delay > 0:
                    await self._sleep(delay)
                continue
            except PermanentError as exc:
                # A permanent failure will not improve on retry: fall back.
                return None, exc
            except Exception as exc:
                last_error = exc
                if attempt < max_attempts and delay > 0:
                    await self._sleep(delay)
                continue
        return None, last_error

    async def deliver(self, request: DeliveryRequest) -> DeliveryReceipt:
        """Deliver *request* across the chain, returning the first success."""
        if not isinstance(request, DeliveryRequest):
            raise TypeError("request must be a DeliveryRequest")
        last_error: BaseException | None = None
        for name in self._targets():
            provider = self._providers[name]
            result, error = await self._attempt(provider, request)
            if result is not None:
                return result
            if error is not None:
                last_error = error
        if last_error is not None:
            message = (
                f"every delivery provider in the chain failed: "
                f"{list(self._config.chain)}"
            )
            raise ProviderExhaustedError(
                message,
                details={
                    "chain": list(self._config.chain),
                    "last_error": str(last_error),
                },
            )
        raise ProviderExhaustedError(
            "no delivery provider is configured for the chain",
            details={"chain": list(self._config.chain)},
        )
