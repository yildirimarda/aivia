"""Notifier fallback chain: try notifiers in order, retrying transient failures.

Providers are tried in ``chain`` order. A notifier that fails with a retryable
error is retried up to ``max_attempts`` times with ``retry_seconds`` between
attempts; a permanent error moves straight to the next notifier. When every
notifier fails, :class:`~aivia.core.errors.ProviderExhaustedError` is raised.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from aivia.core.errors import PermanentError, ProviderExhaustedError, RetryableError
from aivia.core.models.base import utcnow
from aivia.providers.protocols.notify import (
    NotificationMessage,
    NotificationReceipt,
    Notifier,
)

__all__ = ["NotifyChain"]


class NotifyChain:
    """Notify one message across a configured fallback chain of notifiers.

    Notifiers are tried in :attr:`chain` order. A notifier that fails with a
    retryable error is retried up to ``max_attempts`` times with
    ``retry_seconds`` between attempts; a permanent error moves straight to the
    next notifier. When every notifier fails, :class:`ProviderExhaustedError`
    is raised.
    """

    def __init__(
        self,
        providers: Mapping[str, Notifier],
        *,
        chain: list[str] | None = None,
        default: str | None = None,
        max_attempts: int = 1,
        retry_seconds: float = 0.0,
        sleep: Callable[[float], Any] = asyncio.sleep,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._providers = dict(providers)
        self._chain = list(chain or [])
        self._default = default
        self._max_attempts = max(1, int(max_attempts))
        self._retry_seconds = max(0.0, float(retry_seconds))
        self._sleep = sleep
        self._clock = clock

    @property
    def providers(self) -> dict[str, Notifier]:
        """The configured notifiers keyed by name."""
        return self._providers

    @property
    def chain(self) -> list[str]:
        """The notifier names in fallback order."""
        return list(self._chain)

    def _targets(self) -> list[str]:
        targets = list(self._chain)
        if not targets and self._default:
            targets = [self._default]
        return [name for name in targets if name in self._providers]

    async def _attempt(
        self, provider: Notifier, message: NotificationMessage
    ) -> tuple[NotificationReceipt | None, BaseException | None]:
        """Try *provider* once, returning ``(receipt, None)`` or ``(None, error)``."""
        last_error: BaseException | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await provider.send(message), None
            except RetryableError as exc:
                last_error = exc
                if attempt < self._max_attempts and self._retry_seconds > 0:
                    await self._sleep(self._retry_seconds)
                continue
            except PermanentError as exc:
                # A permanent failure will not improve on retry: fall back.
                return None, exc
            except Exception as exc:
                last_error = exc
                if attempt < self._max_attempts and self._retry_seconds > 0:
                    await self._sleep(self._retry_seconds)
                continue
        return None, last_error

    async def notify(self, message: NotificationMessage) -> NotificationReceipt:
        """Notify *message* across the chain, returning the first success."""
        if not isinstance(message, NotificationMessage):
            raise TypeError("message must be a NotificationMessage")
        last_error: BaseException | None = None
        for name in self._targets():
            provider = self._providers[name]
            result, error = await self._attempt(provider, message)
            if result is not None:
                return result
            if error is not None:
                last_error = error
        if last_error is not None:
            message_text = f"every notifier in the chain failed: {list(self._chain)}"
            raise ProviderExhaustedError(
                message_text,
                details={
                    "chain": list(self._chain),
                    "last_error": str(last_error),
                },
            )
        raise ProviderExhaustedError(
            "no notifier is configured for the chain",
            details={"chain": list(self._chain)},
        )
