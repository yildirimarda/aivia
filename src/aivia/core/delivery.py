"""Scheduled delivery: wait until the due time in the order's timezone,
then hand the output to the fallback chain and record a confirmation.

The transport providers only move artifacts and report the destination they
used. Scheduling (waiting until the requested time), retries and fallback
(belonging to the chain) and the confirmation record are owned by this
service, not by a provider.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from aivia.core.models.base import utcnow
from aivia.core.models.io import Delivery, DeliveryState
from aivia.providers.delivery import DeliveryChain
from aivia.providers.protocols.delivery import (
    DeliveryReceipt,
    DeliveryRequest,
)

__all__ = ["ScheduledDelivery"]


def _to_utc(value: datetime | None, *, timezone: str) -> datetime | None:
    """Return *value* as a UTC instant, or ``None`` when due immediately.

    A naive datetime is read as wall-clock time in *timezone*; an aware
    datetime is converted straight to UTC.
    """

    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=ZoneInfo(timezone))
    return value.astimezone(UTC)


class ScheduledDelivery:
    """Deliver a :class:`~aivia.core.models.io.Delivery` at its due time.

    *chain* is the fallback chain that actually moves the artifacts and retries
    transient failures. *clock* supplies the current instant and *sleep* is
    used between polls, so tests can drive time deterministically.

    The order's *timezone* interprets a naive ``scheduled_for`` as local wall
    clock time. An aware ``scheduled_for`` is converted straight to UTC. The
    service polls *clock* until the due instant, then delivers through the
    chain and records the provider's acknowledgement as a confirmation on the
    delivery.
    """

    def __init__(
        self,
        chain: DeliveryChain,
        *,
        clock: Callable[[], datetime] = utcnow,
        sleep: Callable[[float], Any] = asyncio.sleep,
        poll_interval: float = 0.5,
    ) -> None:
        self._chain = chain
        self._clock = clock
        self._sleep = sleep
        self._poll_interval = max(0.0, float(poll_interval))

    @property
    def chain(self) -> DeliveryChain:
        """The fallback chain this service delivers through."""
        return self._chain

    def due_at(self, delivery: Delivery, *, timezone: str = "UTC") -> datetime | None:
        """The UTC instant *delivery* is due, or ``None`` when due now."""
        return _to_utc(delivery.scheduled_for, timezone=timezone)

    def is_due(
        self,
        delivery: Delivery,
        *,
        timezone: str = "UTC",
        now: datetime | None = None,
    ) -> bool:
        """Whether the due instant has passed.

        A delivery without a scheduled time is always due.
        """
        due = self.due_at(delivery, timezone=timezone)
        if due is None:
            return True
        current = now if now is not None else self._clock()
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        return current.astimezone(UTC) >= due

    async def deliver(
        self,
        delivery: Delivery,
        request: DeliveryRequest,
        *,
        timezone: str = "UTC",
    ) -> DeliveryReceipt:
        """Wait until *delivery* is due, then deliver and record a confirmation.

        Retries and provider fallback are handled by *chain*. On success the
        provider's reference is stored on *delivery* as its confirmation,
        ``delivered_at`` is stamped and the state moves to ``delivered``.
        """
        while not self.is_due(delivery, timezone=timezone, now=self._clock()):
            await self._sleep(self._poll_interval)
        receipt = await self._chain.deliver(request)
        delivery.confirmation = receipt.reference or ""
        delivery.delivered_at = receipt.delivered_at or self._clock()
        delivery.state = DeliveryState.DELIVERED
        return receipt
