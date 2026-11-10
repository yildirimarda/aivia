"""Scheduled delivery: wait until the due time in the order's timezone,
then hand the output to the fallback chain and record a confirmation.

The transport providers only move artifacts and report the destination they
used. Scheduling (waiting until the requested time), retries and fallback
(belonging to the chain) and the confirmation record are owned by this
service, not by a provider.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from aivia.core.errors import ProviderExhaustedError
from aivia.core.models.asset import AssetLicense
from aivia.core.models.base import utcnow
from aivia.core.models.io import Delivery, DeliveryState
from aivia.media.music_service import (
    DEFAULT_LICENSE_GATE,
    MusicLicenseError,
    MusicLicenseGate,
)
from aivia.providers.console import default_notifier
from aivia.providers.delivery import DeliveryChain
from aivia.providers.protocols.delivery import (
    DeliveryArtifact,
    DeliveryReceipt,
    DeliveryRequest,
)
from aivia.providers.protocols.notify import (
    NotificationButton,
    NotificationLevel,
    NotificationMessage,
    Notifier,
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
        gate: MusicLicenseGate | None = None,
        notifier: Notifier | None = None,
    ) -> None:
        self._chain = chain
        self._clock = clock
        self._sleep = sleep
        self._poll_interval = max(0.0, float(poll_interval))
        self._gate = gate if gate is not None else DEFAULT_LICENSE_GATE
        self._notifier = notifier

    @property
    def chain(self) -> DeliveryChain:
        """The fallback chain this service delivers through."""
        return self._chain

    @property
    def gate(self) -> MusicLicenseGate:
        """The license gate consulted before delivering an order."""
        return self._gate

    @property
    def notifier(self) -> Notifier:
        """The notifier used to report delivery failures, defaulting to the console."""
        if self._notifier is None:
            self._notifier = default_notifier()
        return self._notifier

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
        order: Mapping[str, Any] | None = None,
    ) -> DeliveryReceipt:
        """Wait until *delivery* is due, then deliver and record a confirmation.

        Retries and provider fallback are handled by *chain*. On success the
        provider's reference is stored on *delivery* as its confirmation,
        ``delivered_at`` is stamped and the state moves to ``delivered``.

        When *order* is given, its music license metadata is checked first. An
        order whose music is not cleared for commercial delivery is blocked:
        the chain is never called, the error is recorded on the delivery and
        :class:`~aivia.media.music_service.MusicLicenseError` is raised.
        """
        while not self.is_due(delivery, timezone=timezone, now=self._clock()):
            await self._sleep(self._poll_interval)
        if order is not None:
            licenses = _order_music_licenses(order)
            if licenses:
                decision = self._gate.check(*licenses, action="deliver")
                if not decision.allowed:
                    delivery.error_code = "music_license"
                    delivery.error_message = "music is not cleared for delivery"
                    raise MusicLicenseError(
                        "music is not cleared for delivery",
                        action="deliver",
                        blockers=decision.blockers,
                        licenses=decision.licenses,
                    )
        try:
            receipt = await self._chain.deliver(request)
        except ProviderExhaustedError as error:
            self._record_failure(delivery, error)
            await self._notify_failure(delivery, error)
            raise
        self._record_success(delivery, receipt)
        return receipt

    async def redo(
        self,
        delivery: Delivery,
        request: DeliveryRequest,
        *,
        timezone: str = "UTC",
    ) -> DeliveryReceipt:
        """Retry a failed delivery reusing its cached artifact files.

        The request's artifacts are replaced by the files already on disk
        (``delivery.files``) so the finished output is not regenerated. A
        delivered delivery cannot be redone and a delivery with no attempts
        left raises :class:`ValueError`. On success the confirmation is
        recorded; on exhaustion the failure is recorded, the console is
        notified and :class:`ProviderExhaustedError` is raised.
        """
        if delivery.is_delivered:
            raise ValueError(f"delivery {delivery.id} is already delivered")
        if not delivery.has_attempts_left:
            raise ValueError(f"delivery {delivery.id} has no attempts left")
        artifacts = [
            DeliveryArtifact(path=str(path), filename=Path(path).name)
            for path in delivery.files
        ]
        retried = DeliveryRequest(
            method=request.method,
            target=request.target,
            order_id=request.order_id,
            run_id=request.run_id,
            subject=request.subject,
            message=request.message,
            artifacts=artifacts,
            expires_seconds=request.expires_seconds,
            idempotency_key=request.idempotency_key,
            metadata=dict(request.metadata),
        )
        try:
            receipt = await self._chain.deliver(retried)
        except ProviderExhaustedError as error:
            self._record_failure(delivery, error)
            await self._notify_failure(delivery, error)
            raise
        self._record_success(delivery, receipt)
        return receipt

    def _record_success(self, delivery: Delivery, receipt: DeliveryReceipt) -> None:
        """Mark *delivery* as delivered using *receipt*."""
        delivery.confirmation = receipt.reference or ""
        delivery.delivered_at = receipt.delivered_at or self._clock()
        delivery.state = DeliveryState.DELIVERED

    def _record_failure(self, delivery: Delivery, error: BaseException) -> None:
        """Record one failed delivery attempt on *delivery*."""
        delivery.attempts += 1
        delivery.state = DeliveryState.FAILED
        delivery.error_code = getattr(error, "code", "") or ""
        delivery.error_message = str(error)

    async def _notify_failure(self, delivery: Delivery, error: BaseException) -> None:
        """Send a console fallback notification that *delivery* failed."""
        message = NotificationMessage(
            title="Delivery failed",
            text=(
                f"Delivery {delivery.id} for order {delivery.order_id} failed "
                f"after {delivery.attempts} attempt(s): {delivery.error_message}"
            ),
            level=NotificationLevel.ERROR,
            buttons=[
                NotificationButton(
                    label="Redo",
                    callback_data=f"delivery-redo:{delivery.id}",
                )
            ],
            metadata={
                "delivery_id": delivery.id,
                "order_id": delivery.order_id,
                "error_code": delivery.error_code,
                "retryable": delivery.has_attempts_left,
            },
        )
        await self.notifier.send(message)


def _order_music_licenses(order: Mapping[str, Any]) -> tuple[AssetLicense, ...]:
    """Return the music licenses recorded on *order*'s metadata."""
    metadata = order.get("metadata")
    if not isinstance(metadata, Mapping):
        return ()
    raw = metadata.get("music_licenses")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    licenses: list[AssetLicense] = []
    for entry in raw:
        if isinstance(entry, Mapping):
            licenses.append(AssetLicense.model_validate(entry))
    return tuple(licenses)
