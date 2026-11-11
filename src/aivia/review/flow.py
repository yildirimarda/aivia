"""Human review flow: send a preview for approval through a notifier.

The flow assembles a notification carrying the preview image, title, sources,
cost and warnings, plus approve, reject and regenerate buttons, and hands it to
a notifier. Interpreting the callbacks is the review state machine's job; this
flow only builds and sends the message.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from aivia.core.config.ops import ReviewConfig
from aivia.providers.protocols.notify import (
    NotificationButton,
    NotificationLevel,
    NotificationMedia,
    NotificationMessage,
    NotificationReceipt,
    Notifier,
)

APPROVE = "approve"
REJECT = "reject"
REGENERATE = "regenerate"

_BUTTONS: tuple[NotificationButton, ...] = (
    NotificationButton(label="Approve", callback_data=APPROVE),
    NotificationButton(label="Reject", callback_data=REJECT),
    NotificationButton(label="Regenerate", callback_data=REGENERATE),
)


@dataclass(frozen=True)
class ReviewResult:
    """Outcome of sending (or skipping) a review notification."""

    sent: bool
    """Whether a notification was actually sent."""
    receipt: NotificationReceipt | None
    """The channel acknowledgement, only set when sent."""
    message: NotificationMessage | None
    """The message that was sent, only set when sent."""


class ReviewFlow:
    """Send a review preview for human approval through a notifier."""

    def __init__(self, notifier: Notifier, config: ReviewConfig) -> None:
        self._notifier = notifier
        self._config = config

    @property
    def notifier(self) -> Notifier:
        """The notifier the flow delivers review messages through."""
        return self._notifier

    @property
    def config(self) -> ReviewConfig:
        """The review configuration governing whether to send."""
        return self._config

    def enabled(self) -> bool:
        """Whether review notifications are configured to be sent."""
        return (
            self._config.enabled
            and self._config.notify
            and self._config.approval_mode != "never"
        )

    def _build_message(
        self,
        *,
        preview: NotificationMedia | None,
        title: str,
        sources: Sequence[str],
        cost_usd: float,
        warnings: Sequence[str],
    ) -> NotificationMessage:
        parts: list[str] = []
        if sources:
            parts.append("Sources:")
            parts.extend(f"- {source}" for source in sources)
        parts.append(f"Cost: ${cost_usd:.4f}")
        if warnings:
            parts.append("")
            parts.append("Warnings:")
            parts.extend(f"- {warning}" for warning in warnings)
        media = [preview] if preview is not None else []
        return NotificationMessage(
            title=title,
            text="\n".join(parts),
            level=NotificationLevel.WARNING,
            media=media,
            buttons=list(_BUTTONS),
        )

    async def send_review(
        self,
        *,
        preview: NotificationMedia | None,
        title: str,
        sources: Sequence[str],
        cost_usd: float,
        warnings: Sequence[str],
    ) -> ReviewResult:
        """Send the review notification, or report that it was skipped."""
        if not self.enabled():
            return ReviewResult(sent=False, receipt=None, message=None)
        message = self._build_message(
            preview=preview,
            title=title,
            sources=list(sources),
            cost_usd=cost_usd,
            warnings=list(warnings),
        )
        receipt = await self._notifier.send(message)
        return ReviewResult(sent=True, receipt=receipt, message=message)
