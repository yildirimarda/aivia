"""Console notifier used as the fallback when no channel is configured.

The adapter writes one redacted line per message to a text stream and returns a
receipt, so a module can always notify a human even when Telegram, Slack or
email is unavailable. It is a normal provider: it declares the ``notify``
capability, has metadata and supports the shared health and close lifecycle.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any, TextIO

from aivia.core.config.http import HttpClientConfig
from aivia.core.ids import ulid
from aivia.core.models.base import utcnow
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.core.redact import redact
from aivia.providers.base import Provider
from aivia.providers.protocols.notify import (
    CAPABILITY_NOTIFY,
    NotificationMessage,
    NotificationReceipt,
)

__all__ = [
    "ConsoleNotificationProvider",
    "ConsoleNotifier",
    "console_notifier",
    "default_notifier",
]

_DEFAULT_METADATA = ProviderMeta(
    name="console",
    capabilities=[CAPABILITY_NOTIFY],
    pricing={CAPABILITY_NOTIFY: ProviderPrice(amount_usd=0.0)},
)


class ConsoleNotifier(Provider):
    """A notifier that prints messages to a text stream.

    ``stream`` defaults to :data:`sys.stdout` and is looked up when a message is
    sent, so a test can replace ``sys.stdout`` after construction. Every line is
    passed through :func:`~aivia.core.redact.redact` before it is written.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        metadata: ProviderMeta | Mapping[str, Any] | None = None,
        *,
        name: str | None = None,
        stream: TextIO | None = None,
        health_timeout_seconds: float | None = None,
        close_timeout_seconds: float | None = None,
        http_config: HttpClientConfig | None = None,
    ) -> None:
        super().__init__(
            metadata,
            name=name,
            health_timeout_seconds=health_timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            http_config=http_config,
        )
        self._stream = stream
        self._sent: list[NotificationMessage] = []

    @property
    def stream(self) -> TextIO:
        """The stream messages are written to."""
        return self._stream if self._stream is not None else sys.stdout

    @property
    def sent(self) -> tuple[NotificationMessage, ...]:
        """Every message this instance has written, in order."""
        return tuple(self._sent)

    @staticmethod
    def format_message(message: NotificationMessage) -> str:
        """Render *message* as one redacted console line."""
        parts = [f"[{message.level.value.upper()}]"]
        if message.title:
            parts.append(redact(message.title))
        if message.text:
            parts.append(redact(message.text))
        if message.media:
            count = len(message.media)
            parts.append(f"({count} attachment{'s' if count != 1 else ''})")
        return " ".join(parts)

    async def send(self, message: NotificationMessage) -> NotificationReceipt:
        """Write *message* to the console and acknowledge it."""
        if not isinstance(message, NotificationMessage):
            raise TypeError("message must be a NotificationMessage")
        line = self.format_message(message)
        print(line, file=self.stream, flush=True)
        self._sent.append(message)
        return NotificationReceipt(
            message_id=f"console-{ulid()}",
            channel="console",
            provider=self.name,
            delivered_at=utcnow(),
            metadata={"level": message.level.value},
        )


ConsoleNotificationProvider = ConsoleNotifier
"""Alias for :class:`ConsoleNotifier` for callers that prefer provider wording."""


def console_notifier(**kwargs: Any) -> ConsoleNotifier:
    """Return a console notifier, optionally configured with keyword arguments."""
    return ConsoleNotifier(**kwargs)


def default_notifier(**kwargs: Any) -> ConsoleNotifier:
    """Alias for :func:`console_notifier`, the fallback every module can use."""
    return console_notifier(**kwargs)
