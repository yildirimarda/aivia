"""Slack notifier backed by an incoming webhook.

The adapter POSTs a JSON payload to a Slack incoming webhook URL. A message
without media becomes a simple ``text`` payload; buttons are rendered as block
kit ``actions`` and media as ``attachments``. The webhook URL and an optional
default channel are supplied at construction time; an ``httpx`` transport may
be injected for the offline suite.

Slack incoming webhooks are POST-only and cannot be probed with a GET, so the
health check reports healthy whenever a webhook URL is configured.
"""

from __future__ import annotations

from typing import Any

import httpx

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import InvalidRequestError
from aivia.core.models.base import utcnow
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.providers.base import HealthStatus, Provider, ProviderHealth
from aivia.providers.protocols.notify import (
    CAPABILITY_NOTIFY,
    NotificationButton,
    NotificationMedia,
    NotificationMessage,
    NotificationReceipt,
)
from aivia.providers.registry import register_provider

__all__ = [
    "DEFAULT_SLACK_BASE_URL",
    "SlackNotificationProvider",
    "SlackNotifier",
]

DEFAULT_SLACK_BASE_URL = "https://hooks.slack.com"
"""The Slack incoming webhook root, without a trailing slash."""


def _plain_text(text: str) -> dict[str, str]:
    return {"type": "plain_text", "text": text}


def _blocks(buttons: list[NotificationButton]) -> list[dict[str, Any]]:
    """Render notification buttons as block kit action blocks."""
    elements: list[dict[str, Any]] = []
    for button in buttons:
        element: dict[str, Any] = {"type": "button", "text": _plain_text(button.label)}
        if button.callback_data is not None:
            element["action_id"] = button.callback_data
        else:
            element["url"] = button.url
        elements.append(element)
    return [{"type": "actions", "elements": elements}] if elements else []


class SlackNotifier(Provider):
    """A notifier that delivers messages to Slack via an incoming webhook.

    ``webhook_url`` is required. ``channel`` may override the webhook's default
    channel. ``transport`` injects an ``httpx.AsyncBaseTransport`` for offline
    tests. ``base_url`` may point at a local test webhook server.
    """

    default_metadata = ProviderMeta(
        name="slack",
        capabilities=[CAPABILITY_NOTIFY],
        pricing={CAPABILITY_NOTIFY: ProviderPrice(amount_usd=0.0)},
    )

    def __init__(
        self,
        *,
        webhook_url: str | None = None,
        channel: str | None = None,
        username: str = "aivia",
        icon_emoji: str | None = None,
        base_url: str = DEFAULT_SLACK_BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
        name: str | None = None,
        health_timeout_seconds: float | None = None,
        close_timeout_seconds: float | None = None,
        http_config: HttpClientConfig | None = None,
    ) -> None:
        super().__init__(
            name=name,
            health_timeout_seconds=health_timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            http_config=http_config,
        )
        if transport is not None and not isinstance(
            transport, httpx.AsyncBaseTransport
        ):
            raise TypeError("transport must be an httpx.AsyncBaseTransport")
        if webhook_url is None or not str(webhook_url).strip():
            raise InvalidRequestError(
                "slack notifier requires a webhook_url",
                details={"field": "webhook_url"},
            )
        self._webhook_url = str(webhook_url).strip()
        self._channel = str(channel).strip() if channel else None
        self._username = username
        self._icon_emoji = icon_emoji
        self._base_url = base_url.rstrip("/")
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def webhook_url(self) -> str:
        """The Slack incoming webhook URL messages are POSTed to."""
        return self._webhook_url

    @property
    def channel(self) -> str | None:
        """The optional default channel messages are delivered to."""
        return self._channel

    def _text(self, message: NotificationMessage) -> str:
        title = message.title.strip() if message.title else ""
        text = message.text.strip() if message.text else ""
        if title and text:
            return f"{title}\n{text}"
        return title or text

    def _attachments(self, media: list[NotificationMedia]) -> list[dict[str, Any]]:
        attachments: list[dict[str, Any]] = []
        for item in media:
            attachment: dict[str, Any] = {
                "fallback": item.caption or item.media_type or "attachment",
                "color": "#3AA3E3",
            }
            if item.caption:
                attachment["text"] = item.caption
            if item.url is not None:
                attachment["image_url"] = item.url
            attachments.append(attachment)
        return attachments

    async def check_health(self) -> ProviderHealth:
        """Report healthy when a webhook URL is configured."""
        if self._webhook_url:
            return ProviderHealth(status=HealthStatus.HEALTHY, message="ok")
        return ProviderHealth(
            status=HealthStatus.UNHEALTHY, message="no webhook_url configured"
        )

    async def send(self, message: NotificationMessage) -> NotificationReceipt:
        """Send *message* to the configured Slack webhook."""
        if not isinstance(message, NotificationMessage):
            raise TypeError("message must be a NotificationMessage")
        payload: dict[str, Any] = {"text": self._text(message)}
        if self._channel:
            payload["channel"] = self._channel
        if self._username:
            payload["username"] = self._username
        if self._icon_emoji:
            payload["icon_emoji"] = self._icon_emoji
        blocks = _blocks(message.buttons)
        if blocks:
            payload["blocks"] = blocks
        if message.media:
            payload["attachments"] = self._attachments(message.media)
        response = await self.http.post(
            self._webhook_url, json=payload, raise_for_status=True
        )
        return self._receipt(response)

    def _receipt(self, response: httpx.Response) -> NotificationReceipt:
        body = response.text.strip()
        return NotificationReceipt(
            message_id=body or "",
            channel=self._channel or "slack",
            provider=self.name,
            delivered=True,
            delivered_at=utcnow(),
            metadata={"response": body},
        )


SlackNotificationProvider = SlackNotifier
"""Alias for :class:`SlackNotifier` for callers that prefer provider wording."""

register_provider("slack", SlackNotifier, module=__name__)
register_provider("slack_notifier", SlackNotifier, module=__name__)
