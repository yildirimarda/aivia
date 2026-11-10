"""Telegram notifier backed by the Telegram Bot API.

The adapter sends messages through ``https://api.telegram.org/bot<token>/``.
A message without media becomes ``sendMessage``; a message with media becomes
``sendPhoto``/``sendVideo``/``sendAudio``/``sendDocument``. Inline buttons are
sent as an ``inline_keyboard`` where every button carries either ``callback_data``
or ``url``.

The bot token is a credential: it is supplied either directly (``bot_token``) or
by reference (``bot_token_ref``) resolved through
:class:`~aivia.core.secrets.SecretsLoader`, never stored in configuration. The
target chat is fixed at construction time through ``chat_id``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import InvalidRequestError
from aivia.core.models.base import utcnow
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.core.secrets import SecretsLoader
from aivia.providers.base import HealthStatus, Provider, ProviderHealth
from aivia.providers.protocols.notify import (
    CAPABILITY_NOTIFY,
    NotificationButton,
    NotificationMessage,
    NotificationReceipt,
)
from aivia.providers.registry import register_provider

__all__ = [
    "DEFAULT_BASE_URL",
    "TELEGRAM_BASE_URL",
    "TelegramNotificationProvider",
    "TelegramNotifier",
]

DEFAULT_BASE_URL = "https://api.telegram.org"
"""The Telegram Bot API root, without a trailing slash."""

TELEGRAM_BASE_URL = DEFAULT_BASE_URL
"""Alias for :data:`DEFAULT_BASE_URL`."""

_MEDIA_METHODS = (
    ("image/", "sendPhoto"),
    ("video/", "sendVideo"),
    ("audio/", "sendAudio"),
)

_FIELD_BY_METHOD = {
    "sendPhoto": "photo",
    "sendVideo": "video",
    "sendAudio": "audio",
    "sendDocument": "document",
}

_DEFAULT_FILENAME = "attachment"
_DEFAULT_CONTENT_TYPE = "application/octet-stream"


def _resolve_bot_token(
    bot_token: str | None,
    bot_token_ref: str | None,
    secrets: SecretsLoader | None,
) -> str:
    """Resolve the bot token from a literal value or a secret reference."""
    if bot_token is not None:
        text = str(bot_token).strip()
        if not text:
            raise InvalidRequestError("telegram bot token must not be blank")
        return text
    if bot_token_ref is None or not str(bot_token_ref).strip():
        raise InvalidRequestError(
            "telegram notifier requires a bot token or bot_token_ref",
            details={"field": "bot_token_ref"},
        )
    loader = secrets or SecretsLoader()
    return loader.get(str(bot_token_ref).strip())


def _media_method(media_type: str | None) -> str:
    """Pick the Telegram method for a media type, defaulting to a document."""
    if media_type:
        text = media_type.lower()
        for prefix, method in _MEDIA_METHODS:
            if text.startswith(prefix):
                return method
    return "sendDocument"


class TelegramNotifier(Provider):
    """A notifier that delivers messages to a Telegram chat.

    ``chat_id`` names the target chat and is required. ``bot_token`` and
    ``bot_token_ref`` are mutually exclusive ways to supply the bot token;
    ``bot_token_ref`` names a secret resolved through ``secrets``. ``base_url``
    may point at a local test server and ``transport`` injects an ``httpx`` mock
    transport for the offline suite.
    """

    default_metadata = ProviderMeta(
        name="telegram",
        capabilities=[CAPABILITY_NOTIFY],
        pricing={CAPABILITY_NOTIFY: ProviderPrice(amount_usd=0.0)},
    )

    def __init__(
        self,
        *,
        chat_id: str | int | None = None,
        bot_token: str | None = None,
        bot_token_ref: str | None = None,
        secrets: SecretsLoader | None = None,
        base_url: str = DEFAULT_BASE_URL,
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
        if chat_id is None or not str(chat_id).strip():
            raise InvalidRequestError(
                "telegram notifier requires a chat_id",
                details={"field": "chat_id"},
            )
        self._chat_id = str(chat_id).strip()
        self._bot_token = _resolve_bot_token(bot_token, bot_token_ref, secrets)
        self._base_url = base_url.rstrip("/")
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def chat_id(self) -> str:
        """The target chat id messages are delivered to."""
        return self._chat_id

    @property
    def base_url(self) -> str:
        """The configured Telegram Bot API root, without a trailing slash."""
        return self._base_url

    def _endpoint(self, method: str) -> str:
        return f"{self._base_url}/bot{self._bot_token}/{method}"

    @staticmethod
    def _keyboard(buttons: list[NotificationButton]) -> list[list[dict[str, str]]]:
        """Render notification buttons as one-row inline keyboard entries."""
        rows: list[list[dict[str, str]]] = []
        for button in buttons:
            entry: dict[str, str] = {"text": button.label}
            if button.callback_data is not None:
                entry["callback_data"] = button.callback_data
            else:
                url = button.url
                assert url is not None
                entry["url"] = url
            rows.append([entry])
        return rows

    async def check_health(self) -> ProviderHealth:
        """Probe the bot with ``getMe`` and report whether the token works."""
        response = await self.http.get(
            f"{self._base_url}/bot{self._bot_token}/getMe",
            raise_for_status=True,
        )
        try:
            body = response.json()
        except ValueError:
            return ProviderHealth(
                status=HealthStatus.UNHEALTHY,
                message="getMe returned a non-JSON response",
            )
        if body.get("ok"):
            return ProviderHealth(status=HealthStatus.HEALTHY, message="ok")
        return ProviderHealth(
            status=HealthStatus.UNHEALTHY,
            message=str(body.get("description") or "getMe returned ok=false"),
        )

    async def send(self, message: NotificationMessage) -> NotificationReceipt:
        """Send *message* to the configured Telegram chat."""
        if not isinstance(message, NotificationMessage):
            raise TypeError("message must be a NotificationMessage")
        if message.media:
            return await self._send_media(message)
        return await self._send_message(message)

    def _text(self, message: NotificationMessage) -> str:
        title = message.title.strip() if message.title else ""
        text = message.text.strip() if message.text else ""
        if title and text:
            return f"{title}\n{text}"
        return title or text

    async def _send_message(self, message: NotificationMessage) -> NotificationReceipt:
        data: dict[str, Any] = {"chat_id": self._chat_id, "text": self._text(message)}
        if message.buttons:
            data["reply_markup"] = json.dumps(
                {"inline_keyboard": self._keyboard(message.buttons)}
            )
        response = await self.http.post(
            self._endpoint("sendMessage"),
            data=data,
            raise_for_status=True,
        )
        return self._receipt(response, "sendMessage")

    async def _send_media(self, message: NotificationMessage) -> NotificationReceipt:
        media = message.media[0]
        method = _media_method(media.media_type)
        field = _FIELD_BY_METHOD[method]
        data: dict[str, Any] = {"chat_id": self._chat_id}
        if media.caption:
            data["caption"] = media.caption
        if message.buttons:
            data["reply_markup"] = json.dumps(
                {"inline_keyboard": self._keyboard(message.buttons)}
            )
        files: dict[str, tuple[str, bytes, str] | str] | None = None
        if media.url is not None:
            data[field] = media.url
        else:
            payload = media.data if media.data is not None else self._read_path(media)
            filename = media.filename or _default_filename(media.media_type)
            content_type = media.media_type or _DEFAULT_CONTENT_TYPE
            files = {field: (filename, payload, content_type)}
        response = await self.http.post(
            self._endpoint(method),
            data=data,
            files=files,
            raise_for_status=True,
        )
        return self._receipt(response, method)

    @staticmethod
    def _read_path(media: Any) -> bytes:
        from pathlib import Path

        return Path(media.path).read_bytes()

    def _receipt(self, response: httpx.Response, method: str) -> NotificationReceipt:
        try:
            body = response.json()
        except ValueError as exc:
            raise InvalidRequestError(
                "telegram returned a non-JSON response",
                details={"provider": self.name, "method": method},
            ) from exc
        if not body.get("ok"):
            description = body.get("description") or "unknown error"
            raise InvalidRequestError(
                f"telegram rejected message: {description}",
                details={
                    "provider": self.name,
                    "method": method,
                    "error_code": body.get("error_code"),
                },
            )
        result = body.get("result") or {}
        message_id = result.get("message_id")
        return NotificationReceipt(
            message_id=str(message_id) if message_id is not None else "",
            channel=self._chat_id,
            provider=self.name,
            delivered=True,
            delivered_at=utcnow(),
            metadata={"method": method},
        )


def _default_filename(media_type: str | None) -> str:
    if not media_type:
        return _DEFAULT_FILENAME
    kind = media_type.split("/", 1)[0]
    extension = {
        "image": "png",
        "video": "mp4",
        "audio": "mp3",
    }.get(kind, "bin")
    return f"{_DEFAULT_FILENAME}.{extension}"


TelegramNotificationProvider = TelegramNotifier
"""Alias for :class:`TelegramNotifier` for callers that prefer provider wording."""

register_provider("telegram", TelegramNotifier, module=__name__)
register_provider("telegram_notifier", TelegramNotifier, module=__name__)
