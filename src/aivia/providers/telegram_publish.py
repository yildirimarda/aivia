"""Telegram channel publisher backed by the Telegram Bot API.

The adapter posts a validated :class:`~aivia.providers.protocols.publish.PublishRequest`
to ``sendMessage`` and returns the platform identifiers. The remote message id
becomes the ``remote_id`` so a successful result is live
(``PublishResult.is_live`` is ``True``), while a dry run is not.

The bot token is a credential supplied either directly (``bot_token``) or by
reference (``bot_token_ref``) resolved through
:class:`~aivia.core.secrets.SecretsLoader`, never stored in configuration. The
target channel is fixed at construction time through ``chat_id``.
"""

from __future__ import annotations

from typing import Any

import httpx

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import InvalidRequestError
from aivia.core.models.base import utcnow
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.core.secrets import SecretsLoader
from aivia.providers.base import HealthStatus, Provider, ProviderHealth
from aivia.providers.protocols.publish import (
    CAPABILITY_PUBLISH,
    PublishRequest,
    PublishResult,
)
from aivia.providers.registry import register_provider

__all__ = [
    "DEFAULT_BASE_URL",
    "TELEGRAM_BASE_URL",
    "TelegramChannelPublisher",
    "TelegramPublisher",
]

DEFAULT_BASE_URL = "https://api.telegram.org"
"""The Telegram Bot API root, without a trailing slash."""

TELEGRAM_BASE_URL = DEFAULT_BASE_URL
"""Alias for :data:`DEFAULT_BASE_URL`."""


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
            "telegram channel publisher requires a bot token or bot_token_ref",
            details={"field": "bot_token_ref"},
        )
    loader = secrets or SecretsLoader()
    return loader.get(str(bot_token_ref).strip())


class TelegramChannelPublisher(Provider):
    """A publisher that posts to a Telegram channel via ``sendMessage``.

    ``chat_id`` names the target channel (a ``@username`` or a ``-100...`` id)
    and is required. ``bot_token`` and ``bot_token_ref`` are mutually exclusive
    ways to supply the bot token; ``bot_token_ref`` names a secret resolved
    through ``secrets``. ``base_url`` may point at a local test server and
    ``transport`` injects an ``httpx`` mock transport for the offline suite.
    """

    default_metadata = ProviderMeta(
        name="telegram_channel",
        capabilities=[CAPABILITY_PUBLISH],
        pricing={CAPABILITY_PUBLISH: ProviderPrice(amount_usd=0.0)},
    )

    def __init__(
        self,
        *,
        chat_id: str | None = None,
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
        if http_config is not None and not isinstance(http_config, HttpClientConfig):
            raise TypeError("http_config must be an HttpClientConfig")
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
                "telegram channel publisher requires a chat_id",
                details={"field": "chat_id"},
            )
        self._chat_id = str(chat_id).strip()
        self._bot_token = _resolve_bot_token(bot_token, bot_token_ref, secrets)
        self._base_url = base_url.rstrip("/")
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def chat_id(self) -> str:
        """The target channel posts are delivered to."""
        return self._chat_id

    @property
    def base_url(self) -> str:
        """The configured Telegram Bot API root, without a trailing slash."""
        return self._base_url

    def _endpoint(self, method: str) -> str:
        return f"{self._base_url}/bot{self._bot_token}/{method}"

    def _text(self, request: PublishRequest) -> str:
        parts: list[str] = []
        if request.title:
            parts.append(request.title.strip())
        if request.caption:
            parts.append(request.caption.strip())
        if request.hashtags:
            parts.append(" ".join(f"#{tag}" for tag in request.hashtags))
        return "\n".join(parts)

    def _post_url(self, message_id: int | str) -> str:
        chat = self._chat_id
        if chat.startswith("@"):
            return f"https://t.me/{chat[1:]}/{message_id}"
        if chat.startswith("-100"):
            return f"https://t.me/c/{chat[4:]}/{message_id}"
        return f"https://t.me/{chat}/{message_id}"

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

    async def publish(self, request: PublishRequest) -> PublishResult:
        """Publish *request* to the configured channel and return identifiers."""
        if not isinstance(request, PublishRequest):
            raise TypeError("request must be a PublishRequest")
        if request.dry_run:
            return PublishResult(
                platform=request.platform,
                account=request.account,
                provider=self.name,
                dry_run=True,
            )
        data: dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": self._text(request),
        }
        response = await self.http.post(
            self._endpoint("sendMessage"),
            data=data,
            raise_for_status=True,
        )
        try:
            body = response.json()
        except ValueError as exc:
            raise InvalidRequestError(
                "telegram returned a non-JSON response",
                details={"provider": self.name, "method": "sendMessage"},
            ) from exc
        if not body.get("ok"):
            description = body.get("description") or "unknown error"
            raise InvalidRequestError(
                f"telegram rejected post: {description}",
                details={
                    "provider": self.name,
                    "method": "sendMessage",
                    "error_code": body.get("error_code"),
                },
            )
        result = body.get("result") or {}
        message_id = result.get("message_id")
        remote_id = str(message_id) if message_id is not None else ""
        url = self._post_url(message_id) if message_id is not None else None
        return PublishResult(
            platform=request.platform,
            account=request.account,
            provider=self.name,
            remote_id=remote_id,
            url=url,
            published_at=utcnow(),
            metadata={"method": "sendMessage"},
        )


TelegramPublisher = TelegramChannelPublisher
"""Alias for :class:`TelegramChannelPublisher` for provider-style callers."""

register_provider("telegram_channel", TelegramChannelPublisher, module=__name__)
register_provider("telegram_publish", TelegramChannelPublisher, module=__name__)
