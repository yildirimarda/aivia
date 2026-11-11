"""Telegram channel publisher adapter tests."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, cast

import httpx
import pytest

from aivia.core.errors import InvalidRequestError
from aivia.core.secrets import SecretsLoader
from aivia.providers import provider_registry
from aivia.providers.protocols.publish import (
    CAPABILITY_PUBLISH,
    PublishRequest,
)
from aivia.providers.telegram_publish import (
    DEFAULT_BASE_URL,
    TELEGRAM_BASE_URL,
    TelegramChannelPublisher,
    TelegramPublisher,
)

BOT_TOKEN = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
PUBLIC_CHAT_ID = "@aivia_channel"
PRIVATE_CHAT_ID = "-1001234567890"

OK_RESPONSE = {"ok": True, "result": {"message_id": 42, "date": 1700000000}}


def _capture(
    response: httpx.Response | dict[str, Any] = OK_RESPONSE,
) -> tuple[list[httpx.Request], list[int], Callable[[httpx.Request], httpx.Response]]:
    requests: list[httpx.Request] = []
    calls = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        calls[0] += 1
        if isinstance(response, httpx.Response):
            return response
        return httpx.Response(200, json=response)

    return requests, calls, handler


def _provider(
    handler: Any,
    *,
    chat_id: str = PUBLIC_CHAT_ID,
    bot_token: str | None = BOT_TOKEN,
    bot_token_ref: str | None = None,
    secrets: SecretsLoader | None = None,
    **kwargs: Any,
) -> TelegramChannelPublisher:
    options: dict[str, Any] = {
        "chat_id": chat_id,
        "bot_token": bot_token,
        "bot_token_ref": bot_token_ref,
        "secrets": secrets,
        "transport": httpx.MockTransport(handler),
    }
    options.update(kwargs)
    return TelegramChannelPublisher(**options)


def _request(
    *,
    platform: str = "telegram_channel",
    title: str = "Review needed",
    caption: str = "A run is waiting",
    hashtags: list[str] | None = None,
    dry_run: bool = False,
) -> PublishRequest:
    return PublishRequest(
        platform=platform,
        title=title,
        caption=caption,
        hashtags=hashtags or [],
        dry_run=dry_run,
    )


def _form_body(content: bytes) -> dict[str, list[str]]:
    from urllib.parse import parse_qs

    return parse_qs(content.decode())


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_metadata_and_registration_describe_the_provider() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=OK_RESPONSE))

    assert provider.name == "telegram_channel"
    assert provider.capabilities == (CAPABILITY_PUBLISH,)
    assert provider.chat_id == PUBLIC_CHAT_ID
    assert provider.base_url == DEFAULT_BASE_URL
    assert TELEGRAM_BASE_URL == DEFAULT_BASE_URL
    assert TelegramPublisher is TelegramChannelPublisher
    assert provider_registry.resolve("telegram_channel") is TelegramChannelPublisher
    assert provider_registry.resolve("telegram_publish") is TelegramChannelPublisher


def test_publish_text_returns_a_live_result() -> None:
    requests, calls, handler = _capture()
    provider = _provider(handler)
    request = _request(title="", caption="A run is waiting", hashtags=[])

    result = _run(provider.publish(request))

    assert calls[0] == 1
    submitted = requests[0]
    assert submitted.method == "POST"
    assert str(submitted.url) == f"{DEFAULT_BASE_URL}/bot{BOT_TOKEN}/sendMessage"
    assert submitted.headers["content-type"].startswith(
        "application/x-www-form-urlencoded"
    )
    body = _form_body(submitted.content)
    assert body["chat_id"] == [PUBLIC_CHAT_ID]
    assert body["text"] == ["A run is waiting"]
    assert result.is_live is True
    assert result.remote_id == "42"
    assert result.provider == "telegram_channel"
    assert result.platform == "telegram_channel"
    assert result.url == "https://t.me/aivia_channel/42"
    assert result.published_at is not None
    assert result.published_at.tzinfo is not None
    assert result.dry_run is False


def test_publish_title_caption_and_hashtags_compose_the_text() -> None:
    requests, _, handler = _capture()
    provider = _provider(handler)
    request = _request(
        title="Review needed",
        caption="A run is waiting",
        hashtags=["launch", "video"],
    )

    _run(provider.publish(request))

    body = _form_body(requests[0].content)
    text = body["text"][0]
    assert "Review needed" in text
    assert "A run is waiting" in text
    assert "#launch" in text
    assert "#video" in text


def test_dry_run_returns_a_not_live_result_without_posting() -> None:
    requests, calls, handler = _capture()
    provider = _provider(handler)
    request = _request(dry_run=True)

    result = _run(provider.publish(request))

    assert calls[0] == 0
    assert requests == []
    assert result.is_live is False
    assert result.remote_id == ""
    assert result.dry_run is True
    assert result.provider == "telegram_channel"


def test_publish_rejection_is_raised() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            200,
            json={
                "ok": False,
                "error_code": 400,
                "description": "Bad Request: chat not found",
            },
        )
    )

    with pytest.raises(InvalidRequestError, match="rejected"):
        _run(provider.publish(_request()))


def test_non_json_response_is_raised() -> None:
    provider = _provider(lambda request: httpx.Response(200, text="not json"))

    with pytest.raises(InvalidRequestError, match="non-JSON"):
        _run(provider.publish(_request()))


def test_publish_requires_a_publish_request() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=OK_RESPONSE))

    with pytest.raises(TypeError, match="PublishRequest"):
        _run(provider.publish(cast(PublishRequest, "not a request")))


def test_bot_token_resolved_from_reference() -> None:
    requests, _, handler = _capture()
    secrets = SecretsLoader(environ={"AIVA_TELEGRAM_BOT_TOKEN": BOT_TOKEN})
    provider = _provider(
        handler,
        bot_token=None,
        bot_token_ref="TELEGRAM_BOT_TOKEN",
        secrets=secrets,
    )

    _run(provider.publish(_request()))

    assert str(requests[0].url) == f"{DEFAULT_BASE_URL}/bot{BOT_TOKEN}/sendMessage"


def test_missing_bot_token_is_rejected() -> None:
    with pytest.raises(InvalidRequestError):
        TelegramChannelPublisher(chat_id=PUBLIC_CHAT_ID)


def test_missing_chat_id_is_rejected() -> None:
    with pytest.raises(InvalidRequestError):
        TelegramChannelPublisher(bot_token=BOT_TOKEN)


def test_health_check_reports_healthy_when_get_me_succeeds() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            200, json={"ok": True, "result": {"username": "bot"}}
        )
    )

    health = _run(provider.health())

    assert health.healthy is True
    assert health.status.value == "healthy"


def test_health_check_reports_unhealthy_when_get_me_fails() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            200,
            json={"ok": False, "error_code": 401, "description": "Unauthorized"},
        )
    )

    health = _run(provider.health())

    assert health.healthy is False
    assert health.status.value == "unhealthy"


def test_post_url_for_public_channel() -> None:
    requests, _, handler = _capture()
    provider = _provider(handler, chat_id=PUBLIC_CHAT_ID)

    _run(provider.publish(_request()))

    assert requests[0].url is not None
    result = _run(provider.publish(_request()))
    assert result.url == "https://t.me/aivia_channel/42"


def test_post_url_for_private_channel() -> None:
    provider = _provider(
        lambda request: httpx.Response(200, json=OK_RESPONSE), chat_id=PRIVATE_CHAT_ID
    )

    result = _run(provider.publish(_request()))

    assert result.url == "https://t.me/c/1234567890/42"
