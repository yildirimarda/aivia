"""Telegram notifier adapter tests."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest

from aivia.core.errors import InvalidRequestError
from aivia.core.secrets import SecretsLoader
from aivia.providers import provider_registry
from aivia.providers.protocols.notify import (
    CAPABILITY_NOTIFY,
    NotificationButton,
    NotificationLevel,
    NotificationMedia,
    NotificationMessage,
)
from aivia.providers.telegram import (
    DEFAULT_BASE_URL,
    TELEGRAM_BASE_URL,
    TelegramNotificationProvider,
    TelegramNotifier,
)

BOT_TOKEN = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
CHAT_ID = "-1001234567890"

OK_RESPONSE = {"ok": True, "result": {"message_id": 42, "date": 1700000000}}
IMAGE_URL = "https://example.com/preview.png"


def _capture(
    response: httpx.Response | dict[str, Any] = OK_RESPONSE,
) -> tuple[list[httpx.Request], Any]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if isinstance(response, httpx.Response):
            return response
        return httpx.Response(200, json=response)

    return requests, handler


def _provider(
    handler: Any,
    **kwargs: Any,
) -> TelegramNotifier:
    options: dict[str, Any] = {
        "chat_id": CHAT_ID,
        "bot_token": BOT_TOKEN,
        "transport": httpx.MockTransport(handler),
    }
    options.update(kwargs)
    return TelegramNotifier(**options)


def _message(
    *,
    text: str = "A run is waiting",
    title: str = "Review needed",
    media: NotificationMedia | None = None,
    buttons: list[NotificationButton] | None = None,
) -> NotificationMessage:
    return NotificationMessage(
        title=title,
        text=text,
        level=NotificationLevel.WARNING,
        media=[media] if media is not None else [],
        buttons=buttons or [],
    )


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_metadata_and_registration_describe_the_provider() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=OK_RESPONSE))

    assert provider.name == "telegram"
    assert provider.capabilities == (CAPABILITY_NOTIFY,)
    assert provider.chat_id == CHAT_ID
    assert provider.base_url == DEFAULT_BASE_URL
    assert TELEGRAM_BASE_URL == DEFAULT_BASE_URL
    assert TelegramNotificationProvider is TelegramNotifier
    assert provider_registry.resolve("telegram") is TelegramNotifier
    assert provider_registry.resolve("telegram_notifier") is TelegramNotifier


def _form_body(content: bytes) -> dict[str, list[str]]:
    return parse_qs(content.decode())


def test_send_text_message_posts_send_message() -> None:
    requests, handler = _capture()
    provider = _provider(handler)
    message = _message(title="")

    receipt = _run(provider.send(message))

    submitted = requests[0]
    assert submitted.method == "POST"
    assert str(submitted.url) == f"{DEFAULT_BASE_URL}/bot{BOT_TOKEN}/sendMessage"
    assert submitted.headers["content-type"].startswith(
        "application/x-www-form-urlencoded"
    )
    body = _form_body(submitted.content)
    assert body["chat_id"] == [CHAT_ID]
    assert body["text"] == ["A run is waiting"]
    assert "reply_markup" not in body
    assert receipt.message_id == "42"
    assert receipt.channel == CHAT_ID
    assert receipt.provider == "telegram"
    assert receipt.delivered is True


def test_title_and_text_are_combined_into_one_message() -> None:
    requests, handler = _capture()
    provider = _provider(handler)
    message = _message(title="  Review needed  ", text="  A run is waiting  ")

    _run(provider.send(message))

    body = _form_body(requests[0].content)
    assert body["text"] == ["Review needed\nA run is waiting"]


def test_buttons_become_an_inline_keyboard() -> None:
    requests, handler = _capture()
    provider = _provider(handler)
    message = _message(
        buttons=[
            NotificationButton(label="Approve", callback_data="approve"),
            NotificationButton(label="Open", url=IMAGE_URL),
        ]
    )

    _run(provider.send(message))

    body = _form_body(requests[0].content)
    markup = json.loads(body["reply_markup"][0])
    assert markup == {
        "inline_keyboard": [
            [{"text": "Approve", "callback_data": "approve"}],
            [{"text": "Open", "url": IMAGE_URL}],
        ]
    }


def test_send_photo_by_data_uploads_multipart() -> None:
    requests, handler = _capture()
    provider = _provider(handler)
    payload = b"\x89PNG\r\n\x1a\n"
    media = NotificationMedia(
        data=payload,
        media_type="image/png",
        filename="preview.png",
        caption="preview",
    )
    message = _message(media=media)

    receipt = _run(provider.send(message))

    submitted = requests[0]
    assert str(submitted.url).endswith("/sendPhoto")
    assert submitted.headers["content-type"].startswith("multipart/form-data")
    content = submitted.content
    assert b'name="photo"' in content
    assert b'filename="preview.png"' in content
    assert b"image/png" in content
    assert payload in content
    assert b'name="caption"' in content and b"preview" in content
    assert receipt.message_id == "42"


def test_send_photo_by_url_passes_the_url() -> None:
    requests, handler = _capture()
    provider = _provider(handler)
    media = NotificationMedia(url=IMAGE_URL, media_type="image/png", caption="preview")
    message = _message(media=media)

    _run(provider.send(message))

    submitted = requests[0]
    assert str(submitted.url).endswith("/sendPhoto")
    assert submitted.headers["content-type"].startswith(
        "application/x-www-form-urlencoded"
    )
    body = _form_body(submitted.content)
    assert body["photo"] == [IMAGE_URL]
    assert b'name="photo"' not in submitted.content


def test_send_video_and_audio_use_the_right_method() -> None:
    requests, handler = _capture()
    provider = _provider(handler)
    payload = b"video-bytes"
    for media_type, method in (("video/mp4", "sendVideo"), ("audio/mpeg", "sendAudio")):
        requests.clear()
        message = _message(media=NotificationMedia(data=payload, media_type=media_type))
        _run(provider.send(message))
        assert str(requests[0].url).endswith(f"/{method}")


def test_bot_token_resolved_from_reference() -> None:
    requests, handler = _capture()
    secrets = SecretsLoader(environ={"AIVA_TELEGRAM_BOT_TOKEN": BOT_TOKEN})
    provider = _provider(
        handler,
        bot_token=None,
        bot_token_ref="TELEGRAM_BOT_TOKEN",
        secrets=secrets,
    )

    _run(provider.send(_message()))

    assert str(requests[0].url) == f"{DEFAULT_BASE_URL}/bot{BOT_TOKEN}/sendMessage"


def test_missing_bot_token_is_rejected() -> None:
    with pytest.raises(InvalidRequestError):
        TelegramNotifier(chat_id=CHAT_ID)


def test_missing_chat_id_is_rejected() -> None:
    with pytest.raises(InvalidRequestError):
        TelegramNotifier(bot_token=BOT_TOKEN)


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


def test_telegram_rejection_is_raised() -> None:
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

    with pytest.raises(Exception, match="rejected"):
        _run(provider.send(_message()))
