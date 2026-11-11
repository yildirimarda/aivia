"""Review flow sends preview, title, sources, cost, warnings and buttons."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import parse_qs

import httpx

from aivia.core.config.ops import ReviewConfig
from aivia.providers.protocols.notify import (
    NotificationMedia,
    NotificationReceipt,
)
from aivia.providers.telegram import TelegramNotifier
from aivia.review import ReviewFlow, ReviewResult
from aivia.review.flow import APPROVE, REGENERATE, REJECT

BOT_TOKEN = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
CHAT_ID = "-1001234567890"
IMAGE_URL = "https://example.com/preview.png"

OK_RESPONSE = {"ok": True, "result": {"message_id": 42, "date": 1700000000}}


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


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


def _provider(handler: Any, **kwargs: Any) -> TelegramNotifier:
    options: dict[str, Any] = {
        "chat_id": CHAT_ID,
        "bot_token": BOT_TOKEN,
        "transport": httpx.MockTransport(handler),
    }
    options.update(kwargs)
    return TelegramNotifier(**options)


def _form_body(content: bytes) -> dict[str, list[str]]:
    return parse_qs(content.decode())


def _multipart_field(content: bytes, name: str) -> bytes | None:
    marker = b'name="' + name.encode() + b'"'
    idx = content.find(marker)
    if idx == -1:
        return None
    start = content.find(b"\r\n\r\n", idx) + 4
    end = content.find(b"\r\n--", start)
    if end == -1:
        end = len(content)
    return content[start:end]


class _RecordingNotifier:
    """A notifier that records every message it is asked to send."""

    def __init__(self) -> None:
        self.sent: list[Any] = []

    async def send(self, message: Any) -> NotificationReceipt:
        self.sent.append(message)
        return NotificationReceipt(
            message_id="1", channel="test", provider="fake", delivered=True
        )


def _config(**overrides: Any) -> ReviewConfig:
    data: dict[str, Any] = {
        "enabled": True,
        "approval_mode": "ask",
        "notify": True,
    }
    data.update(overrides)
    return ReviewConfig(**data)


def _preview() -> NotificationMedia:
    return NotificationMedia(
        data=b"\x89PNG\r\n\x1a\n",
        media_type="image/png",
        filename="preview.png",
        caption="preview",
    )


def _flow(
    config: ReviewConfig | None = None,
    notifier: Any | None = None,
) -> ReviewFlow:
    used: Any = notifier if notifier is not None else _RecordingNotifier()
    return ReviewFlow(notifier=used, config=config or _config())


def test_send_review_carries_title_sources_cost_warnings_and_buttons() -> None:
    notifier = _RecordingNotifier()
    flow = _flow(notifier=notifier)

    result = _run(
        flow.send_review(
            preview=_preview(),
            title="Bridge video",
            sources=["https://example.com/story", "https://other.test/background"],
            cost_usd=0.1234,
            warnings=["subtitle style resized", "external asset fetched"],
        )
    )

    assert isinstance(result, ReviewResult)
    assert result.sent is True
    assert result.receipt is not None
    assert result.receipt.delivered is True
    assert result.message is not None
    message = result.message
    assert message.title == "Bridge video"
    assert "https://example.com/story" in message.text
    assert "https://other.test/background" in message.text
    assert "Cost: $0.1234" in message.text
    assert "subtitle style resized" in message.text
    assert "external asset fetched" in message.text
    assert len(message.media) == 1
    assert message.media[0].media_type == "image/png"
    assert [button.label for button in message.buttons] == [
        "Approve",
        "Reject",
        "Regenerate",
    ]
    assert [button.callback_data for button in message.buttons] == [
        APPROVE,
        REJECT,
        REGENERATE,
    ]
    assert notifier.sent == [message]


def test_send_review_without_preview_sends_text_only() -> None:
    notifier = _RecordingNotifier()
    flow = _flow(notifier=notifier)

    result = _run(
        flow.send_review(
            preview=None,
            title="Title only",
            sources=[],
            cost_usd=0.0,
            warnings=[],
        )
    )

    assert result.sent is True
    assert result.message is not None
    assert result.message.title == "Title only"
    assert "Cost: $0.0000" in result.message.text
    assert result.message.media == []
    assert len(result.message.buttons) == 3


def test_send_review_buttons_are_approve_reject_regenerate() -> None:
    notifier = _RecordingNotifier()
    flow = _flow(notifier=notifier)

    _run(
        flow.send_review(
            preview=None,
            title="T",
            sources=[],
            cost_usd=1.0,
            warnings=[],
        )
    )

    buttons = notifier.sent[0].buttons
    assert [(button.label, button.callback_data) for button in buttons] == [
        ("Approve", APPROVE),
        ("Reject", REJECT),
        ("Regenerate", REGENERATE),
    ]


def test_review_is_skipped_when_disabled() -> None:
    notifier = _RecordingNotifier()
    flow = _flow(config=_config(enabled=False), notifier=notifier)

    result = _run(
        flow.send_review(preview=None, title="T", sources=[], cost_usd=1.0, warnings=[])
    )

    assert result.sent is False
    assert result.receipt is None
    assert result.message is None
    assert notifier.sent == []


def test_review_is_skipped_when_approval_mode_is_never() -> None:
    notifier = _RecordingNotifier()
    flow = _flow(config=_config(approval_mode="never"), notifier=notifier)

    result = _run(
        flow.send_review(preview=None, title="T", sources=[], cost_usd=1.0, warnings=[])
    )

    assert result.sent is False
    assert notifier.sent == []


def test_review_is_skipped_when_notify_is_false() -> None:
    notifier = _RecordingNotifier()
    flow = _flow(config=_config(notify=False), notifier=notifier)

    result = _run(
        flow.send_review(preview=None, title="T", sources=[], cost_usd=1.0, warnings=[])
    )

    assert result.sent is False
    assert notifier.sent == []


def test_send_review_via_telegram_sends_photo_with_buttons() -> None:
    requests, handler = _capture()
    flow = _flow(notifier=_provider(handler))

    result = _run(
        flow.send_review(
            preview=_preview(),
            title="Bridge video",
            sources=["https://example.com/story"],
            cost_usd=0.5,
            warnings=["subtitle style resized"],
        )
    )

    assert result.sent is True
    assert result.receipt is not None
    assert result.receipt.message_id == "42"

    submitted = requests[0]
    assert submitted.method == "POST"
    assert str(submitted.url).endswith(f"/bot{BOT_TOKEN}/sendPhoto")
    assert submitted.headers["content-type"].startswith("multipart/form-data")
    assert b'name="photo"' in submitted.content
    assert b'filename="preview.png"' in submitted.content
    assert b"image/png" in submitted.content
    markup_raw = _multipart_field(submitted.content, "reply_markup")
    assert markup_raw is not None
    markup = json.loads(markup_raw)
    assert markup == {
        "inline_keyboard": [
            [{"text": "Approve", "callback_data": "approve"}],
            [{"text": "Reject", "callback_data": "reject"}],
            [{"text": "Regenerate", "callback_data": "regenerate"}],
        ]
    }


def test_send_review_via_telegram_text_only_uses_send_message() -> None:
    requests, handler = _capture()
    flow = _flow(notifier=_provider(handler))

    _run(
        flow.send_review(
            preview=None,
            title="Bridge video",
            sources=["https://example.com/story"],
            cost_usd=0.5,
            warnings=["subtitle style resized"],
        )
    )

    submitted = requests[0]
    assert str(submitted.url).endswith(f"/bot{BOT_TOKEN}/sendMessage")
    body = _form_body(submitted.content)
    assert body["chat_id"] == [CHAT_ID]
    text = body["text"][0]
    assert "Bridge video" in text
    assert "Cost: $0.5000" in text
    assert "https://example.com/story" in text
    markup = json.loads(body["reply_markup"][0])
    assert len(markup["inline_keyboard"]) == 3
