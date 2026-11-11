"""Slack and email notifier tests sharing the same message model with chain fallback."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest import mock

import httpx
import pytest

from aivia.core.errors import (
    InvalidRequestError,
    PermanentError,
    ProviderExhaustedError,
    RetryableError,
)
from aivia.providers import provider_registry
from aivia.providers.notify_chain import NotifyChain
from aivia.providers.notify_email import EmailNotifier
from aivia.providers.protocols.notify import (
    CAPABILITY_NOTIFY,
    NotificationButton,
    NotificationMedia,
    NotificationMessage,
    NotificationReceipt,
)
from aivia.providers.slack import SlackNotifier

WEBHOOK_URL = "https://example.com/slack/webhook"
IMAGE_URL = "https://example.com/preview.png"


def _capture(
    response: str | httpx.Response = "ok",
) -> tuple[list[httpx.Request], str | httpx.Response]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if isinstance(response, httpx.Response):
            return response
        return httpx.Response(200, text=response)

    return requests, handler


def _slack(handler: Any, **kwargs: Any) -> SlackNotifier:
    options: dict[str, Any] = {
        "webhook_url": WEBHOOK_URL,
        "channel": "#alerts",
        "transport": httpx.MockTransport(handler),
    }
    options.update(kwargs)
    return SlackNotifier(**options)


def _smtp_capture() -> tuple[dict[str, Any], mock.MagicMock]:
    sent: dict[str, Any] = {}
    smtp = mock.MagicMock()
    smtp.has_extn.return_value = False
    smtp.__enter__.return_value = smtp

    def sendmail(from_addr: str, to_addrs: list[str], msg_bytes: bytes) -> None:
        sent["from"] = from_addr
        sent["to"] = list(to_addrs)
        sent["msg"] = msg_bytes

    smtp.sendmail.side_effect = sendmail
    return sent, smtp


def _email(smtp: mock.MagicMock, **kwargs: Any) -> EmailNotifier:
    options: dict[str, Any] = {
        "smtp_host": "smtp.example.com",
        "smtp_port": 587,
        "from_address": "aivia@example.com",
        "target": "user@example.com",
    }
    options.update(kwargs)
    return EmailNotifier(**options)


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
        media=[media] if media is not None else [],
        buttons=buttons or [],
    )


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class _RecordingNotifier:
    def __init__(
        self,
        name: str,
        *,
        receipt: NotificationReceipt | None = None,
        exc: BaseException | None = None,
    ) -> None:
        self.name = name
        self.capabilities = (CAPABILITY_NOTIFY,)
        self.receipt = receipt or NotificationReceipt(
            message_id="1",
            channel=name,
            provider=name,
            delivered=True,
        )
        self.exc = exc
        self.sent: list[NotificationMessage] = []

    async def send(self, message: NotificationMessage) -> NotificationReceipt:
        self.sent.append(message)
        if self.exc is not None:
            raise self.exc
        return self.receipt


class _FlakyNotifier:
    def __init__(self, name: str = "slack") -> None:
        self.name = name
        self.capabilities = (CAPABILITY_NOTIFY,)
        self.calls = 0

    async def send(self, message: NotificationMessage) -> NotificationReceipt:
        self.calls += 1
        if self.calls == 1:
            raise RetryableError("temporary failure")
        return NotificationReceipt(
            message_id="2",
            channel=self.name,
            provider=self.name,
            delivered=True,
        )


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------


def test_slack_metadata_and_registration_describe_the_provider() -> None:
    provider = _slack(lambda request: httpx.Response(200, text="ok"))

    assert provider.name == "slack"
    assert provider.capabilities == (CAPABILITY_NOTIFY,)
    assert provider.webhook_url == WEBHOOK_URL
    assert provider.channel == "#alerts"
    assert provider_registry.resolve("slack") is SlackNotifier
    assert provider_registry.resolve("slack_notifier") is SlackNotifier


def test_slack_send_text_message_posts_json_to_webhook() -> None:
    requests, handler = _capture()
    provider = _slack(handler)
    message = _message(title="")

    receipt = _run(provider.send(message))

    submitted = requests[0]
    assert submitted.method == "POST"
    assert str(submitted.url) == WEBHOOK_URL
    assert submitted.headers["content-type"].startswith("application/json")
    body = json.loads(submitted.content)
    assert body["text"] == "A run is waiting"
    assert body["channel"] == "#alerts"
    assert body["username"] == "aivia"
    assert "blocks" not in body
    assert "attachments" not in body
    assert receipt.channel == "#alerts"
    assert receipt.provider == "slack"
    assert receipt.delivered is True


def test_slack_title_and_text_are_combined_into_one_message() -> None:
    requests, handler = _capture()
    provider = _slack(handler)
    message = _message(title="  Review needed  ", text="  A run is waiting  ")

    _run(provider.send(message))

    body = json.loads(requests[0].content)
    assert body["text"] == "Review needed\nA run is waiting"


def test_slack_buttons_become_block_kit_actions() -> None:
    requests, handler = _capture()
    provider = _slack(handler)
    message = _message(
        buttons=[
            NotificationButton(label="Approve", callback_data="approve"),
            NotificationButton(label="Open", url=IMAGE_URL),
        ]
    )

    _run(provider.send(message))

    body = json.loads(requests[0].content)
    assert body["blocks"] == [
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Approve"},
                    "action_id": "approve",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Open"},
                    "url": IMAGE_URL,
                },
            ],
        }
    ]


def test_slack_media_url_becomes_an_attachment() -> None:
    requests, handler = _capture()
    provider = _slack(handler)
    media = NotificationMedia(url=IMAGE_URL, media_type="image/png", caption="preview")
    message = _message(media=media)

    _run(provider.send(message))

    body = json.loads(requests[0].content)
    attachment = body["attachments"][0]
    assert attachment["image_url"] == IMAGE_URL
    assert attachment["text"] == "preview"
    assert attachment["fallback"] == "preview"


def test_slack_missing_webhook_url_is_rejected() -> None:
    with pytest.raises(InvalidRequestError):
        SlackNotifier(webhook_url=None)


def test_slack_http_error_is_raised() -> None:
    _, handler = _capture(response=httpx.Response(400, text="invalid payload"))
    provider = _slack(handler)

    with pytest.raises(InvalidRequestError, match="HTTP 400"):
        _run(provider.send(_message()))


def test_slack_health_check_reports_healthy_with_a_webhook_url() -> None:
    provider = _slack(lambda request: httpx.Response(200, text="ok"))

    health = _run(provider.health())

    assert health.healthy is True


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


def test_email_metadata_and_registration_describe_the_provider() -> None:
    smtp, _ = _smtp_capture()
    provider = _email(smtp)

    assert provider.name == "email_notifier"
    assert provider.capabilities == (CAPABILITY_NOTIFY,)
    assert provider.smtp_host == "smtp.example.com"
    assert provider.from_address == "aivia@example.com"
    assert provider.target == "user@example.com"
    assert provider_registry.resolve("email_notifier") is EmailNotifier


def test_email_send_message_over_smtp() -> None:
    sent, smtp = _smtp_capture()
    provider = _email(smtp)
    message = _message(title="")

    with mock.patch("aivia.providers.notify_email.smtplib.SMTP", return_value=smtp):
        receipt = _run(provider.send(message))

    assert smtp.sendmail.called
    assert sent["from"] == "aivia@example.com"
    assert sent["to"] == ["user@example.com"]
    msg_bytes = sent["msg"]
    assert b"Subject: Aivia notification" in msg_bytes
    assert b"A run is waiting" in msg_bytes
    assert b"To: user@example.com" in msg_bytes
    assert receipt.channel == "user@example.com"
    assert receipt.provider == "email_notifier"
    assert receipt.delivered is True


def test_email_title_and_text_are_combined_in_the_body() -> None:
    sent, smtp = _smtp_capture()
    provider = _email(smtp)
    message = _message(title="  Review needed  ", text="  A run is waiting  ")

    with mock.patch("aivia.providers.notify_email.smtplib.SMTP", return_value=smtp):
        _run(provider.send(message))

    assert b"Review needed" in sent["msg"]
    assert b"A run is waiting" in sent["msg"]


def test_email_missing_fields_are_rejected() -> None:
    with pytest.raises(InvalidRequestError, match="smtp_host"):
        EmailNotifier(smtp_host="", from_address="a@b.com", target="c@d.com")
    with pytest.raises(InvalidRequestError, match="from_address"):
        EmailNotifier(smtp_host="h", from_address="", target="c@d.com")
    with pytest.raises(InvalidRequestError, match="target"):
        EmailNotifier(smtp_host="h", from_address="a@b.com", target="")


def test_email_health_check_reports_healthy_when_configured() -> None:
    smtp, _ = _smtp_capture()
    provider = _email(smtp)

    health = _run(provider.health())

    assert health.healthy is True


def test_email_check_health_reports_unhealthy_when_not_configured() -> None:
    smtp, _ = _smtp_capture()
    provider = _email(smtp)
    provider._smtp_host = ""

    report = _run(provider.check_health())

    assert report.healthy is False


# ---------------------------------------------------------------------------
# Chain fallback
# ---------------------------------------------------------------------------


def test_chain_sends_to_the_first_provider() -> None:
    slack = _RecordingNotifier("slack")
    email = _RecordingNotifier("email")
    chain = NotifyChain({"slack": slack, "email": email}, chain=["slack", "email"])

    receipt = _run(chain.notify(_message()))

    assert receipt.provider == "slack"
    assert len(slack.sent) == 1
    assert len(email.sent) == 0


def test_chain_falls_back_to_the_next_provider() -> None:
    slack = _RecordingNotifier("slack", exc=PermanentError("slack is down"))
    email = _RecordingNotifier("email")
    chain = NotifyChain({"slack": slack, "email": email}, chain=["slack", "email"])

    receipt = _run(chain.notify(_message()))

    assert receipt.provider == "email"
    assert len(slack.sent) == 1
    assert len(email.sent) == 1


def test_chain_retries_a_transient_failure_before_falling_back() -> None:
    flaky = _FlakyNotifier("slack")
    email = _RecordingNotifier("email")
    chain = NotifyChain(
        {"slack": flaky, "email": email},
        chain=["slack", "email"],
        max_attempts=2,
        retry_seconds=0.0,
    )

    receipt = _run(chain.notify(_message()))

    assert receipt.provider == "slack"
    assert flaky.calls == 2
    assert len(email.sent) == 0


def test_chain_is_exhausted_when_every_provider_fails() -> None:
    slack = _RecordingNotifier("slack", exc=PermanentError("slack is down"))
    email = _RecordingNotifier("email", exc=PermanentError("email is down"))
    chain = NotifyChain({"slack": slack, "email": email}, chain=["slack", "email"])

    with pytest.raises(
        ProviderExhaustedError, match="every notifier in the chain failed"
    ):
        _run(chain.notify(_message()))


def test_chain_without_a_configured_provider_is_exhausted() -> None:
    chain = NotifyChain({}, chain=[])

    with pytest.raises(ProviderExhaustedError, match="no notifier is configured"):
        _run(chain.notify(_message()))


def test_chain_uses_the_default_when_no_chain_is_given() -> None:
    slack = _RecordingNotifier("slack")
    chain = NotifyChain({"slack": slack}, default="slack")

    receipt = _run(chain.notify(_message()))

    assert receipt.provider == "slack"
    assert len(slack.sent) == 1


def test_chain_shares_the_same_message_model_as_the_notifiers() -> None:
    slack = _RecordingNotifier("slack")
    email = _RecordingNotifier("email")
    chain = NotifyChain({"slack": slack, "email": email}, chain=["slack", "email"])

    media = NotificationMedia(url=IMAGE_URL, media_type="image/png", caption="preview")
    message = _message(
        text="A run is waiting",
        title="Review needed",
        media=media,
        buttons=[NotificationButton(label="Approve", callback_data="approve")],
    )

    _run(chain.notify(message))

    assert isinstance(slack.sent[0], NotificationMessage)
    assert slack.sent[0].text == "A run is waiting"
    assert slack.sent[0].title == "Review needed"
    assert slack.sent[0].has_media
    assert slack.sent[0].has_buttons
