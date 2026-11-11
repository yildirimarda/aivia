"""Email notifier: send a notification message to a recipient over SMTP.

This adapter delivers a :class:`~aivia.providers.protocols.notify.NotificationMessage`
to a fixed recipient, unlike the delivery email provider which sends finished
artifacts. Credentials are resolved by the caller and passed in as plain
strings; the provider never reads a secrets file itself. ``timeout_seconds``
bounds the SMTP conversation.
"""

from __future__ import annotations

import asyncio
import smtplib
from email.message import EmailMessage

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import InvalidRequestError
from aivia.core.models.base import utcnow
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.providers.base import HealthStatus, Provider, ProviderHealth
from aivia.providers.protocols.notify import (
    CAPABILITY_NOTIFY,
    NotificationMessage,
    NotificationReceipt,
)
from aivia.providers.registry import register_provider

__all__ = [
    "EmailNotificationProvider",
    "EmailNotifier",
]

_DEFAULT_SUBJECT = "Aivia notification"
_DEFAULT_PORT = 587
_DEFAULT_TIMEOUT_SECONDS = 30.0


def _recipients(target: str) -> list[str]:
    return [address.strip() for address in target.split(",") if address.strip()]


def _build_message(
    message: NotificationMessage,
    from_address: str,
    target: str,
    subject: str,
) -> EmailMessage:
    email = EmailMessage()
    email["Subject"] = subject or _DEFAULT_SUBJECT
    email["From"] = from_address
    email["To"] = ", ".join(_recipients(target))
    parts: list[str] = []
    if message.title:
        parts.append(message.title)
    if message.text:
        parts.append(message.text)
    email.set_content("\n".join(parts) or "")
    return email


def _send_smtp(
    smtp_host: str,
    smtp_port: int,
    from_address: str,
    recipients: list[str],
    email: EmailMessage,
    username: str = "",
    password: str = "",
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> None:
    with smtplib.SMTP(smtp_host, smtp_port, timeout=timeout_seconds) as smtp:
        if username and password and smtp.has_extn("STARTTLS"):
            smtp.starttls()
            smtp.login(username, password)
        smtp.sendmail(from_address, recipients, email.as_bytes())


class EmailNotifier(Provider):
    """A notifier that delivers notification messages to a recipient over SMTP.

    ``smtp_host``, ``from_address`` and ``target`` (the recipient) are required.
    ``username``/``password`` authenticate when the server requires it.
    """

    default_metadata = ProviderMeta(
        name="email_notifier",
        capabilities=[CAPABILITY_NOTIFY],
        pricing={CAPABILITY_NOTIFY: ProviderPrice(amount_usd=0.0)},
    )

    def __init__(
        self,
        *,
        smtp_host: str,
        smtp_port: int = _DEFAULT_PORT,
        from_address: str = "",
        target: str = "",
        subject: str = _DEFAULT_SUBJECT,
        username: str = "",
        password: str = "",
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
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
        if not smtp_host or not str(smtp_host).strip():
            raise InvalidRequestError(
                "email notifier requires an smtp_host",
                details={"field": "smtp_host"},
            )
        if not from_address or not str(from_address).strip():
            raise InvalidRequestError(
                "email notifier requires a from_address",
                details={"field": "from_address"},
            )
        if not target or not str(target).strip():
            raise InvalidRequestError(
                "email notifier requires a recipient target",
                details={"field": "target"},
            )
        self._smtp_host = str(smtp_host).strip()
        self._smtp_port = int(smtp_port)
        self._from_address = str(from_address).strip()
        self._target = str(target).strip()
        self._subject = subject
        self._username = username
        self._password = password
        self._timeout_seconds = float(timeout_seconds)
        self._sent: list[NotificationMessage] = []

    @property
    def smtp_host(self) -> str:
        """The SMTP host the provider connects to."""
        return self._smtp_host

    @property
    def from_address(self) -> str:
        """The envelope ``From`` address."""
        return self._from_address

    @property
    def target(self) -> str:
        """The recipient notification messages are delivered to."""
        return self._target

    @property
    def subject(self) -> str:
        """The default message subject."""
        return self._subject

    @property
    def sent(self) -> tuple[NotificationMessage, ...]:
        """Every message this instance has sent, in order."""
        return tuple(self._sent)

    async def check_health(self) -> ProviderHealth:
        """Report healthy when host, sender and recipient are configured."""
        if self._smtp_host and self._from_address and self._target:
            return ProviderHealth(status=HealthStatus.HEALTHY, message="ok")
        return ProviderHealth(
            status=HealthStatus.UNHEALTHY,
            message="smtp_host, from_address and target are required",
        )

    async def send(self, message: NotificationMessage) -> NotificationReceipt:
        """Send *message* to the configured recipient and acknowledge it."""
        if not isinstance(message, NotificationMessage):
            raise TypeError("message must be a NotificationMessage")
        email = _build_message(message, self._from_address, self._target, self._subject)
        recipients = _recipients(self._target)
        await asyncio.to_thread(
            _send_smtp,
            self._smtp_host,
            self._smtp_port,
            self._from_address,
            recipients,
            email,
            self._username,
            self._password,
            self._timeout_seconds,
        )
        self._sent.append(message)
        return NotificationReceipt(
            message_id="",
            channel=self._target,
            provider=self.name,
            delivered=True,
            delivered_at=utcnow(),
            metadata={
                "recipients": recipients,
                "subject": email["Subject"],
            },
        )


EmailNotificationProvider = EmailNotifier
"""Alias for :class:`EmailNotifier` for callers that prefer provider wording."""

register_provider("email_notifier", EmailNotifier, module=__name__)
register_provider("email_notify", EmailNotifier, module=__name__)
