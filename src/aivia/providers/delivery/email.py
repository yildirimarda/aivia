"""Email delivery: send finished artifacts to a recipient over SMTP."""

from __future__ import annotations

import asyncio
import smtplib
from email.message import EmailMessage

from aivia.core.config.http import HttpClientConfig
from aivia.core.models.base import utcnow
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.providers.base import Provider
from aivia.providers.protocols.delivery import (
    CAPABILITY_DELIVERY,
    DeliveryMethod,
    DeliveryReceipt,
    DeliveryRequest,
)

__all__ = ["EmailDeliveryProvider", "MailDeliveryProvider"]

_DEFAULT_METADATA = ProviderMeta(
    name="email",
    capabilities=[CAPABILITY_DELIVERY],
    pricing={CAPABILITY_DELIVERY: ProviderPrice(amount_usd=0.0)},
)


def _recipients(target: str) -> list[str]:
    return [address.strip() for address in target.split(",") if address.strip()]


def _build_message(
    request: DeliveryRequest,
    from_address: str,
    subject: str,
) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = subject or "Your personalized card"
    message["From"] = from_address
    message["To"] = ", ".join(_recipients(request.target))
    message.set_content(request.message or "")
    for index, artifact in enumerate(request.artifacts):
        if artifact.data is not None:
            message.add_attachment(
                artifact.data,
                maintype="application",
                subtype="octet-stream",
                filename=artifact.filename or f"artifact-{index}",
            )
        elif artifact.path is not None:
            with open(artifact.path, "rb") as handle:
                message.add_attachment(
                    handle.read(),
                    maintype="application",
                    subtype="octet-stream",
                    filename=artifact.filename,
                )
    return message


def _send_smtp(
    smtp_host: str,
    smtp_port: int,
    from_address: str,
    recipients: list[str],
    message: EmailMessage,
    username: str = "",
    password: str = "",
    timeout_seconds: float = 30.0,
) -> None:
    with smtplib.SMTP(smtp_host, smtp_port, timeout=timeout_seconds) as smtp:
        if username and password and smtp.has_extn("STARTTLS"):
            smtp.starttls()
            smtp.login(username, password)
        smtp.sendmail(from_address, recipients, message.as_bytes())


class EmailDeliveryProvider(Provider):
    """Send finished artifacts to a recipient over SMTP.

    Credentials are resolved by the caller and passed in as plain strings; the
    provider never reads a secrets file itself. ``timeout_seconds`` bounds the
    SMTP conversation.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        *,
        from_address: str = "",
        subject: str = "Your personalized card",
        username: str = "",
        password: str = "",
        timeout_seconds: float = 30.0,
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
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._from_address = from_address
        self._subject = subject
        self._username = username
        self._password = password
        self._timeout_seconds = timeout_seconds

    @property
    def smtp_host(self) -> str:
        """The SMTP host the provider connects to."""
        return self._smtp_host

    @property
    def from_address(self) -> str:
        """The envelope ``From`` address."""
        return self._from_address

    @property
    def subject(self) -> str:
        """The default message subject."""
        return self._subject

    async def deliver(self, request: DeliveryRequest) -> DeliveryReceipt:
        """Send *request* as an email and acknowledge acceptance."""
        if not isinstance(request, DeliveryRequest):
            raise TypeError("request must be a DeliveryRequest")
        if request.method != DeliveryMethod.EMAIL:
            raise ValueError(
                f"email provider cannot deliver with method {request.method.value!r}"
            )
        if not self._from_address:
            raise ValueError("email provider requires a from_address")
        if not request.target:
            raise ValueError("email provider requires a recipient target")
        subject = request.subject or self._subject
        message = _build_message(request, self._from_address, subject)
        recipients = _recipients(request.target)
        await asyncio.to_thread(
            _send_smtp,
            self._smtp_host,
            self._smtp_port,
            self._from_address,
            recipients,
            message,
            self._username,
            self._password,
            self._timeout_seconds,
        )
        return DeliveryReceipt(
            method=DeliveryMethod.EMAIL,
            target=", ".join(recipients),
            provider=self.name,
            delivered_at=utcnow(),
            metadata={
                "recipients": recipients,
                "subject": message["Subject"],
                "attachments": len(request.artifacts),
            },
        )


MailDeliveryProvider = EmailDeliveryProvider
"""Alias for :class:`EmailDeliveryProvider`."""
