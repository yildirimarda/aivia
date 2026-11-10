"""Signed webhook delivery: POST a signed payload to an allowlisted endpoint."""

from __future__ import annotations

import hashlib
import hmac
import json
from urllib.parse import urlsplit

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import InvalidRequestError
from aivia.core.http import HttpClient
from aivia.core.models.base import utcnow
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.providers.base import Provider
from aivia.providers.protocols.delivery import (
    CAPABILITY_DELIVERY,
    DeliveryMethod,
    DeliveryReceipt,
    DeliveryRequest,
)

__all__ = ["SignedWebhookDeliveryProvider", "WebhookDeliveryProvider"]

_DEFAULT_METADATA = ProviderMeta(
    name="webhook",
    capabilities=[CAPABILITY_DELIVERY],
    pricing={CAPABILITY_DELIVERY: ProviderPrice(amount_usd=0.0)},
)


def _host(url: str) -> str:
    """Return the lower-case host of *url*, with a leading ``www.`` removed."""
    host = (urlsplit(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _is_allowlisted(url: str, allowlist: list[str]) -> bool:
    if not allowlist:
        return True
    return _host(url) in set(allowlist)


def _sign(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


class SignedWebhookDeliveryProvider(Provider):
    """POST a signed JSON payload to an allowlisted webhook endpoint.

    ``allowlist`` restricts the hosts the provider may call; any other host is
    refused before a request is made. The request body is signed with HMAC-SHA
    256 and carried in ``signature_header`` (default ``X-Aivia-Signature``).
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        url: str,
        *,
        secret: str = "",
        allowlist: list[str] | None = None,
        signature_header: str = "X-Aivia-Signature",
        http_client: HttpClient | None = None,
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
        self._url = url
        self._secret = secret
        self._allowlist = list(allowlist or [])
        self._signature_header = signature_header
        self._http_client = http_client or HttpClient(HttpClientConfig())

    @property
    def url(self) -> str:
        """The webhook endpoint."""
        return self._url

    @property
    def secret(self) -> str:
        """The signing secret (redacted by the base class when logged)."""
        return self._secret

    @property
    def allowlist(self) -> list[str]:
        """The hosts the provider may call."""
        return list(self._allowlist)

    @property
    def signature_header(self) -> str:
        """The HTTP header carrying the HMAC signature."""
        return self._signature_header

    def _payload(self, request: DeliveryRequest) -> bytes:
        artifacts = [
            {
                "filename": artifact.filename,
                "media_type": artifact.media_type,
                "size_bytes": artifact.size_bytes,
            }
            for artifact in request.artifacts
        ]
        payload = {
            "order_id": request.order_id,
            "run_id": request.run_id,
            "subject": request.subject,
            "message": request.message,
            "artifacts": artifacts,
            "scheduled_for": (
                request.scheduled_for.isoformat() if request.scheduled_for else None
            ),
        }
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    async def deliver(self, request: DeliveryRequest) -> DeliveryReceipt:
        """Sign and POST *request*, returning the endpoint acknowledgement."""
        if not isinstance(request, DeliveryRequest):
            raise TypeError("request must be a DeliveryRequest")
        if request.method != DeliveryMethod.WEBHOOK:
            raise ValueError(
                f"webhook provider cannot deliver with method {request.method.value!r}"
            )
        if not _is_allowlisted(self._url, self._allowlist):
            raise InvalidRequestError(
                f"webhook host {_host(self._url)!r} is not allowlisted",
                details={"host": _host(self._url), "allowlist": self._allowlist},
            )
        body = self._payload(request)
        headers = {
            "Content-Type": "application/json",
            self._signature_header: _sign(self._secret, body),
        }
        response = await self._http_client.post(
            self._url, content=body, headers=headers, raise_for_status=True
        )
        await response.aclose()
        return DeliveryReceipt(
            method=DeliveryMethod.WEBHOOK,
            target=self._url,
            provider=self.name,
            url=self._url,
            delivered_at=utcnow(),
            metadata={
                "status": response.status_code,
                "signature_header": self._signature_header,
            },
        )


WebhookDeliveryProvider = SignedWebhookDeliveryProvider
"""Alias for :class:`SignedWebhookDeliveryProvider`."""
