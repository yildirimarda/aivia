"""Expiring storage link delivery: store artifacts and hand out expiring links."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from aivia.core.config.http import HttpClientConfig
from aivia.core.models.base import utcnow
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.providers.base import Provider
from aivia.providers.protocols.delivery import (
    CAPABILITY_DELIVERY,
    DeliveryArtifact,
    DeliveryMethod,
    DeliveryReceipt,
    DeliveryRequest,
)
from aivia.providers.protocols.storage import StorageObject, StoragePutRequest

__all__ = ["ExpiringStorageLinkDeliveryProvider", "LinkDeliveryProvider"]

_DEFAULT_METADATA = ProviderMeta(
    name="link",
    capabilities=[CAPABILITY_DELIVERY],
    pricing={CAPABILITY_DELIVERY: ProviderPrice(amount_usd=0.0)},
)


def _artifact_bytes(artifact: DeliveryArtifact, index: int) -> bytes:
    if artifact.data is not None:
        return artifact.data
    if artifact.path is not None:
        return Path(artifact.path).read_bytes()
    raise ValueError(f"link delivery cannot store artifact {index} from a URL")


class ExpiringStorageLinkDeliveryProvider(Provider):
    """Store artifacts in a storage backend and return an expiring link.

    The provider puts every artifact into the configured storage backend and
    reports a storage URL valid for ``expires_seconds``. The link itself is the
    delivery target; the recipient fetches the file before it expires.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        storage: Any,
        *,
        expires_seconds: int = 604_800,
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
        if expires_seconds <= 0:
            raise ValueError("expires_seconds must be positive")
        self._storage = storage
        self._expires_seconds = expires_seconds

    @property
    def storage(self) -> Any:
        """The storage backend artifacts are written to."""
        return self._storage

    @property
    def expires_seconds(self) -> int:
        """Seconds the returned link remains valid."""
        return self._expires_seconds

    async def _expires_at(self) -> datetime:
        return datetime.now(UTC) + timedelta(seconds=self._expires_seconds)

    async def deliver(self, request: DeliveryRequest) -> DeliveryReceipt:
        """Store every artifact and return the expiring link receipt."""
        if not isinstance(request, DeliveryRequest):
            raise TypeError("request must be a DeliveryRequest")
        if request.method != DeliveryMethod.LINK:
            raise ValueError(
                f"link provider cannot deliver with method {request.method.value!r}"
            )
        urls: list[str] = []
        for index, artifact in enumerate(request.artifacts):
            data = _artifact_bytes(artifact, index)
            key = f"deliveries/{index}/{utcnow().timestamp()}"
            metadata: dict[str, Any] = {}
            if artifact.filename:
                metadata["filename"] = artifact.filename
            stored: StorageObject = await self._storage.put(
                StoragePutRequest(
                    data=data,
                    key=key,
                    content_type=artifact.media_type,
                    metadata=metadata,
                )
            )
            urls.append(
                self._storage.url(stored.key, expires_seconds=self._expires_seconds)
            )
        return DeliveryReceipt(
            method=DeliveryMethod.LINK,
            target=urls[0] if urls else "",
            provider=self.name,
            delivered_at=utcnow(),
            url=urls[0] if urls else None,
            expires_at=await self._expires_at(),
            metadata={"urls": urls, "expires_seconds": self._expires_seconds},
        )


LinkDeliveryProvider = ExpiringStorageLinkDeliveryProvider
"""Alias for :class:`ExpiringStorageLinkDeliveryProvider`."""
