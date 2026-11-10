"""Delivery providers: folder, expiring storage link, email and signed webhook."""

from __future__ import annotations

from aivia.providers.delivery.chain import DeliveryChain
from aivia.providers.delivery.email import EmailDeliveryProvider
from aivia.providers.delivery.factory import (
    build_delivery_provider,
    build_delivery_providers,
)
from aivia.providers.delivery.folder import (
    FolderDeliveryProvider,
    LocalFolderDeliveryProvider,
)
from aivia.providers.delivery.link import (
    ExpiringStorageLinkDeliveryProvider,
    LinkDeliveryProvider,
)
from aivia.providers.delivery.webhook import (
    SignedWebhookDeliveryProvider,
    WebhookDeliveryProvider,
)

__all__ = [
    "DeliveryChain",
    "EmailDeliveryProvider",
    "ExpiringStorageLinkDeliveryProvider",
    "FolderDeliveryProvider",
    "LinkDeliveryProvider",
    "LocalFolderDeliveryProvider",
    "SignedWebhookDeliveryProvider",
    "WebhookDeliveryProvider",
    "build_delivery_provider",
    "build_delivery_providers",
]
