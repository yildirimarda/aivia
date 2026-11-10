"""Personalized orders: lifecycle, validation, delivery and packaging.

The orders module turns an occasion, a recipient name and optional variables
into generated media. Name pronunciation is verified by transcription and the
spoken name is regenerated with phonetic hints up to the configured retry limit.
Packaging turns rendered masters into pinned finals: MP4, animated GIF, animated
WebP, MP3, poster and web card at the configured size presets.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from aivia.core.db.repos import OrderRepository
from aivia.orders.animate import animate_still
from aivia.orders.license import (
    order_music_licenses,
    store_order_music_licenses,
)
from aivia.orders.name_overlay import (
    DEFAULT_FONT_FAMILY,
    DEFAULT_FONT_SIZE,
    overlay_name,
)
from aivia.orders.packaging import (
    DEFAULT_SIZE_PRESETS,
    PackagedOutput,
    Packager,
    PackagingError,
    SizePreset,
    package_outputs,
)
from aivia.orders.privacy import (
    DataProcessingEntry,
    DataProcessingRecord,
    apply_placeholders,
    minimize_order,
    placeholder_for,
    redact_order,
)
from aivia.orders.pronunciation import (
    NamePronunciationError,
    NamePronunciationResult,
    verify_name_pronunciation,
)
from aivia.orders.provided_assets import (
    AcceptProvidedAssetsError,
    AssetKind,
    ProvidedAsset,
    ProvidedAssets,
    accept_provided_assets,
)
from aivia.orders.receipt import Receipt, export_receipt
from aivia.orders.review import (
    OrderFeedback,
    OrderReview,
    affected_stages,
    feedback_for_order,
    record_feedback,
    revision_stages,
)
from aivia.orders.song_mode import (
    NamedSong,
    SongMode,
    generate_batch_songs,
)


def visitor_hash(salt: str, day: str, visitor_id: str) -> str:
    """Compute a daily salted hash of a visitor identifier.

    The same *visitor_id* on the same *day* produces the same hash, but a
    different *day* (or a different *salt*) produces a different hash, so
    visitors cannot be linked across days.
    """
    return hashlib.sha256(f"{salt}:{day}:{visitor_id}".encode()).hexdigest()


def today() -> str:
    """Return today's date in ISO format (YYYY-MM-DD) for hit bucketing."""
    return datetime.now(UTC).date().isoformat()


def order(
    repository: OrderRepository,
    *,
    character: str | None = None,
    **values: Any,
) -> dict[str, Any]:
    """Create one durable order, optionally bound to a character.

    *character* is optional. When it is a non-blank string it is stored on the
    order; when it is absent or blank the order is created with no character.
    Both paths run: an order without a character proceeds normally and an
    order with a selected character records that choice on the stored record.
    """
    text = character.strip() if isinstance(character, str) else ""
    values = dict(values)
    values.setdefault("character", text)
    return repository.create(**values)


__all__ = [
    "DEFAULT_FONT_FAMILY",
    "DEFAULT_FONT_SIZE",
    "DEFAULT_SIZE_PRESETS",
    "AcceptProvidedAssetsError",
    "AssetKind",
    "DataProcessingEntry",
    "DataProcessingRecord",
    "NamePronunciationError",
    "NamePronunciationResult",
    "NamedSong",
    "OrderFeedback",
    "OrderReview",
    "PackagedOutput",
    "Packager",
    "PackagingError",
    "ProvidedAsset",
    "ProvidedAssets",
    "Receipt",
    "SizePreset",
    "SongMode",
    "accept_provided_assets",
    "affected_stages",
    "animate_still",
    "apply_placeholders",
    "export_receipt",
    "feedback_for_order",
    "generate_batch_songs",
    "minimize_order",
    "order",
    "order_music_licenses",
    "overlay_name",
    "package_outputs",
    "placeholder_for",
    "record_feedback",
    "redact_order",
    "revision_stages",
    "store_order_music_licenses",
    "today",
    "verify_name_pronunciation",
    "visitor_hash",
]
