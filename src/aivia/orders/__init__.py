"""Personalized orders: lifecycle, validation, delivery and packaging.

The orders module turns an occasion, a recipient name and optional variables
into generated media. Name pronunciation is verified by transcription and the
spoken name is regenerated with phonetic hints up to the configured retry limit.
Packaging turns rendered masters into pinned finals: MP4, animated GIF, animated
WebP, MP3, poster and web card at the configured size presets.
"""

from __future__ import annotations

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
    "SizePreset",
    "SongMode",
    "accept_provided_assets",
    "affected_stages",
    "animate_still",
    "apply_placeholders",
    "feedback_for_order",
    "generate_batch_songs",
    "minimize_order",
    "order_music_licenses",
    "overlay_name",
    "package_outputs",
    "placeholder_for",
    "record_feedback",
    "redact_order",
    "revision_stages",
    "store_order_music_licenses",
    "verify_name_pronunciation",
]
