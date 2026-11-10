"""Personalized orders: lifecycle, validation, delivery and pronunciation.

The orders module turns an occasion, a recipient name and optional variables
into generated media. Name pronunciation is verified by transcription and the
spoken name is regenerated with phonetic hints up to the configured retry
limit.
"""

from __future__ import annotations

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

__all__ = [
    "AcceptProvidedAssetsError",
    "AssetKind",
    "NamePronunciationError",
    "NamePronunciationResult",
    "ProvidedAsset",
    "ProvidedAssets",
    "accept_provided_assets",
    "verify_name_pronunciation",
]
