"""Derive the provenance fields stored beside a generated media asset.

Providers describe the rights attached to their output in provider metadata,
while one request may carry its own license, attribution or alt text. These
helpers read both places in a fixed order so every asset row records the same
kind of license and accessibility text regardless of the capability that made
it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aivia.core.models.asset import AssetLicense

__all__ = [
    "alt_text_from",
    "license_from",
]

_LICENSE_KEYS = (
    "license",
    "licence",
    "spdx",
    "license_url",
    "licence_url",
    "terms_url",
    "attribution",
    "credit",
    "attribution_text",
    "attribution_required",
    "requires_attribution",
    "commercial_use",
    "allow_commercial",
    "share_alike",
)
_ALT_TEXT_KEYS = ("alt_text", "alt", "accessibility_text")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _license_value(value: Any) -> AssetLicense | None:
    if isinstance(value, AssetLicense):
        return value
    metadata = value if isinstance(value, Mapping) else None
    if metadata is None:
        return None
    raw = metadata.get("license", metadata.get("licence"))
    if isinstance(raw, Mapping):
        return AssetLicense.model_validate(dict(raw))
    values = {key: metadata[key] for key in _LICENSE_KEYS if key in metadata}
    if not values:
        return None
    return AssetLicense.model_validate(values)


def license_from(
    *sources: Any,
    provider: Any = None,
) -> AssetLicense:
    """Return the first known license among *sources* or *provider* metadata."""
    for source in sources:
        found = _license_value(source)
        if found is not None and found.is_known:
            return found
    metadata = getattr(provider, "metadata", None)
    found = _license_value(getattr(metadata, "license", None))
    if found is not None:
        return found
    return AssetLicense()


def alt_text_from(*sources: Any) -> str:
    """Return the first non-empty alt text found in *sources*."""
    for source in sources:
        metadata = source if isinstance(source, Mapping) else None
        if metadata is None:
            continue
        for key in _ALT_TEXT_KEYS:
            text = _text(metadata.get(key))
            if text:
                return text
    return ""
