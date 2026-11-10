"""Music license metadata stored on personalized orders.

An order whose media includes music keeps the license of every track in its
``metadata`` under the ``music_licenses`` key. A license only clears commercial
use when it names itself or links to its terms and explicitly states
``commercial_use`` is true; missing or unstated rights are never treated as
cleared.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from aivia.core.models.asset import AssetLicense

__all__ = ["order_music_licenses", "store_order_music_licenses"]

_MUSIC_LICENSES_KEY = "music_licenses"


def order_music_licenses(order: Mapping[str, Any]) -> tuple[AssetLicense, ...]:
    """Return the music licenses recorded on *order*'s metadata.

    Returns an empty tuple when the order records no music licenses, so an
    order without music is never blocked for lacking a license.
    """
    metadata = order.get("metadata")
    if not isinstance(metadata, Mapping):
        return ()
    raw = metadata.get(_MUSIC_LICENSES_KEY)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    licenses: list[AssetLicense] = []
    for entry in raw:
        if isinstance(entry, Mapping):
            licenses.append(AssetLicense.model_validate(entry))
    return tuple(licenses)


def store_order_music_licenses(
    order: Mapping[str, Any], licenses: Sequence[AssetLicense]
) -> dict[str, Any]:
    """Return *order* with *licenses* recorded in its metadata.

    The returned mapping is a shallow copy; the caller is responsible for
    persisting it through the order repository.
    """
    metadata = dict(order.get("metadata") or {})
    metadata[_MUSIC_LICENSES_KEY] = [
        license.model_dump(mode="json") for license in licenses
    ]
    return dict(order, metadata=metadata)
