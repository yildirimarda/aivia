"""Filter providers by reference capability with a prompt-only fallback.

Providers that accept reference images declare the ``image_input`` capability.
When none of the configured providers declares it, references cannot be sent
and the caller falls back to prompt-only style transfer, warning that the
references were dropped.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable

from aivia.providers.base import Provider
from aivia.providers.image import (
    CAPABILITY_IMAGE_INPUT,
    filter_providers_for_reference_input,
)
from aivia.providers.protocols.image import ImageRequest

__all__ = [
    "reference_providers_for_request",
]


def reference_providers_for_request(
    providers: Iterable[Provider],
    request: ImageRequest,
) -> tuple[tuple[Provider, ...], str | None]:
    """Return the providers that may receive *request* with its references.

    Providers are filtered by the ``image_input`` capability. When no provider
    declares support, every provider is returned together with a warning
    message so the caller can fall back to prompt-only style transfer without
    references.

    Returns ``(providers, warning)`` where ``warning`` is ``None`` when no
    fallback is needed.
    """
    candidates = tuple(providers)
    if not request.reference_images:
        return candidates, None
    supported = filter_providers_for_reference_input(candidates, request)
    if supported:
        return supported, None
    message = (
        f"no provider declares {CAPABILITY_IMAGE_INPUT}; "
        "falling back to prompt-only style transfer without references"
    )
    warnings.warn(message, UserWarning, stacklevel=2)
    return candidates, message
