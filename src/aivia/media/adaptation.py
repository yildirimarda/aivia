"""Apply the provider-declared adaptation layer before a media submission.

Every media service selects a provider through the router, so the provider is
only known inside the call the router executes. This module turns the selected
provider's ``ProviderMeta.supported_options`` into :class:`MediaConstraints`
and adapts the request with the shared layer, keeping the capability mapping in
one place instead of repeating it in each service.
"""

from __future__ import annotations

from typing import Any

from aivia.providers.base import Provider
from aivia.providers.media.adapt import (
    AdaptationResult,
    MediaConstraints,
    adapt_request,
)
from aivia.providers.media.dto import MediaKind
from aivia.providers.protocols.image import CAPABILITY_IMAGE_GEN
from aivia.providers.protocols.music import CAPABILITY_MUSIC_GEN
from aivia.providers.protocols.tts import CAPABILITY_TTS
from aivia.providers.protocols.video import CAPABILITY_VIDEO_GEN

__all__ = ["CAPABILITIES", "adapt_for_provider", "constraints_for"]

CAPABILITIES: dict[MediaKind, str] = {
    MediaKind.IMAGE: CAPABILITY_IMAGE_GEN,
    MediaKind.VIDEO: CAPABILITY_VIDEO_GEN,
    MediaKind.MUSIC: CAPABILITY_MUSIC_GEN,
    MediaKind.SPEECH: CAPABILITY_TTS,
}
"""The capability whose option tokens describe each media kind."""


def _kind(value: MediaKind | str) -> MediaKind:
    try:
        return value if isinstance(value, MediaKind) else MediaKind(value)
    except ValueError:
        raise ValueError(f"unsupported media kind: {value!r}") from None


def constraints_for(
    provider: Provider,
    kind: MediaKind | str,
) -> MediaConstraints:
    """Return the constraints *provider* declares for *kind*."""
    if not isinstance(provider, Provider):
        raise TypeError("provider must be a Provider")
    capability = CAPABILITIES[_kind(kind)]
    return MediaConstraints.from_metadata(provider.metadata, capability)


def adapt_for_provider(
    request: Any,
    provider: Provider,
    kind: MediaKind | str,
    *,
    seed: int | None = None,
) -> AdaptationResult[Any]:
    """Adapt *request* to the limits *provider* declares for *kind*.

    *seed* is a value the service generated itself when the caller did not ask
    for one. It is only injected when the provider accepts a seed, so an
    internal reproducibility detail cannot be rejected as an unsupported
    caller option.
    """
    constraints = constraints_for(provider, kind)
    result = adapt_request(request, constraints)
    if (
        seed is None
        or getattr(request, "seed", None) is not None
        or getattr(result.request, "seed", None) is not None
    ):
        return result
    if constraints.supports is not None and "seed" not in constraints.supports:
        return result
    allowed = constraints.supported_options.get("seed")
    if allowed is not None and str(seed) not in allowed:
        return result
    return AdaptationResult(
        request=result.request.model_copy(update={"seed": seed}),
        warnings=result.warnings,
    )
