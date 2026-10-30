"""Reference image support for image generation providers.

A provider opts into reference images by declaring the ``image_input``
capability. Requests that carry references must be routed only to those
providers; otherwise the references would be silently ignored. This module
answers the policy question in front of a call, filters provider candidates and
offers a guard that fails loudly when a caller tries to send references to a
provider that did not declare support.

The check is deliberately conservative: a provider that does not declare the
capability is treated as unable to consume references. An adapter may also
publish the legacy ``reference_images`` option token for the ``image_gen``
capability, which this module accepts as an equivalent declaration.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from aivia.core.errors import InvalidRequestError
from aivia.core.models.provider import ProviderMeta
from aivia.providers.base import Provider
from aivia.providers.protocols.image import (
    CAPABILITY_IMAGE_GEN,
    CAPABILITY_IMAGE_INPUT,
    ImageRequest,
)

__all__ = [
    "CAPABILITY_IMAGE_INPUT",
    "ReferenceInputNotSupportedError",
    "ensure_reference_support",
    "filter_providers_for_reference_input",
    "prepare_image_request",
    "required_image_capabilities",
    "supports_reference_images",
]

_REFERENCE_OPTION = "reference_images"


class ReferenceInputNotSupportedError(InvalidRequestError):
    """A request with references was sent to a provider without image input."""

    code = "reference_input_not_supported"

    def __init__(
        self,
        provider: str,
        *,
        count: int = 0,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.count = count
        merged: dict[str, Any] = {
            "provider": provider,
            "reference_images": count,
            "required_capability": CAPABILITY_IMAGE_INPUT,
        }
        if details:
            merged.update(details)
        super().__init__(
            f"provider {provider!r} does not declare {CAPABILITY_IMAGE_INPUT} support",
            details=merged,
        )


def _metadata(
    source: Provider | ProviderMeta | Mapping[str, Any],
) -> ProviderMeta:
    if isinstance(source, ProviderMeta):
        return source
    if isinstance(source, Provider):
        return source.metadata
    if isinstance(source, Mapping):
        return ProviderMeta.model_validate(source)
    raise TypeError("source must be a Provider, ProviderMeta or mapping")


def supports_reference_images(
    source: Provider | ProviderMeta | Mapping[str, Any],
) -> bool:
    """Whether *source* declares that it accepts reference images."""
    metadata = _metadata(source)
    if metadata.supports(CAPABILITY_IMAGE_INPUT):
        return True
    return _REFERENCE_OPTION in metadata.supported_options.get(CAPABILITY_IMAGE_GEN, ())


def required_image_capabilities(request: ImageRequest) -> tuple[str, ...]:
    """Return the extra capabilities *request* needs from its provider.

    A request without references needs nothing beyond ``image_gen``. A request
    with references requires a provider that declares ``image_input``, so the
    router can skip text-to-image-only providers instead of dropping the
    references.
    """
    if not isinstance(request, ImageRequest):
        raise TypeError("request must be an ImageRequest")
    if not request.reference_images:
        return ()
    return (CAPABILITY_IMAGE_INPUT,)


def filter_providers_for_reference_input(
    providers: Iterable[Provider],
    request: ImageRequest,
) -> tuple[Provider, ...]:
    """Return only the providers that may receive *request* with its references."""
    candidates = tuple(providers)
    if not required_image_capabilities(request):
        return candidates
    return tuple(
        provider for provider in candidates if supports_reference_images(provider)
    )


def ensure_reference_support(
    source: Provider | ProviderMeta | Mapping[str, Any],
    request: ImageRequest,
) -> None:
    """Raise when *request* carries references but *source* cannot accept them."""
    if not isinstance(request, ImageRequest):
        raise TypeError("request must be an ImageRequest")
    if not request.reference_images:
        return
    if supports_reference_images(source):
        return
    raise ReferenceInputNotSupportedError(
        _metadata(source).name or "provider",
        count=len(request.reference_images),
    )


def prepare_image_request(
    source: Provider | ProviderMeta | Mapping[str, Any],
    request: ImageRequest,
) -> ImageRequest:
    """Return *request* after proving *source* can receive its references."""
    ensure_reference_support(source, request)
    return request
