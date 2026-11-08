"""Attach character reference images to image generation requests.

A character carries reference images on disk next to its character file. When a
provider declares the ``character_reference`` capability the images are loaded,
converted to media inputs and attached to a copy of the request, adapted to the
provider's declared limits so snapping, clamping and rejected options surface
explicitly instead of being silently dropped. Providers that do not declare the
capability never receive the references.
"""

from __future__ import annotations

import mimetypes
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from aivia.characters.capabilities import (
    CAPABILITY_CHARACTER_REFERENCE,
    enforce_character_reference_limit,
    filter_providers_for_character_references,
)
from aivia.characters.schema import Character
from aivia.core.errors import InvalidRequestError
from aivia.providers.base import Provider
from aivia.providers.media.adapt import MediaConstraints, adapt_image_request
from aivia.providers.protocols.image import (
    CAPABILITY_IMAGE_GEN,
    ImageRequest,
)
from aivia.providers.protocols.media import MediaInput

__all__ = [
    "CharacterReferenceAttachError",
    "attach_character_references",
    "load_character_references",
    "prepare_character_image_request",
]


class CharacterReferenceAttachError(InvalidRequestError):
    """A character's reference images could not be attached to a request."""

    code = "character_reference_attach"

    def __init__(
        self,
        provider: str,
        *,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = {"provider": provider, "reason": reason}
        if details:
            merged.update(details)
        super().__init__(
            f"could not attach character references to {provider!r}: {reason}",
            details=merged,
        )
        self.provider = provider
        self.reason = reason


def _media_type_for(path: Path) -> str | None:
    """Return a media MIME type for *path*, guessed from its extension."""
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed is not None and guessed.startswith(("image/", "video/", "audio/")):
        return guessed
    return None


def _constraints_for(provider: Provider) -> MediaConstraints:
    """Derive adaptation constraints for *provider*.

    A provider that declares :data:`CAPABILITY_CHARACTER_REFERENCE` can consume
    character reference images, so the ``reference_images`` option is treated as
    accepted even when the provider did not spell it out in its published
    options. When the provider published no allowlist at all (``supports is
    None``) nothing is rejected on support grounds.
    """
    constraints = MediaConstraints.from_metadata(
        provider.metadata, CAPABILITY_IMAGE_GEN
    )
    if CAPABILITY_CHARACTER_REFERENCE not in provider.metadata.capabilities:
        return constraints
    if constraints.supports is None:
        return constraints
    if "reference_images" in constraints.supports:
        return constraints
    supports = set(constraints.supports)
    supports.add("reference_images")
    return constraints.model_copy(update={"supports": frozenset(supports)})


def load_character_references(
    character: Character, base_dir: str | Path
) -> list[MediaInput]:
    """Load *character*'s reference images from *base_dir* as media inputs.

    Each reference image path is relative to the character file's directory, so
    *base_dir* is that directory. The file bytes are read inline so a provider
    receives self-contained media instead of a path it cannot resolve.
    """
    base = Path(base_dir)
    inputs: list[MediaInput] = []
    for index, relative in enumerate(character.reference_images):
        path = base / relative
        if not path.is_file():
            raise CharacterReferenceAttachError(
                character.name,
                reason=f"reference image {relative!r} is missing on disk",
                details={
                    "character": character.name,
                    "reference_index": index,
                    "asset": relative,
                    "file": str(path),
                },
            )
        media_type = _media_type_for(path) or "image/png"
        inputs.append(
            MediaInput(
                data=path.read_bytes(),
                media_type=media_type,
                filename=path.name,
            )
        )
    return inputs


def attach_character_references(
    request: ImageRequest, references: list[MediaInput]
) -> ImageRequest:
    """Return a copy of *request* with *references* attached as reference images.

    The original request is left untouched.
    """
    return request.model_copy(update={"reference_images": list(references)})


def prepare_character_image_request(
    character: Character,
    request: ImageRequest,
    providers: Iterable[Provider],
    max_reference_images: int,
    base_dir: str | Path,
) -> tuple[list[tuple[Provider, ImageRequest, MediaConstraints]], str | None]:
    """Prepare an image request for every provider that supports character references.

    Enforces the reference-image limit, then filters providers by the
    ``character_reference`` capability. For each surviving provider the
    character's reference images are loaded from *base_dir*, attached to a copy
    of *request* and adapted to that provider's declared limits, so every change
    (size snapping, count clamping, rejected options) is explicit.

    Returns ``(prepared, warning)`` where ``prepared`` is a list of
    ``(provider, adapted_request, constraints)`` triples and ``warning`` is
    ``None`` unless no provider declares the capability and the character carries
    references, in which case every provider is returned unprepared with a
    fallback warning.

    When the character carries no reference images every provider is returned
    with an unadapted copy of *request* and no references attached.
    """
    enforce_character_reference_limit(character, max_reference_images)
    candidates = tuple(providers)
    if not character.reference_images:
        return [
            (provider, request, _constraints_for(provider)) for provider in candidates
        ], None

    supported = filter_providers_for_character_references(candidates, character)
    if not supported:
        message = (
            f"no provider declares {CAPABILITY_CHARACTER_REFERENCE}; "
            "falling back to prompt-only generation without character references"
        )
        return [
            (provider, request, _constraints_for(provider)) for provider in candidates
        ], message

    references = load_character_references(character, base_dir)
    prepared: list[tuple[Provider, ImageRequest, MediaConstraints]] = []
    for provider in supported:
        constraints = _constraints_for(provider)
        attached = attach_character_references(request, references)
        try:
            adapted = adapt_image_request(attached, constraints)
        except Exception as exc:
            raise CharacterReferenceAttachError(
                provider.name,
                reason=str(exc),
                details={
                    "character": character.name,
                    "provider": provider.name,
                },
            ) from exc
        prepared.append((provider, adapted.request, constraints))

    return prepared, None
