"""Filter providers by character capability flags and enforce the reference limit.

A provider that can render a recurring character must declare the
``character_reference`` capability so a run sends the character's reference
images only to providers that can consume them. When no provider declares it,
the references are dropped and a warning is emitted so the caller can fall
back to prompt-only generation. The character's configured
``max_reference_images`` cap is enforced: a character carrying more references
than allowed is rejected loudly instead of silently truncated.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Mapping
from typing import Any

from aivia.characters.schema import Character
from aivia.core.errors import InvalidRequestError
from aivia.core.models.provider import ProviderMeta
from aivia.providers.base import Provider

__all__ = [
    "CAPABILITY_CHARACTER_REFERENCE",
    "CharacterReferenceLimitError",
    "CharacterReferenceNotSupportedError",
    "enforce_character_reference_limit",
    "filter_providers_for_character_references",
    "prepare_character_request",
    "supports_character_references",
]

CAPABILITY_CHARACTER_REFERENCE = "character_reference"
"""The capability a provider declares when it accepts character reference images."""

_LOGGER = logging.getLogger(__name__)


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


def supports_character_references(
    source: Provider | ProviderMeta | Mapping[str, Any],
) -> bool:
    """Whether *source* declares that it accepts character reference images."""
    return _metadata(source).supports(CAPABILITY_CHARACTER_REFERENCE)


class CharacterReferenceNotSupportedError(InvalidRequestError):
    """A character with references was sent to a provider without the flag."""

    code = "character_reference_not_supported"

    def __init__(
        self,
        provider: str,
        *,
        count: int = 0,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = {
            "provider": provider,
            "reference_images": count,
            "required_capability": CAPABILITY_CHARACTER_REFERENCE,
        }
        if details:
            merged.update(details)
        super().__init__(
            (
                f"provider {provider!r} does not declare "
                f"{CAPABILITY_CHARACTER_REFERENCE} support"
            ),
            details=merged,
        )
        self.provider = provider
        self.count = count


class CharacterReferenceLimitError(InvalidRequestError):
    """A character carries more reference images than ``max_reference_images``."""

    code = "character_reference_limit"

    def __init__(
        self,
        character: str,
        *,
        count: int,
        limit: int,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = {
            "character": character,
            "reference_images": count,
            "max_reference_images": limit,
        }
        if details:
            merged.update(details)
        super().__init__(
            f"character {character!r} carries {count} reference images "
            f"but max_reference_images is {limit}",
            details=merged,
        )
        self.character = character
        self.count = count
        self.limit = limit


def enforce_character_reference_limit(
    character: Character, max_reference_images: int
) -> None:
    """Raise when *character* carries more references than *max_reference_images*.

    ``max_reference_images`` must be non-negative; a character with no
    references is always allowed.
    """
    if isinstance(max_reference_images, bool) or not isinstance(
        max_reference_images, int
    ):
        raise TypeError("max_reference_images must be an integer")
    if max_reference_images < 0:
        raise ValueError("max_reference_images must not be negative")
    count = len(character.reference_images)
    if count > max_reference_images:
        raise CharacterReferenceLimitError(
            character.name, count=count, limit=max_reference_images
        )


def filter_providers_for_character_references(
    providers: Iterable[Provider],
    character: Character,
) -> tuple[Provider, ...]:
    """Return only the providers that may receive *character* with its references.

    When the character carries no reference images every provider is returned.
    Otherwise only providers that declare
    :data:`CAPABILITY_CHARACTER_REFERENCE` are returned.
    """
    candidates = tuple(providers)
    if not character.reference_images:
        return candidates
    return tuple(
        provider for provider in candidates if supports_character_references(provider)
    )


def prepare_character_request(
    character: Character,
    providers: Iterable[Provider],
    max_reference_images: int,
) -> tuple[tuple[Provider, ...], str | None]:
    """Prepare *character* for delivery to providers.

    Enforces the reference-image limit, then filters providers by the character
    capability flag. When no provider declares
    :data:`CAPABILITY_CHARACTER_REFERENCE` and the character carries references,
    every provider is returned together with a warning so the caller can fall
    back to prompt-only generation without references.

    Returns ``(providers, warning)`` where ``warning`` is ``None`` when no
    fallback is needed.
    """
    enforce_character_reference_limit(character, max_reference_images)
    candidates = tuple(providers)
    if not character.reference_images:
        return candidates, None
    supported = filter_providers_for_character_references(candidates, character)
    if supported:
        return supported, None
    message = (
        f"no provider declares {CAPABILITY_CHARACTER_REFERENCE}; "
        "falling back to prompt-only generation without character references"
    )
    _LOGGER.warning(message)
    warnings.warn(message, UserWarning, stacklevel=2)
    return candidates, message
