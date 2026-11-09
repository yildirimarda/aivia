"""Character rights policy: ownership, likeness consent and resemblance.

A character may only be used when its rights flags are satisfied: an owned
character is required when ``ownership_required`` is set, real-person likeness
is blocked unless ``allow_real_person_likeness`` is set, and an optional
resemblance warning is emitted when ``resemblance_warning`` is set.
"""

from __future__ import annotations

import logging
import warnings

from aivia.characters.schema import Character
from aivia.core.errors import ContentPolicyError

__all__ = [
    "CharacterRightsError",
    "enforce_character_rights",
]

_LOGGER = logging.getLogger(__name__)


class CharacterRightsError(ContentPolicyError):
    """A character cannot be used in the requested way."""

    code = "character_rights"
    retryable = False


def enforce_character_rights(
    character: Character,
    *,
    owned: bool = False,
    likeness: bool = False,
    resemblance: bool = False,
) -> None:
    """Raise when *character* may not be used as requested.

    Ownership is required when ``character.ownership_required`` is set and the
    character is not owned. Real-person likeness is blocked when
    ``character.allow_real_person_likeness`` is not set and likeness is
    requested. When ``character.resemblance_warning`` is set and resemblance is
    detected, a non-blocking warning is emitted so the caller can proceed with
    an advisory notice.
    """
    if character.ownership_required and not owned:
        raise CharacterRightsError(
            "character rights require ownership",
            details={
                "character": character.name,
                "ownership_required": character.ownership_required,
                "owned": owned,
            },
        )
    if likeness and not character.allow_real_person_likeness:
        raise CharacterRightsError(
            "real person likeness requires consent",
            details={
                "character": character.name,
                "allow_real_person_likeness": character.allow_real_person_likeness,
                "likeness": likeness,
            },
        )
    if resemblance and character.resemblance_warning:
        message = f"character {character.name!r} resembles a real person"
        _LOGGER.warning(message)
        warnings.warn(message, UserWarning, stacklevel=2)
