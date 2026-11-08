"""Similarity levels and the style spec sections they enforce.

``similarity_level`` selects how closely generated content follows a reference.
Each level enforces a superset of the style spec sections: ``low`` enforces
only structure, ``medium`` adds pacing, and ``high`` enforces every section.
The effective level is resolved with global, then profile, then CLI precedence
by the configuration resolver; this module maps a resolved level to the fields
a generator must honor.
"""

from __future__ import annotations

from enum import StrEnum

from aivia.core.config.reference import SimilarityLevel

__all__ = [
    "ENFORCED_SECTIONS",
    "SIMILARITY_ENFORCED_FIELDS",
    "EnforcedSection",
    "SimilarityLevel",
    "enforced_fields",
]


class EnforcedSection(StrEnum):
    """One section of a style spec that a similarity level may enforce."""

    STRUCTURE = "structure"
    PACING = "pacing"
    PALETTE = "palette"
    TYPOGRAPHY = "typography"
    MUSIC = "music"
    ASPECT = "aspect"


ENFORCED_SECTIONS: frozenset[str] = frozenset(
    section.value for section in EnforcedSection
)
"""Every style spec section, enforced only at the ``high`` level."""

SIMILARITY_ENFORCED_FIELDS: dict[SimilarityLevel, frozenset[str]] = {
    SimilarityLevel.LOW: frozenset({EnforcedSection.STRUCTURE}),
    SimilarityLevel.MEDIUM: frozenset(
        {EnforcedSection.STRUCTURE, EnforcedSection.PACING}
    ),
    SimilarityLevel.HIGH: ENFORCED_SECTIONS,
}
"""The style spec sections enforced at each similarity level.

``low`` keeps only the reference's structure (scene count, durations and shot
types). ``medium`` additionally pins the pacing (cut rate, shot length and
speech rate). ``high`` enforces every section: structure, pacing, palette,
typography, music mood and aspect.
"""


def enforced_fields(similarity_level: SimilarityLevel) -> frozenset[str]:
    """Return the style spec sections enforced for *similarity_level*."""
    try:
        return SIMILARITY_ENFORCED_FIELDS[similarity_level]
    except KeyError as error:
        raise ValueError(f"unknown similarity level: {similarity_level!r}") from error
