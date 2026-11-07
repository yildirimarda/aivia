"""Rights policy for references.

An unowned reference may guide style and structure only: it can never be
matched closely. High similarity therefore requires an owned flag.
"""

from __future__ import annotations

from aivia.core.config.reference import SimilarityLevel
from aivia.core.errors import ValidationError
from aivia.reference.schema import ReferenceSpec

__all__ = [
    "ReferenceRightsError",
    "enforce_reference_rights",
]


class ReferenceRightsError(ValidationError):
    """A reference cannot be used at the requested similarity level."""

    code = "reference_rights"
    retryable = False


def _level(value: SimilarityLevel | str) -> SimilarityLevel:
    if isinstance(value, SimilarityLevel):
        return value
    return SimilarityLevel(value)


def enforce_reference_rights(
    reference: ReferenceSpec, similarity_level: SimilarityLevel | str
) -> None:
    """Raise when *similarity_level* is not permitted for *reference*.

    High similarity requires an owned flag; an unowned reference may guide
    style and structure (low and medium similarity) but never high similarity.
    """
    level = _level(similarity_level)
    if level is SimilarityLevel.HIGH and not reference.owned:
        raise ReferenceRightsError(
            "high similarity requires an owned reference",
            details={
                "similarity_level": str(level),
                "owned": reference.owned,
            },
        )
