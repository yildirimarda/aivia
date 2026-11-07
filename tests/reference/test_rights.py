"""Rights policy: unowned references are style and structure only.

High similarity requires an owned flag; an unowned reference may guide
style and structure (low and medium similarity) but never high similarity.
"""

from __future__ import annotations

import pytest

from aivia.core.errors import ValidationError
from aivia.reference import (
    ReferenceRightsError,
    ReferenceSpec,
    enforce_reference_rights,
)
from aivia.reference.similarity import SimilarityLevel


def _reference(
    *, owned: bool = False, path: str = "references/ref.png"
) -> ReferenceSpec:
    return ReferenceSpec(path=path, owned=owned)


def test_high_similarity_requires_an_owned_flag() -> None:
    reference = _reference(owned=False)
    with pytest.raises(ReferenceRightsError, match="owned"):
        enforce_reference_rights(reference, SimilarityLevel.HIGH)


def test_high_similarity_requires_an_owned_flag_for_string_level() -> None:
    reference = _reference(owned=False)
    with pytest.raises(ReferenceRightsError, match="owned"):
        enforce_reference_rights(reference, "high")


def test_high_similarity_is_allowed_when_owned() -> None:
    reference = _reference(owned=True)
    enforce_reference_rights(reference, SimilarityLevel.HIGH)


def test_unowned_reference_is_style_and_structure_only_at_low() -> None:
    reference = _reference(owned=False)
    enforce_reference_rights(reference, SimilarityLevel.LOW)


def test_unowned_reference_is_style_and_structure_only_at_medium() -> None:
    reference = _reference(owned=False)
    enforce_reference_rights(reference, SimilarityLevel.MEDIUM)


def test_owned_reference_is_allowed_at_low_and_medium() -> None:
    reference = _reference(owned=True)
    enforce_reference_rights(reference, SimilarityLevel.LOW)
    enforce_reference_rights(reference, SimilarityLevel.MEDIUM)


def test_error_is_a_validation_error_with_code_and_details() -> None:
    reference = _reference(owned=False)
    with pytest.raises(ReferenceRightsError) as exc_info:
        enforce_reference_rights(reference, SimilarityLevel.HIGH)

    error = exc_info.value
    assert isinstance(error, ValidationError)
    assert error.code == "reference_rights"
    assert error.retryable is False
    assert error.details["owned"] is False
    assert error.details["similarity_level"] == "high"


def test_default_reference_is_unowned_and_blocks_high_similarity() -> None:
    reference = ReferenceSpec(path="references/ref.png")
    assert reference.owned is False
    with pytest.raises(ReferenceRightsError):
        enforce_reference_rights(reference, SimilarityLevel.HIGH)
