"""Filter providers by character capability flags and enforce the reference limit."""

from __future__ import annotations

from typing import Any, cast

import pytest

from aivia.characters.capabilities import (
    CAPABILITY_CHARACTER_REFERENCE,
    CharacterReferenceLimitError,
    CharacterReferenceNotSupportedError,
    enforce_character_reference_limit,
    filter_providers_for_character_references,
    prepare_character_request,
    supports_character_references,
)
from aivia.characters.schema import Character
from aivia.core.models import ProviderMeta
from aivia.providers import Provider


class FakeProvider(Provider):
    def __init__(
        self,
        name: str,
        capabilities: list[str],
    ) -> None:
        super().__init__(
            ProviderMeta(
                name=name,
                capabilities=capabilities,
            )
        )


def _character(*, references: int = 2, name: str = "Mascot") -> Character:
    images = [f"ref-{i}.png" for i in range(references)]
    return Character.model_validate({"name": name, "reference_images": images})


def test_providers_declaring_character_capability_are_returned_with_references() -> (
    None
):
    without = FakeProvider("without", [])
    with_flag = FakeProvider("with-flag", [CAPABILITY_CHARACTER_REFERENCE])

    providers, warning = prepare_character_request(
        _character(), [without, with_flag], max_reference_images=4
    )

    assert warning is None
    assert providers == (with_flag,)


def test_providers_without_character_capability_are_filtered_out() -> None:
    without = FakeProvider("without", [])
    with_flag = FakeProvider("with-flag", [CAPABILITY_CHARACTER_REFERENCE])

    providers = filter_providers_for_character_references(
        [without, with_flag], _character()
    )

    assert providers == (with_flag,)


def test_no_references_returns_all_providers_without_warning() -> None:
    without = FakeProvider("without", [])
    with_flag = FakeProvider("with-flag", [CAPABILITY_CHARACTER_REFERENCE])

    providers, warning = prepare_character_request(
        _character(references=0), [without, with_flag], max_reference_images=4
    )

    assert warning is None
    assert providers == (without, with_flag)


def test_no_character_capable_provider_falls_back_to_prompt_only_with_warning() -> None:
    without = FakeProvider("without", [])

    providers, warning = prepare_character_request(
        _character(), [without], max_reference_images=4
    )

    assert warning is not None
    assert CAPABILITY_CHARACTER_REFERENCE in warning
    assert "prompt-only generation" in warning
    assert providers == (without,)


def test_fallback_emits_a_user_warning() -> None:
    without = FakeProvider("without", [])

    with pytest.warns(UserWarning, match="prompt-only generation"):
        prepare_character_request(_character(), [without], max_reference_images=4)


def test_empty_provider_list_with_references_falls_back() -> None:
    providers, warning = prepare_character_request(
        _character(), [], max_reference_images=4
    )

    assert warning is not None
    assert CAPABILITY_CHARACTER_REFERENCE in warning
    assert providers == ()


def test_supports_character_references_reads_provider_metadata() -> None:
    with_flag = FakeProvider("with-flag", [CAPABILITY_CHARACTER_REFERENCE])
    without = FakeProvider("without", [])

    assert supports_character_references(with_flag) is True
    assert supports_character_references(without) is False


def test_character_within_reference_limit_passes() -> None:
    enforce_character_reference_limit(_character(references=2), max_reference_images=4)


def test_character_at_reference_limit_passes() -> None:
    enforce_character_reference_limit(_character(references=4), max_reference_images=4)


def test_character_over_reference_limit_raises() -> None:
    with pytest.raises(CharacterReferenceLimitError) as caught:
        enforce_character_reference_limit(
            _character(references=5), max_reference_images=4
        )

    assert caught.value.code == "character_reference_limit"
    assert caught.value.count == 5
    assert caught.value.limit == 4
    assert caught.value.character == "Mascot"


def test_character_over_reference_limit_names_field() -> None:
    with pytest.raises(CharacterReferenceLimitError) as caught:
        enforce_character_reference_limit(
            _character(references=6, name="Sidekick"), max_reference_images=2
        )

    assert caught.value.details["character"] == "Sidekick"
    assert caught.value.details["reference_images"] == 6
    assert caught.value.details["max_reference_images"] == 2


def test_max_reference_images_zero_allows_no_references_only() -> None:
    enforce_character_reference_limit(_character(references=0), max_reference_images=0)

    with pytest.raises(CharacterReferenceLimitError):
        enforce_character_reference_limit(
            _character(references=1), max_reference_images=0
        )


def test_max_reference_images_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        enforce_character_reference_limit(_character(), max_reference_images=-1)


def test_max_reference_images_must_be_an_integer() -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        enforce_character_reference_limit(
            _character(), max_reference_images=cast(Any, 4.5)
        )


def test_prepare_enforces_limit_before_filtering_providers() -> None:
    with_flag = FakeProvider("with-flag", [CAPABILITY_CHARACTER_REFERENCE])

    with pytest.raises(CharacterReferenceLimitError):
        prepare_character_request(
            _character(references=5), [with_flag], max_reference_images=4
        )


def test_character_reference_not_supported_error_can_be_raised() -> None:
    with pytest.raises(CharacterReferenceNotSupportedError) as caught:
        raise CharacterReferenceNotSupportedError("no-flag-provider", count=3)

    assert caught.value.code == "character_reference_not_supported"
    assert caught.value.provider == "no-flag-provider"
    assert caught.value.count == 3
    assert CAPABILITY_CHARACTER_REFERENCE in str(caught.value)
