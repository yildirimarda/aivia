"""Character rights: ownership required, likeness blocked without consent,
and an optional resemblance warning."""

from __future__ import annotations

import warnings

import pytest

from aivia.characters.rights import (
    CharacterRightsError,
    enforce_character_rights,
)
from aivia.characters.schema import Character


def _character(
    *,
    name: str = "Mascot",
    ownership_required: bool = True,
    allow_real_person_likeness: bool = False,
    resemblance_warning: bool = True,
) -> Character:
    return Character.model_validate(
        {
            "name": name,
            "ownership_required": ownership_required,
            "allow_real_person_likeness": allow_real_person_likeness,
            "resemblance_warning": resemblance_warning,
        }
    )


def test_ownership_required_blocks_an_unowned_character() -> None:
    with pytest.raises(CharacterRightsError, match="ownership"):
        enforce_character_rights(_character(), owned=False)


def test_ownership_required_allows_an_owned_character() -> None:
    enforce_character_rights(_character(), owned=True)


def test_ownership_not_required_passes_without_ownership() -> None:
    enforce_character_rights(_character(ownership_required=False), owned=False)


def test_likeness_without_consent_is_blocked() -> None:
    with pytest.raises(CharacterRightsError, match="consent"):
        enforce_character_rights(_character(), owned=True, likeness=True)


def test_likeness_with_consent_is_allowed() -> None:
    enforce_character_rights(
        _character(allow_real_person_likeness=True), owned=True, likeness=True
    )


def test_likeness_not_requested_passes_without_consent() -> None:
    enforce_character_rights(_character(), owned=True, likeness=False)


def test_ownership_takes_precedence_over_likeness() -> None:
    with pytest.raises(CharacterRightsError, match="ownership"):
        enforce_character_rights(_character(), owned=False, likeness=True)


def test_error_is_a_content_policy_error_with_code_and_details() -> None:
    with pytest.raises(CharacterRightsError) as caught:
        enforce_character_rights(_character(), owned=False)

    error = caught.value
    assert error.code == "character_rights"
    assert error.retryable is False
    assert error.details["character"] == "Mascot"
    assert error.details["owned"] is False
    assert error.details["ownership_required"] is True


def test_likeness_error_names_the_flag_and_request() -> None:
    with pytest.raises(CharacterRightsError) as caught:
        enforce_character_rights(_character(), owned=True, likeness=True)

    error = caught.value
    assert error.details["character"] == "Mascot"
    assert error.details["allow_real_person_likeness"] is False
    assert error.details["likeness"] is True


def test_resemblance_warning_emits_a_warning_when_enabled() -> None:
    with pytest.warns(UserWarning, match="resembles a real person"):
        enforce_character_rights(_character(), owned=True, resemblance=True)


def test_resemblance_warning_is_silent_when_disabled() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        enforce_character_rights(
            _character(resemblance_warning=False), owned=True, resemblance=True
        )


def test_no_resemblance_produces_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        enforce_character_rights(_character(), owned=True, resemblance=False)
