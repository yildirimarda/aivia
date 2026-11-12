"""Publish attestation: who confirmed rights and disclosure, when, the
applied disclosure text and the license summary, blocking under strict mode
when anything is missing."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aivia.core.models import PublishAttestation

_NOW = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)


def _publish_attestation(**overrides: object) -> PublishAttestation:
    values: dict[str, object] = {
        "rights_confirmed_by": "ada",
        "disclosure_confirmed_by": "ada",
        "rights_confirmed": True,
        "disclosure_confirmed": True,
        "confirmed_at": _NOW,
        "disclosure_text": "AI generated",
        "license_summary": "CC BY 4.0",
    }
    values.update(overrides)
    return PublishAttestation.model_validate(values)


def test_publish_attestation_records_who_rights_disclosure_when_text_and_license() -> (
    None
):
    attestation = _publish_attestation()

    assert attestation.rights_confirmed_by == "ada"
    assert attestation.disclosure_confirmed_by == "ada"
    assert attestation.rights_confirmed is True
    assert attestation.disclosure_confirmed is True
    assert attestation.confirmed_at == attestation.when == _NOW
    assert attestation.disclosure_text == attestation.disclosure == "AI generated"
    assert attestation.license_summary == attestation.license == "CC BY 4.0"
    assert attestation.missing == []
    assert attestation.is_complete is True
    assert attestation.confirmed is True
    assert attestation.is_blocking is False


def test_publish_attestation_defaults_to_incomplete() -> None:
    attestation = PublishAttestation()

    assert attestation.rights_confirmed_by == ""
    assert attestation.disclosure_confirmed_by == ""
    assert attestation.rights_confirmed is False
    assert attestation.disclosure_confirmed is False
    assert attestation.confirmed_at is None
    assert attestation.disclosure_text == ""
    assert attestation.license_summary == ""
    assert attestation.strict is False
    assert attestation.is_complete is False
    assert attestation.missing == [
        "confirmed_by",
        "rights_confirmed",
        "disclosure_confirmed",
        "confirmed_at",
        "disclosure_text",
        "license_summary",
    ]


@pytest.mark.parametrize(
    "field,overrides",
    [
        ("confirmed_by", {"confirmed_by": "", "rights_confirmed_by": ""}),
        ("rights_confirmed", {"rights_confirmed": False}),
        ("disclosure_confirmed", {"disclosure_confirmed": False}),
        ("confirmed_at", {"confirmed_at": None}),
        ("disclosure_text", {"disclosure_text": ""}),
        ("license_summary", {"license_summary": ""}),
    ],
)
def test_publish_attestation_strict_blocks_when_anything_missing(
    field: str, overrides: dict[str, object]
) -> None:
    attestation = _publish_attestation(**overrides, strict=True)

    assert attestation.is_blocking is True
    assert attestation.blocks is True
    assert field in attestation.missing
    assert field in attestation.missing_fields


def test_publish_attestation_strict_does_not_block_when_complete() -> None:
    attestation = _publish_attestation(strict=True)

    assert attestation.is_blocking is False
    assert attestation.blocks is False
    assert attestation.is_complete is True
