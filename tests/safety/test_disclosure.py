"""AI disclosure metadata and on-screen label per platform.

A platform's disclosure rules (required, text, on-screen label and metadata
field) are applied to a post and stored with it and its attestation: the
disclosure text and on-screen label live on the post, the disclosure metadata is
recorded under the platform's metadata field in ``Post.metadata``, and the
applied disclosure text is mirrored onto the post's attestation.
"""

from __future__ import annotations

from datetime import UTC, datetime

from aivia.core.models import Attestation, Post
from aivia.core.platforms import DisclosureRules, Platform

_NOW = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)


def _platform(**overrides: object) -> Platform:
    values: dict[str, object] = {
        "id": "youtube",
        "disclosure": DisclosureRules(
            required=True,
            text="This content was generated with AI.",
            on_screen_label="AI-generated",
            metadata=True,
            metadata_field="containsSyntheticMedia",
        ),
    }
    values.update(overrides)
    return Platform(**values)


def _post(**overrides: object) -> Post:
    values: dict[str, object] = {
        "platform": "youtube",
        "caption": "A bridge at dusk",
        "attestation": Attestation(
            confirmed_by="ada",
            rights_confirmed=True,
            disclosure_confirmed=True,
            confirmed_at=_NOW,
            disclosure_text="",
            license_summary="CC BY 4.0",
        ),
    }
    values.update(overrides)
    return Post(**values)


def test_apply_disclosure_stores_text_label_metadata_and_attestation() -> None:
    post = _post()
    post.apply_disclosure(_platform())

    assert post.disclosure_required is True
    assert post.disclosure_applied is True
    assert post.disclosure_text == "This content was generated with AI."
    assert post.on_screen_label == "AI-generated"
    assert post.metadata["containsSyntheticMedia"] is True
    assert post.attestation.disclosure_text == "This content was generated with AI."


def test_apply_disclosure_records_on_screen_label_optionally() -> None:
    platform = _platform(disclosure=DisclosureRules(required=True, text="AI"))
    post = _post()
    post.apply_disclosure(platform)

    assert post.disclosure_text == "AI"
    assert post.on_screen_label == ""
    assert post.disclosure_applied is True


def test_apply_disclosure_keeps_existing_label_when_platform_has_none() -> None:
    platform = _platform(disclosure=DisclosureRules(required=True, text="AI"))
    post = _post(on_screen_label="Custom label")
    post.apply_disclosure(platform)

    assert post.on_screen_label == "Custom label"


def test_apply_disclosure_keeps_existing_text_when_platform_has_none() -> None:
    platform = _platform(disclosure=DisclosureRules(required=True))
    post = _post(disclosure_text="Previously applied")
    post.apply_disclosure(platform)

    assert post.disclosure_text == "Previously applied"
    assert post.disclosure_applied is True


def test_non_required_disclosure_leaves_post_unchanged() -> None:
    platform = _platform(
        disclosure=DisclosureRules(
            required=False, text="AI", metadata=True, metadata_field="x"
        )
    )
    post = _post()
    post.apply_disclosure(platform)

    assert post.disclosure_required is False
    assert post.disclosure_applied is False
    assert post.disclosure_text == ""
    assert post.on_screen_label == ""
    assert post.metadata == {}


def test_disclosure_metadata_is_not_stored_when_metadata_unsupported() -> None:
    platform = _platform(
        disclosure=DisclosureRules(
            required=True, text="AI", on_screen_label="AI", metadata=False
        )
    )
    post = _post()
    post.apply_disclosure(platform)

    assert post.disclosure_text == "AI"
    assert post.on_screen_label == "AI"
    assert post.metadata == {}


def test_disclosure_metadata_is_not_stored_without_a_metadata_field() -> None:
    platform = _platform(
        disclosure=DisclosureRules(required=True, text="AI", metadata=True)
    )
    post = _post()
    post.apply_disclosure(platform)

    assert post.disclosure_text == "AI"
    assert post.metadata == {}


def test_apply_disclosure_mirrors_text_onto_existing_attestation() -> None:
    attestation = Attestation(
        confirmed_by="ada",
        rights_confirmed=True,
        disclosure_confirmed=True,
        confirmed_at=_NOW,
        disclosure_text="",
        license_summary="CC BY 4.0",
    )
    post = _post(attestation=attestation)
    post.apply_disclosure(_platform())

    assert post.attestation is attestation
    assert attestation.disclosure_text == "This content was generated with AI."


def test_apply_disclosure_is_idempotent() -> None:
    post = _post()
    post.apply_disclosure(_platform())
    first_text = post.disclosure_text
    first_label = post.on_screen_label
    post.apply_disclosure(_platform())

    assert post.disclosure_text == first_text
    assert post.on_screen_label == first_label
    assert post.metadata["containsSyntheticMedia"] is True
