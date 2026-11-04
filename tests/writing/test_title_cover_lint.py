"""A title, caption and cover text are linted together for publication."""

from __future__ import annotations

from typing import ClassVar

import pytest

from aivia.core.platforms import CaptionLimits, Platform
from aivia.writing import (
    DEFAULT_COVER_MAX_LENGTH,
    TitleCoverFinding,
    TitleCoverFindings,
    TitleCoverLinter,
    TitleCoverPairing,
    lint_caption_set,
    lint_pairing,
    lint_script,
    lint_title_cover,
)

CLEAN_TITLE = "The secret life of bees"
CLEAN_CAPTION = "Bees can recognise human faces and remember them for days."
CLEAN_COVER = "Faces bees never forget"


def _platform(
    *,
    max_length: int = 5000,
    max_title_length: int | None = 100,
) -> Platform:
    return Platform(
        id="example",
        name="Example",
        caption_limits=CaptionLimits(
            max_length=max_length,
            max_title_length=max_title_length,
        ),
    )


def test_a_clean_pairing_has_no_findings() -> None:
    report = lint_title_cover(CLEAN_TITLE, CLEAN_CAPTION, CLEAN_COVER)

    assert isinstance(report, TitleCoverFindings)
    assert report.ok is True
    assert report.findings == []
    assert report.errors == []
    assert report.warnings == []
    assert report.to_dict()["ok"] is True


def test_a_field_over_its_own_limit_is_truncated() -> None:
    report = lint_title_cover(
        CLEAN_TITLE,
        CLEAN_CAPTION,
        CLEAN_COVER,
        max_title_length=12,
        max_caption_length=20,
        max_cover_length=10,
    )

    assert {finding.field for finding in report.truncated} == {
        "title",
        "caption",
        "cover",
    }
    title = report.by_field("title")[0]
    assert isinstance(title, TitleCoverFinding)
    assert title.code == "truncated"
    assert title.severity == "error"
    assert title.length == len(CLEAN_TITLE)
    assert title.limit == 12
    assert "over the 12 character limit" in title.message


def test_platform_and_caption_limits_are_used_without_explicit_numbers() -> None:
    platform = _platform(max_length=30, max_title_length=12)

    report = lint_title_cover(
        CLEAN_TITLE,
        CLEAN_CAPTION,
        CLEAN_COVER,
        platform=platform,
    )

    assert report.limits["max_title_length"] == 12
    assert report.limits["max_caption_length"] == 30
    assert {finding.field for finding in report.truncated} == {"title", "caption"}


def test_a_trailing_ellipsis_is_reported_as_truncation() -> None:
    report = lint_title_cover("A story cut off\u2026", CLEAN_CAPTION, CLEAN_COVER)
    dotted = lint_title_cover("A story cut off...", CLEAN_CAPTION, CLEAN_COVER)

    assert [finding.code for finding in report] == ["truncated"]
    assert "ellipsis" in report[0].message
    assert [finding.code for finding in dotted] == ["truncated"]


def test_exact_and_near_duplicates_across_fields_are_reported() -> None:
    exact = lint_title_cover(CLEAN_TITLE, CLEAN_CAPTION, CLEAN_TITLE)
    near = lint_title_cover(CLEAN_TITLE, "Secret life of bees, the", CLEAN_COVER)

    assert [(finding.field, finding.related) for finding in exact.duplicates] == [
        ("cover", "title")
    ]
    assert [(finding.field, finding.related) for finding in near.duplicates] == [
        ("caption", "title")
    ]
    assert "duplicates title" in exact.duplicates[0].message


def test_repeated_words_and_sentences_inside_a_field_are_duplicates() -> None:
    words = lint_title_cover(
        "The the secret life of bees",
        CLEAN_CAPTION,
        CLEAN_COVER,
    )
    sentences = lint_title_cover(
        CLEAN_TITLE,
        "Bees can recognise human faces. Bees can recognise human faces.",
        CLEAN_COVER,
    )

    assert any("repeats the word 'The'" in item.message for item in words.duplicates)
    assert any("repeats the sentence" in item.message for item in sentences.duplicates)


@pytest.mark.parametrize(
    "title",
    [
        "You won't believe what happened",
        "This is why bees matter",
        "Amazing bees",
        "This is it",
        "",
    ],
)
def test_vague_or_empty_titles_are_reported(title: str) -> None:
    report = lint_title_cover(title, CLEAN_CAPTION, CLEAN_COVER)

    assert report.vague
    assert all(finding.code == "vague" for finding in report.vague)
    assert all(finding.field == "title" for finding in report.vague)


def test_vague_cover_text_lists_every_matched_term() -> None:
    report = lint_title_cover(CLEAN_TITLE, CLEAN_CAPTION, "Amazing stuff")

    cover = report.by_field("cover")
    assert len(cover) == 1
    assert cover[0].code == "vague"
    assert "amazing" in cover[0].message
    assert "stuff" in cover[0].message


def test_custom_vague_terms_extend_the_default_list() -> None:
    default = lint_title_cover("Bees use synergy", CLEAN_CAPTION, CLEAN_COVER)
    custom = lint_title_cover(
        "Bees use synergy",
        CLEAN_CAPTION,
        CLEAN_COVER,
        vague_terms=["synergy"],
    )

    assert default.ok is True
    assert [finding.code for finding in custom] == ["vague"]
    assert "synergy" in custom[0].message


def test_a_linter_keeps_defaults_and_allows_a_per_call_override() -> None:
    linter = TitleCoverLinter(max_title_length=5)

    default = linter.lint(CLEAN_TITLE, CLEAN_CAPTION, CLEAN_COVER)
    overridden = linter.lint(
        CLEAN_TITLE,
        CLEAN_CAPTION,
        CLEAN_COVER,
        max_title_length=100,
    )

    assert [finding.field for finding in default.truncated] == ["title"]
    assert overridden.truncated == []


def test_the_cover_alias_and_pairing_helpers_share_one_result() -> None:
    direct = lint_title_cover(
        CLEAN_TITLE,
        CLEAN_CAPTION,
        cover=CLEAN_COVER,
    )
    mapping = lint_pairing(
        {
            "title": CLEAN_TITLE,
            "caption": CLEAN_CAPTION,
            "cover_text": CLEAN_COVER,
        }
    )
    pairing = TitleCoverPairing(CLEAN_TITLE, CLEAN_CAPTION, CLEAN_COVER).lint()

    assert direct.ok is mapping.ok is pairing.ok is True
    assert mapping.title == CLEAN_TITLE
    assert mapping.cover_text == CLEAN_COVER


def test_script_and_caption_set_helpers_use_their_own_fields() -> None:
    class Script:
        title = CLEAN_TITLE
        hook = CLEAN_CAPTION

    class CaptionSet:
        title = CLEAN_TITLE
        caption = CLEAN_CAPTION
        limits: ClassVar[dict[str, int]] = {
            "max_title_length": 12,
            "max_length": 20,
        }

    script = lint_script(Script(), cover_text=CLEAN_COVER)
    captions = lint_caption_set(CaptionSet(), cover_text=CLEAN_COVER)

    assert script.ok is True
    assert captions.limits["max_title_length"] == 12
    assert captions.limits["max_caption_length"] == 20
    assert {finding.field for finding in captions.truncated} == {"title", "caption"}


def test_findings_expose_codes_fields_and_a_json_mapping() -> None:
    report = lint_title_cover(
        CLEAN_TITLE,
        "Secret life of bees, the",
        CLEAN_TITLE,
        max_title_length=5,
    )

    assert report.by_code("duplicate")
    assert report.by_field("cover")
    assert report.by_code("missing") == []
    assert report.duplicates and report.truncated
    assert all(item.to_dict()["code"] in {"duplicate", "truncated"} for item in report)
    payload = report.to_dict()
    assert payload["title"] == CLEAN_TITLE
    assert payload["cover_text"] == CLEAN_TITLE
    assert payload["limits"]["max_title_length"] == 5
    assert len(payload["findings"]) == len(report)


def test_invalid_limits_and_thresholds_fail_clearly() -> None:
    with pytest.raises(TypeError, match="max_title_length"):
        lint_title_cover(CLEAN_TITLE, max_title_length="10")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_cover_length"):
        lint_title_cover(CLEAN_TITLE, max_cover_length=-1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        lint_title_cover(CLEAN_TITLE, similarity_threshold=2.0)
    with pytest.raises(TypeError, match="text must be a string"):
        lint_title_cover(123)


def test_the_default_cover_limit_is_used_when_none_is_given() -> None:
    report = lint_title_cover(CLEAN_TITLE, CLEAN_CAPTION, "x" * 61)

    assert DEFAULT_COVER_MAX_LENGTH == 60
    assert [finding.field for finding in report.truncated] == ["cover"]
    assert report.truncated[0].limit == 60
