"""Originality rewrites above the overlap limit and profanity is blocked."""

from __future__ import annotations

import pytest

from aivia.core.config.sections import SafetyConfig
from aivia.core.errors import ConfigError, ContentPolicyError
from aivia.writing import (
    DEFAULT_ORIGINALITY_THRESHOLD,
    DEFAULT_PROFANITY_TERMS,
    LyricsMode,
    LyricsPayload,
    LyricsSafetyChecker,
    OriginalityChecker,
    Script,
    apply_lyrics_safety,
    block_profanity,
    check_lyrics_safety,
    check_originality,
    check_profanity,
    enforce_lyrics_safety,
    enforce_originality,
    find_profanity,
    has_profanity,
    is_original,
    lyrics_text,
    overlap_score,
    rewrite_above_threshold,
)

REFERENCE = (
    "The city lights are calling me home tonight\n"
    "Every street remembers where we used to walk\n"
    "I hear the river singing through the rain"
)

ORIGINAL = (
    "Paper boats are drifting on a quiet stream\n"
    "A lantern hums a tune I never knew\n"
    "Tomorrow keeps a secret in its sleeve"
)


def _payload(script: Script) -> LyricsPayload:
    assert isinstance(script.payload, LyricsPayload)
    return script.payload


def test_overlap_score_uses_the_highest_reference_and_defaults_to_zero() -> None:
    assert DEFAULT_ORIGINALITY_THRESHOLD == 0.6
    assert overlap_score(ORIGINAL, [REFERENCE]) == 0.0
    assert overlap_score(REFERENCE, [REFERENCE, ORIGINAL]) == 1.0
    assert overlap_score(ORIGINAL, None) == 0.0
    assert is_original(ORIGINAL, [REFERENCE])
    assert not is_original(REFERENCE, [REFERENCE])


def test_check_originality_passes_text_below_the_threshold() -> None:
    result = check_originality(ORIGINAL, [REFERENCE])

    assert result == ORIGINAL
    assert result.text == ORIGINAL
    assert result.passed
    assert result.is_original
    assert result.rewritten is False
    assert result.attempts == 0
    assert result.reason == ""
    assert result.threshold == DEFAULT_ORIGINALITY_THRESHOLD


def test_check_originality_rewrites_above_the_threshold() -> None:
    calls: list[tuple[str, float, int]] = []

    def rewrite(text: str, overlap: float, attempt: int) -> str:
        calls.append((text, overlap, attempt))
        return ORIGINAL

    result = check_originality(REFERENCE, [REFERENCE], rewrite=rewrite)

    assert result == ORIGINAL
    assert result.passed
    assert result.rewritten is True
    assert result.attempts == 1
    assert result.overlap == 0.0
    assert calls == [(REFERENCE, 1.0, 1)]


def test_rewriter_may_accept_only_the_text() -> None:
    result = check_originality(
        REFERENCE,
        [REFERENCE],
        rewrite=lambda text: ORIGINAL,
    )

    assert result == ORIGINAL
    assert result.rewritten


def test_enforce_originality_refuses_when_no_rewriter_is_supplied() -> None:
    with pytest.raises(ContentPolicyError) as refused:
        enforce_originality(REFERENCE, [REFERENCE])

    details = refused.value.details
    assert details["reason"] == "overlap"
    assert details["overlap"] == 1.0
    assert details["threshold"] == DEFAULT_ORIGINALITY_THRESHOLD
    assert details["attempts"] == 0


def test_enforce_originality_refuses_after_the_rewrite_limit() -> None:
    with pytest.raises(ContentPolicyError, match="overlap"):
        enforce_originality(
            REFERENCE,
            [REFERENCE],
            rewrite=lambda text: REFERENCE,
            max_rewrites=2,
        )

    result = check_originality(
        REFERENCE,
        [REFERENCE],
        rewrite=lambda text: REFERENCE,
        max_rewrites=2,
    )
    assert result.passed is False
    assert result.attempts == 2
    assert result.reason == "overlap"


def test_non_strict_check_reports_a_failure_instead_of_raising() -> None:
    result = check_originality(REFERENCE, [REFERENCE])

    assert result == REFERENCE
    assert result.passed is False
    assert result.reason == "overlap"
    assert rewrite_above_threshold is enforce_originality


def test_check_originality_accepts_scripts_payloads_and_mappings() -> None:
    script = LyricsMode().build_script(ORIGINAL, safety=False)
    payload = _payload(script)

    assert lyrics_text(script) == ORIGINAL
    assert lyrics_text(payload) == ORIGINAL
    assert lyrics_text({"lyrics": ORIGINAL}) == ORIGINAL
    assert check_originality(script, [REFERENCE]).passed
    assert check_originality(payload, [REFERENCE]).passed
    assert check_originality({"lines": ORIGINAL.splitlines()}, [REFERENCE]).passed


def test_reference_and_threshold_validation() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        check_originality(ORIGINAL, [REFERENCE], threshold=1.5)
    with pytest.raises(TypeError, match="threshold must be a number"):
        check_originality(ORIGINAL, [REFERENCE], threshold=True)
    with pytest.raises(ValueError, match="max_rewrites"):
        check_originality(ORIGINAL, [REFERENCE], max_rewrites=-1)
    with pytest.raises(TypeError, match="must be an integer"):
        check_originality(ORIGINAL, [REFERENCE], max_rewrites=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="metric"):
        overlap_score(ORIGINAL, [REFERENCE], metric="cosine")


def test_profanity_baseline_blocks_whole_words_without_false_positives() -> None:
    assert "damn" in DEFAULT_PROFANITY_TERMS
    assert find_profanity("This damn chorus is stuck in my head") == ("damn",)
    assert find_profanity("A classic song for the class") == ()
    assert find_profanity("Scunthorpe is a real place") == ()
    assert has_profanity("You fucking ruined the bridge")
    assert not has_profanity("A clean and gentle verse")


def test_explicit_terms_extend_the_baseline_and_config_terms_extend_them() -> None:
    assert find_profanity("damn", terms=["nonsense"]) == ("damn",)
    assert find_profanity("nonsense", terms=["nonsense"]) == ("nonsense",)
    assert find_profanity("damn", terms=["nonsense"], use_defaults=False) == ()
    config = SafetyConfig(blocked_terms=["forbidden"])
    assert find_profanity("forbidden damn", terms=["nonsense"], config=config) == (
        "damn",
        "forbidden",
    )
    assert find_profanity("forbidden", config=SafetyConfig(enabled=False)) == ()


def test_check_and_block_profanity() -> None:
    result = check_profanity("This is shit")
    assert result.passed is False
    assert result.reason == "profanity"
    assert result.blocked_terms == ("shit",)
    assert result.text == "This is shit"

    clean = check_profanity("This is fine")
    assert clean.passed
    assert clean.blocked_terms == ()

    with pytest.raises(ContentPolicyError) as refused:
        block_profanity("This is shit")
    assert refused.value.details["blocked_terms"] == ["shit"]


def test_lyrics_safety_blocks_profanity_even_when_it_could_be_rewritten() -> None:
    calls: list[str] = []

    def rewrite(text: str) -> str:
        calls.append(text)
        return ORIGINAL

    result = check_lyrics_safety(
        "This damn chorus",
        [REFERENCE],
        rewrite=rewrite,
    )

    assert result.passed is False
    assert result.reason == "profanity"
    assert result.blocked_terms == ("damn",)
    assert result.text == "This damn chorus"
    assert calls == []

    with pytest.raises(ContentPolicyError, match="profanity"):
        enforce_lyrics_safety("This damn chorus", [REFERENCE], rewrite=rewrite)


def test_lyrics_safety_rewrites_overlap_and_rechecks_the_result() -> None:
    def rewrite(text: str) -> str:
        return "This damn replacement"

    result = check_lyrics_safety(REFERENCE, [REFERENCE], rewrite=rewrite)

    assert result.passed is False
    assert result.reason == "profanity"
    assert result.blocked_terms == ("damn",)
    assert result.rewritten is True
    assert result.attempts == 1

    with pytest.raises(ContentPolicyError, match="profanity"):
        enforce_lyrics_safety(REFERENCE, [REFERENCE], rewrite=rewrite)


def test_lyrics_safety_accepts_config_terms() -> None:
    config = SafetyConfig(blocked_terms=["forbidden"])
    result = check_lyrics_safety("A forbidden line", config=config)

    assert result.passed is False
    assert result.blocked_terms == ("forbidden",)


def test_apply_lyrics_safety_rewrites_each_line_and_keeps_sections() -> None:
    payload = LyricsPayload.model_validate(
        {
            "sections": [
                {"name": "verse", "lines": REFERENCE.splitlines()},
                {"name": "chorus", "lines": ["A clean hook"]},
            ]
        }
    )

    def rewrite(text: str) -> str:
        return ORIGINAL.splitlines()[0]

    safe = apply_lyrics_safety(payload, [REFERENCE], rewrite=rewrite)

    assert [section.name for section in safe.sections] == ["verse", "chorus"]
    assert safe.sections[0].lines == [ORIGINAL.splitlines()[0]] * 3
    assert safe.sections[1].lines == ["A clean hook"]


def test_lyrics_mode_applies_safety_by_default_and_can_opt_out() -> None:
    with pytest.raises(ContentPolicyError, match="profanity"):
        LyricsMode().build_script("A damn chorus")

    unsafe = LyricsMode().build_script("A damn chorus", safety=False)
    assert _payload(unsafe).sections[0].lines == ["A damn chorus"]


def test_lyrics_mode_rewrites_overlap_when_references_are_supplied() -> None:
    replacements = dict(zip(REFERENCE.splitlines(), ORIGINAL.splitlines(), strict=True))
    script = LyricsMode().build_script(
        REFERENCE,
        references=[REFERENCE],
        rewrite=lambda text: replacements.get(text, text),
    )

    assert [beat.text for beat in script.beats] == ORIGINAL.splitlines()


def test_checker_classes_bind_their_options() -> None:
    checker = OriginalityChecker(
        [REFERENCE],
        rewrite=lambda text: ORIGINAL,
    )
    assert checker.check(REFERENCE).passed
    assert checker.enforce(REFERENCE).passed

    safety = LyricsSafetyChecker(
        [REFERENCE],
        rewrite=lambda text: ORIGINAL,
        terms=["forbidden"],
    )
    assert safety.check(REFERENCE).passed
    assert safety.enforce(REFERENCE).passed
    with pytest.raises(ContentPolicyError):
        safety.enforce("forbidden")


def test_rewriter_that_returns_empty_text_is_rejected() -> None:
    with pytest.raises(ConfigError, match="empty"):
        check_originality(REFERENCE, [REFERENCE], rewrite=lambda text: "   ")
