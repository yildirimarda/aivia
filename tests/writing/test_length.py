"""Length estimates use per-language speaking rates and rewrite until they fit."""

from __future__ import annotations

import math

import pytest

from aivia.core.config.sections import LanguageConfig, LanguagesConfig
from aivia.core.errors import ConfigError
from aivia.writing import (
    DEFAULT_MAX_LENGTH_REWRITES,
    DEFAULT_WORDS_PER_SECOND,
    LengthRange,
    check_length,
    count_words,
    enforce_length,
    estimate_duration,
    estimate_duration_range,
    fit_length,
    reading_time_seconds,
    resolve_words_per_second,
    resolve_words_per_second_range,
    rewrite_to_length,
)

LANGUAGES = LanguagesConfig(
    default="en",
    languages={
        "en": LanguageConfig(
            name="English",
            words_per_second={"min": 2.0, "max": 4.0},  # type: ignore[arg-type]
        ),
        "de": LanguageConfig(
            name="German",
            words_per_second={"min": 1.0, "max": 2.0},  # type: ignore[arg-type]
        ),
    },
)

SIX_WORDS = "one two three four five six"


def test_estimates_use_the_midpoint_of_the_language_rate_bounds() -> None:
    assert resolve_words_per_second("en", languages=LANGUAGES) == pytest.approx(3.0)
    assert resolve_words_per_second("de", languages=LANGUAGES) == pytest.approx(1.5)
    assert estimate_duration(SIX_WORDS, language="en", languages=LANGUAGES) == 2.0
    assert estimate_duration(SIX_WORDS, language="de", languages=LANGUAGES) == 4.0
    assert reading_time_seconds(SIX_WORDS, language="de", languages=LANGUAGES) == 4.0


def test_estimates_fall_back_to_the_default_rate_without_a_language_source() -> None:
    assert resolve_words_per_second("en") == DEFAULT_WORDS_PER_SECOND
    assert estimate_duration(SIX_WORDS) == pytest.approx(
        count_words(SIX_WORDS) / DEFAULT_WORDS_PER_SECOND
    )


def test_duration_range_uses_the_slowest_and_fastest_configured_rates() -> None:
    assert resolve_words_per_second_range("en", languages=LANGUAGES) == (2.0, 4.0)
    estimated = estimate_duration_range(SIX_WORDS, language="en", languages=LANGUAGES)
    assert estimated == LengthRange(1.5, 3.0)
    assert estimate_duration_range(SIX_WORDS) == LengthRange(
        count_words(SIX_WORDS) / DEFAULT_WORDS_PER_SECOND,
        count_words(SIX_WORDS) / DEFAULT_WORDS_PER_SECOND,
    )


def test_explicit_rate_bounds_and_models_override_the_language_source() -> None:
    assert resolve_words_per_second(
        "de", languages=LANGUAGES, words_per_second=5.0
    ) == pytest.approx(5.0)
    assert resolve_words_per_second(
        min_words_per_second=2.0, max_words_per_second=4.0
    ) == pytest.approx(3.0)
    assert resolve_words_per_second(
        words_per_second={"min": 1.0, "max": 3.0}
    ) == pytest.approx(2.0)
    assert estimate_duration(SIX_WORDS, words_per_second=3.0) == 2.0


def test_a_script_inside_the_target_is_not_rewritten() -> None:
    result = check_length(SIX_WORDS, (1.0, 3.0), words_per_second=3.0)

    assert result == SIX_WORDS
    assert result.text == SIX_WORDS
    assert result.seconds == pytest.approx(2.0)
    assert result.measured_seconds == result.seconds
    assert result.words == 6
    assert result.passed is True
    assert result.rewritten is False
    assert result.attempts == 0
    assert result.reason == ""


def test_a_script_outside_the_target_is_rewritten_until_it_fits() -> None:
    calls: list[tuple[str, float, int]] = []

    def rewrite(text: str, measured: float, attempt: int) -> str:
        calls.append((text, measured, attempt))
        return "one two three"

    result = check_length(
        SIX_WORDS,
        (0.0, 1.5),
        words_per_second=3.0,
        rewrite=rewrite,
    )

    assert result == "one two three"
    assert result.passed is True
    assert result.rewritten is True
    assert result.attempts == 1
    assert result.seconds == pytest.approx(1.0)
    assert calls == [(SIX_WORDS, 2.0, 1)]


def test_a_rewriter_may_accept_only_the_text() -> None:
    result = check_length(
        SIX_WORDS,
        (0.0, 1.5),
        words_per_second=3.0,
        rewrite=lambda text: "one two three",
    )

    assert result == "one two three"
    assert result.rewritten is True


def test_rewriting_stops_at_the_limit_and_reports_the_final_measurement() -> None:
    result = check_length(
        SIX_WORDS,
        (0.0, 1.0),
        words_per_second=3.0,
        rewrite=lambda text: text,
        max_rewrites=2,
    )

    assert result == SIX_WORDS
    assert result.passed is False
    assert result.rewritten is True
    assert result.attempts == 2
    assert result.reason == "length"
    assert result.seconds == pytest.approx(2.0)
    assert DEFAULT_MAX_LENGTH_REWRITES == 2


def test_enforce_length_fails_with_measured_versus_target() -> None:
    with pytest.raises(ConfigError, match="outside the target range") as refused:
        enforce_length(
            SIX_WORDS,
            (0.0, 1.0),
            words_per_second=3.0,
            rewrite=lambda text: text,
            max_rewrites=1,
        )

    details = refused.value.details
    assert details["reason"] == "length"
    assert details["measured_seconds"] == pytest.approx(2.0)
    assert details["min_seconds"] == 0.0
    assert details["max_seconds"] == 1.0
    assert details["attempts"] == 1
    assert details["rewritten"] is True
    assert "2.0s" in str(refused.value)
    assert "0.0-1.0s" in str(refused.value)


def test_language_rates_drive_the_rewrite_and_the_failure() -> None:
    result = check_length(
        SIX_WORDS,
        (0.0, 1.5),
        language="de",
        languages=LANGUAGES,
        rewrite=lambda text: "one two three",
        max_rewrites=1,
    )

    assert result.seconds == pytest.approx(2.0)
    assert result.passed is False

    with pytest.raises(ConfigError) as refused:
        enforce_length(
            SIX_WORDS,
            (0.0, 3.0),
            language="de",
            languages=LANGUAGES,
        )
    assert refused.value.details["words_per_second"] == pytest.approx(1.5)


def test_no_target_measures_without_failing_or_rewriting() -> None:
    result = check_length(
        SIX_WORDS,
        words_per_second=3.0,
        rewrite=lambda text: "one",
    )

    assert result.passed is True
    assert result.rewritten is False
    assert result.target is None
    assert result.min_seconds is None
    assert result.max_seconds is None


def test_explicit_seconds_bounds_resolve_a_target() -> None:
    result = check_length(
        SIX_WORDS, min_seconds=1.0, max_seconds=3.0, words_per_second=3.0
    )
    assert result.passed is True
    assert result.target == LengthRange(1.0, 3.0)


def test_results_serialize_to_a_json_compatible_mapping() -> None:
    result = check_length(SIX_WORDS, (1.0, 3.0), words_per_second=3.0)
    payload = result.to_dict()

    assert payload["text"] == SIX_WORDS
    assert payload["words"] == 6
    assert payload["seconds"] == pytest.approx(2.0)
    assert payload["target"] == {"min_seconds": 1.0, "max_seconds": 3.0}
    assert payload["passed"] is True


def test_rewriter_returning_empty_text_or_a_non_string_is_rejected() -> None:
    with pytest.raises(ConfigError, match="empty text"):
        check_length(
            SIX_WORDS,
            (0.0, 1.0),
            words_per_second=3.0,
            rewrite=lambda text: "   ",
        )
    with pytest.raises(ConfigError, match="must return text"):
        check_length(
            SIX_WORDS,
            (0.0, 1.0),
            words_per_second=3.0,
            rewrite=lambda text: 3,
        )


def test_invalid_rates_and_rewrite_limits_are_rejected() -> None:
    with pytest.raises(ConfigError, match="positive"):
        estimate_duration(SIX_WORDS, words_per_second=0)
    with pytest.raises(ConfigError, match="must not exceed"):
        resolve_words_per_second_range(
            min_words_per_second=4.0, max_words_per_second=2.0
        )
    with pytest.raises(ValueError, match="max_rewrites"):
        check_length(SIX_WORDS, (1.0, 2.0), max_rewrites=-1)
    with pytest.raises(TypeError, match="must be an integer"):
        check_length(SIX_WORDS, (1.0, 2.0), max_rewrites=1.5)  # type: ignore[arg-type]


def test_infinite_target_bounds_are_reported_as_unbounded() -> None:
    result = check_length(SIX_WORDS, {"min_seconds": 1.0}, words_per_second=3.0)
    assert result.passed is True
    assert math.isinf(result.max_seconds or 0.0)


def test_strict_aliases_enforce_and_non_strict_aliases_return_a_result() -> None:
    assert rewrite_to_length is enforce_length
    assert fit_length is enforce_length
    assert check_length is not enforce_length
