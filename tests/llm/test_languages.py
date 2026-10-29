# ruff: noqa: RUF001
"""Language codes, detection and per-language settings."""

from __future__ import annotations

import pytest

from aivia.core.config import LanguageConfig, LanguagesConfig
from aivia.core.errors import ConfigError, ValidationError
from aivia.llm import (
    DEFAULT_LANGUAGE,
    LanguageResolver,
    TextDirection,
    detect_language,
    hashtag_style,
    is_rtl_language,
    language_code,
    language_direction,
    language_locale,
    language_name,
    normalize_language_code,
    normalize_languages,
    resolve_language,
    subtitle_font,
    words_per_second,
)

CONFIG = {
    "default": "en",
    "en": {
        "name": "English",
        "locale": "en_US",
        "words_per_second": {"min": 2.0, "max": 3.5},
        "subtitle_font": "Lato",
        "hashtag_style": "lower",
    },
    "de": {
        "name": "Deutsch",
        "locale": "de_DE",
        "words_per_second": {"min": 1.8, "max": 3.0},
        "subtitle_font": "Inter",
        "hashtag_style": "title",
    },
    "tr": {
        "name": "Türkçe",
        "locale": "tr_TR",
        "words_per_second": {"min": 1.6, "max": 2.8},
        "subtitle_font": "Noto Sans",
        "hashtag_style": "upper",
    },
}


def _resolver() -> LanguageResolver:
    return LanguageResolver(LanguagesConfig.from_mapping(CONFIG))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("en", "en"),
        ("EN", "en"),
        (" en ", "en"),
        ("en_US", "en"),
        ("en-US", "en"),
        ("en_US.UTF-8", "en"),
        ("English", "en"),
        ("English (United States)", "en"),
        ("eng", "en"),
        ("German", "de"),
        ("Deutsch", "de"),
        ("ger", "de"),
        ("deu", "de"),
        ("de_DE", "de"),
        ("Turkish", "tr"),
        ("Türkçe", "tr"),
        ("turkce", "tr"),
        ("tur", "tr"),
        ("tr_TR", "tr"),
        ("zh-Hant", "zh"),
        ("pt_BR", "pt"),
        ("sr-Latn-RS", "sr"),
    ],
)
def test_codes_normalize_to_one_canonical_base(value: str, expected: str) -> None:
    assert normalize_language_code(value) == expected
    assert language_code(value) == expected


@pytest.mark.parametrize("value", ["", "   ", "C", "POSIX", "123", "!!!"])
def test_invalid_language_names_are_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        normalize_language_code(value)


def test_non_text_language_names_are_rejected() -> None:
    with pytest.raises(ValidationError):
        normalize_language_code(42)  # type: ignore[arg-type]


def test_normalizing_a_list_dedupes_and_keeps_the_first_order() -> None:
    assert normalize_languages(["EN", "en_US", "German", "de_DE", "tr"]) == [
        "en",
        "de",
        "tr",
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "This is the best day and we are happy with the result.",
            "en",
        ),
        (
            "Das ist der beste Tag und wir sind sehr glücklich über das Ergebnis.",
            "de",
        ),
        ("Bu gün en iyi gün ve biz çok mutluyuz.", "tr"),
        ("Iğdır'da yaşıyorum ve burada mutluyum.", "tr"),
        ("Grüße aus München", "de"),
        ("İstanbul", "tr"),
    ],
)
def test_detection_recognises_the_shipped_languages(
    text: str,
    expected: str,
) -> None:
    assert detect_language(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "123 456", "Hotel", "!!!"])
def test_detection_returns_none_for_empty_or_ambiguous_text(text: str) -> None:
    assert detect_language(text) is None


def test_detection_respects_the_allowed_languages() -> None:
    german = "Das ist der beste Tag und wir sind sehr glücklich."

    assert detect_language(german, allowed=["de"]) == "de"
    assert detect_language(german, allowed=["en", "tr"]) is None
    assert detect_language(german, allowed=["DE"]) == "de"
    assert detect_language(german, allowed=[]) is None


def test_detection_is_deterministic() -> None:
    text = "Das ist der beste Tag und wir sind sehr glücklich."

    assert detect_language(text) == detect_language(text)


def test_the_default_language_is_english() -> None:
    assert DEFAULT_LANGUAGE == "en"


def test_direction_defaults_to_left_to_right_and_knows_right_to_left() -> None:
    assert language_direction("en") is TextDirection.LTR
    assert language_direction("de_DE") is TextDirection.LTR
    assert language_direction("tr") is TextDirection.LTR
    assert language_direction("ar") is TextDirection.RTL
    assert language_direction("he") is TextDirection.RTL
    assert language_direction("fa_IR") is TextDirection.RTL
    assert is_rtl_language("ur") is True
    assert is_rtl_language("en") is False


def test_the_resolver_falls_back_to_the_configured_default() -> None:
    resolver = _resolver()

    assert resolver.default.code == "en"
    assert resolver.resolve().code == "en"
    assert resolver.resolve("").code == "en"
    assert resolver.resolve("   ").code == "en"
    assert resolver.resolve("de").code == "de"


def test_the_resolver_accepts_codes_locales_and_names() -> None:
    resolver = _resolver()

    assert resolver.get("de") is not None
    assert resolver.get("de_DE").code == "de"  # type: ignore[union-attr]
    assert resolver.get("Deutsch").code == "de"  # type: ignore[union-attr]
    assert resolver.get("Türkçe").code == "tr"  # type: ignore[union-attr]
    assert resolver.get("tr_TR").code == "tr"  # type: ignore[union-attr]
    assert resolver.get("fr") is None
    assert resolver.get("") is None
    assert resolver.get("not a language") is None


def test_an_unknown_language_names_the_available_choices() -> None:
    resolver = _resolver()

    with pytest.raises(ConfigError) as caught:
        resolver.require("fr")

    assert "unknown language: fr" in str(caught.value)
    assert caught.value.details["available"] == ["en", "de", "tr"]


def test_per_language_accessors_return_the_configured_values() -> None:
    resolver = _resolver()

    assert resolver.words_per_second("de").min == 1.8
    assert resolver.words_per_second("de").max == 3.0
    assert resolver.subtitle_font("de") == "Inter"
    assert resolver.hashtag_style("tr") == "upper"
    assert resolver.locale("tr") == "tr_TR"
    assert resolver.name("tr") == "Türkçe"
    assert resolver.name("en") == "English"
    assert resolver.direction("en") is TextDirection.LTR

    settings = resolver.resolve("de")
    assert settings.min_words_per_second == 1.8
    assert settings.max_words_per_second == 3.0
    assert settings.subtitle_font == "Inter"
    assert settings.hashtag_style == "title"
    assert settings.locale == "de_DE"
    assert settings.direction is TextDirection.LTR


def test_a_language_without_a_name_falls_back_to_its_code() -> None:
    resolver = LanguageResolver(LanguagesConfig.from_mapping({"en": {}}))

    assert resolver.name("en") == "en"


def test_module_level_accessors_use_the_same_configuration() -> None:
    config = LanguagesConfig.from_mapping(CONFIG)

    assert resolve_language(config, "tr").code == "tr"
    assert words_per_second(config, "de").min == 1.8
    assert subtitle_font(config, "de") == "Inter"
    assert hashtag_style(config, "tr") == "upper"
    assert language_locale(config, "tr") == "tr_TR"
    assert language_name(config, "tr") == "Türkçe"


def test_the_resolver_limits_detection_to_configured_languages() -> None:
    english = "This is the best day and we are happy."
    resolver = LanguageResolver(
        LanguagesConfig.from_mapping(
            {
                "default": "de",
                "de": {"name": "German"},
                "tr": {"name": "Turkish"},
            }
        )
    )

    assert resolver.detect(english) is None
    detected = resolver.detect("Das ist der beste Tag und wir sind glücklich.")
    assert detected is not None
    assert detected.code == "de"


def test_the_resolver_iterates_configured_languages_in_order() -> None:
    resolver = _resolver()

    assert resolver.codes == ("en", "de", "tr")
    assert [settings.code for settings in resolver] == ["en", "de", "tr"]
    assert len(resolver) == 3
    assert "de_DE" in resolver
    assert "fr" not in resolver


def test_duplicate_canonical_codes_are_a_configuration_error() -> None:
    config = LanguagesConfig.from_mapping({"en": {}, "en_US": {}})

    with pytest.raises(ConfigError) as caught:
        LanguageResolver(config)

    assert "duplicate language code: en" in str(caught.value)


def test_a_default_outside_the_configured_codes_is_a_configuration_error() -> None:
    config = LanguagesConfig.model_construct(
        default="de",
        languages={"en": LanguageConfig()},
    )

    with pytest.raises(ConfigError) as caught:
        LanguageResolver(config)

    assert "default language is not configured" in str(caught.value)
