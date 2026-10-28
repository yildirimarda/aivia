"""Language configuration: locale, speech rate, subtitle font and hashtag style."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    LanguageConfig,
    LanguagesConfig,
    WordsPerSecondConfig,
)
from aivia.core.errors import ConfigError


def test_language_defaults_cover_locale_rate_font_and_hashtags() -> None:
    language = LanguageConfig.from_mapping({})

    assert language.locale == "en_US"
    assert language.words_per_second == WordsPerSecondConfig(min=2.0, max=3.5)
    assert language.subtitle_font == "Lato"
    assert language.hashtag_style == "lower"


def test_language_accepts_explicit_values() -> None:
    language = LanguageConfig.from_mapping(
        {
            "name": "Turkish",
            "locale": "tr_TR",
            "words_per_second": {"min": 1.8, "max": 3.2},
            "subtitle_font": "Lato",
            "hashtag_style": "title",
        }
    )

    assert language.name == "Turkish"
    assert language.locale == "tr_TR"
    assert language.words_per_second.min == 1.8
    assert language.words_per_second.max == 3.2
    assert language.subtitle_font == "Lato"
    assert language.hashtag_style == "title"


@pytest.mark.parametrize("style", ["lower", "title", "upper"])
def test_every_hashtag_style_is_accepted(style: str) -> None:
    language = LanguageConfig.from_mapping({"hashtag_style": style})

    assert language.hashtag_style == style


def test_unknown_hashtag_style_reports_the_language_field() -> None:
    with pytest.raises(ConfigError) as caught:
        LanguagesConfig.from_mapping({"en": {"hashtag_style": "camel"}})

    assert "languages.en.hashtag_style" in str(caught.value)


@pytest.mark.parametrize("value", [0, -1.5])
def test_words_per_second_bounds_must_be_positive(value: float) -> None:
    with pytest.raises(ConfigError) as caught:
        LanguageConfig.from_mapping({"words_per_second": {"min": value}})

    assert "words_per_second.min" in str(caught.value)


def test_words_per_second_min_must_not_exceed_max() -> None:
    with pytest.raises(ConfigError) as caught:
        LanguageConfig.from_mapping({"words_per_second": {"min": 4.0, "max": 3.0}})

    assert "words_per_second" in str(caught.value)
    assert "must not exceed max" in str(caught.value)


def test_words_per_second_equal_bounds_are_allowed() -> None:
    language = LanguageConfig.from_mapping(
        {"words_per_second": {"min": 2.5, "max": 2.5}}
    )

    assert language.words_per_second.min == 2.5
    assert language.words_per_second.max == 2.5


def test_subtitle_font_must_not_be_empty() -> None:
    with pytest.raises(ConfigError) as caught:
        LanguageConfig.from_mapping({"subtitle_font": ""})

    assert "subtitle_font" in str(caught.value)


def test_unknown_rate_keys_report_the_full_dotted_path() -> None:
    with pytest.raises(ConfigError) as caught:
        LanguagesConfig.from_mapping({"en": {"words_per_second": {"median": 2.5}}})

    assert "languages.en.words_per_second.median" in str(caught.value)


def test_nested_rate_defaults_are_independent_between_instances() -> None:
    first = LanguageConfig.from_mapping({})
    second = LanguageConfig.from_mapping({})

    first.words_per_second.min = 1.0

    assert second.words_per_second.min == 2.0


def test_languages_keep_per_language_rate_font_and_hashtag_style() -> None:
    config = LanguagesConfig.from_mapping(
        {
            "default": "de",
            "en": {
                "name": "English",
                "locale": "en_US",
                "words_per_second": {"min": 2.2, "max": 3.6},
                "subtitle_font": "Lato",
                "hashtag_style": "lower",
            },
            "de": {
                "name": "German",
                "locale": "de_DE",
                "words_per_second": {"min": 1.8, "max": 3.2},
                "subtitle_font": "Lato",
                "hashtag_style": "title",
            },
        }
    )

    assert config.default == "de"
    assert config.languages["en"].words_per_second.max == 3.6
    assert config.languages["de"].words_per_second.min == 1.8
    assert config.languages["de"].subtitle_font == "Lato"
    assert config.languages["de"].hashtag_style == "title"


def test_languages_default_must_name_a_configured_language() -> None:
    with pytest.raises(ConfigError) as caught:
        LanguagesConfig.from_mapping({"default": "de", "en": {"name": "English"}})

    message = str(caught.value)
    assert "languages.default" in message
    assert "de" in message
    assert "available: en" in message


@pytest.mark.parametrize("payload", [{}, {"default": "en"}])
def test_languages_reject_a_default_without_entries(
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ConfigError) as caught:
        LanguagesConfig.from_mapping(payload)

    message = str(caught.value)
    assert "languages.default" in message
    assert "at least one configured language" in message


def test_languages_accept_a_configured_default_in_the_nested_shape() -> None:
    config = LanguagesConfig.from_mapping(
        {"default": "tr", "languages": {"tr": {"name": "Turkish"}}}
    )

    assert config.default == "tr"
    assert config.languages["tr"].name == "Turkish"


def test_example_config_declares_the_language_controls() -> None:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    languages = LanguagesConfig.from_mapping(payload["languages"])

    assert languages.languages["en"].words_per_second.min == 2.0
    assert languages.languages["en"].words_per_second.max == 3.5
    assert languages.languages["en"].subtitle_font == "Lato"
    assert languages.languages["de"].hashtag_style == "title"
