"""Multiple languages per occasion for lyrics, speech, on screen text and
age and date phrasing."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    OccasionConfig,
    OccasionLanguageConfig,
    OccasionsConfig,
)
from aivia.core.errors import ConfigError


def example_section(name: str) -> dict[str, Any]:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    return cast(dict[str, Any], payload[name])


def birthday_with_languages() -> dict[str, Any]:
    return {
        "label": "Birthday",
        "languages": {
            "en": {
                "lyrics": "Happy birthday to {name}",
                "speech": "Happy birthday, {name}, turning {age}",
                "on_screen_text": "{name} is {age} today!",
                "age_phrasing": "turning {age}",
                "date_phrasing": "on {date}",
            },
            "de": {
                "lyrics": "Alles Gute zum Geburtstag, {name}",
                "speech": "Alles Gute, {name}, zum {age}. Geburtstag",
                "on_screen_text": "{name} wird heute {age}",
                "age_phrasing": "zum {age}. Geburtstag",
                "date_phrasing": "am {date}",
            },
        },
    }


def test_occasion_accepts_per_language_content() -> None:
    occasion = OccasionConfig.from_mapping(birthday_with_languages())

    assert occasion.language_codes() == ["en", "de"]
    english = occasion.language_for("en")
    assert english.lyrics == "Happy birthday to {name}"
    assert english.speech == "Happy birthday, {name}, turning {age}"
    assert english.on_screen_text == "{name} is {age} today!"
    assert english.age_phrasing == "turning {age}"
    assert english.date_phrasing == "on {date}"

    german = occasion.language_for("de")
    assert german.speech == "Alles Gute, {name}, zum {age}. Geburtstag"
    assert german.age_phrasing == "zum {age}. Geburtstag"


def test_occasion_language_list_form_still_works() -> None:
    occasion = OccasionConfig.from_mapping(
        {"label": "Birthday", "languages": ["en", "de", "fr"]}
    )

    assert occasion.language_codes() == ["en", "de", "fr"]
    assert occasion.language_for("en") == OccasionLanguageConfig()
    assert occasion.language_for("de").lyrics == ""
    assert occasion.language_for("fr").speech == ""


def test_unknown_language_returns_empty_content() -> None:
    occasion = OccasionConfig.from_mapping(birthday_with_languages())

    missing = occasion.language_for("pt")
    assert missing == OccasionLanguageConfig()
    assert missing.lyrics == ""
    assert missing.date_phrasing == ""


def test_languages_reject_non_string_content() -> None:
    with pytest.raises(ConfigError):
        OccasionConfig.from_mapping(
            {
                "label": "Birthday",
                "languages": {"en": {"lyrics": 123}},
            }
        )


def test_languages_reject_invalid_shape() -> None:
    with pytest.raises(ConfigError):
        OccasionConfig.from_mapping({"label": "Birthday", "languages": "en"})


def test_languages_accept_none_and_empty() -> None:
    occasion = OccasionConfig.from_mapping({"label": "Birthday"})

    assert occasion.language_codes() == []
    assert occasion.language_for("en") == OccasionLanguageConfig()

    occasion = OccasionConfig.from_mapping({"label": "Birthday", "languages": None})
    assert occasion.language_codes() == []


def test_occasions_config_exposes_per_language_content() -> None:
    occasions = OccasionsConfig.from_mapping(
        {
            "default": "birthday",
            "birthday": birthday_with_languages(),
        }
    )

    assert occasions.default == "birthday"
    german = occasions.occasions["birthday"].language_for("de")
    assert german.on_screen_text == "{name} wird heute {age}"
    assert occasions.occasions["birthday"].language_for("en").date_phrasing == (
        "on {date}"
    )


def test_packaged_example_occasions_declare_no_languages() -> None:
    occasions = OccasionsConfig.from_mapping(example_section("occasions"))

    for occasion in occasions.occasions.values():
        assert occasion.language_codes() == []
        assert occasion.language_for("en") == OccasionLanguageConfig()
