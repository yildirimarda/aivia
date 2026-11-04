"""The style bible loads from user files with packaged defaults as fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aivia.core import paths
from aivia.core.errors import ConfigError
from aivia.writing import (
    DEFAULT_STYLE_BIBLE,
    DEFAULT_STYLE_BIBLE_FILENAME,
    DEFAULT_STYLE_BIBLE_NAME,
    STYLE_BIBLE_DIR,
    StyleBible,
    StyleBibleError,
    StyleBibleLibrary,
    StyleBibleNotFoundError,
    load_default_style_bible,
    load_style_bible,
    resolve_style_bible_path,
)


def write_bible(root: Path, name: str, text: str) -> Path:
    path = root / STYLE_BIBLE_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def user_bible() -> str:
    return """\
schema_version: 1
name: default
description: The user's own voice.
entries:
  - key: voice
    title: Voice
    text: Warm, plain and specific.
    prompt_suffix: Soft natural light.
"""


def test_the_packaged_default_bible_loads_with_entries_and_suffixes(
    tmp_path: Path,
) -> None:
    bible = load_style_bible(
        config_dir=tmp_path / "config",
        defaults_dir=paths.package_defaults_dir(),
    )

    assert isinstance(bible, StyleBible)
    assert bible.name == DEFAULT_STYLE_BIBLE_NAME
    assert bible.source == paths.package_defaults_dir() / STYLE_BIBLE_DIR / (
        DEFAULT_STYLE_BIBLE_FILENAME
    )
    assert DEFAULT_STYLE_BIBLE.source == bible.source
    assert len(DEFAULT_STYLE_BIBLE) >= 1
    assert DEFAULT_STYLE_BIBLE.source == bible.source
    assert bible.entries.keys() == tuple(entry.key for entry in bible.entries)
    assert bible.entries.get("voice") is not None
    assert "voice" in bible
    assert bible.require("Voice").text
    assert bible.to_prompt()
    assert bible.text == bible.to_prompt()
    assert bible.suffix
    assert "voice" in bible.to_prompt_dict()["entries"]


def test_a_user_bible_shadows_the_packaged_default(tmp_path: Path) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    write_bible(
        defaults, "default.yaml", "entries:\n  - key: voice\n    text: packaged\n"
    )
    user = write_bible(config, "default.yaml", user_bible())

    bible = load_style_bible("default", config_dir=config, defaults_dir=defaults)

    assert bible.source == user
    assert bible.description == "The user's own voice."
    assert bible.require("voice").text == "Warm, plain and specific."
    assert bible.suffix == "Soft natural light."


def test_a_packaged_default_is_used_when_the_user_has_no_copy(tmp_path: Path) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    packaged = write_bible(
        defaults,
        "default.yaml",
        "name: default\nentries:\n  - key: voice\n    text: packaged\n",
    )

    bible = load_style_bible("default", config_dir=config, defaults_dir=defaults)

    assert bible.source == packaged
    assert bible.require("voice").text == "packaged"
    assert (
        resolve_style_bible_path("default", config_dir=config, defaults_dir=defaults)
        == packaged
    )


def test_a_user_only_bible_loads_without_a_packaged_copy(tmp_path: Path) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    user = write_bible(config, "custom.yaml", user_bible())

    bible = load_style_bible("custom", config_dir=config, defaults_dir=defaults)

    assert bible.source == user
    assert bible.name == "default"


def test_the_name_may_omit_or_include_a_supported_extension(tmp_path: Path) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    user = write_bible(config, "plain.md", "# Voice\n\nWarm and plain.\n")

    assert (
        load_style_bible("plain", config_dir=config, defaults_dir=defaults).source
        == user
    )
    assert (
        load_style_bible("plain.md", config_dir=config, defaults_dir=defaults).source
        == user
    )
    assert load_style_bible(user).require("style").text == "# Voice\n\nWarm and plain."


def test_a_missing_bible_names_the_bible_and_every_directory_searched(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"

    with pytest.raises(StyleBibleNotFoundError) as caught:
        load_style_bible("missing", config_dir=config, defaults_dir=defaults)

    assert isinstance(caught.value, ConfigError)
    assert caught.value.code == "style_bible_not_found"
    assert "missing" in caught.value.message
    assert str(config / STYLE_BIBLE_DIR) in caught.value.message
    assert str(defaults / STYLE_BIBLE_DIR) in caught.value.message
    assert caught.value.details["searched"] == [
        str(config / STYLE_BIBLE_DIR),
        str(defaults / STYLE_BIBLE_DIR),
    ]


def test_an_explicit_missing_path_fails_clearly(tmp_path: Path) -> None:
    missing = tmp_path / "styles" / "missing.yaml"

    with pytest.raises(StyleBibleNotFoundError, match="not found") as caught:
        load_style_bible(missing)

    assert str(missing) in caught.value.message


def test_a_requested_source_that_does_not_exist_fails_clearly(tmp_path: Path) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    write_bible(
        defaults, "default.yaml", "entries:\n  - key: voice\n    text: packaged\n"
    )

    with pytest.raises(StyleBibleNotFoundError, match="no user version"):
        load_style_bible(
            "default", version="user", config_dir=config, defaults_dir=defaults
        )


def test_a_malformed_bible_fails_clearly(tmp_path: Path) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    write_bible(config, "broken.yaml", "entries: [\n")

    with pytest.raises(StyleBibleError, match="invalid style bible"):
        load_style_bible("broken", config_dir=config, defaults_dir=defaults)

    with pytest.raises(StyleBibleError, match="unsupported style bible extension"):
        load_style_bible(write_bible(config, "broken.ini", "x"))


def test_a_bible_without_entries_or_text_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    write_bible(config, "empty.yaml", "name: empty\n")

    with pytest.raises(ValueError, match="at least one entry"):
        load_style_bible("empty", config_dir=config, defaults_dir=defaults)


def test_a_mapping_with_named_entries_builds_a_bible() -> None:
    bible = load_style_bible(
        {
            "name": "custom",
            "description": "A compact bible.",
            "prompt_suffix": "Matte texture.",
            "entries": {
                "voice": "Plain and warm.",
                "visual": {"text": "Soft light.", "suffix": "Soft light."},
            },
        }
    )

    assert bible.name == "custom"
    assert bible.entries.keys() == ("voice", "visual")
    assert bible.entries["voice"].text == "Plain and warm."
    assert bible.entries["visual"].suffix == "Soft light."
    assert bible.suffixes() == ("Soft light.", "Matte texture.")
    assert "Matte texture." in bible.to_prompt()


def test_the_library_lists_user_and_packaged_names(tmp_path: Path) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    write_bible(config, "custom.yaml", user_bible())
    write_bible(defaults, "default.yaml", user_bible())
    library = StyleBibleLibrary(config_dir=config, defaults_dir=defaults)

    assert library.names() == ("custom", "default")
    assert "custom" in library
    assert "missing" not in library
    assert len(library) == 2


def test_load_default_style_bible_matches_the_packaged_default() -> None:
    bible = load_default_style_bible()

    assert bible.name == DEFAULT_STYLE_BIBLE_NAME
    assert bible.source == paths.package_defaults_dir() / STYLE_BIBLE_DIR / (
        DEFAULT_STYLE_BIBLE_FILENAME
    )
    assert bible.entries.keys()


def test_the_packaged_default_bible_has_no_absolute_or_template_data() -> None:
    bible = load_default_style_bible()
    payload: Any = bible.to_dict()

    assert payload["name"] == "default"
    assert all(entry["text"] for entry in payload["entries"])
