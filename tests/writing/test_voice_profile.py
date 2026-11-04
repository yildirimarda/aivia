"""The per profile voice profile loads from user files before writing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aivia.core import paths
from aivia.core.config.profile import ContentProfileConfig
from aivia.core.errors import ConfigError
from aivia.writing import (
    DEFAULT_VOICE_PROFILE,
    DEFAULT_VOICE_PROFILE_FILENAME,
    DEFAULT_VOICE_PROFILE_NAME,
    VOICE_PROFILE_DIR,
    VOICE_PROFILE_TEMPLATE_FILENAME,
    VoiceProfile,
    VoiceProfileError,
    VoiceProfileLibrary,
    VoiceProfileNotFoundError,
    load_default_voice_profile,
    load_voice_profile,
    load_voice_profile_for_profile,
    resolve_voice_profile_path,
)


def write_profile(root: Path, name: str, text: str) -> Path:
    path = root / VOICE_PROFILE_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def user_profile() -> str:
    return """\
schema_version: 1
name: default
description: The user's own voice.
language: en
persona: A friendly historian.
tone: Warm, plain and specific.
audience: Curious adults.
point_of_view: second person
style: Short declarative sentences.
vocabulary: [concrete, vivid]
forbidden_words: [synergy]
forbidden_phrases: [in today's video]
signature_phrases: [Here is the part that matters.]
avoid: [Clickbait, shouting]
rules:
  - Explain the idea the way a friend would.
examples:
  - Start with one concrete detail.
instructions: Never invent a number.
prompt_suffix: Calm and factual.
entries:
  - key: rhythm
    title: Rhythm
    text: Leave room for each idea to land.
    prompt_suffix: Deliberate pacing.
"""


def test_the_packaged_default_is_an_empty_template(tmp_path: Path) -> None:
    profile = load_voice_profile(
        config_dir=tmp_path / "config",
        defaults_dir=paths.package_defaults_dir(),
    )

    assert isinstance(profile, VoiceProfile)
    assert profile.name == DEFAULT_VOICE_PROFILE_NAME
    assert profile.source == paths.package_defaults_dir() / VOICE_PROFILE_DIR / (
        DEFAULT_VOICE_PROFILE_FILENAME
    )
    assert DEFAULT_VOICE_PROFILE.source == profile.source
    assert DEFAULT_VOICE_PROFILE.is_empty
    assert profile.is_empty
    assert profile.to_prompt() == ""
    assert load_default_voice_profile().is_empty


def test_the_packaged_empty_template_is_shipped(tmp_path: Path) -> None:
    path = (
        paths.package_defaults_dir()
        / VOICE_PROFILE_DIR
        / VOICE_PROFILE_TEMPLATE_FILENAME
    )

    assert path.is_file()
    profile = load_voice_profile("template", config_dir=tmp_path / "config")

    assert profile.is_empty
    assert profile.entries == ()


def test_a_user_profile_shadows_the_packaged_default(tmp_path: Path) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    write_profile(
        defaults,
        "default.yaml",
        "name: default\ntone: packaged\n",
    )
    user = write_profile(config, "default.yaml", user_profile())

    profile = load_voice_profile("default", config_dir=config, defaults_dir=defaults)

    assert profile.source == user
    assert profile.description == "The user's own voice."
    assert profile.tone == "Warm, plain and specific."
    assert profile.audience == "Curious adults."
    assert profile.vocabulary == ("concrete", "vivid")
    assert profile.forbidden_words == ("synergy",)
    assert profile.forbidden_phrases == ("in today's video",)
    assert profile.require("rhythm").text == "Leave room for each idea to land."
    assert profile.suffixes == ("Deliberate pacing.", "Calm and factual.")
    assert "Warm, plain and specific." in profile.to_prompt()


def test_a_packaged_default_is_used_when_the_user_has_no_copy(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    packaged = write_profile(
        defaults,
        "default.yaml",
        "name: default\ntone: packaged\n",
    )

    profile = load_voice_profile("default", config_dir=config, defaults_dir=defaults)

    assert profile.source == packaged
    assert profile.tone == "packaged"
    assert (
        resolve_voice_profile_path("default", config_dir=config, defaults_dir=defaults)
        == packaged
    )


def test_the_profile_voice_name_selects_the_file_before_writing(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    selected = write_profile(
        config,
        "historical.yaml",
        "name: historical\ntone: Measured and vivid.\n",
    )
    profile = ContentProfileConfig(voice="historical")

    loaded = load_voice_profile_for_profile(
        profile, config_dir=config, defaults_dir=defaults
    )

    assert loaded.source == selected
    assert loaded.tone == "Measured and vivid."
    assert (
        load_voice_profile(profile, config_dir=config, defaults_dir=defaults).tone
        == "Measured and vivid."
    )


def test_a_profile_without_a_voice_loads_the_empty_template(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    write_profile(defaults, "default.yaml", "name: default\n")

    profile = load_voice_profile_for_profile(
        ContentProfileConfig(),
        config_dir=config,
        defaults_dir=defaults,
    )

    assert profile.name == DEFAULT_VOICE_PROFILE_NAME
    assert profile.is_empty


def test_the_name_may_omit_or_include_a_supported_extension(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    user = write_profile(config, "plain.md", "Warm and plain.\n")

    assert (
        load_voice_profile("plain", config_dir=config, defaults_dir=defaults).source
        == user
    )
    assert (
        load_voice_profile("plain.md", config_dir=config, defaults_dir=defaults).source
        == user
    )
    assert load_voice_profile(user).instructions == "Warm and plain."


def test_a_missing_profile_names_the_profile_and_every_directory_searched(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"

    with pytest.raises(VoiceProfileNotFoundError) as caught:
        load_voice_profile("missing", config_dir=config, defaults_dir=defaults)

    assert isinstance(caught.value, ConfigError)
    assert caught.value.code == "voice_profile_not_found"
    assert "missing" in caught.value.message
    assert str(config / VOICE_PROFILE_DIR) in caught.value.message
    assert str(defaults / VOICE_PROFILE_DIR) in caught.value.message
    assert caught.value.details["searched"] == [
        str(config / VOICE_PROFILE_DIR),
        str(defaults / VOICE_PROFILE_DIR),
    ]


def test_an_explicit_missing_path_fails_clearly(tmp_path: Path) -> None:
    missing = tmp_path / "voices" / "missing.yaml"

    with pytest.raises(VoiceProfileNotFoundError, match="not found") as caught:
        load_voice_profile(missing)

    assert str(missing) in caught.value.message


def test_a_requested_source_that_does_not_exist_fails_clearly(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    write_profile(defaults, "default.yaml", "name: default\n")

    with pytest.raises(VoiceProfileNotFoundError, match="no user version"):
        load_voice_profile(
            "default", version="user", config_dir=config, defaults_dir=defaults
        )


def test_a_malformed_profile_fails_clearly(tmp_path: Path) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    write_profile(config, "broken.yaml", "tone: [\n")

    with pytest.raises(VoiceProfileError, match="invalid voice profile"):
        load_voice_profile("broken", config_dir=config, defaults_dir=defaults)

    with pytest.raises(VoiceProfileError, match="unsupported voice profile extension"):
        load_voice_profile(write_profile(config, "broken.ini", "x"))


def test_a_mapping_list_and_text_build_a_profile() -> None:
    profile = load_voice_profile(
        {
            "name": "custom",
            "description": "A compact voice.",
            "tone": "Dry.",
            "forbidden_words": "synergy, leverage",
            "entries": {"rhythm": "Let each idea land."},
        }
    )

    assert profile.name == "custom"
    assert profile.tone == "Dry."
    assert profile.forbidden_words == ("synergy, leverage",)
    assert profile.entries.keys() == ("rhythm",)
    assert load_voice_profile(["Keep it short.", "Avoid jargon."]).rules == (
        "Keep it short.",
        "Avoid jargon.",
    )
    assert VoiceProfile.from_mapping("Plain text guidance.").instructions == (
        "Plain text guidance."
    )


def test_the_library_lists_user_and_packaged_names(tmp_path: Path) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    write_profile(config, "custom.yaml", user_profile())
    write_profile(defaults, "default.yaml", user_profile())
    library = VoiceProfileLibrary(config_dir=config, defaults_dir=defaults)

    assert library.names() == ("custom", "default")
    assert "custom" in library
    assert "missing" not in library
    assert len(library) == 2


def test_the_packaged_template_has_no_guidance_data() -> None:
    profile = load_default_voice_profile()
    payload: Any = profile.to_dict()

    assert payload["name"] == DEFAULT_VOICE_PROFILE_NAME
    assert payload["entries"] == []
    assert not any(
        payload[key]
        for key in (
            "description",
            "language",
            "persona",
            "tone",
            "audience",
            "point_of_view",
            "tense",
            "formality",
            "style",
            "voice",
            "pacing",
            "sentence_style",
            "vocabulary",
            "forbidden_words",
            "forbidden_phrases",
            "signature_phrases",
            "avoid",
            "rules",
            "examples",
            "instructions",
            "prompt_suffix",
        )
    )
