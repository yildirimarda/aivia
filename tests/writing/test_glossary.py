"""A per language brand glossary loads from files and reaches every writer output."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.core import paths
from aivia.core.errors import ConfigError
from aivia.core.models.script import Beat, DialogueLine
from aivia.core.platforms import CaptionLimits, DisclosureRules, Platform
from aivia.writing import (
    DEFAULT_GLOSSARY,
    DEFAULT_GLOSSARY_FILENAME,
    DEFAULT_GLOSSARY_NAME,
    GLOSSARY_DIR,
    Glossary,
    GlossaryApplication,
    GlossaryError,
    GlossaryLibrary,
    GlossaryNotFoundError,
    GlossaryViolation,
    MediaPrompt,
    Script,
    apply_glossary,
    apply_glossary_to_captions,
    apply_glossary_to_media_prompts,
    apply_glossary_to_script,
    build_captions,
    build_media_prompts,
    glossary_violations,
    load_default_glossary,
    load_glossary,
    resolve_glossary_path,
)


def write_glossary(root: Path, name: str, text: str) -> Path:
    path = root / GLOSSARY_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def user_glossary() -> str:
    return """\
schema_version: 1
name: default
language: en
terms:
  - term: Aivia
    preferred: AIVIA
    definition: a small blue robot narrator
    avoid: [Avia]
  - term: AI
    preferred: artificial intelligence
    aliases: [A.I.]
"""


def _platform() -> Platform:
    return Platform(
        id="example",
        name="Example",
        caption_limits=CaptionLimits(
            max_length=5000,
            max_title_length=100,
            max_description_length=5000,
            max_hashtags=15,
        ),
        disclosure=DisclosureRules(
            required=True,
            text="This content was generated with AI.",
        ),
    )


def _script() -> Script:
    return Script(
        title="Avia explains AI",
        hook="Avia is an Aivia character.",
        beats=[
            Beat(
                text="Avia uses AI to explain the hive.",
                dialogue=[DialogueLine(speaker="narrator", text="Avia speaks.")],
            )
        ],
        outro="Avia says goodbye.",
    )


def test_the_packaged_default_glossary_is_empty(tmp_path: Path) -> None:
    glossary = load_glossary(
        config_dir=tmp_path / "config",
        defaults_dir=paths.package_defaults_dir(),
    )

    assert isinstance(glossary, Glossary)
    assert glossary.name == DEFAULT_GLOSSARY_NAME
    assert glossary.source == paths.package_defaults_dir() / GLOSSARY_DIR / (
        DEFAULT_GLOSSARY_FILENAME
    )
    assert DEFAULT_GLOSSARY.source == glossary.source
    assert glossary.terms == []
    assert load_default_glossary().terms == []


def test_a_user_glossary_shadows_the_packaged_default(tmp_path: Path) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    write_glossary(
        defaults,
        "default.yaml",
        "terms:\n  - term: Aivia\n    preferred: packaged\n",
    )
    user = write_glossary(config, "default.yaml", user_glossary())

    glossary = load_glossary("default", config_dir=config, defaults_dir=defaults)

    assert glossary.source == user
    assert glossary.name == "default"
    assert glossary.terms[0].preferred == "AIVIA"
    assert glossary.terms[0].definition == "a small blue robot narrator"
    assert glossary.terms[1].aliases == ("A.I.",)


def test_a_packaged_glossary_is_used_without_a_user_copy(tmp_path: Path) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    packaged = write_glossary(
        defaults,
        "default.yaml",
        "terms:\n  - term: Aivia\n    preferred: AIVIA\n",
    )

    glossary = load_glossary("default", config_dir=config, defaults_dir=defaults)

    assert glossary.source == packaged
    assert (
        resolve_glossary_path("default", config_dir=config, defaults_dir=defaults)
        == packaged
    )
    assert GlossaryLibrary(config_dir=config, defaults_dir=defaults).names() == (
        "default",
    )


def test_a_language_file_is_preferred_for_that_language(tmp_path: Path) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    write_glossary(config, "default.yaml", user_glossary())
    german = write_glossary(
        config,
        "de.yaml",
        "terms:\n  - term: honey\n    preferred: Honig\n    language: de\n",
    )

    glossary = load_glossary(language="de", config_dir=config, defaults_dir=defaults)

    assert glossary.source == german
    assert glossary.apply("honey", language="de") == "Honig"
    assert glossary.apply("honey", language="en") == "honey"


def test_a_language_filters_terms_in_the_default_file(tmp_path: Path) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    write_glossary(
        config,
        "default.yaml",
        """\
terms:
  - term: honey
    preferred: Honig
    language: de
  - term: honey
    preferred: bal
    language: tr
""",
    )

    german = load_glossary(language="de", config_dir=config, defaults_dir=defaults)
    turkish = load_glossary(language="tr", config_dir=config, defaults_dir=defaults)

    assert german.apply("honey", language="de") == "Honig"
    assert turkish.apply("honey", language="tr") == "bal"
    assert german.apply("honey", language="en") == "honey"


def test_a_missing_glossary_names_every_directory_searched(tmp_path: Path) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"

    with pytest.raises(GlossaryNotFoundError) as caught:
        load_glossary("missing", config_dir=config, defaults_dir=defaults)

    assert isinstance(caught.value, ConfigError)
    assert caught.value.code == "glossary_not_found"
    assert "missing" in caught.value.message
    assert caught.value.details["searched"] == [
        str(config / GLOSSARY_DIR),
        str(defaults / GLOSSARY_DIR),
    ]


def test_a_malformed_glossary_fails_clearly(tmp_path: Path) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    write_glossary(config, "broken.yaml", "terms: [\n")

    with pytest.raises(GlossaryError, match="invalid glossary") as caught:
        load_glossary("broken", config_dir=config, defaults_dir=defaults)

    assert caught.value.code == "glossary_invalid"


def test_applying_a_glossary_rewrites_text_and_reports_violations() -> None:
    glossary = load_glossary(user_glossary())

    result = apply_glossary("Avia uses AI and Aivia.", glossary)

    assert result == "AIVIA uses artificial intelligence and AIVIA."
    assert isinstance(result, GlossaryApplication)
    assert [violation.found for violation in result.violations] == [
        "Avia",
        "AI",
        "Aivia",
    ]
    assert all(isinstance(item, GlossaryViolation) for item in result.violations)
    assert result.violations[0].expected == "AIVIA"
    assert result.violations[0].to_dict()["code"] == "glossary"
    assert glossary.violations("Avia uses AI").__len__() == 2


def test_a_script_title_hook_beats_dialogue_and_outro_are_rewritten() -> None:
    glossary = load_glossary(user_glossary())
    script = _script()

    result = apply_glossary_to_script(script, glossary)

    assert result is not script
    assert result.title == "AIVIA explains artificial intelligence"
    assert result.hook == "AIVIA is an AIVIA character."
    assert result.beats[0].text == (
        "AIVIA uses artificial intelligence to explain the hive."
    )
    assert result.beats[0].dialogue[0].text == "AIVIA speaks."
    assert result.outro == "AIVIA says goodbye."
    assert script.title == "Avia explains AI"
    violations = glossary_violations(script, glossary)
    assert {item.found for item in violations} == {"Avia", "AI", "Aivia"}


def test_media_prompts_accept_a_glossary_loaded_from_a_file(tmp_path: Path) -> None:
    config = tmp_path / "config"
    defaults = tmp_path / "defaults"
    write_glossary(config, "default.yaml", user_glossary())
    glossary = load_glossary(config_dir=config, defaults_dir=defaults)
    script = build_media_prompts(_script(), glossary=glossary)

    image = script.beats[0].image_prompt
    assert image is not None
    assert "AIVIA" in image.prompt
    assert "Avia" not in image.prompt
    assert "AIVIA: a small blue robot narrator" in image.prompt

    rewritten = apply_glossary_to_media_prompts(script, glossary)
    assert rewritten.beats[0].image_prompt is not None
    assert "Avia" not in rewritten.beats[0].image_prompt.prompt


def test_captions_apply_the_glossary_and_report_violations() -> None:
    glossary = load_glossary(user_glossary())

    result = build_captions(_script(), _platform(), glossary=glossary)

    assert result.title == "AIVIA explains artificial intelligence"
    assert "AIVIA" in result.caption
    assert "Avia" not in result.caption
    assert result.glossary_violations
    assert result.warnings
    assert any("glossary" in warning for warning in result.warnings)

    rewritten = apply_glossary_to_captions(result, glossary)
    assert "Avia" not in rewritten.caption


def test_a_glossary_name_reaches_media_prompts_and_captions(tmp_path: Path) -> None:
    config = tmp_path / "config"
    write_glossary(config, "default.yaml", user_glossary())
    script = _script()

    prompts = build_media_prompts(
        script,
        glossary="default",
        config_dir=config,
    )
    captions = build_captions(
        script,
        _platform(),
        glossary="default",
        config_dir=config,
    )

    assert prompts.beats[0].image_prompt is not None
    assert "AIVIA" in prompts.beats[0].image_prompt.prompt
    assert captions.title == "AIVIA explains artificial intelligence"


def test_applying_to_a_script_also_rewrites_its_media_prompts() -> None:
    glossary = load_glossary(user_glossary())
    script = _script()
    script.beats[0].media_prompts = [MediaPrompt(prompt="Avia stands in a hive.")]

    result = apply_glossary_to_script(script, glossary)

    assert result.beats[0].image_prompt is not None
    assert result.beats[0].image_prompt.prompt == "AIVIA stands in a hive."
