"""Prompt loading: user files, packaged fallbacks and sandboxed rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.core import paths
from aivia.core.bootstrap import bootstrap
from aivia.core.errors import ConfigError, ValidationError
from aivia.llm import (
    PromptLibrary,
    PromptNotFoundError,
    PromptRenderError,
    PromptTemplate,
    create_prompt_environment,
    load_prompt,
    prompt_names,
    render_prompt,
    resolve_prompt_path,
)

DEFAULT_PROMPT = "Write a concise image prompt for {{ subject }}.\n"


def write_prompt(root: Path, name: str, text: str) -> Path:
    path = root / "prompts" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    return tmp_path / "config"


@pytest.fixture
def defaults_dir(tmp_path: Path) -> Path:
    root = tmp_path / "defaults"
    write_prompt(root, "image.md", DEFAULT_PROMPT)
    write_prompt(root, "plain.md", "Describe the scene.\n")
    return root


@pytest.fixture
def library(config_dir: Path, defaults_dir: Path) -> PromptLibrary:
    return PromptLibrary(config_dir=config_dir, defaults_dir=defaults_dir)


def test_a_packaged_default_is_used_when_the_user_has_no_copy(
    library: PromptLibrary,
    defaults_dir: Path,
) -> None:
    template = library.load("image")

    assert isinstance(template, PromptTemplate)
    assert template.name == "image.md"
    assert template.is_default is True
    assert template.is_user is False
    assert template.text == DEFAULT_PROMPT
    assert template.path == defaults_dir / "prompts" / "image.md"
    assert library.render("image", {"subject": "a lighthouse"}) == (
        "Write a concise image prompt for a lighthouse.\n"
    )


def test_the_shipped_image_prompt_loads_and_renders_without_variables() -> None:
    template = load_prompt("image")

    assert template.is_default is True
    assert template.path == paths.package_defaults_dir() / "prompts" / "image.md"
    assert "image" in template.render()


def test_a_user_prompt_shadows_the_packaged_default(
    library: PromptLibrary,
    config_dir: Path,
    defaults_dir: Path,
) -> None:
    user = write_prompt(config_dir, "image.md", "User prompt for {{ subject }}.\n")

    template = library.load("image")

    assert template.path == user
    assert template.is_user is True
    assert template.source == "user"
    assert library.render("image", subject="a fox") == "User prompt for a fox.\n"
    assert (defaults_dir / "prompts" / "image.md").is_file()


def test_a_user_prompt_without_a_default_still_loads(
    library: PromptLibrary,
    config_dir: Path,
) -> None:
    user = write_prompt(config_dir, "custom.j2", "Custom {{ value }}.\n")

    assert (
        resolve_prompt_path(
            "custom",
            config_dir=config_dir,
            defaults_dir=library.defaults_dir,
        )
        == user
    )
    assert library.render("custom", value=3) == "Custom 3.\n"


def test_the_extension_may_be_omitted_or_given(
    library: PromptLibrary,
    config_dir: Path,
) -> None:
    user = write_prompt(config_dir, "image.md", "User image.\n")

    assert library.resolve("image") == user
    assert library.resolve("image.md") == user
    assert library.load("image.md").path == user


def test_missing_variables_fail_strictly_and_name_the_template(
    library: PromptLibrary,
    config_dir: Path,
) -> None:
    write_prompt(
        config_dir,
        "writing.md",
        "Write about {{ topic }} for {{ audience }}.\n",
    )

    with pytest.raises(PromptRenderError) as caught:
        library.render("writing", {"topic": "birds"})

    assert caught.value.code == "prompt_render"
    assert "writing.md" in caught.value.message
    assert "audience" in caught.value.message
    assert isinstance(caught.value, ValidationError)


def test_a_template_syntax_error_is_a_validation_error_naming_the_prompt(
    library: PromptLibrary,
    config_dir: Path,
) -> None:
    write_prompt(config_dir, "broken.md", "Hello {% if name %}\n")

    with pytest.raises(PromptRenderError) as caught:
        library.render("broken", name="Ada")

    assert "broken.md" in caught.value.message
    assert caught.value.details["prompt"] == "broken.md"


def test_the_sandbox_blocks_access_to_python_internals(
    library: PromptLibrary,
    config_dir: Path,
) -> None:
    write_prompt(config_dir, "unsafe.md", "{{ ''.__class__.__mro__ }}\n")

    with pytest.raises(PromptRenderError) as caught:
        library.render("unsafe")

    assert "unsafe.md" in caught.value.message
    assert "unsafe" in caught.value.message.lower()


@pytest.mark.parametrize(
    ("name", "text", "tag"),
    [
        ("include.md", '{% include "../secret.txt" %}\n', "include"),
        ("extends.md", '{% extends "../base.md" %}\n', "extends"),
        ("import.md", '{% import "../macros.j2" as m %}\n', "import"),
        ("from.md", '{% from "../macros.j2" import shared %}\n', "from"),
    ],
)
def test_a_prompt_cannot_read_files_outside_the_prompts_directory(
    library: PromptLibrary,
    config_dir: Path,
    name: str,
    text: str,
    tag: str,
) -> None:
    write_prompt(config_dir, name, text)
    (config_dir / "secret.txt").write_text("TOP-SECRET", encoding="utf-8")
    (config_dir / "base.md").write_text("TOP-SECRET", encoding="utf-8")
    (config_dir / "macros.j2").write_text("TOP-SECRET", encoding="utf-8")

    with pytest.raises(PromptRenderError) as caught:
        library.render(name)

    assert caught.value.details["prompt"] == name
    assert caught.value.details["tag"] == tag
    assert tag in caught.value.message
    assert "TOP-SECRET" not in caught.value.message


def test_a_prompt_cannot_include_even_a_file_inside_the_prompts_directory(
    library: PromptLibrary,
    config_dir: Path,
) -> None:
    write_prompt(config_dir, "partial.md", "Shared text.\n")
    write_prompt(config_dir, "main.md", '{% include "partial.md" %}\n')

    with pytest.raises(PromptRenderError) as caught:
        library.render("main")

    assert caught.value.details["tag"] == "include"
    assert "main.md" in caught.value.message


def test_a_missing_prompt_lists_the_searched_directories(
    library: PromptLibrary,
) -> None:
    with pytest.raises(PromptNotFoundError) as caught:
        library.load("does-not-exist")

    assert isinstance(caught.value, ConfigError)
    assert "does-not-exist" in caught.value.message
    assert "image.md" in caught.value.message
    assert caught.value.details["searched"]


def test_a_prompt_name_cannot_escape_the_prompts_directory(
    library: PromptLibrary,
) -> None:
    with pytest.raises(ConfigError):
        library.load("../secrets.md")
    with pytest.raises(ConfigError):
        library.load("/etc/passwd")
    with pytest.raises(ConfigError):
        library.load("nested/../../secrets.md")


def test_names_merge_user_and_default_prompts_with_user_files_shadowing(
    library: PromptLibrary,
    config_dir: Path,
) -> None:
    write_prompt(config_dir, "image.md", "User image.\n")
    write_prompt(config_dir, "video.md", "User video.\n")

    names = library.names()

    assert names == ("image.md", "plain.md", "video.md")
    assert (
        prompt_names(
            config_dir=config_dir,
            defaults_dir=library.defaults_dir,
        )
        == names
    )
    assert "video.md" in library
    assert "missing.md" not in library


def test_versions_returns_the_user_copy_before_the_default(
    library: PromptLibrary,
    config_dir: Path,
) -> None:
    write_prompt(config_dir, "image.md", "User image.\n")

    versions = library.versions("image")

    assert [template.source for template in versions] == ["user", "default"]
    assert versions[0].is_user is True
    assert versions[1].is_default is True


def test_rendering_keyword_values_override_the_mapping(library: PromptLibrary) -> None:
    assert library.render("image", {"subject": "old"}, subject="new") == (
        "Write a concise image prompt for new.\n"
    )
    assert (
        render_prompt(
            "image",
            {"subject": "old"},
            subject="new",
            config_dir=library.config_dir,
            defaults_dir=library.defaults_dir,
        )
        == "Write a concise image prompt for new.\n"
    )


def test_bootstrap_copies_prompts_to_the_user_directory_before_loading(
    tmp_path: Path,
    defaults_dir: Path,
) -> None:
    config = tmp_path / "bootstrapped"
    bootstrap(config_dir=config, defaults_dir=defaults_dir)

    library = PromptLibrary(config_dir=config, defaults_dir=defaults_dir)
    template = library.load("image")

    assert template.path == config / "prompts" / "image.md"
    assert template.is_user is True
    assert library.render("image", subject="a castle") == (
        "Write a concise image prompt for a castle.\n"
    )


def test_the_default_library_uses_the_xdg_config_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    write_prompt(tmp_path / "xdg" / "aivia", "local.md", "Local {{ value }}.\n")

    library = PromptLibrary()

    assert library.config_dir == tmp_path / "xdg" / "aivia"
    assert library.render("local", value="prompt") == "Local prompt.\n"


def test_the_environment_is_sandboxed_and_strict() -> None:
    environment = create_prompt_environment()

    assert environment.sandboxed is True
    assert environment.undefined.__name__ == "StrictUndefined"
    assert create_prompt_environment() is not environment
