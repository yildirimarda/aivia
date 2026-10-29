"""``aivia prompts``: select versions per config/profile and list them."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.cli import main
from aivia.core import paths
from aivia.core.errors import ConfigError
from aivia.llm import PromptLibrary, PromptNotFoundError

USER_IMAGE = "User image prompt for {{ subject }}.\n"
DEFAULT_IMAGE = paths.package_defaults_dir() / "prompts" / "image.md"


def write_prompt(root: Path, name: str, text: str) -> Path:
    path = root / "prompts" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_config(root: Path, text: str) -> Path:
    path = root / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def run(
    capsys: pytest.CaptureFixture[str],
    *arguments: str,
) -> tuple[int, str, str]:
    code = main(["prompts", *arguments])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_list_shows_user_default_and_selected_versions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_prompt(tmp_path, "image.md", USER_IMAGE)
    config = write_config(tmp_path, "schema_version: 1\n")

    code, output, err = run(capsys, "list", "--config", str(config))

    assert code == 0
    assert err == ""
    lines = output.splitlines()
    assert lines[0] == "NAME\tUSER\tDEFAULT\tSELECTED"
    assert "image.md\tyes\tyes\tuser" in lines


def test_a_default_only_prompt_lists_the_packaged_version(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path, "schema_version: 1\n")

    code, output, err = run(capsys, "list", "--config", str(config))

    assert code == 0
    assert err == ""
    assert "image.md\tno\tyes\tdefault" in output.splitlines()


def test_config_selects_the_default_over_a_user_copy(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_prompt(tmp_path, "image.md", USER_IMAGE)
    config = write_config(
        tmp_path,
        """
schema_version: 1
prompts:
  versions:
    image: default
""".lstrip(),
    )

    code, output, err = run(capsys, "list", "--config", str(config))

    assert code == 0
    assert err == ""
    assert "image.md\tyes\tyes\tdefault" in output.splitlines()


def test_config_accepts_a_direct_prompt_mapping(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_prompt(tmp_path, "image.md", USER_IMAGE)
    config = write_config(
        tmp_path,
        """
schema_version: 1
prompts:
  image: default
""".lstrip(),
    )

    code, output, _ = run(capsys, "list", "--config", str(config))

    assert code == 0
    assert "image.md\tyes\tyes\tdefault" in output.splitlines()


def test_profile_overrides_select_a_different_version(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_prompt(tmp_path, "image.md", USER_IMAGE)
    config = write_config(
        tmp_path,
        """
schema_version: 1
prompts:
  versions:
    image: default
profiles:
  short:
    overrides:
      prompts:
        versions:
          image: user
""".lstrip(),
    )

    code, output, err = run(
        capsys,
        "list",
        "--config",
        str(config),
        "--profile",
        "short",
    )

    assert code == 0
    assert err == ""
    assert "image.md\tyes\tyes\tuser" in output.splitlines()


def test_profile_overrides_can_select_the_default_for_one_prompt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_prompt(tmp_path, "image.md", USER_IMAGE)
    config = write_config(
        tmp_path,
        """
schema_version: 1
profiles:
  short:
    overrides:
      prompts:
        image: default
""".lstrip(),
    )

    code, output, _ = run(
        capsys,
        "list",
        "--config",
        str(config),
        "--profile",
        "short",
    )

    assert code == 0
    assert "image.md\tyes\tyes\tdefault" in output.splitlines()


def test_set_flag_selects_a_version_after_the_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_prompt(tmp_path, "image.md", USER_IMAGE)
    config = write_config(
        tmp_path,
        """
schema_version: 1
prompts:
  versions:
    image: user
""".lstrip(),
    )

    code, output, _ = run(
        capsys,
        "list",
        "--config",
        str(config),
        "--set",
        "prompts.versions.image=default",
    )

    assert code == 0
    assert "image.md\tyes\tyes\tdefault" in output.splitlines()


def test_list_all_prints_one_row_per_version(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_prompt(tmp_path, "image.md", USER_IMAGE)
    config = write_config(tmp_path, "schema_version: 1\n")

    code, output, err = run(capsys, "list", "--all", "--config", str(config))

    assert code == 0
    assert err == ""
    lines = output.splitlines()
    assert lines[0] == "NAME\tSOURCE\tSELECTED\tPATH"
    assert any(line.startswith("image.md\tuser\tyes\t") for line in lines)
    assert any(line.startswith("image.md\tdefault\tno\t") for line in lines)


def test_list_groups_mixed_extensions_as_one_logical_prompt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_prompt(tmp_path, "image.j2", USER_IMAGE)
    config = write_config(tmp_path, "schema_version: 1\n")

    code, output, err = run(capsys, "list", "--config", str(config))

    assert code == 0
    assert err == ""
    rows = [line for line in output.splitlines() if line.startswith("image")]
    assert rows == ["image.j2\tyes\tyes\tuser"]


def test_list_paths_prints_the_selected_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    user = write_prompt(tmp_path, "image.md", USER_IMAGE)
    config = write_config(tmp_path, "schema_version: 1\n")

    code, output, _ = run(
        capsys,
        "list",
        "--paths",
        "--config",
        str(config),
        "image",
    )

    assert code == 0
    assert str(user) in output
    assert str(DEFAULT_IMAGE) in output


def test_list_rejects_an_unknown_configured_prompt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(
        tmp_path,
        """
schema_version: 1
prompts:
  versions:
    missing: default
""".lstrip(),
    )

    code, output, err = run(capsys, "list", "--config", str(config))

    assert code == 1
    assert output == ""
    assert "unknown prompt version: missing" in err


def test_list_rejects_a_version_that_does_not_exist(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(
        tmp_path,
        """
schema_version: 1
prompts:
  versions:
    image: user
""".lstrip(),
    )

    code, output, err = run(capsys, "list", "--config", str(config))

    assert code == 1
    assert output == ""
    assert "has no user version" in err


def test_show_prints_the_selected_version_and_can_print_content(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    user = write_prompt(tmp_path, "image.md", USER_IMAGE)
    config = write_config(
        tmp_path,
        """
schema_version: 1
prompts:
  versions:
    image: default
""".lstrip(),
    )

    code, output, err = run(capsys, "show", "image", "--config", str(config))

    assert code == 0
    assert err == ""
    assert str(DEFAULT_IMAGE) in output
    assert str(user) not in output
    assert "\tdefault\t" in output

    code, output, err = run(
        capsys,
        "show",
        "image",
        "--config",
        str(config),
        "--content",
    )

    assert code == 0
    assert err == ""
    assert output == DEFAULT_IMAGE.read_text(encoding="utf-8")


def test_the_library_selects_a_version_explicitly(tmp_path: Path) -> None:
    write_prompt(tmp_path, "image.md", USER_IMAGE)
    library = PromptLibrary(config_dir=tmp_path)

    assert library.select("image").is_user is True
    assert library.select("image", "user").is_user is True
    assert library.select("image", "default").is_default is True
    assert library.select("image", "packaged").is_default is True

    selection = library.selection("image", "default")
    assert selection.source == "default"
    assert selection.user is not None
    assert selection.default is not None
    assert selection.selected.path == DEFAULT_IMAGE


def test_the_library_rejects_a_missing_selected_version(tmp_path: Path) -> None:
    library = PromptLibrary(config_dir=tmp_path)

    with pytest.raises(PromptNotFoundError) as caught:
        library.select("image", "user")

    assert "has no user version" in caught.value.message
    assert caught.value.details["available"] == ["default"]
    assert isinstance(caught.value, ConfigError)
