"""``aivia config validate`` contract: every error, its key path and exit code."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.cli import main
from aivia.core.model_ids import CATALOG_FILENAME, ModelIdWarning

CATALOG = """
schema_version: 1
models:
  alpha-llm:
    capabilities: [llm]
""".lstrip()

DECOY_CATALOG = """
schema_version: 1
models:
  zzzz-llm:
    capabilities: [llm]
""".lstrip()


def write_config(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def validate(
    config: Path,
    capsys: pytest.CaptureFixture[str],
    *extra: str,
) -> tuple[int, str, str]:
    code = main(["config", "validate", "--config", str(config), *extra])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_validate_accepts_a_valid_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path / "config.yaml",
        """
schema_version: 1
pipeline:
  timezone: UTC
""".lstrip(),
    )

    code, output, err = validate(config, capsys)

    assert code == 0
    assert output.strip() == "configuration is valid"
    assert err == ""


def test_validate_accepts_the_packaged_defaults_without_a_user_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, output, err = validate(tmp_path / "missing" / "config.yaml", capsys)

    assert code == 0
    assert output.strip() == "configuration is valid"
    assert err == ""


def test_validate_prints_every_invalid_key_path_and_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path / "config.yaml",
        """
schema_version: 1
pipeline:
  bogus: 1
render:
  font_size: 0
video:
  fps: nope
""".lstrip(),
    )

    code, output, err = validate(config, capsys)

    assert code == 1
    assert output == ""
    for path in ("pipeline.bogus", "render.font_size", "video.fps"):
        assert path in err
    assert err.count("error:") == 3
    assert "Traceback" not in err


def test_validate_reports_every_unknown_section_with_its_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path / "config.yaml",
        """
schema_version: 1
alpha: {}
zulu: {}
""".lstrip(),
    )

    code, output, err = validate(config, capsys)

    assert code == 1
    assert output == ""
    assert "unknown config section: alpha" in err
    assert "unknown config section: zulu" in err


def test_validate_reports_unknown_sections_and_invalid_keys_together(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path / "config.yaml",
        """
schema_version: 1
unknown_section: {}
render:
  font_size: 0
""".lstrip(),
    )

    code, output, err = validate(config, capsys)

    assert code == 1
    assert output == ""
    assert "unknown config section: unknown_section" in err
    assert "render.font_size" in err


def test_validate_rejects_an_invalid_schema_version(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path / "config.yaml", 'schema_version: "one"\n')

    code, output, err = validate(config, capsys)

    assert code == 1
    assert output == ""
    assert "schema_version" in err
    assert "positive integer" in err


def test_validate_rejects_a_schema_version_from_a_newer_release(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path / "config.yaml", "schema_version: 99\n")

    code, output, err = validate(config, capsys)

    assert code == 1
    assert output == ""
    assert "schema_version" in err
    assert "newer than supported version" in err
    assert "upgrade aivia" in err


def test_validate_reports_invalid_yaml_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path / "config.yaml", "pipeline: [unclosed\n")

    code, output, err = validate(config, capsys)

    assert code == 1
    assert output == ""
    assert "invalid YAML" in err
    assert "Traceback" not in err


def test_validate_applies_the_profile_and_set_overrides(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path / "config.yaml",
        """
schema_version: 1
profiles:
  short:
    overrides:
      render:
        font_size: 0
""".lstrip(),
    )

    code, output, err = validate(
        config, capsys, "--profile", "short", "--set", "video.fps=60"
    )

    assert code == 1
    assert output == ""
    assert "render.font_size" in err


def test_validate_rejects_an_unknown_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path / "config.yaml",
        """
schema_version: 1
profiles:
  alpha:
    overrides: {}
  beta:
    overrides: {}
""".lstrip(),
    )

    code, output, err = validate(config, capsys, "--profile", "missing")

    assert code == 1
    assert output == ""
    assert "unknown profile: missing" in err
    assert "available: alpha, beta" in err


def test_validate_reports_cross_reference_errors_with_their_key_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path / "config.yaml",
        """
schema_version: 1
platforms:
  accounts:
    youtube/en: {}
profiles:
  default:
    accounts: [tiktok/en]
""".lstrip(),
    )

    code, output, err = validate(config, capsys)

    assert code == 1
    assert output == ""
    assert "profiles.default.accounts" in err
    assert "tiktok/en" in err


def test_validate_rejects_a_selected_prompt_that_does_not_exist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path / "config.yaml",
        """
schema_version: 1
prompts:
  versions:
    missing: default
""".lstrip(),
    )

    code, output, err = validate(config, capsys)

    assert code == 1
    assert output == ""
    assert "prompts.versions.missing" in err
    assert "unknown prompt: missing" in err


def test_validate_rejects_a_selected_prompt_source_that_does_not_exist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path / "config.yaml",
        """
schema_version: 1
prompts:
  versions:
    image: user
""".lstrip(),
    )

    code, output, err = validate(config, capsys)

    assert code == 1
    assert output == ""
    assert "prompts.versions.image" in err
    assert "has no user version" in err


def test_validate_resolves_the_default_models_file_beside_the_explicit_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(
        tmp_path / "project" / "config.yaml",
        "schema_version: 1\nllm:\n  roles:\n    writing: alpha-lm\n",
    )
    write_config(tmp_path / "project" / CATALOG_FILENAME, CATALOG)
    write_config(tmp_path / "aivia" / CATALOG_FILENAME, DECOY_CATALOG)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    with pytest.warns(ModelIdWarning, match="closest match is 'alpha-llm'"):
        code, output, err = validate(config, capsys)

    assert code == 0
    assert output.strip() == "configuration is valid"
    assert err == ""


def test_validate_accepts_the_research_section_from_an_installed_module(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path / "config.yaml",
        """
schema_version: 1
research:
  dedupe_window_days: 7
  embedding_threshold: 0.9
  max_topics: 3
  max_per_category: 1
  sensitivity:
    blocked_terms: [spoilers]
""".lstrip(),
    )

    code, output, err = validate(config, capsys)

    assert code == 0
    assert output.strip() == "configuration is valid"
    assert err == ""


def test_validate_accepts_research_settings_from_a_profile_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path / "config.yaml",
        """
schema_version: 1
research:
  dedupe_window_days: 365
profiles:
  news:
    overrides:
      research:
        dedupe_window_days: 7
""".lstrip(),
    )

    code, output, err = validate(config, capsys, "--profile", "news")

    assert code == 0
    assert output.strip() == "configuration is valid"
    assert err == ""


def test_validate_reports_an_invalid_key_inside_the_research_section(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path / "config.yaml",
        """
schema_version: 1
research:
  dedupe_window_days: 7
  bogus: true
""".lstrip(),
    )

    code, output, err = validate(config, capsys)

    assert code == 1
    assert output == ""
    assert "research.bogus" in err
    assert err.count("error:") == 1
