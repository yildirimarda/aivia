"""``aivia config validate`` contract: every error, its key path and exit code."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.cli import main


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
