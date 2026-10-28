"""Readable configuration errors: file, path, expectation, hint, no traceback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aivia.core.config import ConfigModel, parse_app_config, resolve_config
from aivia.core.errors import ConfigError


def write_yaml(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def resolved_error(
    tmp_path: Path,
    *,
    defaults: str = "schema_version: 1\n",
    user: str,
    environ: dict[str, str] | None = None,
) -> ConfigError:
    defaults_path = write_yaml(tmp_path / "defaults.yaml", defaults)
    config_path = write_yaml(tmp_path / "config.yaml", user)
    resolved = resolve_config(
        defaults_file=defaults_path,
        config_file=config_path,
    )
    with pytest.raises(ConfigError) as caught:
        parse_app_config(
            resolved.values,
            environ={} if environ is None else environ,
            context=resolved.error_context(),
        )
    return caught.value


def test_validation_error_names_file_path_expected_actual_and_hint(
    tmp_path: Path,
) -> None:
    error = resolved_error(
        tmp_path,
        user="""
schema_version: 1
pipeline:
  bogus: 1
render:
  font_size: 0
""".lstrip(),
    )

    message = str(error)
    assert str(tmp_path / "config.yaml") in message
    assert "pipeline.bogus" in message
    assert "render.font_size" in message
    assert "expected" in message
    assert "actual" in message
    assert "hint:" in message
    assert "Traceback" not in message

    details: list[dict[str, Any]] = error.details["errors"]
    assert {detail["path"] for detail in details} == {
        "pipeline.bogus",
        "render.font_size",
    }
    assert all(detail["file"] == str(tmp_path / "config.yaml") for detail in details)
    assert all(detail["expected"] for detail in details)
    assert all(detail["actual"] for detail in details)
    assert all(detail["hint"] for detail in details)


def test_invalid_value_contrasts_expected_type_with_the_actual_value(
    tmp_path: Path,
) -> None:
    error = resolved_error(
        tmp_path,
        user="""
schema_version: 1
render:
  font_size: huge
""".lstrip(),
    )

    message = str(error)
    assert "render.font_size" in message
    assert "expected an integer" in message
    assert "actual 'huge'" in message
    assert "hint: use a whole number" in message


def test_a_missing_required_key_is_reported_with_a_value_and_a_hint() -> None:
    class RequiredConfig(ConfigModel):
        name: str

    with pytest.raises(ConfigError) as caught:
        RequiredConfig.from_mapping({}, context={"file": "config.yaml"})

    message = str(caught.value)
    assert message.startswith("config.yaml:name:")
    assert "missing required config key: name" in message
    assert "expected a value" in message
    assert "actual nothing" in message
    assert "hint: add the required key" in message


def test_an_unknown_section_names_the_file_and_how_to_fix_it(
    tmp_path: Path,
) -> None:
    error = resolved_error(
        tmp_path,
        user="""
schema_version: 1
research:
  enabled: true
""".lstrip(),
    )

    message = str(error)
    assert str(tmp_path / "config.yaml") in message
    assert "unknown config section: research" in message
    assert "hint: register the module" in message


def test_a_missing_environment_variable_names_the_file_and_a_hint(
    tmp_path: Path,
) -> None:
    error = resolved_error(
        tmp_path,
        user="""
schema_version: 1
pipeline:
  timezone: ${AIVA_TIMEZONE}
""".lstrip(),
        environ={},
    )

    message = str(error)
    assert str(tmp_path / "config.yaml") in message
    assert "pipeline.timezone" in message
    assert "missing environment variable AIVA_TIMEZONE" in message
    assert "hint: set the variable" in message
    assert "Traceback" not in message


def test_the_file_is_the_layer_that_supplied_the_invalid_value(
    tmp_path: Path,
) -> None:
    error = resolved_error(
        tmp_path,
        defaults="schema_version: 1\nvideo:\n  fps: nope\n",
        user="schema_version: 1\npipeline:\n  timezone: UTC\n",
    )

    message = str(error)
    assert str(tmp_path / "defaults.yaml") in message
    assert str(tmp_path / "config.yaml") not in message
    assert "video.fps" in message
