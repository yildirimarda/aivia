"""``aivia config migrate`` contract: version upgrades, diff and key safety."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from aivia.cli import main
from aivia.core.config.migrations import (
    CURRENT_SCHEMA_VERSION,
    ConfigMigration,
    migrate_config,
)


def write_config(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def migrate(
    config: Path,
    capsys: pytest.CaptureFixture[str],
    *extra: str,
) -> tuple[int, str, str]:
    code = main(["config", "migrate", "--config", str(config), *extra])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(path.read_text(encoding="utf-8")))


def test_migrate_adds_the_schema_version_and_preserves_unknown_keys(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path / "config.yaml",
        """
# my local notes
pipeline:
  timezone: UTC
future_module:
  enabled: true
""".lstrip(),
    )

    code, output, err = migrate(config, capsys)

    assert code == 0
    assert err == ""
    assert "schema_version 0 -> 1" in output
    assert "+schema_version: 1" in output
    assert load(config) == {
        "schema_version": 1,
        "pipeline": {"timezone": "UTC"},
        "future_module": {"enabled": True},
    }
    assert "# my local notes" in config.read_text(encoding="utf-8")


def test_migrate_reports_an_up_to_date_config_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = f"schema_version: {CURRENT_SCHEMA_VERSION}\npipeline:\n  timezone: UTC\n"
    config = write_config(tmp_path / "config.yaml", text)

    code, output, err = migrate(config, capsys)

    assert code == 0
    assert err == ""
    assert "already at schema_version 1" in output
    assert config.read_text(encoding="utf-8") == text


def test_migrate_dry_run_prints_the_diff_and_leaves_the_file_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = "pipeline:\n  timezone: UTC\n"
    config = write_config(tmp_path / "config.yaml", text)

    code, output, err = migrate(config, capsys, "--dry-run")

    assert code == 0
    assert err == ""
    assert "+schema_version: 1" in output
    assert "Dry run: no files written." in output
    assert "Would migrate" in output
    assert config.read_text(encoding="utf-8") == text


def test_migrate_upgrades_an_explicit_version_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path / "config.yaml",
        "schema_version: 0\nunknown_but_valid: keep-me\n",
    )

    code, output, err = migrate(config, capsys)

    assert code == 0
    assert err == ""
    assert "-schema_version: 0" in output
    assert "+schema_version: 1" in output
    assert load(config) == {
        "schema_version": 1,
        "unknown_but_valid": "keep-me",
    }


def test_migrate_refuses_a_config_from_a_newer_release(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = "schema_version: 99\npipeline:\n  timezone: UTC\n"
    config = write_config(tmp_path / "config.yaml", text)

    code, output, err = migrate(config, capsys)

    assert code == 1
    assert output == ""
    assert "newer than supported version 1" in err
    assert "Traceback" not in err
    assert config.read_text(encoding="utf-8") == text


def test_migrate_rejects_an_invalid_schema_version(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path / "config.yaml", 'schema_version: "one"\n')

    code, output, err = migrate(config, capsys)

    assert code == 1
    assert output == ""
    assert "invalid schema_version" in err
    assert "positive integer" in err


def test_migrate_fails_clearly_when_the_config_file_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing" / "config.yaml"

    code, output, err = migrate(missing, capsys)

    assert code == 1
    assert output == ""
    assert "config file not found" in err


def test_migrate_runs_each_version_step_in_order_and_keeps_other_keys() -> None:
    def add_version(values: dict[str, Any]) -> None:
        values["schema_version"] = 1

    def rename_old_key(values: dict[str, Any]) -> None:
        values["new_key"] = values.pop("old_key")
        values["schema_version"] = 2

    migrations = (
        ConfigMigration(0, 1, "add version", add_version),
        ConfigMigration(1, 2, "rename old_key", rename_old_key),
    )

    result = migrate_config(
        {"old_key": "value", "unknown": {"keep": True}},
        migrations=migrations,
        target_version=2,
    )

    assert result.from_version == 0
    assert result.to_version == 2
    assert [step.description for step in result.steps] == [
        "add version",
        "rename old_key",
    ]
    assert result.values == {
        "new_key": "value",
        "unknown": {"keep": True},
        "schema_version": 2,
    }
