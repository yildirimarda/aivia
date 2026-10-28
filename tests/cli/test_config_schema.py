"""``aivia config schema`` contract: deterministic export and staleness check."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from aivia.cli import main
from aivia.core.config import CORE_SECTION_KEYS, ConfigModel, SectionRegistry
from aivia.core.config.schema import (
    SCHEMA_DIALECT,
    build_config_schema,
    dump_config_schema,
    packaged_schema_path,
)


def run_schema(
    capsys: pytest.CaptureFixture[str],
    *extra: str,
) -> tuple[int, str, str]:
    code = main(["config", "schema", *extra])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_schema_prints_every_registered_section_as_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, output, err = run_schema(capsys)

    assert code == 0
    assert err == ""
    schema = cast(dict[str, Any], json.loads(output))
    properties = cast(dict[str, Any], schema["properties"])

    assert schema["$schema"] == SCHEMA_DIALECT
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(properties) == set(CORE_SECTION_KEYS) | {"schema_version"}
    assert properties["schema_version"]["type"] == "integer"
    for key in CORE_SECTION_KEYS:
        assert key in properties


def test_schema_output_is_byte_for_byte_deterministic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    first_code, first, first_err = run_schema(capsys)
    second_code, second, second_err = run_schema(capsys)

    assert (first_code, first_err) == (0, "")
    assert (second_code, second_err) == (0, "")
    assert first == second
    assert first == dump_config_schema()


def test_schema_is_independent_of_registration_order() -> None:
    class AlphaConfig(ConfigModel):
        enabled: bool = True

    class ZuluConfig(ConfigModel):
        enabled: bool = True

    first = SectionRegistry()
    first.register("alpha", AlphaConfig)
    first.register("zulu", ZuluConfig)
    second = SectionRegistry()
    second.register("zulu", ZuluConfig)
    second.register("alpha", AlphaConfig)

    assert dump_config_schema(build_config_schema(first)) == dump_config_schema(
        build_config_schema(second)
    )


def test_schema_includes_a_module_registered_section() -> None:
    class CustomConfig(ConfigModel):
        enabled: bool = True

    sections = SectionRegistry()
    sections.register("custom", CustomConfig)

    schema = build_config_schema(sections)

    assert "custom" in schema["properties"]
    assert "CustomConfig" in schema["$defs"]


def test_the_committed_schema_is_current() -> None:
    path = packaged_schema_path()

    assert path.is_file(), f"missing {path}"
    assert path.read_text(encoding="utf-8") == dump_config_schema(), (
        "config schema is stale; regenerate it with: "
        "aivia config schema --output src/aivia/data/config.schema.json"
    )


def test_schema_check_accepts_the_committed_copy(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, output, err = run_schema(capsys, "--check")

    assert code == 0
    assert err == ""
    assert "up to date" in output


def test_schema_check_fails_for_a_stale_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stale = tmp_path / "config.schema.json"
    stale.write_text("{}\n", encoding="utf-8")

    code, output, err = run_schema(capsys, "--check", "--output", str(stale))

    assert code == 1
    assert output == ""
    assert "stale" in err
    assert str(stale) in err


def test_schema_check_reports_a_missing_file_as_stale(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing" / "config.schema.json"

    code, output, err = run_schema(capsys, "--check", "--output", str(missing))

    assert code == 1
    assert output == ""
    assert "stale" in err
    assert str(missing) in err


def test_schema_output_writes_the_generated_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "nested" / "config.schema.json"

    code, output, err = run_schema(capsys, "--output", str(target))

    assert code == 0
    assert err == ""
    assert str(target) in output
    assert target.read_text(encoding="utf-8") == dump_config_schema()
