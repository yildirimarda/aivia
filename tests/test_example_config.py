"""Packaged default configuration: one canonical file covering every area."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.bootstrap import DEFAULT_CONFIG_SOURCE
from aivia.core.config import (
    CORE_SECTION_KEYS,
    AppConfig,
    SectionRegistry,
    build_config_schema,
    register_core_sections,
    resolve_config,
)


def defaults_dir() -> Path:
    return paths.package_defaults_dir()


def default_config_path() -> Path:
    return defaults_dir() / DEFAULT_CONFIG_SOURCE


def default_config() -> dict[str, Any]:
    payload = yaml.safe_load(default_config_path().read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def core_schema() -> dict[str, Any]:
    return build_config_schema(register_core_sections(SectionRegistry()))


def test_the_package_ships_exactly_one_default_config() -> None:
    configs = sorted(
        path.name
        for path in defaults_dir().iterdir()
        if path.is_file() and path.suffix in {".yaml", ".yml"}
    )

    assert configs == [DEFAULT_CONFIG_SOURCE]


def test_the_default_config_covers_every_configuration_area() -> None:
    payload = default_config()
    properties = cast(dict[str, Any], core_schema()["properties"])

    assert set(payload) == set(CORE_SECTION_KEYS) | {"schema_version"}
    assert set(payload) == set(properties)


def test_the_default_config_declares_the_schema_version() -> None:
    assert default_config()["schema_version"] == 1


def test_the_default_config_validates_as_one_root_configuration() -> None:
    config = AppConfig.from_mapping(default_config())

    assert config.schema_version == 1
    assert config.pipeline is not None  # type: ignore[attr-defined]
    assert config.pipeline.timezone == "Europe/Berlin"  # type: ignore[attr-defined]
    assert config.retention.delete_final_outputs is False  # type: ignore[attr-defined]


def test_resolving_the_packaged_defaults_validates(tmp_path: Path) -> None:
    resolved = resolve_config(config_file=tmp_path / "missing" / "config.yaml")

    assert resolved.sources == (default_config_path(),)
    assert resolved.app_config.schema_version == 1


@pytest.mark.parametrize("key", CORE_SECTION_KEYS)
def test_every_area_is_a_mapping_in_the_default_config(key: str) -> None:
    assert isinstance(default_config()[key], dict)
