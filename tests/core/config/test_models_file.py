"""Model catalog file configuration: registration and path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    CORE_SECTION_KEYS,
    AppConfig,
    ModelsFileConfig,
    ModelsFileSettings,
    SectionRegistry,
    register_core_sections,
)
from aivia.core.errors import ConfigError
from aivia.core.model_ids import (
    CATALOG_FILENAME,
    load_model_catalog,
    model_catalog_path,
    resolve_model_catalog_path,
    validate_model_ids,
)

CATALOG = "schema_version: 1\nmodels:\n  alpha-llm: {}\n"


def test_models_file_defaults_to_the_catalog_filename() -> None:
    config = ModelsFileConfig.from_mapping({})

    assert config.path == CATALOG_FILENAME


def test_models_file_accepts_a_path_or_the_scalar_shorthand() -> None:
    configured = ModelsFileConfig.from_mapping({"path": "custom/models.yaml"})
    shorthand = ModelsFileConfig.from_mapping("custom/models.yaml")

    assert configured.path == "custom/models.yaml"
    assert shorthand.path == "custom/models.yaml"


def test_models_file_accepts_a_filename_alias() -> None:
    config = ModelsFileConfig.from_mapping({"filename": "other.yaml"})

    assert config.path == "other.yaml"


def test_models_file_settings_alias_points_at_the_same_model() -> None:
    assert ModelsFileSettings is ModelsFileConfig


@pytest.mark.parametrize(
    "values",
    [
        {"path": ""},
        {"path": "   "},
        {"path": 42},
        {"unknown": True},
    ],
)
def test_invalid_models_file_settings_are_configuration_errors(
    values: dict[str, Any],
) -> None:
    with pytest.raises(ConfigError):
        ModelsFileConfig.from_mapping(values)


def test_models_file_section_is_registered_and_parsed_from_the_root() -> None:
    assert "models_file" in CORE_SECTION_KEYS
    assert "models_file" in register_core_sections(SectionRegistry())

    config = AppConfig.from_mapping({"models_file": {"path": "catalog.yaml"}})

    assert config.models_file is not None  # type: ignore[attr-defined]
    assert config.models_file.path == "catalog.yaml"  # type: ignore[attr-defined]


def test_models_file_section_accepts_the_scalar_form_at_the_root() -> None:
    config = AppConfig.from_mapping({"models_file": "catalog.yaml"})

    assert config.models_file is not None  # type: ignore[attr-defined]
    assert config.models_file.path == "catalog.yaml"  # type: ignore[attr-defined]


def test_example_config_declares_the_models_file_section() -> None:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    config = ModelsFileConfig.from_mapping(payload["models_file"])

    assert config.path == CATALOG_FILENAME


def test_default_resolution_prefers_the_user_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    user_catalog = tmp_path / "aivia" / CATALOG_FILENAME
    user_catalog.parent.mkdir(parents=True)
    user_catalog.write_text(CATALOG, encoding="utf-8")

    assert model_catalog_path() == user_catalog
    assert ModelsFileConfig.from_mapping({}).resolve() == user_catalog


def test_default_resolution_falls_back_to_the_packaged_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    expected = paths.package_defaults_dir() / CATALOG_FILENAME
    assert model_catalog_path() == expected
    assert resolve_model_catalog_path({"path": CATALOG_FILENAME}) == expected


def test_a_relative_path_resolves_against_the_config_directory(tmp_path: Path) -> None:
    config = ModelsFileConfig.from_mapping({"path": "catalog/custom.yaml"})

    resolved = config.resolve(config_dir=tmp_path)

    assert resolved == tmp_path / "catalog" / "custom.yaml"
    assert model_catalog_path(config, tmp_path) == resolved


def test_an_absolute_path_is_used_unchanged(tmp_path: Path) -> None:
    catalog = tmp_path / "elsewhere" / "models.yaml"
    config = ModelsFileConfig.from_mapping({"path": str(catalog)})

    assert config.resolve(config_dir=tmp_path / "ignored") == catalog
    assert resolve_model_catalog_path(config) == catalog


def test_load_model_catalog_uses_the_configured_path(tmp_path: Path) -> None:
    catalog = tmp_path / "custom.yaml"
    catalog.write_text(CATALOG, encoding="utf-8")
    config = ModelsFileConfig.from_mapping({"path": str(catalog)})

    loaded = load_model_catalog(models_file=config)

    assert loaded.source == catalog
    assert "alpha-llm" in loaded


def test_validation_uses_the_catalog_selected_by_the_root_config(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "custom.yaml"
    catalog.write_text(CATALOG, encoding="utf-8")
    config = {
        "models_file": {"path": str(catalog)},
        "llm": {"roles": {"writing": "alpha-llm"}},
    }

    assert validate_model_ids(config) == []


def test_a_string_path_can_be_resolved_directly(tmp_path: Path) -> None:
    catalog = tmp_path / "custom.yaml"

    assert model_catalog_path(str(catalog)) == catalog
