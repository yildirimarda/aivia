"""Layered resolution: precedence, origins, environment files and --set flags."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from aivia.core.config import (
    AppConfig,
    Layer,
    merge_layers,
    parse_set_overrides,
    parse_set_value,
    resolve_app_config,
    resolve_config,
)
from aivia.core.config.resolve import (
    CLI_ORIGIN,
    DEFAULT_ORIGIN,
    ENV_ORIGIN_PREFIX,
    PROFILE_ORIGIN_PREFIX,
    USER_ORIGIN,
)
from aivia.core.errors import ConfigError


def write_yaml(path: Path, payload: dict[str, Any] | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else yaml.safe_dump(payload)
    path.write_text(text, encoding="utf-8")
    return path


def layered_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    defaults = write_yaml(
        tmp_path / "defaults" / "config.example.yaml",
        {
            "schema_version": 1,
            "pipeline": {"timezone": "UTC", "stages": ["research", "render"]},
            "video": {"fps": 30},
            "render": {"require_bundled_fonts": True},
        },
    )
    config = write_yaml(
        tmp_path / "user" / "config.yaml",
        {
            "pipeline": {"timezone": "Europe/Berlin", "stages": ["write", "render"]},
            "video": {"fps": 24},
            "profiles": {
                "short": {
                    "overrides": {
                        "pipeline": {"timezone": "Europe/Istanbul"},
                        "render": {"require_bundled_fonts": False},
                    }
                }
            },
        },
    )
    environment = write_yaml(
        tmp_path / "user" / "config.dev.yaml",
        {
            "pipeline": {"timezone": "Europe/Paris"},
            "video": {"fps": 25},
        },
    )
    return defaults, config, environment


def test_the_full_stack_applies_every_layer_in_precedence_order(
    tmp_path: Path,
) -> None:
    defaults, config, _ = layered_files(tmp_path)

    resolved = resolve_config(
        defaults_file=defaults,
        config_file=config,
        environment="dev",
        profile="short",
        set_values=["pipeline.timezone=Europe/London", "video.fps=60"],
    )

    assert resolved.values["pipeline"]["timezone"] == "Europe/London"
    assert resolved.values["pipeline"]["stages"] == ["write", "render"]
    assert resolved.values["video"]["fps"] == 60
    assert resolved.values["render"]["require_bundled_fonts"] is False
    assert resolved.environment == "dev"
    assert resolved.profile == "short"

    assert resolved.origins["schema_version"] == DEFAULT_ORIGIN
    assert resolved.origins["pipeline.stages"] == USER_ORIGIN
    assert resolved.origins["video.fps"] == CLI_ORIGIN
    assert resolved.origins["render.require_bundled_fonts"] == (
        f"{PROFILE_ORIGIN_PREFIX}short"
    )
    assert resolved.origins["pipeline.timezone"] == CLI_ORIGIN


def test_the_environment_file_sits_between_user_config_and_profile(
    tmp_path: Path,
) -> None:
    defaults, config, _ = layered_files(tmp_path)

    resolved = resolve_config(
        defaults_file=defaults,
        config_file=config,
        environment="dev",
    )

    assert resolved.values["pipeline"]["timezone"] == "Europe/Paris"
    assert resolved.origins["pipeline.timezone"] == f"{ENV_ORIGIN_PREFIX}dev"
    assert resolved.origins["video.fps"] == f"{ENV_ORIGIN_PREFIX}dev"
    assert resolved.origins["render.require_bundled_fonts"] == DEFAULT_ORIGIN


def test_a_profile_can_override_values_introduced_by_the_environment_file(
    tmp_path: Path,
) -> None:
    defaults, config, _ = layered_files(tmp_path)

    resolved = resolve_config(
        defaults_file=defaults,
        config_file=config,
        environment="dev",
        profile="short",
    )

    assert resolved.values["pipeline"]["timezone"] == "Europe/Istanbul"
    assert resolved.origins["pipeline.timezone"] == (f"{PROFILE_ORIGIN_PREFIX}short")


def test_the_environment_can_override_the_profile_definition(
    tmp_path: Path,
) -> None:
    defaults = write_yaml(tmp_path / "defaults.yaml", {"pipeline": {"timezone": "UTC"}})
    config = write_yaml(
        tmp_path / "config.yaml",
        {
            "profiles": {
                "short": {"overrides": {"pipeline": {"timezone": "UserProfile"}}}
            }
        },
    )
    write_yaml(
        tmp_path / "config.dev.yaml",
        {
            "profiles": {
                "short": {"overrides": {"pipeline": {"timezone": "EnvProfile"}}}
            }
        },
    )

    resolved = resolve_config(
        defaults_file=defaults,
        config_file=config,
        environment="dev",
        profile="short",
    )

    assert resolved.values["pipeline"]["timezone"] == "EnvProfile"
    assert resolved.origins["pipeline.timezone"] == (f"{PROFILE_ORIGIN_PREFIX}short")


def test_the_environment_can_be_selected_from_aiva_env(tmp_path: Path) -> None:
    defaults, config, _ = layered_files(tmp_path)

    resolved = resolve_config(
        defaults_file=defaults,
        config_file=config,
        environ={"AIVA_ENV": "dev"},
    )

    assert resolved.environment == "dev"
    assert resolved.values["pipeline"]["timezone"] == "Europe/Paris"


def test_an_explicit_environment_wins_over_the_environment_variable(
    tmp_path: Path,
) -> None:
    defaults = write_yaml(tmp_path / "defaults.yaml", {})
    config = write_yaml(tmp_path / "config.yaml", {})
    write_yaml(tmp_path / "config.dev.yaml", {"pipeline": {"timezone": "Dev"}})
    write_yaml(tmp_path / "config.prod.yaml", {"pipeline": {"timezone": "Prod"}})

    resolved = resolve_config(
        defaults_file=defaults,
        config_file=config,
        environment="prod",
        environ={"AIVA_ENV": "dev"},
    )

    assert resolved.values["pipeline"]["timezone"] == "Prod"


def test_no_environment_file_is_read_without_a_selected_environment(
    tmp_path: Path,
) -> None:
    defaults = write_yaml(tmp_path / "defaults.yaml", {"pipeline": {"timezone": "UTC"}})
    config = write_yaml(tmp_path / "config.yaml", {})
    write_yaml(tmp_path / "config.dev.yaml", {"pipeline": {"timezone": "Dev"}})

    resolved = resolve_config(
        defaults_file=defaults,
        config_file=config,
        environ={},
    )

    assert resolved.environment is None
    assert resolved.values["pipeline"]["timezone"] == "UTC"


def test_a_selected_environment_requires_its_file(tmp_path: Path) -> None:
    defaults = write_yaml(tmp_path / "defaults.yaml", {})
    config = write_yaml(tmp_path / "config.yaml", {})

    with pytest.raises(ConfigError) as caught:
        resolve_config(
            defaults_file=defaults,
            config_file=config,
            environment="missing",
        )

    assert "environment config file not found" in str(caught.value)
    assert caught.value.details["environment"] == "missing"


@pytest.mark.parametrize("name", ["../prod", "dev/prod", "dev\\prod", " "])
def test_environment_names_cannot_escape_the_config_directory(
    tmp_path: Path, name: str
) -> None:
    defaults = write_yaml(tmp_path / "defaults.yaml", {})
    config = write_yaml(tmp_path / "config.yaml", {})

    with pytest.raises(ConfigError, match="environment name"):
        resolve_config(
            defaults_file=defaults,
            config_file=config,
            environment=name,
        )


def test_a_custom_config_name_uses_its_own_environment_file(
    tmp_path: Path,
) -> None:
    defaults = write_yaml(tmp_path / "defaults.yaml", {})
    config = write_yaml(tmp_path / "custom.yaml", {})
    write_yaml(tmp_path / "custom.dev.yaml", {"pipeline": {"timezone": "Dev"}})

    resolved = resolve_config(
        defaults_file=defaults,
        config_file=config,
        environment="dev",
    )

    assert resolved.values["pipeline"]["timezone"] == "Dev"


def test_layers_deep_merge_mappings_and_replace_lists() -> None:
    values, origins = merge_layers(
        [
            Layer("first", {"pipeline": {"a": 1, "b": 2}, "stages": [1, 2]}),
            Layer("second", {"pipeline": {"b": 3, "c": 4}, "stages": [3]}),
        ]
    )

    assert values == {
        "pipeline": {"a": 1, "b": 3, "c": 4},
        "stages": [3],
    }
    assert origins["pipeline.a"] == "first"
    assert origins["pipeline.b"] == "second"
    assert origins["pipeline.c"] == "second"
    assert origins["stages"] == "second"


def test_merging_does_not_mutate_the_input_layers() -> None:
    base = {"pipeline": {"timezone": "UTC"}, "stages": ["a"]}
    layer = Layer("first", base)

    merge_layers([layer])

    assert base == {"pipeline": {"timezone": "UTC"}, "stages": ["a"]}


def test_set_values_are_parsed_as_yaml_scalars_and_structures() -> None:
    overrides = parse_set_overrides(
        [
            "video.fps=24",
            "render.require_bundled_fonts=false",
            "pipeline.stages=[write, render]",
            "pipeline.timezone=Europe/Paris",
            "note=a=b",
        ]
    )

    assert overrides == {
        "video": {"fps": 24},
        "render": {"require_bundled_fonts": False},
        "pipeline": {"stages": ["write", "render"], "timezone": "Europe/Paris"},
        "note": "a=b",
    }


def test_repeated_set_keys_keep_the_last_value() -> None:
    overrides = parse_set_overrides(["video.fps=24", "video.fps=30"])

    assert overrides == {"video": {"fps": 30}}


def test_set_values_accept_a_dotted_mapping() -> None:
    overrides = parse_set_overrides({"pipeline.timezone": "UTC", "video.fps": 30})

    assert overrides == {"pipeline": {"timezone": "UTC"}, "video": {"fps": 30}}


def test_parse_set_value_returns_the_path_and_value() -> None:
    path, value = parse_set_value("pipeline.stages=[write]")

    assert path == ("pipeline", "stages")
    assert value == ["write"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("missing-equals", "expected key.path=value"),
        ("=value", "key must not be empty"),
        ("a..b=1", "invalid --set key"),
        ("a.=1", "invalid --set key"),
        ("a=[unclosed", "invalid --set value"),
    ],
)
def test_malformed_set_values_fail_clearly(text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_set_overrides([text])


def test_a_set_path_cannot_pass_through_a_scalar() -> None:
    with pytest.raises(ConfigError, match="a is not a mapping"):
        parse_set_overrides(["a=1", "a.b=2"])


def test_an_unknown_profile_lists_the_configured_ones(tmp_path: Path) -> None:
    defaults = write_yaml(tmp_path / "defaults.yaml", {})
    config = write_yaml(
        tmp_path / "config.yaml",
        {"profiles": {"alpha": {"overrides": {}}, "beta": {"overrides": {}}}},
    )

    with pytest.raises(ConfigError, match="available: alpha, beta"):
        resolve_config(
            defaults_file=defaults,
            config_file=config,
            profile="missing",
        )


def test_missing_defaults_fail_and_missing_user_config_is_allowed(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        resolve_config(defaults_file=tmp_path / "missing.yaml")

    defaults = write_yaml(tmp_path / "defaults.yaml", {"pipeline": {"timezone": "UTC"}})
    resolved = resolve_config(
        defaults_file=defaults,
        config_file=tmp_path / "missing-user.yaml",
    )

    assert resolved.values["pipeline"]["timezone"] == "UTC"


def test_invalid_yaml_in_any_layer_is_reported_with_its_origin(
    tmp_path: Path,
) -> None:
    defaults = write_yaml(tmp_path / "defaults.yaml", {})
    config = write_yaml(tmp_path / "config.yaml", "pipeline: [unclosed\n")

    with pytest.raises(ConfigError, match="invalid YAML in user config"):
        resolve_config(defaults_file=defaults, config_file=config)


def test_a_non_mapping_root_is_rejected(tmp_path: Path) -> None:
    defaults = write_yaml(tmp_path / "defaults.yaml", "- not\n- a\n- mapping\n")

    with pytest.raises(ConfigError, match="root must be a mapping"):
        resolve_config(defaults_file=defaults)


def test_sources_record_the_loaded_layers_in_order(tmp_path: Path) -> None:
    defaults, config, environment = layered_files(tmp_path)

    resolved = resolve_config(
        defaults_file=defaults,
        config_file=config,
        environment="dev",
    )

    assert resolved.sources == (defaults, config, environment)


def test_the_resolved_values_can_be_validated_as_the_root_config(
    tmp_path: Path,
) -> None:
    defaults = write_yaml(tmp_path / "defaults.yaml", {})
    config = write_yaml(
        tmp_path / "config.yaml",
        {"pipeline": {"timezone": "Europe/Berlin"}},
    )

    resolved = resolve_config(defaults_file=defaults, config_file=config)
    validated = resolved.to_app_config()

    assert isinstance(validated, AppConfig)
    assert validated.pipeline.timezone == "Europe/Berlin"  # type: ignore[attr-defined]
    assert isinstance(resolved.app_config, AppConfig)


def test_resolve_app_config_validates_a_mapping() -> None:
    validated = resolve_app_config({"pipeline": {"timezone": "UTC"}})

    assert isinstance(validated, AppConfig)
    assert validated.pipeline.timezone == "UTC"  # type: ignore[attr-defined]
