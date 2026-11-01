"""Named presets: selection, precedence, CLI overrides and inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from aivia.cli import main
from aivia.core import paths
from aivia.core.config import PRESET_ORIGIN_PREFIX, PresetsConfig, resolve_config
from aivia.core.config.resolve import CLI_ORIGIN, PROFILE_ORIGIN_PREFIX
from aivia.core.errors import ConfigError
from aivia.core.presets import (
    apply_preset,
    default_preset,
    get_preset,
    iter_presets,
    list_presets,
    preset_names,
    preset_overrides,
    presets_config,
    select_preset,
)


@pytest.fixture(autouse=True)
def isolate_user_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep CLI tests away from the developer's real config directory."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)


def write_yaml(path: Path, payload: dict[str, Any] | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else yaml.safe_dump(payload)
    path.write_text(text, encoding="utf-8")
    return path


def preset_config(tmp_path: Path) -> tuple[Path, Path]:
    defaults = write_yaml(
        tmp_path / "defaults.yaml",
        {
            "pipeline": {"timezone": "UTC", "stages": ["write", "render"]},
            "video": {"fps": 30},
            "render": {"require_bundled_fonts": True},
        },
    )
    config = write_yaml(
        tmp_path / "config.yaml",
        {
            "profiles": {
                "short": {
                    "overrides": {
                        "pipeline": {"timezone": "Europe/Paris"},
                        "video": {"fps": 24},
                    }
                },
                "long": {"overrides": {"video": {"fps": 24}}},
            },
            "presets": {
                "default": "short",
                "short": {
                    "description": "Short vertical video",
                    "profile": "short",
                    "overrides": {
                        "pipeline": {"timezone": "Europe/Istanbul"},
                        "render": {"require_bundled_fonts": False},
                    },
                },
                "long": {
                    "description": "Long landscape video",
                    "profile": "long",
                    "overrides": {"video": {"fps": 25}},
                },
            },
        },
    )
    return defaults, config


def test_presets_config_accepts_a_section_root_or_mapping() -> None:
    section = PresetsConfig.from_mapping(
        {
            "default": "short",
            "short": {"description": "Short", "overrides": {"video": {"fps": 24}}},
        }
    )

    assert presets_config(section) is section
    assert presets_config({"presets": section}) is section
    assert preset_names({"presets": {"long": {}, "short": {}}}) == ("long", "short")
    assert list_presets(section) == ("short",)
    assert default_preset(section) == "short"
    assert [preset.name for preset in iter_presets(section)] == ["short"]


def test_get_and_select_preset_fail_with_the_available_names() -> None:
    section = PresetsConfig.from_mapping(
        {"short": {}, "long": {"enabled": False}, "default": "short"}
    )

    assert get_preset(section, "short").name == "short"
    default = select_preset(section)
    assert default is not None and default.name == "short"
    selected = select_preset(section, "long")
    assert selected is not None and selected.name == "long"
    assert select_preset(section, "", use_default=False) is None
    assert preset_overrides(section, "short") == {}

    with pytest.raises(ConfigError, match="available: long, short"):
        get_preset(section, "missing")

    disabled = PresetsConfig.from_mapping({"enabled": False, "short": {}})
    assert default_preset(disabled) is None
    assert select_preset(disabled) is None
    assert preset_names(disabled) == ()
    assert preset_names(disabled, include_disabled=True) == ("short",)
    with pytest.raises(ConfigError, match="presets are disabled"):
        select_preset(disabled, "short")


def test_apply_preset_merges_cli_overrides_last_and_does_not_mutate() -> None:
    values: dict[str, Any] = {
        "video": {"fps": 30, "width": 1920},
        "presets": {
            "short": {
                "overrides": {
                    "video": {"fps": 24},
                    "render": {"require_bundled_fonts": False},
                }
            }
        },
    }

    applied = apply_preset(
        values,
        "short",
        overrides=["video.fps=60", "render.require_bundled_fonts=true"],
    )

    assert applied["video"] == {"fps": 60, "width": 1920}
    assert applied["render"]["require_bundled_fonts"] is True
    assert values["video"]["fps"] == 30
    assert values["presets"]["short"]["overrides"]["video"]["fps"] == 24


def test_the_default_preset_applies_with_its_profile_and_preset_values(
    tmp_path: Path,
) -> None:
    defaults, config = preset_config(tmp_path)

    resolved = resolve_config(
        defaults_file=defaults,
        config_file=config,
    )

    assert resolved.preset == "short"
    assert resolved.profile == "short"
    assert resolved.values["pipeline"]["timezone"] == "Europe/Istanbul"
    assert resolved.values["video"]["fps"] == 24
    assert resolved.values["render"]["require_bundled_fonts"] is False
    assert resolved.origins["pipeline.timezone"] == f"{PRESET_ORIGIN_PREFIX}short"
    assert resolved.origins["video.fps"] == f"{PROFILE_ORIGIN_PREFIX}short"
    assert resolved.origins["render.require_bundled_fonts"] == (
        f"{PRESET_ORIGIN_PREFIX}short"
    )


def test_an_explicit_preset_wins_over_the_default_and_cli_set_wins_over_both(
    tmp_path: Path,
) -> None:
    defaults, config = preset_config(tmp_path)

    selected = resolve_config(
        defaults_file=defaults,
        config_file=config,
        preset="long",
    )
    overridden = resolve_config(
        defaults_file=defaults,
        config_file=config,
        preset="long",
        set_values=["video.fps=60", "pipeline.timezone=Europe/London"],
    )

    assert selected.preset == "long"
    assert selected.profile == "long"
    assert selected.values["video"]["fps"] == 25
    assert overridden.values["video"]["fps"] == 60
    assert overridden.values["pipeline"]["timezone"] == "Europe/London"
    assert overridden.origins["video.fps"] == CLI_ORIGIN
    assert overridden.origins["pipeline.timezone"] == CLI_ORIGIN


def test_an_explicit_profile_wins_over_the_profile_named_by_a_preset(
    tmp_path: Path,
) -> None:
    defaults, config = preset_config(tmp_path)

    resolved = resolve_config(
        defaults_file=defaults,
        config_file=config,
        preset="short",
        profile="long",
    )

    assert resolved.preset == "short"
    assert resolved.profile == "long"
    assert resolved.values["video"]["fps"] == 24
    assert resolved.values["pipeline"]["timezone"] == "Europe/Istanbul"


def test_a_blank_preset_disables_the_configured_default(tmp_path: Path) -> None:
    defaults, config = preset_config(tmp_path)

    resolved = resolve_config(
        defaults_file=defaults,
        config_file=config,
        preset="",
    )

    assert resolved.preset is None
    assert resolved.profile is None
    assert resolved.values["pipeline"]["timezone"] == "UTC"
    assert resolved.values["video"]["fps"] == 30


def test_unknown_presets_and_disabled_sections_fail_clearly(tmp_path: Path) -> None:
    defaults, config = preset_config(tmp_path)

    with pytest.raises(ConfigError, match="available: long, short"):
        resolve_config(
            defaults_file=defaults,
            config_file=config,
            preset="missing",
        )

    disabled = write_yaml(
        tmp_path / "disabled.yaml",
        {"presets": {"enabled": False, "short": {"overrides": {"video": {"fps": 24}}}}},
    )
    assert (
        resolve_config(
            defaults_file=defaults,
            config_file=disabled,
        ).preset
        is None
    )
    with pytest.raises(ConfigError, match="presets are disabled"):
        resolve_config(
            defaults_file=defaults,
            config_file=disabled,
            preset="short",
        )


def test_presets_list_marks_the_default_and_show_renders_yaml(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, config = preset_config(tmp_path)

    assert main(["presets", "list", "--config", str(config)]) == 0
    listed = capsys.readouterr()
    assert listed.err == ""
    rows = [line.split("\t") for line in listed.out.splitlines()]
    assert rows[0] == ["NAME", "DESCRIPTION", "DEFAULT"]
    assert ["short", "Short vertical video", "yes"] in rows
    assert ["long", "Long landscape video", "no"] in rows

    assert main(["presets", "show", "short", "--config", str(config)]) == 0
    shown = capsys.readouterr()
    assert shown.err == ""
    payload = cast(dict[str, Any], yaml.safe_load(shown.out))
    assert payload["name"] == "short"
    assert payload["profile"] == "short"
    assert payload["overrides"]["pipeline"]["timezone"] == "Europe/Istanbul"

    assert (
        main(["presets", "show", "short", "--overrides", "--config", str(config)]) == 0
    )
    overrides = cast(dict[str, Any], yaml.safe_load(capsys.readouterr().out))
    assert overrides == {
        "pipeline": {"timezone": "Europe/Istanbul"},
        "render": {"require_bundled_fonts": False},
    }


def test_presets_show_masks_secrets_and_reports_an_unknown_name(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_yaml(
        tmp_path / "config.yaml",
        {
            "presets": {
                "leaky": {"overrides": {"providers": {"token": "super-secret"}}},
            }
        },
    )

    assert main(["presets", "show", "leaky", "--config", str(config)]) == 0
    shown = capsys.readouterr()
    assert "super-secret" not in shown.out
    assert "REDACTED" in shown.out

    assert main(["presets", "show", "missing", "--config", str(config)]) == 1
    failed = capsys.readouterr()
    assert failed.out == ""
    assert "unknown preset: missing" in failed.err


def test_the_global_preset_flag_overrides_values_and_cli_set_wins(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, config = preset_config(tmp_path)

    assert main(["config", "show", "--config", str(config)]) == 0
    default_output = capsys.readouterr().out
    default = cast(dict[str, Any], yaml.safe_load(default_output))
    assert default["pipeline"]["timezone"] == "Europe/Istanbul"
    assert "# preset: short" in default_output

    assert (
        main(
            [
                "config",
                "show",
                "--config",
                str(config),
                "--preset",
                "long",
                "--set",
                "video.fps=60",
                "--origins",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    selected = cast(dict[str, Any], yaml.safe_load(output))
    packaged = cast(
        dict[str, Any],
        yaml.safe_load(
            (paths.package_defaults_dir() / "config.example.yaml").read_text(
                encoding="utf-8"
            )
        ),
    )
    assert selected["pipeline"]["timezone"] == packaged["pipeline"]["timezone"]
    assert selected["video"]["fps"] == 60
    assert "# preset: long" in output
    assert "# video.fps: cli" in output
