"""Select named presets and merge their overrides into configuration values.

A preset is a named bundle of configuration overrides. It is deliberately
separate from a content profile: a preset may *name* the profile it belongs to,
but its ``overrides`` mapping is the part that changes the merged
configuration. This module owns the lookup rules, so the resolver, the CLI and
library callers all agree on which preset is selected and how a command line
``--set`` value wins over it.

The functions accept either a validated :class:`~aivia.core.config.PresetsConfig`,
a root application config, or a plain mapping whose ``presets`` key holds the
section. That keeps them usable before the root config has been validated.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from aivia.core.config.sections import PresetConfig, PresetsConfig
from aivia.core.errors import ConfigError

__all__ = [
    "PRESET_ORIGIN_PREFIX",
    "apply_preset",
    "apply_preset_overrides",
    "default_preset",
    "describe_preset",
    "get_preset",
    "iter_presets",
    "list_presets",
    "preset_names",
    "preset_overrides",
    "presets_config",
    "resolve_preset",
    "select_preset",
    "show_preset",
]

PRESET_ORIGIN_PREFIX = "preset:"
"""Origin label used for values supplied by a preset."""

_GLOBAL_KEYS = frozenset({"enabled", "default", "presets", "items", "definitions"})


def _is_section_mapping(value: Mapping[str, Any]) -> bool:
    if not value:
        return True
    return set(value).issubset(_GLOBAL_KEYS)


def presets_config(config: Any) -> PresetsConfig:
    """Return the preset section from a root config, section or mapping."""
    if config is None:
        return PresetsConfig()
    if isinstance(config, PresetsConfig):
        return config
    if isinstance(config, PresetConfig):
        return PresetsConfig(presets={config.name or "preset": config})

    section = getattr(config, "presets", None)
    if isinstance(section, PresetsConfig):
        return section
    if isinstance(section, Mapping):
        return PresetsConfig.from_mapping(section)

    if isinstance(config, Mapping):
        raw = config.get("presets")
        if isinstance(raw, PresetsConfig):
            return raw
        if isinstance(raw, Mapping):
            return PresetsConfig.from_mapping(raw)
        if _is_section_mapping(config):
            return PresetsConfig.from_mapping(config)
        return PresetsConfig()
    return PresetsConfig()


def default_preset(config: Any) -> str | None:
    """Return the configured default preset name, or None when there is none."""
    section = presets_config(config)
    if not section.enabled:
        return None
    return section.default.strip() or None


def preset_names(
    config: Any,
    *,
    include_disabled: bool = False,
) -> tuple[str, ...]:
    """Return configured preset names in deterministic, sorted order."""
    section = presets_config(config)
    if not section.enabled and not include_disabled:
        return ()
    return tuple(sorted(section.presets))


def iter_presets(
    config: Any,
    *,
    include_disabled: bool = False,
) -> tuple[PresetConfig, ...]:
    """Return the configured preset definitions in name order."""
    section = presets_config(config)
    if not section.enabled and not include_disabled:
        return ()
    return tuple(section.presets[name] for name in sorted(section.presets))


def list_presets(
    config: Any,
    *,
    include_disabled: bool = False,
) -> tuple[str, ...]:
    """Alias for :func:`preset_names`, the names shown by ``presets list``."""
    return preset_names(config, include_disabled=include_disabled)


def get_preset(config: Any, name: str) -> PresetConfig:
    """Return one preset by name or fail with the configured alternatives."""
    section = presets_config(config)
    requested = name.strip() if isinstance(name, str) else ""
    if not requested:
        raise ConfigError("preset name must not be empty")
    try:
        return section.presets[requested]
    except KeyError:
        available = ", ".join(sorted(section.presets)) or "none"
        raise ConfigError(
            f"unknown preset: {requested} (available: {available})",
            details={
                "preset": requested,
                "available": sorted(section.presets),
            },
        ) from None


def select_preset(
    config: Any,
    name: str | None = None,
    *,
    use_default: bool = True,
) -> PresetConfig | None:
    """Return the requested preset, the configured default, or None.

    A blank *name* disables the default when *use_default* is false, which is
    what ``presets list`` and ``presets show`` use to inspect every definition
    without applying one first.
    """
    section = presets_config(config)
    requested = name.strip() if isinstance(name, str) else ""
    if not section.enabled:
        if requested:
            raise ConfigError(
                f"presets are disabled; cannot select preset: {requested}",
                details={"preset": requested},
            )
        return None
    if not requested and use_default:
        requested = section.default.strip()
    if not requested:
        return None
    return get_preset(section, requested)


def resolve_preset(
    config: Any,
    name: str | None = None,
    *,
    use_default: bool = True,
) -> PresetConfig | None:
    """Alias for :func:`select_preset`."""
    return select_preset(config, name, use_default=use_default)


def show_preset(config: Any, name: str) -> PresetConfig:
    """Alias for :func:`get_preset`, used by ``presets show``."""
    return get_preset(config, name)


def _merge(target: dict[str, Any], incoming: Mapping[str, Any]) -> None:
    for key, value in incoming.items():
        existing = target.get(key)
        if isinstance(value, Mapping) and isinstance(existing, Mapping):
            child = dict(existing)
            target[key] = child
            _merge(child, value)
        else:
            target[key] = copy.deepcopy(value)


def _preset_for(
    config: Any,
    preset: str | PresetConfig | None,
    *,
    use_default: bool,
) -> PresetConfig | None:
    if isinstance(preset, PresetConfig):
        return preset
    return select_preset(config, preset, use_default=use_default)


def preset_overrides(
    config: Any,
    name: str | None = None,
    *,
    use_default: bool = True,
) -> dict[str, Any]:
    """Return a copy of the selected preset's overrides, or an empty mapping."""
    selected = select_preset(config, name, use_default=use_default)
    return copy.deepcopy(selected.overrides) if selected is not None else {}


def apply_preset(
    values: Mapping[str, Any],
    preset: str | PresetConfig | None = None,
    *,
    overrides: Sequence[str] | Mapping[str, Any] | None = None,
    use_default: bool = True,
) -> dict[str, Any]:
    """Return *values* with a preset and then CLI overrides merged on top.

    ``overrides`` accepts the same forms as the resolver: a mapping, one
    ``key.path=value`` string or a sequence of them. The command line layer is
    always applied last, so an explicit flag beats the preset value.
    """
    if not isinstance(values, Mapping):
        raise ConfigError("config values must be a mapping")
    merged = copy.deepcopy(dict(values))
    selected = _preset_for(values, preset, use_default=use_default)
    if selected is not None:
        _merge(merged, selected.overrides)

    if overrides is not None:
        if isinstance(overrides, Mapping):
            cli_values = copy.deepcopy(dict(overrides))
        else:
            from aivia.core.config.resolve import parse_set_overrides

            cli_values = parse_set_overrides(overrides)
        if cli_values:
            _merge(merged, cli_values)
    return merged


def apply_preset_overrides(
    values: Mapping[str, Any],
    preset: str | PresetConfig | None = None,
    *,
    overrides: Sequence[str] | Mapping[str, Any] | None = None,
    use_default: bool = True,
) -> dict[str, Any]:
    """Alias for :func:`apply_preset`."""
    return apply_preset(
        values,
        preset,
        overrides=overrides,
        use_default=use_default,
    )


def describe_preset(preset: PresetConfig) -> dict[str, Any]:
    """Return a deterministic, serialisable description of *preset*."""
    return {
        "name": preset.name,
        "description": preset.description,
        "enabled": preset.enabled,
        "profile": preset.profile,
        "profiles": list(preset.profiles),
        "format": preset.format,
        "brand_kit": preset.brand_kit,
        "character": preset.character,
        "accounts": list(preset.accounts),
        "overrides": copy.deepcopy(preset.overrides),
    }
