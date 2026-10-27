"""Layered configuration loading and rendering for ``aivia config show``.

The command is intentionally independent of the validated configuration model
that later milestones introduce. It merges the packaged defaults with the user
config and an optional profile override, tracks which layer supplied every leaf
value and masks values whose key looks secret before printing YAML.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from aivia.core import paths
from aivia.core.bootstrap import DEFAULT_CONFIG_SOURCE
from aivia.core.errors import ConfigError
from aivia.core.redact import REDACTED, redact

__all__ = [
    "DEFAULT_ORIGIN",
    "PROFILE_ORIGIN_PREFIX",
    "USER_ORIGIN",
    "Layer",
    "ResolvedConfig",
    "dump_config",
    "mask_secrets",
    "merge_layers",
    "profile_overrides",
    "resolve_config",
]

DEFAULT_ORIGIN = "defaults"
USER_ORIGIN = "user"
PROFILE_ORIGIN_PREFIX = "profile:"

_MISSING = object()

_SECRET_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "auth_token",
        "bearer_token",
        "client_secret",
        "credential",
        "credentials",
        "key",
        "keys",
        "passwd",
        "password",
        "private_key",
        "pwd",
        "refresh_token",
        "secret",
        "secret_key",
        "token",
        "access_token",
    }
)
_SECRET_SUFFIXES = _SECRET_NAMES - {"key", "keys"}
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class Layer:
    """One configuration layer and the label shown as its origin."""

    origin: str
    values: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """The merged values, the origin of every leaf and the selected profile."""

    values: dict[str, Any]
    origins: dict[str, str]
    profile: str | None = None


def _load_mapping(path: Path, *, origin: str, required: bool) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise ConfigError(f"config file not found: {path}")
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file: {path}") from exc
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {origin} config {path}: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ConfigError(f"config root must be a mapping: {path}")
    return dict(payload)


def _dotted(prefix: tuple[str, ...], key: object) -> str:
    return ".".join((*prefix, str(key)))


def _clear_origins(
    origins: dict[str, str], prefix: tuple[str, ...], key: object
) -> None:
    dotted = _dotted(prefix, key)
    for path in tuple(origins):
        if path == dotted or path.startswith(f"{dotted}."):
            del origins[path]


def _merge_mapping(
    target: dict[str, Any],
    origins: dict[str, str],
    incoming: Mapping[str, Any],
    *,
    origin: str,
    prefix: tuple[str, ...] = (),
) -> None:
    for key, value in incoming.items():
        existing = target.get(key, _MISSING)
        if isinstance(value, Mapping):
            if isinstance(existing, Mapping):
                child = cast(dict[str, Any], existing)
            else:
                if existing is not _MISSING:
                    _clear_origins(origins, prefix, key)
                child = {}
                target[key] = child
            _merge_mapping(
                child,
                origins,
                value,
                origin=origin,
                prefix=(*prefix, str(key)),
            )
            if not value:
                origins[_dotted(prefix, key)] = origin
            continue

        if existing is not _MISSING:
            _clear_origins(origins, prefix, key)
        target[key] = copy.deepcopy(value)
        origins[_dotted(prefix, key)] = origin


def merge_layers(layers: Sequence[Layer]) -> tuple[dict[str, Any], dict[str, str]]:
    """Deep-merge *layers* in order, replacing lists and tracking leaf origins."""
    values: dict[str, Any] = {}
    origins: dict[str, str] = {}
    for layer in layers:
        _merge_mapping(values, origins, layer.values, origin=layer.origin)
    return values, origins


def profile_overrides(values: Mapping[str, Any], profile: str) -> Mapping[str, Any]:
    """Return the ``overrides`` mapping for *profile*, or fail with its peers."""
    profiles = values.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ConfigError(f"unknown profile: {profile} (no profiles are configured)")
    if profile not in profiles:
        available = ", ".join(sorted(str(name) for name in profiles)) or "none"
        raise ConfigError(f"unknown profile: {profile} (available: {available})")
    entry = profiles[profile]
    if not isinstance(entry, Mapping):
        raise ConfigError(f"profile must be a mapping: {profile}")
    overrides = entry.get("overrides", {})
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise ConfigError(f"profile overrides must be a mapping: {profile}")
    return overrides


def resolve_config(
    *,
    config_file: str | Path | None = None,
    defaults_file: str | Path | None = None,
    profile: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> ResolvedConfig:
    """Merge packaged defaults, the user config and an optional profile."""
    resolved_config = paths.resolve_config_file(config_file, environ)
    resolved_defaults = (
        paths.package_defaults_dir() / DEFAULT_CONFIG_SOURCE
        if defaults_file is None
        else Path(defaults_file).expanduser().resolve()
    )

    layers = [
        Layer(
            DEFAULT_ORIGIN,
            _load_mapping(resolved_defaults, origin=DEFAULT_ORIGIN, required=True),
        )
    ]
    user_values = _load_mapping(resolved_config, origin=USER_ORIGIN, required=False)
    if user_values:
        layers.append(Layer(USER_ORIGIN, user_values))

    values, origins = merge_layers(layers)
    selected: str | None = None
    if profile is not None:
        overrides = profile_overrides(values, profile)
        if overrides:
            _merge_mapping(
                values,
                origins,
                overrides,
                origin=f"{PROFILE_ORIGIN_PREFIX}{profile}",
            )
        selected = profile

    return ResolvedConfig(values=values, origins=origins, profile=selected)


def _normalise_key(key: object) -> str:
    return _NON_ALPHANUMERIC.sub("_", str(key).lower()).strip("_")


def _is_secret_key(key: object) -> bool:
    normalised = _normalise_key(key)
    if normalised in _SECRET_NAMES:
        return True
    return any(normalised.endswith(f"_{name}") for name in _SECRET_SUFFIXES)


def mask_secrets(value: Any) -> Any:
    """Return *value* with secret-looking keys and token-like strings masked."""
    if isinstance(value, Mapping):
        masked: dict[Any, Any] = {}
        for key, item in value.items():
            masked[key] = REDACTED if _is_secret_key(key) else mask_secrets(item)
        return masked
    if isinstance(value, list):
        return [mask_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(mask_secrets(item) for item in value)
    if isinstance(value, str):
        return redact(value)
    return value


def dump_config(
    resolved: ResolvedConfig,
    *,
    include_origins: bool = False,
) -> str:
    """Render a resolved configuration as deterministic, masked YAML."""
    lines: list[str] = []
    if resolved.profile is not None:
        lines.append(f"# profile: {resolved.profile}")
    body = yaml.safe_dump(
        mask_secrets(resolved.values),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    ).rstrip("\n")
    lines.append(body)
    if include_origins:
        lines.append("# origins")
        for path in sorted(resolved.origins):
            lines.append(f"# {path}: {resolved.origins[path]}")
    return "\n".join(lines) + "\n"
