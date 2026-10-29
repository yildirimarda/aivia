"""Layered configuration resolution with an explicit, documented precedence.

Layers are applied from lowest to highest precedence:

1. packaged defaults (``config.example.yaml``)
2. the user config (``config.yaml``, or the file selected with ``--config``)
3. the environment file (``config.<env>.yaml`` beside the user config)
4. the selected profile's ``overrides`` mapping
5. command line ``--set key.path=value`` flags, in the order given

Later layers win. Mappings deep-merge key by key; lists and scalars replace the
previous value. Every leaf records the layer that supplied it, so ``config
show --origins`` can explain the result.

The environment is selected explicitly with ``--env``/``--environment`` or,
when neither is given, from ``AIVA_ENV`` (then ``AIVA_ENVIRONMENT``) in the
process environment. Selecting an environment requires its file to exist: a
typo in the name must not silently fall back to the lower layers. Without a
selected environment no environment file is read.

Values in a ``--set`` flag are parsed as YAML scalars, so ``fps=24`` is an
integer, ``enabled=false`` a boolean and ``stages=[write, render]`` a list.
Dotted keys build nested mappings, for example ``pipeline.timezone=UTC``.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml

from aivia.core import paths
from aivia.core.bootstrap import DEFAULT_CONFIG_SOURCE
from aivia.core.errors import ConfigError

if TYPE_CHECKING:
    from aivia.core.config.app import RootConfig
    from aivia.core.config.registry import SectionRegistry

__all__ = [
    "CLI_ORIGIN",
    "DEFAULT_ORIGIN",
    "ENVIRONMENT_VARIABLE",
    "ENVIRONMENT_VARIABLES",
    "ENV_ORIGIN_PREFIX",
    "PRECEDENCE",
    "PROFILE_ORIGIN_PREFIX",
    "USER_ORIGIN",
    "Layer",
    "ResolvedConfig",
    "environment_file",
    "load_config_file",
    "merge_layers",
    "parse_set_overrides",
    "parse_set_value",
    "profile_overrides",
    "resolve_app_config",
    "resolve_config",
    "selected_environment",
    "selected_profile",
]

DEFAULT_ORIGIN = "defaults"
USER_ORIGIN = "user"
ENV_ORIGIN_PREFIX = "env:"
PROFILE_ORIGIN_PREFIX = "profile:"
CLI_ORIGIN = "cli"

ENVIRONMENT_VARIABLE = "AIVA_ENV"
ENVIRONMENT_VARIABLES = (ENVIRONMENT_VARIABLE, "AIVA_ENVIRONMENT")

PRECEDENCE = (
    DEFAULT_ORIGIN,
    USER_ORIGIN,
    f"{ENV_ORIGIN_PREFIX}<environment>",
    f"{PROFILE_ORIGIN_PREFIX}<profile>",
    CLI_ORIGIN,
)

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MISSING = object()


@dataclass(frozen=True, slots=True)
class Layer:
    """One configuration layer and the label shown as its origin."""

    origin: str
    values: Mapping[str, Any]
    path: Path | None = None


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """The merged values, the origin of every leaf and the selected options."""

    values: dict[str, Any]
    origins: dict[str, str]
    profile: str | None = None
    environment: str | None = None
    sources: tuple[Path, ...] = field(default_factory=tuple)
    origin_files: dict[str, str] = field(default_factory=dict)

    def error_context(self) -> dict[str, Any]:
        """Context that lets validation errors name the file behind a path."""
        fallback = self.origin_files.get(USER_ORIGIN)
        if fallback is None and self.sources:
            fallback = str(self.sources[-1])
        return {
            "origins": self.origins,
            "origin_files": self.origin_files,
            "file": fallback,
        }

    def to_app_config(
        self,
        *,
        sections: SectionRegistry | None = None,
    ) -> RootConfig:
        """Validate the merged values into the root application config."""
        return resolve_app_config(
            self.values,
            sections=sections,
            context=self.error_context(),
        )

    @property
    def app_config(self) -> RootConfig:
        """The merged values validated against the registered sections."""
        return self.to_app_config()

    @property
    def config(self) -> RootConfig:
        """Alias for :attr:`app_config`."""
        return self.to_app_config()


def load_config_file(
    path: Path,
    *,
    origin: str,
    required: bool = True,
) -> dict[str, Any]:
    """Read one YAML config layer, failing with a readable :class:`ConfigError`."""
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


def _normalise_environment(environment: str) -> str:
    name = environment.strip()
    if not name:
        raise ConfigError("environment name must not be empty")
    if not _ENVIRONMENT_NAME.fullmatch(name):
        raise ConfigError(
            f"invalid environment name: {environment!r} "
            "(use letters, digits, dots, dashes or underscores)"
        )
    return name


def _selected_environment(
    environment: str | None,
    environ: Mapping[str, str],
) -> str | None:
    if environment is not None:
        return _normalise_environment(environment)
    for name in ENVIRONMENT_VARIABLES:
        value = environ.get(name)
        if value is not None and value.strip():
            return _normalise_environment(value)
    return None


def selected_environment(
    environment: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Return the environment selected by *environment* or the process env.

    Unlike :func:`resolve_config` this reads no config file. Callers use it to
    record which environment was selected before that environment is applied.
    A blank *environment* means no selection, matching an unset option.
    """
    selected = environment.strip() if isinstance(environment, str) else environment
    return _selected_environment(
        selected or None,
        os.environ if environ is None else environ,
    )


def selected_profile(profile: str | None = None) -> str | None:
    """Return the profile selected by *profile*, or None for a blank option.

    Unlike :func:`resolve_config` this reads no config file. Callers use it to
    record which profile was selected before its overrides are applied. A blank
    *profile* means no selection, matching an unset option.
    """
    if not isinstance(profile, str):
        return profile
    return profile.strip() or None


def environment_file(
    config_file: str | Path,
    environment: str,
) -> Path:
    """Return the documented ``config.<env>.yaml`` path beside *config_file*.

    Resolution also accepts ``<stem>.<env>.yaml`` as a fallback when an
    explicit ``--config`` file has another stem, but this helper names the
    documented location.
    """
    target = Path(config_file).expanduser()
    name = _normalise_environment(environment)
    return target.with_name(f"config.{name}.yaml")


def _environment_candidates(config_file: Path, environment: str) -> tuple[Path, ...]:
    primary = config_file.with_name(f"config.{environment}.yaml")
    if config_file.stem == "config":
        return (primary,)
    return (primary, config_file.with_name(f"{config_file.stem}.{environment}.yaml"))


def _split_set_key(key: str) -> tuple[str, ...]:
    text = key.strip()
    if not text:
        raise ConfigError("--set key must not be empty")
    parts = tuple(part.strip() for part in text.split("."))
    if any(not part for part in parts):
        raise ConfigError(f"invalid --set key: {key!r}")
    return parts


def _parse_set_value(text: str) -> Any:
    if text.strip() == "":
        return ""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid --set value: {text!r}: {exc}") from exc


def parse_set_value(text: str) -> tuple[tuple[str, ...], Any]:
    """Parse one ``key.path=value`` flag into a path and a YAML value."""
    if not isinstance(text, str):
        raise ConfigError("--set values must be strings")
    key, separator, raw = text.partition("=")
    if not separator:
        raise ConfigError(f"invalid --set value: {text!r} (expected key.path=value)")
    return _split_set_key(key), _parse_set_value(raw)


def _assign(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = target
    for part in path[:-1]:
        existing = current.get(part, _MISSING)
        if existing is _MISSING:
            child: dict[str, Any] = {}
            current[part] = child
            current = child
        elif isinstance(existing, Mapping):
            if isinstance(existing, dict):
                current = existing
            else:
                child = dict(existing)
                current[part] = child
                current = child
        else:
            dotted = ".".join(path)
            raise ConfigError(f"cannot set {dotted}: {part} is not a mapping")
    current[path[-1]] = copy.deepcopy(value)


def _build_overrides(
    assignments: Sequence[tuple[tuple[str, ...], Any]],
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for path, value in assignments:
        _assign(overrides, path, value)
    return overrides


def parse_set_overrides(
    values: Sequence[str] | Mapping[str, Any] | str | None,
) -> dict[str, Any]:
    """Turn ``--set`` flags or a dotted-key mapping into nested overrides.

    Repeated keys keep the last value. A mapping is treated as already parsed,
    with dotted keys allowed for convenience.
    """
    if values is None:
        return {}
    assignments: list[tuple[tuple[str, ...], Any]] = []
    if isinstance(values, Mapping):
        for key, value in values.items():
            assignments.append((_split_set_key(str(key)), value))
    elif isinstance(values, str):
        assignments.append(parse_set_value(values))
    else:
        for item in values:
            assignments.append(parse_set_value(item))
    return _build_overrides(assignments)


def _pick[T](
    primary: T | None,
    alias: T | None,
    *,
    name: str,
    alias_name: str,
) -> T | None:
    if primary is not None and alias is not None and primary != alias:
        raise ConfigError(f"give either {name} or {alias_name}, not both")
    return primary if primary is not None else alias


def resolve_config(
    *,
    config_file: str | Path | None = None,
    config: str | Path | None = None,
    defaults_file: str | Path | None = None,
    defaults: str | Path | None = None,
    environment: str | None = None,
    env: str | None = None,
    profile: str | None = None,
    set_values: Sequence[str] | Mapping[str, Any] | str | None = None,
    set_overrides: Sequence[str] | Mapping[str, Any] | str | None = None,
    overrides: Sequence[str] | Mapping[str, Any] | str | None = None,
    environ: Mapping[str, str] | None = None,
) -> ResolvedConfig:
    """Resolve every configuration layer and record where each value came from.

    See the module docstring for the precedence order. ``config``, ``defaults``,
    ``env`` and ``overrides`` are aliases kept for callers that read better with
    those names.
    """
    environment_values = os.environ if environ is None else environ
    selected_config = cast(
        "str | Path | None",
        _pick(config_file, config, name="config_file", alias_name="config"),
    )
    selected_defaults = cast(
        "str | Path | None",
        _pick(defaults_file, defaults, name="defaults_file", alias_name="defaults"),
    )
    selected_environment = _selected_environment(
        cast(
            "str | None",
            _pick(environment, env, name="environment", alias_name="env"),
        ),
        environment_values,
    )

    resolved_config = paths.resolve_config_file(selected_config, environment_values)
    resolved_defaults = (
        paths.package_defaults_dir() / DEFAULT_CONFIG_SOURCE
        if selected_defaults is None
        else Path(selected_defaults).expanduser().resolve()
    )

    layers: list[Layer] = [
        Layer(
            DEFAULT_ORIGIN,
            load_config_file(
                resolved_defaults,
                origin=DEFAULT_ORIGIN,
                required=True,
            ),
            resolved_defaults,
        )
    ]
    sources: list[Path] = [resolved_defaults]

    user_values = load_config_file(
        resolved_config,
        origin=USER_ORIGIN,
        required=False,
    )
    if user_values:
        layers.append(Layer(USER_ORIGIN, user_values, resolved_config))
        sources.append(resolved_config)

    if selected_environment is not None:
        candidates = _environment_candidates(resolved_config, selected_environment)
        environment_path = next(
            (path for path in candidates if path.exists()),
            candidates[0],
        )
        if not environment_path.exists():
            expected = ", ".join(str(path) for path in candidates)
            raise ConfigError(
                f"environment config file not found: {expected}",
                details={
                    "environment": selected_environment,
                    "paths": [str(path) for path in candidates],
                },
            )
        environment_values_layer = load_config_file(
            environment_path,
            origin=f"{ENV_ORIGIN_PREFIX}{selected_environment}",
            required=True,
        )
        if environment_values_layer:
            layers.append(
                Layer(
                    f"{ENV_ORIGIN_PREFIX}{selected_environment}",
                    environment_values_layer,
                    environment_path,
                )
            )
            sources.append(environment_path)

    values, origins = merge_layers(layers)

    selected_profile: str | None = None
    if profile is not None:
        selected_profile = profile
        profile_values = profile_overrides(values, profile)
        if profile_values:
            _merge_mapping(
                values,
                origins,
                profile_values,
                origin=f"{PROFILE_ORIGIN_PREFIX}{profile}",
            )

    cli_overrides = parse_set_overrides(
        _pick_set_values(set_values, set_overrides, overrides)
    )
    if cli_overrides:
        _merge_mapping(values, origins, cli_overrides, origin=CLI_ORIGIN)

    origin_files = {
        layer.origin: str(layer.path) for layer in layers if layer.path is not None
    }
    if selected_profile is not None:
        profile_origin = origins.get("profiles") or origins.get(
            f"profiles.{selected_profile}"
        )
        profile_source = origin_files.get(profile_origin or "")
        if profile_source is None:
            profile_source = origin_files.get(USER_ORIGIN) or origin_files.get(
                DEFAULT_ORIGIN
            )
        if profile_source is not None:
            origin_files[f"{PROFILE_ORIGIN_PREFIX}{selected_profile}"] = profile_source
    if cli_overrides:
        origin_files[CLI_ORIGIN] = "<command line>"

    return ResolvedConfig(
        values=values,
        origins=origins,
        profile=selected_profile,
        environment=selected_environment,
        sources=tuple(sources),
        origin_files=origin_files,
    )


def _pick_set_values(
    set_values: Sequence[str] | Mapping[str, Any] | str | None,
    set_overrides: Sequence[str] | Mapping[str, Any] | str | None,
    overrides: Sequence[str] | Mapping[str, Any] | str | None,
) -> Sequence[str] | Mapping[str, Any] | str | None:
    provided = [
        value for value in (set_values, set_overrides, overrides) if value is not None
    ]
    if not provided:
        return None
    if len(provided) > 1:
        raise ConfigError("give only one of set_values, set_overrides or overrides")
    return provided[0]


def resolve_app_config(
    values: Mapping[str, Any] | None = None,
    *,
    sections: SectionRegistry | None = None,
    context: Mapping[str, Any] | None = None,
    **options: Any,
) -> RootConfig:
    """Resolve configuration and validate it into the root application model.

    When *values* is given it is used as the merged mapping; otherwise the
    keyword options are passed to :func:`resolve_config`. *context* carries
    source information so validation errors can name the offending file.
    """
    from aivia.core.config.app import parse_app_config

    resolved = (
        ResolvedConfig(values=dict(values), origins={})
        if values is not None
        else resolve_config(**options)
    )
    merged_context: dict[str, Any] = dict(resolved.error_context())
    if context:
        merged_context.update(context)
    return parse_app_config(
        resolved.values,
        sections=sections,
        context=merged_context,
    )
