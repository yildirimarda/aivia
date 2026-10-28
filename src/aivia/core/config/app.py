"""Root application configuration assembled from every registered section.

The root model is generated from the section registry, so optional modules add
their top-level keys without the core knowing about them. Parsing a root
mapping validates each section and then runs cross-section checks, for example
that a publishing account exists in the platform account list or that a profile
names a configured format.

:data:`AppConfig` is the root model for the global registry. Use
:func:`build_app_config` with a custom registry, or :func:`parse_app_config`
when the registry is not known up front.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Self, cast

from pydantic import Field, create_model

from aivia.core.config.base import ConfigModel
from aivia.core.config.errors import enrich_config_error
from aivia.core.config.migrations import CURRENT_SCHEMA_VERSION
from aivia.core.config.registry import SectionRegistry, registry
from aivia.core.config.sections import register_core_sections
from aivia.core.errors import ConfigError

__all__ = [
    "AppConfig",
    "RootConfig",
    "assemble_root_config",
    "build_app_config",
    "parse_app_config",
    "validate_cross_references",
]


_ROOT_KEYS = frozenset({"schema_version"})


class RootConfig(ConfigModel):
    """Base class for the assembled root configuration model."""

    schema_version: int = Field(default=CURRENT_SCHEMA_VERSION, ge=1)
    _sections: ClassVar[SectionRegistry | None] = None

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        environ: Mapping[str, str] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Self:
        """Validate a root mapping and run the cross-section checks."""
        if not isinstance(values, Mapping):
            raise ConfigError("config root must be a mapping")
        target = cls._sections
        try:
            if target is not None:
                _reject_unknown_sections(values, target)
            parsed = super().from_mapping(values, environ=environ, context=context)
            validate_cross_references(parsed, sections=target)
            return parsed
        except ConfigError as error:
            raise enrich_config_error(error, context=context) from error

    @classmethod
    def parse(
        cls,
        values: Mapping[str, Any],
        *,
        environ: Mapping[str, str] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Self:
        """Alias for :meth:`from_mapping`."""
        return cls.from_mapping(values, environ=environ, context=context)


def build_app_config(
    sections: SectionRegistry | None = None,
) -> type[RootConfig]:
    """Build the root model for *sections*, registering the core sections.

    A custom registry is filled with the core sections first, so callers only
    add the module sections they need.
    """
    target = registry if sections is None else sections
    register_core_sections(target)
    fields: dict[str, Any] = {
        section.key: (section.model | None, None) for section in target
    }
    model = cast(
        type[RootConfig],
        create_model("AppConfig", __base__=RootConfig, **fields),
    )
    model._sections = target
    return model


def assemble_root_config(
    sections: SectionRegistry | None = None,
) -> type[RootConfig]:
    """Alias for :func:`build_app_config`."""
    return build_app_config(sections)


def parse_app_config(
    values: Mapping[str, Any],
    *,
    sections: SectionRegistry | None = None,
    environ: Mapping[str, str] | None = None,
    context: Mapping[str, Any] | None = None,
) -> RootConfig:
    """Parse *values* with the root model for *sections*."""
    model = AppConfig if sections is None else build_app_config(sections)
    return model.from_mapping(values, environ=environ, context=context)


def validate_cross_references(
    config: Any,
    *,
    sections: SectionRegistry | None = None,
) -> Any:
    """Run every section reference hook plus the built-in root checks."""
    target = sections or getattr(type(config), "_sections", None) or registry
    for section in target:
        value = _get_section(config, section.key)
        validator = getattr(value, "validate_references", None)
        if callable(validator):
            validator(config)
    _validate_profile_references(config)
    _validate_preset_references(config)
    return config


def _reject_unknown_sections(
    values: Mapping[str, Any],
    sections: SectionRegistry,
) -> None:
    unknown = sorted(
        str(key) for key in values if key not in sections and key not in _ROOT_KEYS
    )
    if not unknown:
        return
    known = ", ".join(sections.keys()) or "none"
    messages = "; ".join(f"unknown config section: {key}" for key in unknown)
    raise ConfigError(
        f"{messages} (registered: {known})",
        details={"unknown": unknown, "available": list(sections.keys())},
    )


def _get_section(config: Any, key: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(key)
    return getattr(config, key, None)


def _entries(value: Any, field: str) -> Mapping[Any, Any]:
    section = _get_section(value, field)
    if isinstance(section, Mapping):
        return section
    if isinstance(value, Mapping):
        return value
    return {}


def _names(value: Any, field: str) -> set[str] | None:
    section = _get_section(value, field)
    if section is None and isinstance(value, Mapping):
        section = value
    if isinstance(section, Mapping):
        return {str(name) for name in section}
    if isinstance(section, Sequence) and not isinstance(section, (str, bytes)):
        return {str(name) for name in section}
    return None


def _field(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _language_names(value: Any) -> set[str] | None:
    if value is None:
        return None
    entries = _get_section(value, "languages")
    if entries is None and isinstance(value, Mapping):
        entries = value
    if isinstance(entries, Mapping):
        return {str(name) for name in entries if str(name) != "default"}
    if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
        return {str(name) for name in entries}
    return None


def _validate_profile_references(config: Any) -> None:
    profiles = _entries(_get_section(config, "profiles"), "profiles")
    if not profiles:
        return
    accounts = _names(_get_section(config, "platforms"), "accounts")
    formats = _names(_get_section(config, "formats"), "formats")
    languages = _language_names(_get_section(config, "languages"))
    for name, profile in profiles.items():
        if accounts is not None:
            for account in _field(profile, "accounts", None) or []:
                if account not in accounts:
                    available = ", ".join(sorted(accounts)) or "none"
                    raise ConfigError(
                        f"profile {name} references unknown platform account: "
                        f"{account} (available: {available})",
                        details={
                            "profile": name,
                            "account": account,
                            "available": sorted(accounts),
                        },
                    )
        format_name = _field(profile, "format", "") or ""
        if formats is not None and format_name and format_name not in formats:
            available = ", ".join(sorted(formats)) or "none"
            raise ConfigError(
                f"profile {name} references unknown format: {format_name} "
                f"(available: {available})",
                details={
                    "profile": name,
                    "format": format_name,
                    "available": sorted(formats),
                },
            )
        if languages is not None:
            for language in _field(profile, "languages", None) or []:
                if language not in languages:
                    available = ", ".join(sorted(languages)) or "none"
                    raise ConfigError(
                        f"profile {name} references unknown language: {language} "
                        f"(available: {available})",
                        details={
                            "profile": name,
                            "language": language,
                            "available": sorted(languages),
                        },
                    )


def _validate_preset_references(config: Any) -> None:
    presets = _entries(_get_section(config, "presets"), "presets")
    if not presets:
        return
    accounts = _names(_get_section(config, "platforms"), "accounts")
    formats = _names(_get_section(config, "formats"), "formats")
    for name, preset in presets.items():
        if accounts is not None:
            for account in _field(preset, "accounts", None) or []:
                if account not in accounts:
                    available = ", ".join(sorted(accounts)) or "none"
                    raise ConfigError(
                        f"preset {name} references unknown platform account: "
                        f"{account} (available: {available})",
                        details={
                            "preset": name,
                            "account": account,
                            "available": sorted(accounts),
                        },
                    )
        format_name = _field(preset, "format", "") or ""
        if formats is not None and format_name and format_name not in formats:
            available = ", ".join(sorted(formats)) or "none"
            raise ConfigError(
                f"preset {name} references unknown format: {format_name} "
                f"(available: {available})",
                details={
                    "preset": name,
                    "format": format_name,
                    "available": sorted(formats),
                },
            )


AppConfig = build_app_config()
