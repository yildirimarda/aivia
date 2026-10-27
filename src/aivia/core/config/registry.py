"""Registry of top-level configuration sections.

Core code and optional modules register the configuration sections they own
here. A section owns exactly one top-level key and its :class:`ConfigModel`
subclass. Registration rejects duplicate keys, and parsing a mapping with an
unregistered top-level key fails with the key and the registered sections, so a
typo or a module that is not installed never gets silently ignored.
"""

from __future__ import annotations

import keyword
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import create_model

from aivia.core.config.base import ConfigModel
from aivia.core.errors import ConfigError

__all__ = [
    "Section",
    "SectionRegistry",
    "register_section",
    "registry",
]


def _validate_key(key: object) -> str:
    if not isinstance(key, str):
        raise TypeError("config section key must be a string")
    if (
        not key
        or not key.isidentifier()
        or keyword.iskeyword(key)
        or key.startswith("_")
    ):
        raise ConfigError(f"invalid config section key: {key!r}")
    return key


def _validate_model(key: str, model: object) -> type[ConfigModel]:
    if not isinstance(model, type) or not issubclass(model, ConfigModel):
        raise TypeError(
            f"config section {key!r} must be a ConfigModel subclass, "
            f"not {type(model).__name__}"
        )
    return model


@dataclass(frozen=True, slots=True)
class Section:
    """One registered top-level configuration section."""

    key: str
    model: type[ConfigModel]
    module: str | None = None


class SectionRegistry:
    """An ordered, collision-checked collection of configuration sections."""

    def __init__(self) -> None:
        self._sections: dict[str, Section] = {}

    def register(
        self,
        key: str,
        model: type[ConfigModel],
        *,
        module: str | None = None,
    ) -> Section:
        """Register *model* under the top-level *key*.

        A duplicate key is rejected even when it names the same model, because
        two owners for one top-level key would make precedence ambiguous.
        """
        validated_key = _validate_key(key)
        validated_model = _validate_model(validated_key, model)
        existing = self._sections.get(validated_key)
        if existing is not None:
            raise ConfigError(
                f"config section already registered: {validated_key}",
                details={
                    "key": validated_key,
                    "existing": existing.model.__name__,
                    "new": validated_model.__name__,
                },
            )
        section = Section(
            key=validated_key,
            model=validated_model,
            module=module,
        )
        self._sections[validated_key] = section
        return section

    def register_section(
        self,
        key: str,
        model: type[ConfigModel],
        *,
        module: str | None = None,
    ) -> Section:
        """Register a section from an explicit model class."""
        return self.register(key, model, module=module)

    def section(
        self,
        key: str,
        *,
        module: str | None = None,
    ) -> Callable[[type[ConfigModel]], type[ConfigModel]]:
        """Return a decorator that registers the decorated model class."""

        def decorator(model: type[ConfigModel]) -> type[ConfigModel]:
            self.register(key, model, module=module)
            return model

        return decorator

    def get(self, key: str) -> Section:
        """Return the registered section for *key*."""
        try:
            return self._sections[key]
        except KeyError:
            raise KeyError(f"unknown config section: {key}") from None

    def keys(self) -> tuple[str, ...]:
        """Return the registered top-level keys in sorted order."""
        return tuple(sorted(self._sections))

    def models(self) -> dict[str, type[ConfigModel]]:
        """Return a copy of the key-to-model mapping in sorted key order."""
        return {key: self._sections[key].model for key in self.keys()}

    def model(self) -> type[ConfigModel]:
        """Build a root model with one optional field per registered section."""
        fields: dict[str, Any] = {
            section.key: (section.model | None, None) for section in self
        }
        return create_model("RegisteredConfig", __base__=ConfigModel, **fields)

    def parse(
        self,
        values: Mapping[str, Any],
        *,
        environ: Mapping[str, str] | None = None,
    ) -> ConfigModel:
        """Validate *values* into the root model built from registered sections."""
        if not isinstance(values, Mapping):
            raise ConfigError("config root must be a mapping")
        unknown = sorted(str(key) for key in values if key not in self._sections)
        if unknown:
            known = ", ".join(self.keys()) or "none"
            messages = "; ".join(f"unknown config section: {key}" for key in unknown)
            raise ConfigError(
                f"{messages} (registered: {known})",
                details={"unknown": unknown, "available": list(self.keys())},
            )
        parsed = self.model().from_mapping(values, environ=environ)
        for key in self._sections:
            section = getattr(parsed, key, None)
            validator = getattr(section, "validate_references", None)
            if validator is not None:
                validator(parsed)
        return parsed

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._sections

    def __iter__(self) -> Iterator[Section]:
        return iter(self._sections[key] for key in self.keys())

    def __len__(self) -> int:
        return len(self._sections)


registry = SectionRegistry()
register_section = registry.register_section
