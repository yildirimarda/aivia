"""Prompt version selection configuration.

A prompt has up to two versions: the user copy in the config directory and the
packaged default. The ``prompts`` section maps a prompt name to the version that
should be used. Profiles may override individual entries through their
``overrides`` mapping, so one content profile can pin the packaged default while
another keeps the user copy.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel

__all__ = [
    "PromptSource",
    "PromptVersion",
    "PromptVersionSource",
    "PromptsConfig",
    "normalise_prompt_source",
]

_SOURCE_ALIASES: dict[str, str] = {
    "": "auto",
    "auto": "auto",
    "any": "auto",
    "either": "auto",
    "prefer-user": "auto",
    "prefer_user": "auto",
    "user": "user",
    "local": "user",
    "custom": "user",
    "override": "user",
    "default": "default",
    "packaged": "default",
    "package": "default",
    "bundled": "default",
    "builtin": "default",
    "built-in": "default",
}

_CONTAINER_ALIASES = (
    "version",
    "versions",
    "select",
    "selected",
    "selection",
    "selections",
    "source",
    "sources",
    "prompt_versions",
    "prompt_sources",
)

_VALUE_ALIASES = ("source", "version", "from", "origin", "use", "value")


class PromptVersionSource(StrEnum):
    """Which copy of a prompt is selected."""

    AUTO = "auto"
    USER = "user"
    DEFAULT = "default"


PromptVersion = PromptVersionSource
PromptSource = PromptVersionSource


def normalise_prompt_source(value: Any) -> PromptVersionSource:
    """Return *value* as a :class:`PromptVersionSource`.

    ``auto`` keeps the usual user-first fallback, ``user`` requires the user
    copy and ``default`` requires the packaged copy. Common spelling variants
    such as ``packaged`` and ``builtin`` are accepted.
    """
    if isinstance(value, PromptVersionSource):
        return value
    if value is None:
        return PromptVersionSource.AUTO
    if isinstance(value, Mapping):
        for key in _VALUE_ALIASES:
            if key in value:
                return normalise_prompt_source(value[key])
        raise ValueError(
            "prompt version must be a source name or a mapping with a source key"
        )
    if not isinstance(value, str):
        raise ValueError("prompt version must be one of: auto, default, user")
    text = value.strip().lower().replace(" ", "-")
    try:
        return PromptVersionSource(_SOURCE_ALIASES[text])
    except KeyError:
        allowed = ", ".join(source.value for source in PromptVersionSource)
        raise ValueError(
            f"unknown prompt version: {value!r} (allowed: {allowed})"
        ) from None


def _selection_mapping(data: Any) -> dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError("prompts must be a mapping")
    values = dict(data)
    selections: dict[str, Any] = {}
    for key in list(values):
        if key not in _CONTAINER_ALIASES:
            continue
        container = values.pop(key)
        if container is None:
            continue
        if not isinstance(container, Mapping):
            raise ValueError(f"prompts.{key} must be a mapping")
        for name, source in container.items():
            selections.setdefault(name, source)
    for name, source in values.items():
        selections.setdefault(name, source)
    return selections


class PromptsConfig(ConfigModel):
    """The selected version of each prompt, with ``auto`` as the fallback.

    The canonical form is ``prompts: {versions: {image: default}}``. A direct
    mapping (``prompts: {image: default}``) and aliases such as ``select`` are
    accepted as well, because the section is small enough that a nesting level
    should not be a source of configuration errors.
    """

    versions: dict[str, PromptVersionSource] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases_and_direct_mapping(cls, data: Any) -> Any:
        if data is None:
            return {}
        if isinstance(data, Mapping):
            return {"versions": _selection_mapping(data)}
        return data

    @field_validator("versions", mode="before")
    @classmethod
    def _normalise_versions(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        selections: dict[str, PromptVersionSource] = {}
        for name, source in value.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("prompt names must be non-empty strings")
            key = name.strip().replace("\\", "/")
            if key in selections:
                raise ValueError(f"duplicate prompt version: {key}")
            selections[key] = normalise_prompt_source(source)
        return selections

    @property
    def selections(self) -> dict[str, PromptVersionSource]:
        """Alias for :attr:`versions`."""
        return self.versions

    @property
    def sources(self) -> dict[str, PromptVersionSource]:
        """Alias for :attr:`versions`."""
        return self.versions

    def source_for(self, name: str) -> PromptVersionSource:
        """Return the configured source for *name*, or ``auto`` when unset."""
        return self.versions.get(
            name.strip().replace("\\", "/"), PromptVersionSource.AUTO
        )
