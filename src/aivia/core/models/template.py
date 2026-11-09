"""Template records: a saved generation with variables and asset slots.

A template is extracted from a session. Its variables are suggested from the
``{{ var }}`` placeholders found in the session prompt and its composition text
layers, and its asset slots are suggested from the session's labelled assets. A
template is runnable once (``template run``) or in batch from CSV
(``template batch``), with per-row cost and isolated bad rows.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import AliasChoices, Field, field_validator

from aivia.core.ids import ulid
from aivia.core.models.base import ModelBase, utcnow

__all__ = [
    "KINDS",
    "PLACEHOLDER",
    "Template",
    "TemplateAssetSlot",
    "TemplateVariable",
    "extract_template",
    "load_template",
    "save_template",
]

PLACEHOLDER = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")
"""The ``{{ variable }}`` placeholder syntax used in template text."""

KINDS = ("image", "video", "speech", "transcription", "music", "sfx")
"""The media kinds a template can generate."""


class TemplateVariable(ModelBase):
    """One suggested variable filled in when a template is run."""

    name: str = Field(
        min_length=1,
        validation_alias=AliasChoices("name", "variable"),
    )
    default: str = Field(
        default="",
        validation_alias=AliasChoices("default", "value", "suggested"),
    )
    description: str = Field(
        default="",
        validation_alias=AliasChoices("description", "help"),
    )

    @field_validator("name", "default", "description", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class TemplateAssetSlot(ModelBase):
    """One asset slot filled from a session asset when a template is run."""

    name: str = Field(
        min_length=1,
        validation_alias=AliasChoices("name", "slot"),
    )
    asset_label: str = Field(
        default="",
        validation_alias=AliasChoices("asset_label", "asset", "label"),
    )
    description: str = Field(
        default="",
        validation_alias=AliasChoices("description", "help"),
    )

    @field_validator("name", "asset_label", "description", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class Template(ModelBase):
    """A saved generation with variables and asset slots."""

    id: str = Field(
        default_factory=lambda: f"template_{ulid()}",
        min_length=1,
        validation_alias=AliasChoices("id", "template_id"),
    )
    name: str = Field(
        default="",
        validation_alias=AliasChoices("name", "title"),
    )
    source_session_id: str = Field(
        default="",
        validation_alias=AliasChoices(
            "source_session_id", "source_session", "session", "session_id"
        ),
    )
    kind: str = Field(
        default="image",
        validation_alias=AliasChoices("kind", "media_kind"),
    )
    prompt: str = Field(
        default="",
        validation_alias=AliasChoices("prompt", "text"),
    )
    variables: list[TemplateVariable] = Field(
        default_factory=list,
        validation_alias=AliasChoices("variables", "vars"),
    )
    asset_slots: list[TemplateAssetSlot] = Field(
        default_factory=list,
        validation_alias=AliasChoices("asset_slots", "slots", "assets"),
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("options", "params"),
    )
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("kind", mode="before")
    @classmethod
    def _check_kind(cls, value: Any) -> Any:
        if isinstance(value, str) and value and value not in KINDS:
            raise ValueError(
                f"unknown template kind: {value!r} (expected one of {', '.join(KINDS)})"
            )
        return value

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Every variable name in declaration order."""
        return tuple(item.name for item in self.variables)

    def variable_default(self, name: str) -> str:
        """Return the suggested default for *name*, or empty string."""
        for item in self.variables:
            if item.name == name:
                return item.default
        return ""

    def slot_for_label(self, label: str) -> TemplateAssetSlot | None:
        """Return the slot backed by the session asset labelled *label*."""
        for item in self.asset_slots:
            if item.asset_label == label:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return the template as a JSON-compatible mapping."""
        return self.model_dump(mode="json", by_alias=False, exclude_none=True)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _placeholder_names(text: str) -> list[str]:
    """Return placeholder names in *text*, in order of first appearance."""
    found: list[str] = []
    for match in PLACEHOLDER.finditer(text):
        name = match.group(1)
        if name not in found:
            found.append(name)
    return found


def _is_exact_placeholder(text: str, name: str) -> bool:
    match = PLACEHOLDER.fullmatch(text)
    return match is not None and match.group(1) == name


def _text_layers_text(composition: Any) -> str:
    """Concatenate the text of every text layer in *composition*."""
    parts: list[str] = []
    layers = composition.get("layers") if isinstance(composition, Mapping) else None
    if isinstance(layers, Sequence) and not isinstance(layers, (str, bytes)):
        for layer in layers:
            if not isinstance(layer, Mapping):
                continue
            layer_type = str(
                layer.get("type") or layer.get("kind") or layer.get("layer_type") or ""
            )
            if layer_type != "text":
                continue
            text = layer.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _asset_rows(session: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    assets = session.get("assets")
    if isinstance(assets, Mapping):
        assets = [assets]
    if not isinstance(assets, Sequence) or isinstance(assets, (str, bytes)):
        return []
    return [item for item in assets if isinstance(item, Mapping)]


def _slot_name(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-") or "asset"
    return cleaned.lower()


def extract_template(
    session: Mapping[str, Any],
    *,
    name: str = "",
    kind: str | None = None,
    source_session_id: str = "",
) -> Template:
    """Build a template from a session, suggesting variables and asset slots.

    Variables are suggested from every ``{{ var }}`` placeholder found in the
    session prompt and its text layers. When a text field is exactly one
    placeholder, that literal text is the suggested default. Asset slots are
    suggested from every labelled session asset.
    """
    data = dict(session)
    resolved_name = _text(data.get("name")) or name
    resolved_kind = kind or _text(data.get("kind")) or "image"
    if resolved_kind not in KINDS:
        allowed = ", ".join(KINDS)
        message = (
            f"unknown template kind: {resolved_kind!r} (expected one of {allowed})"
        )
        raise ValueError(message)
    source = (
        source_session_id
        or _text(data.get("id"))
        or _text(data.get("session_id"))
        or ""
    )

    prompt = _text(data.get("prompt")) or ""
    current = data.get("current")
    if not prompt and isinstance(current, Mapping):
        prompt = _text_layers_text(current)

    suggested: dict[str, str] = {}
    layer_text = _text_layers_text(current) if isinstance(current, Mapping) else ""
    for text in (prompt, layer_text):
        for variable in _placeholder_names(text):
            if variable not in suggested:
                suggested[variable] = (
                    text if _is_exact_placeholder(text, variable) else ""
                )

    variables = [
        TemplateVariable(name=name_, default=default)
        for name_, default in suggested.items()
    ]

    slots: list[TemplateAssetSlot] = []
    for asset in _asset_rows(data):
        label = _text(asset.get("label")) or _text(asset.get("name"))
        if label:
            slots.append(TemplateAssetSlot(name=_slot_name(label), asset_label=label))

    options = data.get("options")
    if not isinstance(options, Mapping):
        options = {}

    return Template(
        name=resolved_name,
        source_session_id=source,
        kind=resolved_kind,
        prompt=prompt,
        variables=variables,
        asset_slots=slots,
        options=dict(options),
    )


def load_template(path: str | Path) -> Template:
    """Load a template from a YAML or JSON file."""
    target = Path(path).expanduser()
    if not target.is_file():
        raise FileNotFoundError(f"template file not found: {target}")
    text = target.read_text(encoding="utf-8")
    if target.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text)
    if not isinstance(payload, Mapping):
        raise ValueError(f"template file {target} must contain a mapping")
    return Template.model_validate(payload)


def save_template(template: Template, path: str | Path) -> None:
    """Write *template* to a YAML or JSON file."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = template.to_dict()
    if target.suffix.lower() == ".json":
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        target.write_text(
            yaml.safe_dump(payload, sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )
