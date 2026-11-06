"""Render quote cards from typography templates stored as data.

A quote card is a still composition built from a named typography template.
Templates are plain YAML/JSON files: they carry the canvas, the timed layers
and the typography for the quote and its attribution, while the quote text
itself comes from the writing stage. Keeping the design in data means a user
can add or override a template without touching Python, and the same template
always produces the same composition for the same quote.

The packaged templates live under ``data/defaults/templates``. A user file in
``$XDG_CONFIG_HOME/aivia/templates`` with the same name wins, so a profile can
restyle every quote card without a release.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aivia.compose.render import render_still
from aivia.compose.schema import (
    CURRENT_COMPOSITION_SCHEMA_VERSION,
    Canvas,
    Composition,
)
from aivia.core import paths
from aivia.core.errors import ConfigError

__all__ = [
    "DEFAULT_QUOTE_CARD_TEMPLATE",
    "QUOTE_CARD_TEMPLATES_DIR",
    "QUOTE_CARD_TEMPLATES_FILENAME",
    "QuoteCard",
    "QuoteCardError",
    "QuoteCardFormat",
    "QuoteCardRenderer",
    "QuoteCardTemplate",
    "QuoteCardTemplateError",
    "QuoteCardTemplateLibrary",
    "QuoteCardTemplateNotFoundError",
    "QuoteCardTemplateStore",
    "QuoteCardTemplates",
    "TypographyTemplate",
    "TypographyTemplateLibrary",
    "TypographyTemplateStore",
    "TypographyTemplates",
    "build_quote_card",
    "build_quote_card_composition",
    "compose_quote_card",
    "default_quote_card_template_path",
    "default_quote_card_templates",
    "default_templates",
    "load_default_quote_card_templates",
    "load_packaged_quote_card_templates",
    "load_quote_card_template",
    "load_quote_card_templates",
    "load_quote_card_typography",
    "load_template",
    "load_templates",
    "load_typography_templates",
    "quote_card",
    "quote_card_composition",
    "quote_card_renderer",
    "quote_card_template",
    "quote_card_templates",
    "quote_card_templates_path",
    "quote_card_typography",
    "render_quote_card",
    "render_quote_card_still",
    "render_template",
    "save_quote_card",
    "typography_templates",
    "write_quote_card",
    "write_template",
]

DEFAULT_QUOTE_CARD_TEMPLATE = "classic"
"""The template used when a caller does not name one."""

QUOTE_CARD_TEMPLATES_DIR = "templates"
"""The directory under the user and package roots that holds template files."""

QUOTE_CARD_TEMPLATES_FILENAME = "quote_card.yaml"
"""The packaged file that holds the built-in quote card templates."""

_TEMPLATE_FILE_SUFFIXES = frozenset({".yaml", ".yml", ".json"})
_TEMPLATE_FILENAMES = (
    "quote_card.yaml",
    "quote_cards.yaml",
    "quote_card.yml",
    "quote_cards.yml",
    "quote_card.json",
    "quote_cards.json",
)
_TEMPLATE_DIRNAMES = ("quote_card", "quote_cards")
_CONTAINER_KEYS = ("templates", "quote_cards", "typography", "styles")
_STYLE_KEYS = frozenset({"quote", "attribution", "author", "title", "context"})
_LAYER_KEYS = frozenset({"layers", "composition"})
_CANVAS_KEYS = frozenset(
    {"canvas", "resolution", "size", "dimensions", "aspect", "aspect_ratio"}
)
_PLACEHOLDER = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")


class QuoteCardError(ConfigError):
    """A quote card template or composition cannot be built."""

    code = "quote_card"


class QuoteCardTemplateError(QuoteCardError):
    """A quote card typography template is missing or malformed."""


QuoteCardTemplateNotFoundError = QuoteCardTemplateError
"""Alias for :class:`QuoteCardTemplateError` when a name is missing."""


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _aspect_size(value: Any) -> tuple[int, int]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        proportion = float(value)
        if proportion <= 0.0:
            raise QuoteCardTemplateError("aspect ratio must be positive")
        if proportion > 1.0:
            return 1080, max(1, round(1080 / proportion))
        return max(1, round(1080 * proportion)), 1080
    text = _text(value).replace(" ", "").lower()
    if ":" in text:
        left, right = text.split(":", 1)
        try:
            width, height = int(left), int(right)
        except ValueError as exc:
            raise QuoteCardTemplateError(f"invalid aspect ratio: {value!r}") from exc
        if width <= 0 or height <= 0:
            raise QuoteCardTemplateError("aspect ratio must be positive")
        if width >= height:
            return 1080, max(1, round(1080 * height / width))
        return max(1, round(1080 * width / height)), 1080
    try:
        return _aspect_size(float(text))
    except ValueError as exc:
        raise QuoteCardTemplateError(f"invalid aspect ratio: {value!r}") from exc


def _default_canvas(values: Mapping[str, Any]) -> Canvas:
    raw = values.get("canvas")
    if raw is None:
        for key in ("resolution", "size", "dimensions", "aspect", "aspect_ratio"):
            if key in values:
                raw = values[key]
                if key in {"aspect", "aspect_ratio"}:
                    width, height = _aspect_size(raw)
                    raw = {"width": width, "height": height}
                break
    if raw is None:
        raw = {"width": 1080, "height": 1080}
    payload = dict(raw) if isinstance(raw, Mapping) else {"resolution": raw}
    background = values.get("background", values.get("background_color"))
    if background is not None:
        payload.setdefault("background", background)
    try:
        return Canvas.model_validate(payload)
    except ValueError as exc:
        raise QuoteCardTemplateError(f"invalid quote card canvas: {exc}") from exc


def _default_duration(values: Mapping[str, Any]) -> float:
    for key in ("duration_seconds", "duration", "length_seconds"):
        if key in values:
            try:
                value = float(values[key])
            except (TypeError, ValueError) as exc:
                raise QuoteCardTemplateError(
                    f"invalid quote card duration: {values[key]!r}"
                ) from exc
            if value <= 0.0:
                raise QuoteCardTemplateError("quote card duration must be positive")
            return value
    return 6.0


def _default_fps(values: Mapping[str, Any]) -> int:
    for key in ("fps", "frame_rate", "frames_per_second"):
        if key in values:
            try:
                value = int(values[key])
            except (TypeError, ValueError) as exc:
                raise QuoteCardTemplateError(
                    f"invalid quote card fps: {values[key]!r}"
                ) from exc
            if value <= 0:
                raise QuoteCardTemplateError("quote card fps must be positive")
            return value
    return 30


def _style_layer(
    layer_id: str,
    style: Any,
    *,
    text: str,
    x: int,
    y: int,
    width: int,
    height: int,
    z_index: int,
    defaults: Mapping[str, Any],
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "id": layer_id,
        "type": "text",
        "text": text,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "z_index": z_index,
        **defaults,
    }
    if isinstance(style, Mapping):
        values.update(style)
    elif isinstance(style, str) and style.strip():
        values.setdefault("font_family", style.strip())
    values["id"] = values.get("id") or layer_id
    values["type"] = "text"
    values.setdefault("text", text)
    return values


def _layers_from_styles(
    values: Mapping[str, Any],
    canvas: Canvas,
) -> list[dict[str, Any]]:
    width = canvas.width
    height = canvas.height
    margin = max(24, round(min(width, height) * 0.09))
    content_width = max(1, width - margin * 2)
    quote_defaults: dict[str, Any] = {
        "font_family": "Lato",
        "font_size": max(24.0, round(min(width, height) * 0.09)),
        "font_weight": "bold",
        "color": "#ffffff",
        "align": "center",
        "vertical_align": "middle",
        "line_height": 1.08,
        "shrink_to_fit": True,
    }
    attribution_defaults: dict[str, Any] = {
        "font_family": "Lato",
        "font_size": max(16.0, round(min(width, height) * 0.032)),
        "color": "#e2e8f0",
        "align": "center",
    }
    layers: list[dict[str, Any]] = []
    quote_raw = values.get("quote", values.get("quote_style"))
    attribution_raw = values.get("attribution", values.get("author"))
    title_raw = values.get("title")
    context_raw = values.get("context")
    quote_text = _text(values.get("quote_text")) or "{{ quote }}"
    attribution_text = _text(values.get("attribution_text")) or "— {{ attribution }}"
    layers.append(
        _style_layer(
            "quote",
            quote_raw,
            text=quote_text,
            x=margin,
            y=margin,
            width=content_width,
            height=max(1, round(height * 0.55)),
            z_index=1,
            defaults=quote_defaults,
        )
    )
    if attribution_raw is not None or "attribution" not in values:
        layers.append(
            _style_layer(
                "attribution",
                attribution_raw,
                text=attribution_text,
                x=margin,
                y=max(margin, round(height * 0.76)),
                width=content_width,
                height=max(1, round(height * 0.14)),
                z_index=2,
                defaults=attribution_defaults,
            )
        )
    if title_raw is not None:
        layers.append(
            _style_layer(
                "title",
                title_raw,
                text=_text(values.get("title_text")) or "{{ title }}",
                x=margin,
                y=margin,
                width=content_width,
                height=max(1, round(height * 0.12)),
                z_index=3,
                defaults={**attribution_defaults, "font_weight": "bold"},
            )
        )
    if context_raw is not None:
        layers.append(
            _style_layer(
                "context",
                context_raw,
                text=_text(values.get("context_text")) or "{{ context }}",
                x=margin,
                y=max(margin, round(height * 0.9)),
                width=content_width,
                height=max(1, round(height * 0.08)),
                z_index=4,
                defaults=attribution_defaults,
            )
        )
    return layers


class QuoteCardTemplate(BaseModel):
    """One named typography template for a quote card.

    ``layers`` are ordinary composition layer mappings. A text layer may use
    ``{{ quote }}``, ``{{ attribution }}``, ``{{ title }}``, ``{{ context }}``
    or ``{{ language }}`` placeholders. A layer whose id is ``quote`` or
    ``attribution`` is filled even when its template text is empty.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str = ""
    description: str = ""
    schema_version: int = 1
    canvas: Canvas = Field(
        default_factory=lambda: Canvas(width=1080, height=1080, background="#0b1020")
    )
    duration_seconds: float = 6.0
    fps: int = 30
    layers: list[dict[str, Any]] = Field(default_factory=list)
    quote_layer_id: str = "quote"
    attribution_layer_id: str = "attribution"
    title_layer_id: str = "title"
    context_layer_id: str = "context"
    source: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _normalise_payload(cls, data: Any) -> Any:
        if isinstance(data, QuoteCardTemplate):
            return data
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        composition = values.pop("composition", None)
        if isinstance(composition, Composition):
            composition = composition.model_dump(mode="python", by_alias=False)
        if isinstance(composition, Mapping):
            for key in (
                "canvas",
                "duration_seconds",
                "fps",
                "layers",
                "language",
                "platform",
            ):
                if key in composition:
                    values.setdefault(key, composition[key])
        canvas = _default_canvas(values)
        layers = values.get("layers")
        if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
            layers = _layers_from_styles(values, canvas)
        else:
            layers = [
                dict(layer) if isinstance(layer, Mapping) else layer for layer in layers
            ]
        values["canvas"] = canvas
        values["layers"] = layers
        values["duration_seconds"] = _default_duration(values)
        values["fps"] = _default_fps(values)
        if not _text(values.get("name")):
            values["name"] = DEFAULT_QUOTE_CARD_TEMPLATE
        return values

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        name: str | None = None,
        source: str | Path | None = None,
    ) -> QuoteCardTemplate:
        """Build a template from a mapping, applying an optional name/source."""
        payload = dict(value)
        if name is not None:
            payload["name"] = name
        template = cls.model_validate(payload)
        if source is not None:
            template.source = Path(source).expanduser()
        return template

    @classmethod
    def from_file(
        cls, path: str | Path, *, name: str | None = None
    ) -> QuoteCardTemplate:
        """Load one template file, or the default template from a catalog file."""
        target = Path(path).expanduser()
        payload = _read_document(target)
        catalog = _catalog_from_payload(payload, path=target)
        if name is not None:
            return catalog.require(name)
        return catalog.default

    @property
    def quote_layer(self) -> dict[str, Any] | None:
        """The raw layer that carries the quote text."""
        return self.layer(self.quote_layer_id)

    @property
    def attribution_layer(self) -> dict[str, Any] | None:
        """The raw layer that carries the attribution text."""
        return self.layer(self.attribution_layer_id)

    @property
    def quote_style(self) -> dict[str, Any]:
        """The typography fields of the quote layer."""
        return dict(self.quote_layer or {})

    @property
    def attribution_style(self) -> dict[str, Any]:
        """The typography fields of the attribution layer."""
        return dict(self.attribution_layer or {})

    def layer(self, value: str) -> dict[str, Any] | None:
        """Return the raw layer with *value* as its id."""
        key = _text(value)
        for layer in self.layers:
            if _text(layer.get("id")) == key:
                return layer
        return None

    def text_layers(self) -> tuple[dict[str, Any], ...]:
        """Return every text layer in stacking order."""
        return tuple(
            layer
            for layer in self.layers
            if _text(layer.get("type") or layer.get("kind") or layer.get("layer_type"))
            == "text"
        )

    def composition(
        self,
        quote: str,
        attribution: str = "",
        *,
        title: str = "",
        context: str = "",
        language: str = "",
        duration_seconds: float | None = None,
        platform: str = "",
        composition_id: str = "",
    ) -> Composition:
        """Fill the template and return a validated still composition."""
        values = {
            "quote": quote,
            "attribution": attribution,
            "author": attribution,
            "title": title,
            "context": context,
            "language": language,
        }
        layers: list[dict[str, Any]] = []
        for raw in self.layers:
            layer = _substitute(deepcopy(raw), values)
            if not isinstance(layer, Mapping):
                raise QuoteCardTemplateError("quote card layers must be mappings")
            values_map = dict(layer)
            layer_id = _text(values_map.get("id"))
            layer_type = _text(
                values_map.get("type")
                or values_map.get("kind")
                or values_map.get("layer_type")
            )
            if layer_type == "text":
                text = values_map.get("text")
                if not isinstance(text, str) or not text.strip():
                    if layer_id == self.quote_layer_id:
                        values_map["text"] = quote
                    elif layer_id == self.attribution_layer_id:
                        values_map["text"] = attribution
                    elif layer_id == self.title_layer_id:
                        values_map["text"] = title
                    elif layer_id == self.context_layer_id:
                        values_map["text"] = context
                if (
                    not isinstance(values_map.get("text"), str)
                    or not str(values_map.get("text")).strip()
                ):
                    if layer_id in {
                        self.attribution_layer_id,
                        self.title_layer_id,
                        self.context_layer_id,
                    }:
                        continue
                    raise QuoteCardTemplateError(
                        f"quote card layer {layer_id or '<text>'!r} has no text"
                    )
            layers.append(values_map)
        if not any(_text(layer.get("id")) == self.quote_layer_id for layer in layers):
            layers.append(
                {
                    "id": self.quote_layer_id,
                    "type": "text",
                    "text": quote,
                    "x": 0,
                    "y": 0,
                    "width": self.canvas.width,
                    "height": self.canvas.height,
                    "font_family": "Lato",
                    "font_size": max(
                        24.0, round(min(self.canvas.width, self.canvas.height) * 0.09)
                    ),
                    "font_weight": "bold",
                    "color": "#ffffff",
                    "align": "center",
                    "vertical_align": "middle",
                    "shrink_to_fit": True,
                }
            )
        span = self.duration_seconds if duration_seconds is None else duration_seconds
        if span <= 0.0:
            raise QuoteCardTemplateError("quote card duration must be positive")
        payload = {
            "id": composition_id or f"quote-card-{_slug(self.name)}",
            "schema_version": CURRENT_COMPOSITION_SCHEMA_VERSION,
            "canvas": self.canvas.model_dump(mode="python", by_alias=False),
            "duration_seconds": span,
            "fps": self.fps,
            "layers": layers,
            "language": language,
            "platform": platform,
        }
        try:
            return Composition.model_validate(payload)
        except ValueError as exc:
            raise QuoteCardTemplateError(
                f"invalid quote card composition for template {self.name!r}: {exc}"
            ) from exc

    build = composition
    compose = composition
    render = composition

    def to_dict(self) -> dict[str, Any]:
        """Return the template as a JSON-compatible mapping."""
        payload = self.model_dump(mode="json", by_alias=False, exclude_none=True)
        payload.pop("source", None)
        return payload


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", _text(value).lower()).strip("-")
    return cleaned or "quote-card"


def _substitute(value: Any, context: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _PLACEHOLDER.sub(
            lambda match: context.get(match.group(1), match.group(0)),
            value,
        )
    if isinstance(value, list):
        return [_substitute(item, context) for item in value]
    if isinstance(value, tuple):
        return tuple(_substitute(item, context) for item in value)
    if isinstance(value, Mapping):
        return {key: _substitute(item, context) for key, item in value.items()}
    return value


def _read_document(path: Path) -> Any:
    if not path.is_file():
        raise QuoteCardTemplateError(
            f"quote card template file not found: {path}",
            details={"path": str(path)},
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise QuoteCardTemplateError(
            f"cannot read quote card template {path}: {exc}",
            details={"path": str(path)},
        ) from exc
    try:
        if path.suffix.lower() == ".json":
            return json.loads(text)
        return yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise QuoteCardTemplateError(
            f"invalid quote card template {path}: {exc}",
            details={"path": str(path)},
        ) from exc


def _looks_like_template(values: Mapping[str, Any]) -> bool:
    return bool(_LAYER_KEYS & set(values) or _STYLE_KEYS & set(values))


def _catalog_from_payload(
    payload: Any,
    *,
    path: Path | None = None,
) -> QuoteCardTemplates:
    if isinstance(payload, QuoteCardTemplates):
        return payload
    if not isinstance(payload, Mapping):
        raise QuoteCardTemplateError(
            "quote card templates must be a mapping",
            details={"path": str(path) if path else ""},
        )
    values = dict(payload)
    default_name = _text(values.get("default", values.get("default_template", "")))
    container_key = next((key for key in _CONTAINER_KEYS if key in values), None)
    if container_key is not None:
        nested = values[container_key]
        if not isinstance(nested, Mapping):
            raise QuoteCardTemplateError(
                f"quote card templates under {container_key!r} must be a mapping"
            )
        values = dict(nested)
        default_name = default_name or _text(
            values.get("default", values.get("default_template", ""))
        )
    values.pop("default", None)
    values.pop("default_template", None)
    if _looks_like_template(values):
        name = _text(values.get("name") or values.get("id"))
        if not name and path is not None:
            name = path.stem
        name = name or DEFAULT_QUOTE_CARD_TEMPLATE
        template = QuoteCardTemplate.from_mapping(values, name=name, source=path)
        return QuoteCardTemplates({name: template}, default=default_name or name)

    templates: dict[str, QuoteCardTemplate] = {}
    for key, item in values.items():
        name = _text(key)
        if not name:
            raise QuoteCardTemplateError("quote card template names must not be empty")
        if not isinstance(item, Mapping):
            raise QuoteCardTemplateError(
                f"quote card template {name!r} must be a mapping"
            )
        template = QuoteCardTemplate.from_mapping(item, name=name, source=path)
        templates[name] = template
    if not templates:
        raise QuoteCardTemplateError("no quote card templates found")
    return QuoteCardTemplates(templates, default=default_name)


class QuoteCardTemplates(Mapping[str, QuoteCardTemplate]):
    """A named collection of quote card templates with a default."""

    def __init__(
        self,
        templates: Mapping[str, QuoteCardTemplate] | None = None,
        *,
        default: str = "",
    ) -> None:
        self._templates = dict(templates or {})
        self.default_name = _text(default) or next(iter(self._templates), "")
        if self.default_name and self.default_name not in self._templates:
            raise QuoteCardTemplateError(
                f"unknown default quote card template: {self.default_name}",
                details={"available": sorted(self._templates)},
            )

    @property
    def default(self) -> QuoteCardTemplate:
        """The default template, or the first one when none was named."""
        if not self._templates:
            raise QuoteCardTemplateError("no quote card templates are loaded")
        return self._templates[self.default_name or next(iter(self._templates))]

    def names(self) -> tuple[str, ...]:
        """Return every template name in sorted order."""
        return tuple(sorted(self._templates))

    def get(self, name: str, default: Any = None) -> QuoteCardTemplate | Any:
        """Return a template by name, or *default* when it is absent."""
        key = _text(name)
        if key in self._templates:
            return self._templates[key]
        if key in {"", "default"}:
            try:
                return self.default
            except QuoteCardTemplateError:
                return default
        return default

    def require(self, name: str = "") -> QuoteCardTemplate:
        """Return a template or raise an error naming the available ones."""
        key = _text(name)
        if not key or key == "default":
            return self.default
        if key not in self._templates:
            raise QuoteCardTemplateError(
                f"unknown quote card template: {key}",
                details={"template": key, "available": list(self.names())},
            )
        return self._templates[key]

    resolve = require
    template = require

    def __getitem__(self, name: str) -> QuoteCardTemplate:
        return self.require(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._templates)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and _text(name) in self._templates

    def to_dict(self) -> dict[str, Any]:
        """Return every template as a JSON-compatible mapping."""
        return {
            "default": self.default_name,
            "templates": {
                name: template.to_dict() for name, template in self._templates.items()
            },
        }


QuoteCardTemplateStore = QuoteCardTemplates
"""Alias for :class:`QuoteCardTemplates`."""


def _template_file_candidates(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    found: list[Path] = []
    for name in _TEMPLATE_FILENAMES:
        candidate = directory / name
        if candidate.is_file():
            found.append(candidate)
    for name in _TEMPLATE_DIRNAMES:
        child = directory / name
        if not child.is_dir():
            continue
        found.extend(
            path
            for path in sorted(child.iterdir())
            if path.is_file() and path.suffix.lower() in _TEMPLATE_FILE_SUFFIXES
        )
    return found


def _load_directory(directory: Path) -> QuoteCardTemplates:
    templates: dict[str, QuoteCardTemplate] = {}
    default = ""
    for path in _template_file_candidates(directory):
        catalog = _catalog_from_payload(_read_document(path), path=path)
        for name, template in catalog._templates.items():
            templates[name] = template
        if not default:
            default = catalog.default_name
    return QuoteCardTemplates(templates, default=default)


class QuoteCardTemplateLibrary:
    """Quote card templates in the user config directory with packaged fallbacks."""

    def __init__(
        self,
        *,
        config_dir: str | Path | None = None,
        defaults_dir: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.config_dir = (
            paths.config_dir(environ)
            if config_dir is None
            else Path(config_dir).expanduser().resolve()
        )
        self.defaults_dir = (
            paths.package_defaults_dir()
            if defaults_dir is None
            else Path(defaults_dir).expanduser().resolve()
        )
        self.user_dir = self.config_dir / QUOTE_CARD_TEMPLATES_DIR
        self.packaged_dir = self.defaults_dir / QUOTE_CARD_TEMPLATES_DIR
        self._cached: QuoteCardTemplates | None = None

    def _load(self) -> QuoteCardTemplates:
        if self._cached is not None:
            return self._cached
        packaged = _load_directory(self.packaged_dir)
        user = _load_directory(self.user_dir)
        merged: dict[str, QuoteCardTemplate] = dict(packaged._templates)
        merged.update(user._templates)
        default = user.default_name or packaged.default_name
        self._cached = QuoteCardTemplates(merged, default=default)
        return self._cached

    def names(self) -> tuple[str, ...]:
        """Return every visible template name."""
        return self._load().names()

    def load(self, name: str = "") -> QuoteCardTemplate:
        """Load one template by name, preferring a user override."""
        return self._load().require(name)

    require = load
    resolve = load
    template = load

    def templates(self) -> QuoteCardTemplates:
        """Return the whole merged catalog."""
        return self._load()

    catalog = templates

    def __getitem__(self, name: str) -> QuoteCardTemplate:
        return self.load(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self.names())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and _text(name) in self.names()


def quote_card_templates_path(
    name: str | Path = QUOTE_CARD_TEMPLATES_FILENAME,
    *,
    defaults_dir: str | Path | None = None,
) -> Path:
    """Return the packaged path for a template file or name."""
    base = (
        paths.package_defaults_dir()
        if defaults_dir is None
        else Path(defaults_dir).expanduser()
    )
    candidate = Path(name).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.parent != Path("."):
        return base / candidate
    if candidate.suffix.lower() in _TEMPLATE_FILE_SUFFIXES:
        return base / QUOTE_CARD_TEMPLATES_DIR / candidate
    return base / QUOTE_CARD_TEMPLATES_DIR / QUOTE_CARD_TEMPLATES_FILENAME


def default_quote_card_template_path(
    name: str | Path = QUOTE_CARD_TEMPLATES_FILENAME,
    *,
    defaults_dir: str | Path | None = None,
) -> Path:
    """Alias for :func:`quote_card_templates_path`."""
    return quote_card_templates_path(name, defaults_dir=defaults_dir)


def load_quote_card_templates(
    source: Any = None,
    *,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> QuoteCardTemplates:
    """Load quote card templates from data, a file, a mapping or the defaults.

    With no source the user ``templates`` directory is merged over the packaged
    one. A path loads that file. A mapping may be one template or a catalog.
    """
    if isinstance(source, QuoteCardTemplates):
        return source
    if isinstance(source, QuoteCardTemplate):
        return QuoteCardTemplates({source.name: source}, default=source.name)
    if isinstance(source, Mapping):
        return _catalog_from_payload(source)
    if isinstance(source, (str, Path)):
        candidate = Path(source).expanduser()
        if candidate.is_file():
            return _catalog_from_payload(_read_document(candidate), path=candidate)
        text = str(source)
        if not text.strip():
            return QuoteCardTemplateLibrary(
                config_dir=config_dir,
                defaults_dir=defaults_dir,
                environ=environ,
            ).templates()
        if Path(text).suffix.lower() in _TEMPLATE_FILE_SUFFIXES:
            raise QuoteCardTemplateError(
                f"quote card template file not found: {candidate}",
                details={"path": str(candidate)},
            )
        library = QuoteCardTemplateLibrary(
            config_dir=config_dir,
            defaults_dir=defaults_dir,
            environ=environ,
        )
        if text.strip() in library:
            return QuoteCardTemplates(
                {text.strip(): library.load(text.strip())},
                default=text.strip(),
            )
        raise QuoteCardTemplateError(
            f"unknown quote card template: {text.strip()}",
            details={"template": text.strip(), "available": list(library.names())},
        )
    if source is not None:
        raise QuoteCardTemplateError(
            f"cannot load quote card templates from {type(source).__name__}"
        )
    return QuoteCardTemplateLibrary(
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        environ=environ,
    ).templates()


def load_quote_card_template(
    source: Any = None,
    *,
    name: str | None = None,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> QuoteCardTemplate:
    """Load one named quote card template.

    A source mapping or file may hold a catalog; *name* selects from it. When
    the source is a string that is not an existing path it is treated as the
    template name.
    """
    if isinstance(source, QuoteCardTemplate):
        return source
    if isinstance(source, (str, Path)):
        candidate = Path(source).expanduser()
        if (
            not candidate.is_file()
            and Path(str(source)).suffix.lower() not in _TEMPLATE_FILE_SUFFIXES
        ):
            library = QuoteCardTemplateLibrary(
                config_dir=config_dir,
                defaults_dir=defaults_dir,
                environ=environ,
            )
            if name is not None:
                return library.load(name)
            return library.load(str(source))
        catalog = load_quote_card_templates(
            source,
            config_dir=config_dir,
            defaults_dir=defaults_dir,
            environ=environ,
        )
        return catalog.require(name or "")
    if isinstance(source, (Mapping, QuoteCardTemplates)):
        catalog = load_quote_card_templates(
            source,
            config_dir=config_dir,
            defaults_dir=defaults_dir,
            environ=environ,
        )
        return catalog.require(name or "")
    if source is None:
        library = QuoteCardTemplateLibrary(
            config_dir=config_dir,
            defaults_dir=defaults_dir,
            environ=environ,
        )
        return library.load(name or "")
    raise QuoteCardTemplateError(
        f"cannot load a quote card template from {type(source).__name__}"
    )


def load_packaged_quote_card_templates(
    *,
    defaults_dir: str | Path | None = None,
) -> QuoteCardTemplates:
    """Load only the packaged templates, ignoring user overrides."""
    base = (
        paths.package_defaults_dir()
        if defaults_dir is None
        else Path(defaults_dir).expanduser().resolve()
    )
    return _load_directory(base / QUOTE_CARD_TEMPLATES_DIR)


def load_default_quote_card_templates(
    *,
    defaults_dir: str | Path | None = None,
) -> QuoteCardTemplates:
    """Alias for :func:`load_packaged_quote_card_templates`."""
    return load_packaged_quote_card_templates(defaults_dir=defaults_dir)


default_quote_card_templates = load_default_quote_card_templates
quote_card_templates = load_quote_card_templates
quote_card_template = load_quote_card_template
load_typography_templates = load_quote_card_templates
load_quote_card_typography = load_quote_card_templates
typography_templates = load_quote_card_templates
quote_card_typography = load_quote_card_templates


def _quote_values(
    source: Any,
    *,
    attribution: str | None = None,
    title: str | None = None,
    context: str = "",
    language: str = "",
) -> dict[str, str]:
    from aivia.writing.quote import extract_quote
    from aivia.writing.schema import QuotePayload, Script

    script: Script | None = source if isinstance(source, Script) else None
    if script is not None:
        payload = (
            script.payload
            if isinstance(script.payload, QuotePayload)
            else extract_quote(script)
        )
        beat_text = script.beats[0].text if script.beats else ""
        resolved_title = title if title is not None else script.title
        resolved_language = language or script.language
        resolved_attribution = (
            attribution
            if attribution is not None
            else getattr(payload, "attribution", "")
            or script.outro.lstrip("—- ").strip()
        )
        resolved_context = context or getattr(payload, "context", "")
        quote = getattr(payload, "quote", "") or beat_text
    else:
        payload = source if isinstance(source, QuotePayload) else extract_quote(source)
        resolved_title = title or ""
        resolved_language = language or "en"
        resolved_attribution = (
            attribution if attribution is not None else payload.attribution
        )
        resolved_context = context or payload.context
        quote = payload.quote
    if not quote.strip():
        raise QuoteCardError(
            "quote card needs a quote",
            details={"reason": "quote_missing"},
        )
    return {
        "quote": quote.strip(),
        "attribution": (resolved_attribution or "").strip(),
        "title": (resolved_title or "").strip(),
        "context": (resolved_context or "").strip(),
        "language": (resolved_language or "").strip(),
    }


def quote_card_composition(
    source: Any,
    template: Any = None,
    *,
    attribution: str | None = None,
    title: str | None = None,
    context: str = "",
    language: str = "",
    duration_seconds: float | None = None,
    platform: str = "",
    composition_id: str = "",
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Composition:
    """Build the still composition for *source* using a typography template.

    *source* may be a quote string, a mapping, a :class:`QuotePayload` or a
    written :class:`Script`. *template* may be a name, a template, a mapping or
    a file path; the packaged ``classic`` template is used when it is omitted.
    """
    resolved = (
        template
        if isinstance(template, QuoteCardTemplate)
        else load_quote_card_template(
            template,
            config_dir=config_dir,
            defaults_dir=defaults_dir,
            environ=environ,
        )
    )
    values = _quote_values(
        source,
        attribution=attribution,
        title=title,
        context=context,
        language=language,
    )
    return resolved.composition(
        values["quote"],
        values["attribution"],
        title=values["title"],
        context=values["context"],
        language=values["language"],
        duration_seconds=duration_seconds,
        platform=platform,
        composition_id=composition_id,
    )


quote_card = quote_card_composition
build_quote_card = quote_card_composition
build_quote_card_composition = quote_card_composition
compose_quote_card = quote_card_composition


def render_quote_card(
    source: Any,
    template: Any = None,
    *,
    output: str | Path | None = None,
    time_seconds: float = 0.0,
    **kwargs: Any,
) -> Image.Image:
    """Render *source* as a quote card still and optionally save it.

    Extra keyword arguments are forwarded to :func:`quote_card_composition`.
    When *output* is given the image is saved in the format named by its
    suffix, defaulting to PNG.
    """
    composition = quote_card_composition(source, template, **kwargs)
    image = render_still(composition, time_seconds)
    if output is not None:
        path = Path(output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
    return image


render_quote_card_still = render_quote_card


def save_quote_card(
    source: Any,
    output: str | Path,
    template: Any = None,
    *,
    time_seconds: float = 0.0,
    **kwargs: Any,
) -> Path:
    """Render *source* to *output* and return the written path."""
    path = Path(output).expanduser()
    render_quote_card(
        source,
        template,
        output=path,
        time_seconds=time_seconds,
        **kwargs,
    )
    return path


write_quote_card = save_quote_card


class QuoteCardRenderer:
    """Build and render quote cards from one template.

    The renderer is a small convenience wrapper around the module functions so
    a caller can pin a template once and render many quotes with it.
    """

    def __init__(
        self,
        template: Any = None,
        *,
        config_dir: str | Path | None = None,
        defaults_dir: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.template = load_quote_card_template(
            template,
            config_dir=config_dir,
            defaults_dir=defaults_dir,
            environ=environ,
        )
        self.config_dir = config_dir
        self.defaults_dir = defaults_dir
        self.environ = environ

    def composition(self, source: Any, **kwargs: Any) -> Composition:
        """Build a composition for *source* with the pinned template."""
        return quote_card_composition(
            source,
            self.template,
            config_dir=self.config_dir,
            defaults_dir=self.defaults_dir,
            environ=self.environ,
            **kwargs,
        )

    compose = composition
    build = composition

    def render(
        self,
        source: Any,
        *,
        output: str | Path | None = None,
        time_seconds: float = 0.0,
        **kwargs: Any,
    ) -> Image.Image:
        """Render *source* with the pinned template and optionally save it."""
        return render_quote_card(
            source,
            self.template,
            output=output,
            time_seconds=time_seconds,
            config_dir=self.config_dir,
            defaults_dir=self.defaults_dir,
            environ=self.environ,
            **kwargs,
        )

    def save(
        self,
        source: Any,
        output: str | Path,
        *,
        time_seconds: float = 0.0,
        **kwargs: Any,
    ) -> Path:
        """Render *source* with the pinned template to *output*."""
        return save_quote_card(
            source,
            output,
            self.template,
            time_seconds=time_seconds,
            config_dir=self.config_dir,
            defaults_dir=self.defaults_dir,
            environ=self.environ,
            **kwargs,
        )


QuoteCardFormat = QuoteCardRenderer
"""Alias for :class:`QuoteCardRenderer`."""

QuoteCard = QuoteCardTemplate
"""Alias for :class:`QuoteCardTemplate`."""

TypographyTemplate = QuoteCardTemplate
"""Alias for :class:`QuoteCardTemplate`."""

TypographyTemplates = QuoteCardTemplates
"""Alias for :class:`QuoteCardTemplates`."""

TypographyTemplateLibrary = QuoteCardTemplateLibrary
"""Alias for :class:`QuoteCardTemplateLibrary`."""

TypographyTemplateStore = QuoteCardTemplates
"""Alias for :class:`QuoteCardTemplates`."""

load_templates = load_quote_card_templates
load_template = load_quote_card_template
default_templates = load_default_quote_card_templates
render_template = render_quote_card
write_template = save_quote_card
quote_card_renderer = QuoteCardRenderer
