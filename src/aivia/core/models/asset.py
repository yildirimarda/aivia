"""Asset records for generated, downloaded and user supplied files.

An asset is the immutable record of one file used or produced by a run. It
keeps the provenance needed to reproduce and audit it: the provider and model
that made it, the prompt and a hash of that prompt, the license under which it
may be used and links to the cost events it produced. Analysis results and the
written alt text live on the same record.

``pinned`` marks final outputs that retention must never remove. The models
only describe data; they never read files, call providers or import the media,
publishing or assistant modules.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Self

from pydantic import AliasChoices, Field, field_validator, model_validator

from aivia.core.ids import new_asset_id
from aivia.core.models.base import ModelBase, canonical_json

__all__ = [
    "Asset",
    "AssetAnalysis",
    "AssetCost",
    "AssetCostLink",
    "AssetKind",
    "AssetLicense",
    "AssetModel",
    "AssetRecord",
    "CostLink",
    "Face",
    "FaceRegion",
    "License",
    "compute_prompt_hash",
]


class AssetKind(StrEnum):
    """The kind of file an asset record describes."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    MUSIC = "music"
    SFX = "sfx"
    VOICE = "voice"
    STOCK = "stock"
    TEXT = "text"
    FONT = "font"
    OTHER = "other"


_KIND_ALIASES = {
    "picture": AssetKind.IMAGE,
    "photo": AssetKind.IMAGE,
    "still": AssetKind.IMAGE,
    "clip": AssetKind.VIDEO,
    "animation": AssetKind.VIDEO,
    "sound": AssetKind.AUDIO,
    "sound_effect": AssetKind.SFX,
    "effect": AssetKind.SFX,
    "music_track": AssetKind.MUSIC,
    "song": AssetKind.MUSIC,
    "speech": AssetKind.VOICE,
    "voiceover": AssetKind.VOICE,
    "voice_over": AssetKind.VOICE,
    "font_file": AssetKind.FONT,
    "file": AssetKind.OTHER,
    "document": AssetKind.TEXT,
}

_OPTIONAL_TEXT_FIELDS = (
    "label",
    "path",
    "content_hash",
    "provider",
    "model",
    "prompt",
    "prompt_hash",
    "alt_text",
    "run_id",
)


def compute_prompt_hash(
    prompt: str,
    *,
    parameters: Mapping[str, Any] | None = None,
) -> str:
    """Return a stable SHA-256 hash for *prompt* and optional parameters.

    The hash is computed over canonical JSON, so the same prompt and parameter
    mapping always produce the same digest. Whitespace around the prompt is
    ignored. When *parameters* is omitted only the prompt contributes.
    """
    payload: dict[str, Any] = {"prompt": prompt.strip()}
    if parameters is not None:
        payload["parameters"] = dict(parameters)
    encoded = canonical_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _coerce_license(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, AssetLicense):
        return value
    if isinstance(value, str):
        return {"name": value}
    return value


def _coerce_cost_links(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, (str, AssetCostLink)):
        return [value]
    if isinstance(value, Mapping):
        return [value]
    if not isinstance(value, Sequence):
        return value
    links: list[Any] = []
    for raw in value:
        if isinstance(raw, str):
            links.append({"cost_event_id": raw})
        else:
            links.append(raw)
    return links


def _coerce_faces(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, (FaceRegion, Mapping)):
        return [value]
    return value


def _coerce_colors(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return value


def _coerce_detected_text(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)
    return value


class AssetLicense(ModelBase):
    """The rights under which an asset may be used.

    ``name`` is the human readable license name and ``url`` points at its
    terms. ``attribution`` holds the exact credit line a publisher must show
    when ``attribution_required`` is set. ``commercial_use`` is ``None`` when
    the license does not state it, which callers must treat as not cleared.
    """

    name: str = Field(
        default="",
        validation_alias=AliasChoices("name", "license", "licence", "spdx", "type"),
    )
    url: str = Field(
        default="",
        validation_alias=AliasChoices("url", "license_url", "terms_url", "link"),
    )
    attribution: str = Field(
        default="",
        validation_alias=AliasChoices(
            "attribution",
            "credit",
            "attribution_text",
            "required_attribution",
        ),
    )
    commercial_use: bool | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "commercial_use",
            "commercial",
            "allow_commercial",
            "commercial_allowed",
        ),
    )
    attribution_required: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "attribution_required",
            "requires_attribution",
            "require_attribution",
        ),
    )
    share_alike: bool = Field(
        default=False,
        validation_alias=AliasChoices("share_alike", "copyleft"),
    )

    @field_validator("name", "url", "attribution", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @property
    def is_known(self) -> bool:
        """Whether the license names itself or links to its terms."""
        return bool(self.name or self.url)

    @property
    def is_commercial(self) -> bool:
        """Whether commercial use is explicitly allowed."""
        return self.commercial_use is True

    @property
    def requires_attribution(self) -> bool:
        """Whether a credit line must accompany the asset."""
        return self.attribution_required or bool(self.attribution)

    @property
    def credit(self) -> str:
        """Alias for :attr:`attribution`."""
        return self.attribution


class FaceRegion(ModelBase):
    """A rectangle around one detected face, in image pixels."""

    x: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("x", "left"),
    )
    y: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("y", "top"),
    )
    width: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("width", "w"),
    )
    height: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("height", "h"),
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @property
    def area(self) -> float:
        """The area of the rectangle in square pixels."""
        return self.width * self.height

    @property
    def box(self) -> tuple[float, float, float, float]:
        """The region as ``(x, y, width, height)``."""
        return (self.x, self.y, self.width, self.height)


class AssetAnalysis(ModelBase):
    """What analysis learned about an asset's pixels or audio.

    Every field is optional because analysis may run before or after an asset
    is stored. ``width`` and ``height`` are pixels, ``duration_seconds`` only
    applies to timed media, ``colors`` holds dominant colors in first-seen
    order, ``description`` is a vision description, ``detected_text`` is OCR
    output and ``faces`` holds the detected face rectangles.
    """

    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    duration_seconds: float | None = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices("duration_seconds", "duration", "seconds"),
    )
    colors: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("colors", "dominant_colors", "palette"),
    )
    description: str = Field(
        default="",
        validation_alias=AliasChoices(
            "description",
            "vision_description",
            "caption",
            "summary",
        ),
    )
    detected_text: str = Field(
        default="",
        validation_alias=AliasChoices("detected_text", "ocr_text", "text", "ocr"),
    )
    faces: list[FaceRegion] = Field(
        default_factory=list,
        validation_alias=AliasChoices("faces", "face_regions"),
    )

    @field_validator("colors", mode="before")
    @classmethod
    def _validate_colors(cls, value: Any) -> Any:
        colors = _coerce_colors(value)
        if not isinstance(colors, list):
            return colors
        normalized: list[str] = []
        for raw in colors:
            if not isinstance(raw, str):
                normalized.append(raw)
                continue
            color = raw.strip()
            if color and color not in normalized:
                normalized.append(color)
        return normalized

    @field_validator("description", mode="before")
    @classmethod
    def _strip_description(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("detected_text", mode="before")
    @classmethod
    def _validate_detected_text(cls, value: Any) -> Any:
        return _strip(_coerce_detected_text(value))

    @field_validator("faces", mode="before")
    @classmethod
    def _validate_faces(cls, value: Any) -> Any:
        return _coerce_faces(value)

    @property
    def aspect_ratio(self) -> float | None:
        """Width divided by height, or ``None`` until both are known."""
        if not self.width or not self.height:
            return None
        return self.width / self.height

    @property
    def is_landscape(self) -> bool:
        """Whether the asset is wider than it is tall."""
        if self.width is None or self.height is None:
            return False
        return self.width > self.height

    @property
    def is_portrait(self) -> bool:
        """Whether the asset is taller than it is wide."""
        if self.width is None or self.height is None:
            return False
        return self.width < self.height

    @property
    def is_square(self) -> bool:
        """Whether the asset has equal width and height."""
        if self.width is None or self.height is None:
            return False
        return self.width == self.height

    @property
    def has_faces(self) -> bool:
        """Whether analysis detected at least one face."""
        return bool(self.faces)

    @property
    def is_analyzed(self) -> bool:
        """Whether any analysis field carries a value."""
        return bool(
            self.width
            or self.height
            or self.duration_seconds
            or self.colors
            or self.description
            or self.detected_text
            or self.faces
        )

    @property
    def dominant_colors(self) -> list[str]:
        """Alias for :attr:`colors`."""
        return self.colors

    @property
    def ocr_text(self) -> str:
        """Alias for :attr:`detected_text`."""
        return self.detected_text

    @property
    def face_regions(self) -> list[FaceRegion]:
        """Alias for :attr:`faces`."""
        return self.faces


class AssetCostLink(ModelBase):
    """A pointer from an asset to one cost event that paid for it.

    ``cost_event_id`` is the only mandatory value. ``amount_usd`` is copied
    for cheap reporting so a caller does not have to join the cost table for
    every asset. The remaining fields mirror the event for the same reason.
    """

    cost_event_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("cost_event_id", "cost_id", "event_id", "id"),
    )
    amount_usd: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices("amount_usd", "cost_usd", "amount", "cost"),
    )
    provider: str = ""
    model: str = ""
    operation: str = Field(
        default="",
        validation_alias=AliasChoices("operation", "stage", "kind"),
    )

    @field_validator("cost_event_id", "provider", "model", "operation", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @property
    def id(self) -> str:
        """Alias for :attr:`cost_event_id`."""
        return self.cost_event_id

    @property
    def amount(self) -> float | None:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd

    @property
    def cost(self) -> float | None:
        """Alias for :attr:`amount_usd`."""
        return self.amount_usd


class Asset(ModelBase):
    """One file used or produced by a run, with its provenance and analysis.

    ``provider`` and ``model`` stay empty for user supplied files; generated
    assets fill them in. ``prompt_hash`` is derived from ``prompt`` when it is
    omitted, so a generated asset cannot lose the value used for caching.
    ``license``, ``cost_links`` and ``analysis`` default to empty records so
    callers can read nested fields without checking for ``None``.
    """

    id: str = Field(
        default_factory=new_asset_id,
        min_length=1,
        validation_alias=AliasChoices("id", "asset_id"),
    )
    kind: AssetKind = Field(
        default=AssetKind.IMAGE,
        validation_alias=AliasChoices("kind", "type", "media_type", "asset_type"),
    )
    label: str = Field(
        default="",
        validation_alias=AliasChoices("label", "name", "title", "slug"),
    )
    path: str = Field(
        default="",
        validation_alias=AliasChoices(
            "path",
            "file",
            "file_path",
            "uri",
            "url",
            "location",
        ),
    )
    content_hash: str = Field(
        default="",
        validation_alias=AliasChoices(
            "content_hash",
            "hash",
            "sha256",
            "checksum",
            "digest",
        ),
    )
    provider: str = Field(
        default="",
        validation_alias=AliasChoices("provider", "vendor"),
    )
    model: str = Field(
        default="",
        validation_alias=AliasChoices("model", "model_id"),
    )
    prompt: str = Field(
        default="",
        validation_alias=AliasChoices("prompt", "prompt_text", "instruction"),
    )
    prompt_hash: str = Field(
        default="",
        validation_alias=AliasChoices(
            "prompt_hash",
            "prompt_sha256",
            "prompt_digest",
        ),
    )
    license: AssetLicense = Field(
        default_factory=AssetLicense,
        validation_alias=AliasChoices("license", "licence", "rights"),
    )
    cost_links: list[AssetCostLink] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "cost_links",
            "costs",
            "cost_events",
            "cost_ids",
        ),
    )
    analysis: AssetAnalysis = Field(
        default_factory=AssetAnalysis,
        validation_alias=AliasChoices("analysis", "metadata", "properties"),
    )
    alt_text: str = Field(
        default="",
        validation_alias=AliasChoices(
            "alt_text",
            "alt",
            "accessibility_text",
        ),
    )
    pinned: bool = Field(
        default=False,
        validation_alias=AliasChoices("pinned", "is_pinned", "final"),
    )
    quality_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("quality_score", "score", "qa_score"),
    )
    run_id: str = Field(
        default="",
        validation_alias=AliasChoices("run_id", "run"),
    )

    @field_validator("kind", mode="before")
    @classmethod
    def _validate_kind(cls, value: Any) -> Any:
        if isinstance(value, AssetKind):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            return _KIND_ALIASES.get(text, text)
        return value

    @field_validator(*_OPTIONAL_TEXT_FIELDS, mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("license", mode="before")
    @classmethod
    def _validate_license(cls, value: Any) -> Any:
        return _coerce_license(value)

    @field_validator("cost_links", mode="before")
    @classmethod
    def _validate_cost_links(cls, value: Any) -> Any:
        return _coerce_cost_links(value)

    @field_validator("analysis", mode="before")
    @classmethod
    def _validate_analysis(cls, value: Any) -> Any:
        return {} if value is None else value

    @model_validator(mode="after")
    def _derive_prompt_hash(self) -> Self:
        if not self.prompt_hash and self.prompt:
            self.prompt_hash = compute_prompt_hash(self.prompt)
        return self

    @property
    def asset_id(self) -> str:
        """Alias for :attr:`id`."""
        return self.id

    @property
    def media_type(self) -> AssetKind:
        """Alias for :attr:`kind`."""
        return self.kind

    @property
    def type(self) -> AssetKind:
        """Alias for :attr:`kind`."""
        return self.kind

    @property
    def name(self) -> str:
        """Alias for :attr:`label`."""
        return self.label

    @property
    def file(self) -> str:
        """Alias for :attr:`path`."""
        return self.path

    @property
    def uri(self) -> str:
        """Alias for :attr:`path`."""
        return self.path

    @property
    def url(self) -> str:
        """Alias for :attr:`path`."""
        return self.path

    @property
    def hash(self) -> str:
        """Alias for :attr:`content_hash`."""
        return self.content_hash

    @property
    def sha256(self) -> str:
        """Alias for :attr:`content_hash`."""
        return self.content_hash

    @property
    def vendor(self) -> str:
        """Alias for :attr:`provider`."""
        return self.provider

    @property
    def model_id(self) -> str:
        """Alias for :attr:`model`."""
        return self.model

    @property
    def costs(self) -> list[AssetCostLink]:
        """Alias for :attr:`cost_links`."""
        return self.cost_links

    @property
    def cost_event_ids(self) -> list[str]:
        """The cost event identifiers linked to this asset."""
        return [link.cost_event_id for link in self.cost_links]

    @property
    def total_cost_usd(self) -> float:
        """The sum of the linked cost amounts, treating unknown as zero."""
        return sum(link.amount_usd or 0.0 for link in self.cost_links)

    @property
    def license_name(self) -> str:
        """The name of the asset's license, or an empty string."""
        return self.license.name

    @property
    def is_pinned(self) -> bool:
        """Alias for :attr:`pinned`."""
        return self.pinned

    @property
    def has_alt_text(self) -> bool:
        """Whether a non-empty alt text was written."""
        return bool(self.alt_text)

    @property
    def is_ai_generated(self) -> bool:
        """Whether the asset records a provider or model."""
        return bool(self.provider or self.model)

    @property
    def width(self) -> int | None:
        """The analyzed pixel width, if known."""
        return self.analysis.width

    @property
    def height(self) -> int | None:
        """The analyzed pixel height, if known."""
        return self.analysis.height

    @property
    def colors(self) -> list[str]:
        """Alias for :attr:`AssetAnalysis.colors`."""
        return self.analysis.colors

    @property
    def description(self) -> str:
        """Alias for :attr:`AssetAnalysis.description`."""
        return self.analysis.description

    @property
    def detected_text(self) -> str:
        """Alias for :attr:`AssetAnalysis.detected_text`."""
        return self.analysis.detected_text

    @property
    def faces(self) -> list[FaceRegion]:
        """Alias for :attr:`AssetAnalysis.faces`."""
        return self.analysis.faces


AssetModel = Asset
AssetRecord = Asset
AssetCost = AssetCostLink
CostLink = AssetCostLink
License = AssetLicense
Face = FaceRegion
