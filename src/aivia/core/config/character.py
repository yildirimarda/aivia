"""Character and brand kit configuration.

A character is a reusable figure whose look, voice and music stay consistent
across outputs. Characters are strictly opt-in: selecting one is a profile,
order or instruction decision, while this section configures how the shared
character machinery behaves when a character is selected.

``consistency_strategies`` is the ordered chain tried per scene. The first
strategy that produces an acceptable result wins, so the default order is
``composite -> reference -> adapter -> prompt_only``. ``prompt_only`` is a last
resort and may only appear at the end of the chain.

``qa`` holds the thresholds used to accept a generated asset: an overall
minimum score, a vision feature threshold and an embedding similarity
threshold, plus the number of retries before moving to the next strategy.

``consent`` keeps rights decisions explicit. Voice cloning and real-person
likeness are off by default, and ownership is required unless the user opts
out. ``resemblance_warning`` keeps the advisory warning enabled.

A brand kit holds the reusable branding fields: colours, fonts, logo,
watermark, intro and outro clips and the caption style. An empty kit is valid;
binding it to a profile, order or preset is a separate decision.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import Field, ValidationInfo, field_validator, model_validator

from aivia.core.config.base import ConfigModel

__all__ = [
    "BrandConfig",
    "BrandKit",
    "BrandKitConfig",
    "BrandingConfig",
    "BrandsConfig",
    "CharacterConfig",
    "CharacterConsent",
    "CharacterConsentConfig",
    "CharacterConsistencyStrategy",
    "CharacterQA",
    "CharacterQAConfig",
    "CharacterQaConfig",
    "ConsentConfig",
    "ConsistencyStrategy",
    "ConsistencyStrategyOrder",
    "QAConfig",
]


class ConsistencyStrategy(StrEnum):
    """How a scene may be made consistent with a character."""

    COMPOSITE = "composite"
    REFERENCE = "reference"
    ADAPTER = "adapter"
    PROMPT_ONLY = "prompt_only"


CharacterConsistencyStrategy = ConsistencyStrategy
ConsistencyStrategyOrder = ConsistencyStrategy

_STRATEGY_ALIASES = {
    "composite": ConsistencyStrategy.COMPOSITE,
    "compose": ConsistencyStrategy.COMPOSITE,
    "compositing": ConsistencyStrategy.COMPOSITE,
    "cutout": ConsistencyStrategy.COMPOSITE,
    "cutouts": ConsistencyStrategy.COMPOSITE,
    "reference": ConsistencyStrategy.REFERENCE,
    "reference_image": ConsistencyStrategy.REFERENCE,
    "reference_images": ConsistencyStrategy.REFERENCE,
    "ref": ConsistencyStrategy.REFERENCE,
    "adapter": ConsistencyStrategy.ADAPTER,
    "adapters": ConsistencyStrategy.ADAPTER,
    "trainer": ConsistencyStrategy.ADAPTER,
    "lora": ConsistencyStrategy.ADAPTER,
    "fine_tune": ConsistencyStrategy.ADAPTER,
    "prompt_only": ConsistencyStrategy.PROMPT_ONLY,
    "prompt-only": ConsistencyStrategy.PROMPT_ONLY,
    "prompt": ConsistencyStrategy.PROMPT_ONLY,
    "text": ConsistencyStrategy.PROMPT_ONLY,
}

_DEFAULT_STRATEGIES = (
    ConsistencyStrategy.COMPOSITE,
    ConsistencyStrategy.REFERENCE,
    ConsistencyStrategy.ADAPTER,
    ConsistencyStrategy.PROMPT_ONLY,
)

_CHARACTER_ALIASES: dict[str, tuple[str, ...]] = {
    "enabled": ("enable", "active"),
    "consistency_strategies": (
        "strategy_order",
        "consistency_order",
        "consistency_strategy_order",
        "consistency_chain",
        "strategy_chain",
        "strategies",
        "chain",
        "order",
    ),
    "qa": ("qa_thresholds", "quality", "thresholds", "qa_config"),
    "consent": ("consents", "consent_flags", "permissions"),
    "max_reference_images": (
        "max_references",
        "max_reference_count",
        "reference_image_limit",
        "max_images",
    ),
}

_QA_ALIASES: dict[str, tuple[str, ...]] = {
    "min_score": (
        "threshold",
        "min_consistency",
        "consistency_threshold",
        "minimum_score",
        "min_similarity",
        "similarity_threshold",
    ),
    "vision_threshold": (
        "vision_similarity",
        "vision_limit",
        "min_vision_score",
        "vision_score",
    ),
    "embedding_threshold": (
        "embedding_similarity",
        "embedding_limit",
        "min_embedding_score",
        "embedding_score",
    ),
    "max_retries": (
        "retries",
        "retry_limit",
        "max_regenerations",
        "regeneration_limit",
    ),
}

_CONSENT_ALIASES: dict[str, tuple[str, ...]] = {
    "voice_cloning": (
        "voice_clone",
        "allow_voice_cloning",
        "voice_clone_consent",
        "voice_cloning_consent",
        "cloning",
    ),
    "real_person_likeness": (
        "likeness",
        "real_person",
        "allow_likeness",
        "likeness_consent",
        "real_person_consent",
    ),
    "resemblance_warning": (
        "resemblance_warnings",
        "warn_on_resemblance",
        "likeness_warning",
    ),
    "require_ownership": (
        "ownership_required",
        "require_owned",
        "ownership",
    ),
}

_BRAND_ALIASES: dict[str, tuple[str, ...]] = {
    "name": ("title", "id"),
    "colors": ("color", "colour", "colours", "palette", "color_palette"),
    "fonts": ("font", "font_family", "typefaces", "typography"),
    "logo": ("logo_path", "logo_file"),
    "watermark": ("watermark_path", "watermark_file"),
    "intro": ("intro_clip", "intro_file", "intro_path"),
    "outro": ("outro_clip", "outro_file", "outro_path"),
    "caption_style": ("caption", "captions", "caption_style_name"),
}


def _accept_aliases(data: Any, aliases: Mapping[str, tuple[str, ...]]) -> Any:
    if not isinstance(data, Mapping):
        return data
    values = dict(data)
    for canonical, names in aliases.items():
        for name in names:
            if name in values:
                value = values.pop(name)
                values.setdefault(canonical, value)
    return values


def _finite_threshold(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _normalise_names(value: Any, label: str) -> Any:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, Mapping):
        names: dict[str, str] = {}
        for key, entry in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{label} names must be non-empty")
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(f"{label} values must be non-empty")
            names[key.strip()] = entry.strip()
        return names
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        entries: list[str] = []
        seen: set[str] = set()
        for entry in value:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(f"{label} entries must be non-empty")
            name = entry.strip()
            if name in seen:
                raise ValueError(f"duplicate {label}: {name}")
            seen.add(name)
            entries.append(name)
        return entries
    return value


class CharacterQAConfig(ConfigModel):
    """Thresholds and retries used to accept a character-consistent asset."""

    min_score: float = Field(default=0.8, ge=0, le=1)
    vision_threshold: float = Field(default=0.8, ge=0, le=1)
    embedding_threshold: float = Field(default=0.75, ge=0, le=1)
    max_retries: int = Field(default=2, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        return _accept_aliases(data, _QA_ALIASES)

    @field_validator("min_score", "vision_threshold", "embedding_threshold")
    @classmethod
    def _finite_thresholds(cls, value: float) -> float:
        return _finite_threshold(value, "character QA threshold")

    @property
    def threshold(self) -> float:
        """Alias for :attr:`min_score`."""
        return self.min_score

    @property
    def min_consistency(self) -> float:
        """Alias for :attr:`min_score`."""
        return self.min_score

    @property
    def min_similarity(self) -> float:
        """Alias for :attr:`min_score`."""
        return self.min_score

    @property
    def vision_similarity(self) -> float:
        """Alias for :attr:`vision_threshold`."""
        return self.vision_threshold

    @property
    def embedding_similarity(self) -> float:
        """Alias for :attr:`embedding_threshold`."""
        return self.embedding_threshold

    @property
    def retries(self) -> int:
        """Alias for :attr:`max_retries`."""
        return self.max_retries


CharacterQaConfig = CharacterQAConfig
CharacterQA = CharacterQAConfig
QAConfig = CharacterQAConfig


class CharacterConsentConfig(ConfigModel):
    """Rights flags that gate voice cloning and real-person likeness."""

    voice_cloning: bool = False
    real_person_likeness: bool = False
    resemblance_warning: bool = True
    require_ownership: bool = True

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        return _accept_aliases(data, _CONSENT_ALIASES)

    @property
    def voice_clone(self) -> bool:
        """Alias for :attr:`voice_cloning`."""
        return self.voice_cloning

    @property
    def likeness(self) -> bool:
        """Alias for :attr:`real_person_likeness`."""
        return self.real_person_likeness

    @property
    def allow_voice_cloning(self) -> bool:
        """Alias for :attr:`voice_cloning`."""
        return self.voice_cloning

    @property
    def allow_likeness(self) -> bool:
        """Alias for :attr:`real_person_likeness`."""
        return self.real_person_likeness

    @property
    def ownership_required(self) -> bool:
        """Alias for :attr:`require_ownership`."""
        return self.require_ownership


CharacterConsent = CharacterConsentConfig
ConsentConfig = CharacterConsentConfig


class CharacterConfig(ConfigModel):
    """Consistency chain, QA thresholds, consent flags and reference limit."""

    enabled: bool = True
    consistency_strategies: list[ConsistencyStrategy] = Field(
        default_factory=lambda: list(_DEFAULT_STRATEGIES)
    )
    qa: CharacterQAConfig = Field(default_factory=CharacterQAConfig)
    consent: CharacterConsentConfig = Field(default_factory=CharacterConsentConfig)
    max_reference_images: int = Field(default=4, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = _accept_aliases(dict(data), _CHARACTER_ALIASES)

        consent_keys = {
            name for names in _CONSENT_ALIASES.values() for name in names
        } | set(_CONSENT_ALIASES)
        direct_consent: dict[str, Any] = {}
        for key in list(values):
            if key in consent_keys and key != "consent":
                direct_consent[key] = values.pop(key)
        if direct_consent:
            consent = values.get("consent")
            if consent is None:
                consent = {}
            if not isinstance(consent, Mapping):
                raise ValueError("consent must be a mapping")
            merged = dict(consent)
            for key, value in direct_consent.items():
                merged.setdefault(key, value)
            values["consent"] = merged

        if values.get("qa") is None:
            values["qa"] = {}
        if values.get("consent") is None:
            values["consent"] = {}
        return values

    @field_validator("consistency_strategies", mode="before")
    @classmethod
    def _normalise_strategies(cls, value: Any) -> Any:
        if value is None:
            return list(_DEFAULT_STRATEGIES)
        if isinstance(value, (str, ConsistencyStrategy)):
            value = [value]
        if not isinstance(value, Sequence) or isinstance(value, (bytes, Mapping)):
            return value
        strategies: list[ConsistencyStrategy] = []
        for entry in value:
            if isinstance(entry, ConsistencyStrategy):
                strategies.append(entry)
                continue
            if not isinstance(entry, str):
                raise ValueError("consistency strategies must be strings")
            key = entry.strip().lower().replace(" ", "_")
            strategy = _STRATEGY_ALIASES.get(key)
            if strategy is None:
                allowed = ", ".join(strategy.value for strategy in ConsistencyStrategy)
                raise ValueError(
                    f"unknown consistency strategy: {entry} (allowed: {allowed})"
                )
            strategies.append(strategy)
        return strategies

    @field_validator("consistency_strategies")
    @classmethod
    def _validate_strategies(
        cls, value: list[ConsistencyStrategy]
    ) -> list[ConsistencyStrategy]:
        if not value:
            raise ValueError("consistency strategies must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("consistency strategies must not repeat")
        if (
            ConsistencyStrategy.PROMPT_ONLY in value
            and value[-1] is not ConsistencyStrategy.PROMPT_ONLY
        ):
            raise ValueError("prompt_only must be the last consistency strategy")
        return value

    @property
    def strategies(self) -> list[ConsistencyStrategy]:
        """Alias for :attr:`consistency_strategies`."""
        return self.consistency_strategies

    @property
    def strategy_order(self) -> list[ConsistencyStrategy]:
        """Alias for :attr:`consistency_strategies`."""
        return self.consistency_strategies

    @property
    def consistency_order(self) -> list[ConsistencyStrategy]:
        """Alias for :attr:`consistency_strategies`."""
        return self.consistency_strategies

    @property
    def consistency_strategy_order(self) -> list[ConsistencyStrategy]:
        """Alias for :attr:`consistency_strategies`."""
        return self.consistency_strategies

    @property
    def qa_thresholds(self) -> CharacterQAConfig:
        """Alias for :attr:`qa`."""
        return self.qa

    @property
    def consent_flags(self) -> CharacterConsentConfig:
        """Alias for :attr:`consent`."""
        return self.consent

    @property
    def reference_image_limit(self) -> int:
        """Alias for :attr:`max_reference_images`."""
        return self.max_reference_images

    @property
    def voice_cloning(self) -> bool:
        """Whether voice cloning is consented to."""
        return self.consent.voice_cloning

    @property
    def voice_clone(self) -> bool:
        """Alias for :attr:`voice_cloning`."""
        return self.consent.voice_cloning

    @property
    def likeness(self) -> bool:
        """Whether real-person likeness is consented to."""
        return self.consent.real_person_likeness


class BrandKitConfig(ConfigModel):
    """Colours, fonts, logo, watermark, intro/outro and caption style."""

    name: str = ""
    colors: list[str] | dict[str, str] = Field(default_factory=list)
    fonts: list[str] | dict[str, str] = Field(default_factory=list)
    logo: str = ""
    watermark: str = ""
    intro: str = ""
    outro: str = ""
    caption_style: str = ""

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        return _accept_aliases(data, _BRAND_ALIASES)

    @field_validator("colors", "fonts", mode="before")
    @classmethod
    def _coerce_names(cls, value: Any, info: ValidationInfo) -> Any:
        return _normalise_names(value, info.field_name or "brand")

    @field_validator("name", "logo", "watermark", "intro", "outro", "caption_style")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()

    @property
    def palette(self) -> list[str] | dict[str, str]:
        """Alias for :attr:`colors`."""
        return self.colors

    @property
    def typefaces(self) -> list[str] | dict[str, str]:
        """Alias for :attr:`fonts`."""
        return self.fonts

    @property
    def intro_clip(self) -> str:
        """Alias for :attr:`intro`."""
        return self.intro

    @property
    def outro_clip(self) -> str:
        """Alias for :attr:`outro`."""
        return self.outro

    @property
    def captions(self) -> str:
        """Alias for :attr:`caption_style`."""
        return self.caption_style


BrandConfig = BrandKitConfig
BrandKit = BrandKitConfig
BrandingConfig = BrandKitConfig
BrandsConfig = BrandKitConfig
