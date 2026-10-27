"""Core configuration sections registered by the base application.

Each top-level key owns one :class:`ConfigModel`. Optional modules register
their own sections with the same registry, so the core set stays free of
module-specific keys. Sections use defaults for every field: an empty mapping
is a valid configuration and later milestones can tighten individual fields
without making existing configs unreadable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel
from aivia.core.config.budget import BudgetConfig, OnExceed
from aivia.core.config.llm import LLMConfig
from aivia.core.config.ops import (
    JobsConfig,
    RetentionConfig,
    ReviewConfig,
    StorageConfig,
)
from aivia.core.config.quality import QualityConfig, ScoringConfig
from aivia.core.config.registry import SectionRegistry, registry
from aivia.core.config.video import (
    DurationRangeConfig,
    DurationRangeOverrideConfig,
    PlatformVideoConfig,
    ProfileVideoConfig,
    Resolution,
    VideoConfig,
    VideoOverride,
    VideoOverridesConfig,
    VideoSpec,
    VideoSpecConfig,
    deep_merge,
    merge_video_specs,
    parse_resolution,
    resolve_video,
)
from aivia.core.errors import ConfigError

__all__ = [
    "CORE_SECTIONS",
    "CORE_SECTION_KEYS",
    "AccessibilityConfig",
    "ApiConfig",
    "BudgetConfig",
    "DurationRangeConfig",
    "DurationRangeOverrideConfig",
    "JobsConfig",
    "LLMConfig",
    "LanguageConfig",
    "LanguagesConfig",
    "OnExceed",
    "PipelineConfig",
    "PlatformAccountConfig",
    "PlatformVideoConfig",
    "PlatformsConfig",
    "PresetConfig",
    "PresetsConfig",
    "PrivacyConfig",
    "ProfileVideoConfig",
    "QualityConfig",
    "RenderConfig",
    "Resolution",
    "RetentionConfig",
    "ReviewConfig",
    "SafetyConfig",
    "ScoringConfig",
    "StorageConfig",
    "TracingConfig",
    "VideoConfig",
    "VideoOverride",
    "VideoOverridesConfig",
    "VideoSpec",
    "VideoSpecConfig",
    "WordsPerSecondConfig",
    "deep_merge",
    "merge_video_specs",
    "parse_resolution",
    "register_core_sections",
    "resolve_video",
]


class WordsPerSecondConfig(ConfigModel):
    """Inclusive speaking-rate bounds used to estimate and check script length."""

    min: float = Field(default=2.0, gt=0)
    max: float = Field(default=3.5, gt=0)

    @model_validator(mode="after")
    def _ordered_bounds(self) -> Self:
        if self.min > self.max:
            raise ValueError("words per second min must not exceed max")
        return self


class LanguageConfig(ConfigModel):
    """Settings for one language written natively from the shared fact base."""

    name: str = ""
    locale: str = "en_US"
    words_per_second: WordsPerSecondConfig = Field(default_factory=WordsPerSecondConfig)
    subtitle_font: str = Field(default="Lato", min_length=1)
    hashtag_style: Literal["lower", "title", "upper"] = "lower"


class LanguagesConfig(ConfigModel):
    """The available languages plus the one used when none is requested.

    A config file may write the language entries directly under the section
    (``languages: {en: {...}, de: {...}}``) or nest them under ``languages``;
    both forms validate to the same model.
    """

    default: str = "en"
    languages: dict[str, LanguageConfig] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_direct_mapping(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and "languages" not in data:
            values = dict(data)
            default = values.pop("default", "en")
            return {"default": default, "languages": values}
        return data


class PipelineConfig(ConfigModel):
    """The ordered stages a pipeline runs and the timezone used for schedules."""

    stages: list[str] = Field(
        default_factory=lambda: [
            "research",
            "write",
            "voice",
            "visuals",
            "render",
            "review",
            "publish",
        ]
    )
    timezone: str = "UTC"


class RenderConfig(ConfigModel):
    """Deterministic renderer settings that never require an AI call."""

    fonts_dir: str = "package"
    require_bundled_fonts: bool = True
    default_font: str = Field(default="Lato", min_length=1)
    fallback_fonts: list[str] = Field(default_factory=lambda: ["Lato"])
    font_size: int = Field(default=48, gt=0)
    min_contrast: float = Field(default=4.5, ge=1, le=21)
    contrast_strategy: Literal["outline", "shadow", "box"] = "outline"

    @field_validator("fallback_fonts")
    @classmethod
    def _validate_fallback_fonts(cls, value: list[str]) -> list[str]:
        fonts = [font.strip() for font in value]
        if any(not font for font in fonts):
            raise ValueError("fallback font names must not be empty")
        return fonts


class ApiConfig(ConfigModel):
    """The local HTTP API bind address, authentication and request limits."""

    host: str = "127.0.0.1"
    port: int = 8765
    require_auth: bool = True
    api_key_ref: str = "AIVA_API_KEY"
    max_body_bytes: int = 10_000_000
    rate_limit_per_minute: int = 60


class SafetyConfig(ConfigModel):
    """Prompt and media screening before a paid or published call."""

    enabled: bool = True
    screen_prompts: bool = True
    screen_media: bool = True
    block_categories: list[str] = Field(default_factory=list)
    allow_regenerate: bool = True


class PlatformAccountConfig(ConfigModel):
    """One publishing account, bound to exactly one platform and language."""

    platform: str = ""
    language: str = ""
    enabled: bool = True
    credential_refs: list[str] = Field(default_factory=list)


class PlatformsConfig(ConfigModel):
    """Publishing accounts keyed by ``platform/language``.

    Entries may be written directly under the section
    (``platforms: {youtube/en: {...}}``) or nested under ``accounts``.
    """

    accounts: dict[str, PlatformAccountConfig] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_direct_mapping(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and "accounts" not in data:
            return {"accounts": dict(data)}
        return data


class PresetConfig(ConfigModel):
    """A named set of configuration overrides selected by the CLI or a profile."""

    description: str = ""
    overrides: dict[str, Any] = Field(default_factory=dict)


class PresetsConfig(ConfigModel):
    """Named presets keyed by name.

    Entries may be written directly under the section
    (``presets: {short: {...}}``) or nested under ``presets``.
    """

    presets: dict[str, PresetConfig] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_direct_mapping(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and "presets" not in data:
            return {"presets": dict(data)}
        return data


class PrivacyConfig(ConfigModel):
    """Redaction and personal-data handling for logs, prompts and sidecars."""

    redact_logs: bool = True
    redact_prompts: bool = False
    store_prompt_text: bool = True
    allow_personal_data: bool = False
    placeholder_mode: bool = False
    data_processing_record: bool = True


class TracingConfig(ConfigModel):
    """Optional tracing export; disabled by default and vendor neutral."""

    enabled: bool = False
    exporter: Literal["none", "otlp"] = "none"
    endpoint: str | None = None
    service_name: str = "aivia"
    sample_ratio: float = 1.0


class AccessibilityConfig(ConfigModel):
    """Alt text, subtitles and contrast settings for accessible outputs."""

    enabled: bool = True
    alt_text: bool = True
    subtitles: bool = True
    min_contrast: float = Field(default=4.5, ge=1, le=21)
    subtitle_font: str = Field(default="Lato", min_length=1)
    subtitle_font_size: int = Field(default=32, gt=0)
    subtitle_outline: bool = True
    subtitle_background: bool = False


CORE_SECTIONS: tuple[tuple[str, type[ConfigModel]], ...] = (
    ("languages", LanguagesConfig),
    ("pipeline", PipelineConfig),
    ("video", VideoConfig),
    ("render", RenderConfig),
    ("budget", BudgetConfig),
    ("llm", LLMConfig),
    ("quality", QualityConfig),
    ("scoring", ScoringConfig),
    ("storage", StorageConfig),
    ("api", ApiConfig),
    ("jobs", JobsConfig),
    ("safety", SafetyConfig),
    ("platforms", PlatformsConfig),
    ("presets", PresetsConfig),
    ("retention", RetentionConfig),
    ("review", ReviewConfig),
    ("privacy", PrivacyConfig),
    ("tracing", TracingConfig),
    ("accessibility", AccessibilityConfig),
)

CORE_SECTION_KEYS: tuple[str, ...] = tuple(sorted(key for key, _ in CORE_SECTIONS))


def register_core_sections(
    sections: SectionRegistry | None = None,
) -> SectionRegistry:
    """Register every core section on *sections* and return the registry.

    The operation is idempotent for the core models, so importing the package
    and calling this explicitly cannot register the same key twice. A key
    already owned by a different model is still a collision.
    """
    target = registry if sections is None else sections
    for key, model in CORE_SECTIONS:
        if key in target:
            existing = target.get(key)
            if existing.model is not model:
                raise ConfigError(
                    f"config section already registered: {key}",
                    details={
                        "key": key,
                        "existing": existing.model.__name__,
                        "new": model.__name__,
                    },
                )
            continue
        target.register(key, model)
    return target


register_core_sections()
