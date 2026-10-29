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
from aivia.core.config.character import BrandKitConfig, CharacterConfig
from aivia.core.config.circuit import CircuitBreakerConfig
from aivia.core.config.database import DatabaseConfig
from aivia.core.config.formats import FormatsConfig
from aivia.core.config.http import HttpClientConfig
from aivia.core.config.llm import LLMConfig
from aivia.core.config.models_file import ModelsFileConfig
from aivia.core.config.ops import (
    JobsConfig,
    RetentionConfig,
    ReviewConfig,
    StorageConfig,
)
from aivia.core.config.orders import DeliveryConfig, OccasionsConfig, OrdersConfig
from aivia.core.config.planner import PlannerConfig
from aivia.core.config.profile import ProfilesConfig
from aivia.core.config.providers import ProviderConfig, ProvidersConfig
from aivia.core.config.quality import QualityConfig, ScoringConfig
from aivia.core.config.ratelimit import RateLimiterConfig
from aivia.core.config.reconcile import ReconcileConfig
from aivia.core.config.reference import ReferenceConfig
from aivia.core.config.registry import SectionRegistry, registry
from aivia.core.config.schedule import (
    PublishAccountConfig,
    PublishConfig,
    ScheduleConfig,
    ScheduleEntryConfig,
    profile_names,
)
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
    "BrandKitConfig",
    "BudgetConfig",
    "CharacterConfig",
    "CircuitBreakerConfig",
    "DatabaseConfig",
    "DeliveryConfig",
    "DurationRangeConfig",
    "DurationRangeOverrideConfig",
    "FormatsConfig",
    "HttpClientConfig",
    "JobsConfig",
    "LLMConfig",
    "LanguageConfig",
    "LanguagesConfig",
    "OccasionsConfig",
    "OnExceed",
    "OrdersConfig",
    "PipelineConfig",
    "PlannerConfig",
    "PlatformAccountConfig",
    "PlatformVideoConfig",
    "PlatformsConfig",
    "PresetConfig",
    "PresetsConfig",
    "PrivacyConfig",
    "ProfileVideoConfig",
    "ProfilesConfig",
    "ProviderConfig",
    "ProvidersConfig",
    "PublishAccountConfig",
    "PublishConfig",
    "QualityConfig",
    "RateLimiterConfig",
    "ReconcileConfig",
    "ReferenceConfig",
    "RenderConfig",
    "Resolution",
    "RetentionConfig",
    "ReviewConfig",
    "SafetyConfig",
    "ScheduleConfig",
    "ScheduleEntryConfig",
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

    @model_validator(mode="after")
    def _default_exists(self) -> Self:
        if not self.languages:
            raise ValueError(
                "languages.default requires at least one configured language"
            )
        if self.default not in self.languages:
            available = ", ".join(sorted(self.languages))
            raise ValueError(
                f"languages.default is not a configured language: "
                f"{self.default} (available: {available})"
            )
        return self


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
    port: int = Field(default=8765, ge=1, le=65535)
    require_auth: bool = True
    api_key_ref: str = "AIVA_API_KEY"
    max_body_bytes: int = Field(default=10_000_000, gt=0)
    rate_limit_per_minute: int = Field(default=60, ge=0)


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
    """A named set of configuration overrides selected by the CLI or a profile.

    A preset may name the content profile it belongs to. When the configured
    profiles are available, a non-empty profile is checked against them so a
    typo never silently selects the default profile.
    """

    name: str = ""
    description: str = ""
    profile: str = ""
    profiles: list[str] = Field(default_factory=list)
    format: str = ""
    brand_kit: str = ""
    character: str = ""
    accounts: list[str] = Field(default_factory=list)
    enabled: bool = True
    overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias in ("id", "key", "title"):
            if alias in values:
                values.setdefault("name", values.pop(alias))
        for alias in ("profile_name", "profile_id", "content_profile"):
            if alias in values:
                values.setdefault("profile", values.pop(alias))
        for alias in ("recipe", "format_name"):
            if alias in values:
                values.setdefault("format", values.pop(alias))
        for alias in ("brand", "brand_id"):
            if alias in values:
                values.setdefault("brand_kit", values.pop(alias))
        for alias in ("character_id", "persona"):
            if alias in values:
                values.setdefault("character", values.pop(alias))
        for alias in ("account", "publish_accounts"):
            if alias in values:
                values.setdefault("accounts", values.pop(alias))
        for alias in ("config", "config_overrides", "settings"):
            if alias in values:
                values.setdefault("overrides", values.pop(alias))
        if isinstance(values.get("accounts"), Mapping):
            values["accounts"] = list(values["accounts"])
        elif isinstance(values.get("accounts"), str):
            values["accounts"] = [values["accounts"]]
        if isinstance(values.get("profiles"), Mapping):
            values["profiles"] = list(values["profiles"])
        elif isinstance(values.get("profiles"), str):
            values["profiles"] = [values["profiles"]]
        if values.get("overrides") is None:
            values["overrides"] = {}
        return values

    @field_validator("name", "profile", "format", "brand_kit", "character")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("accounts")
    @classmethod
    def _validate_accounts(cls, value: list[str]) -> list[str]:
        accounts: list[str] = []
        seen: set[str] = set()
        for entry in value:
            account = entry.strip() if isinstance(entry, str) else ""
            if not account:
                raise ValueError("preset account names must be non-empty")
            if account in seen:
                raise ValueError(f"duplicate preset account: {account}")
            seen.add(account)
            accounts.append(account)
        return accounts

    @field_validator("profiles")
    @classmethod
    def _validate_profiles(cls, value: list[str]) -> list[str]:
        profiles: list[str] = []
        seen: set[str] = set()
        for entry in value:
            profile = entry.strip() if isinstance(entry, str) else ""
            if not profile:
                raise ValueError("preset profile names must be non-empty")
            if profile in seen:
                raise ValueError(f"duplicate preset profile: {profile}")
            seen.add(profile)
            profiles.append(profile)
        return profiles

    @field_validator("overrides", mode="before")
    @classmethod
    def _accept_overrides(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("preset overrides must be a mapping")
        return dict(value)


class PresetsConfig(ConfigModel):
    """Named presets keyed by name, with an optional default preset.

    Entries may be written directly under the section
    (``presets: {short: {...}}``) or nested under ``presets``.
    """

    enabled: bool = True
    default: str = ""
    presets: dict[str, PresetConfig] = Field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        environ: Mapping[str, str] | None = None,
        context: Mapping[str, Any] | None = None,
        profiles: Any = None,
    ) -> Self:
        """Validate presets and optionally check the selected profiles."""
        merged = dict(context or {})
        if profiles is not None:
            merged["profiles"] = profiles
        return super().from_mapping(values, environ=environ, context=merged)

    @model_validator(mode="before")
    @classmethod
    def _accept_direct_mapping(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias in ("items", "definitions"):
            if alias in values:
                values.setdefault("presets", values.pop(alias))
        if "presets" in values:
            return values
        global_keys = {"enabled", "default"}
        entries = {
            key: value for key, value in values.items() if key not in global_keys
        }
        for key in list(entries):
            values.pop(key)
        values["presets"] = entries
        return values

    @field_validator("presets")
    @classmethod
    def _name_presets(
        cls,
        value: dict[str, PresetConfig],
    ) -> dict[str, PresetConfig]:
        presets: dict[str, PresetConfig] = {}
        for name, preset in value.items():
            key = name.strip() if isinstance(name, str) else ""
            if not key:
                raise ValueError("preset name must be non-empty")
            if key in presets:
                raise ValueError(f"duplicate preset: {key}")
            if not preset.name:
                object.__setattr__(preset, "name", key)
            presets[key] = preset
        return presets

    @model_validator(mode="after")
    def _default_exists(self) -> PresetsConfig:
        if self.default and self.presets and self.default not in self.presets:
            raise ValueError(f"default preset is not configured: {self.default}")
        return self

    @model_validator(mode="after")
    def _validate_context(self, info: Any) -> PresetsConfig:
        context = info.context
        profiles = context.get("profiles") if isinstance(context, Mapping) else None
        return self.validate_profiles(profiles)

    def validate_profiles(self, profiles: Any) -> PresetsConfig:
        """Check every preset profile against *profiles* when provided."""
        available = profile_names(profiles)
        if available is None:
            return self
        for name, preset in self.presets.items():
            referenced = list(preset.profiles)
            if preset.profile:
                referenced.insert(0, preset.profile)
            for profile in referenced:
                if profile not in available:
                    names = ", ".join(sorted(available)) or "none"
                    raise ConfigError(
                        f"preset {name} references unknown profile: "
                        f"{profile} (available: {names})",
                        details={
                            "preset": name,
                            "profile": profile,
                            "available": sorted(available),
                        },
                    )
        return self

    def check_profiles(self, profiles: Any) -> PresetsConfig:
        """Alias for :meth:`validate_profiles`."""
        return self.validate_profiles(profiles)

    def validate_references(self, config: Any) -> PresetsConfig:
        """Validate profiles from a parsed root configuration or mapping."""
        if isinstance(config, Mapping):
            profiles = config.get("profiles")
        else:
            profiles = getattr(config, "profiles", None)
        return self.validate_profiles(profiles)


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
    ("formats", FormatsConfig),
    ("profiles", ProfilesConfig),
    ("video", VideoConfig),
    ("render", RenderConfig),
    ("budget", BudgetConfig),
    ("llm", LLMConfig),
    ("models_file", ModelsFileConfig),
    ("providers", ProvidersConfig),
    ("circuit", CircuitBreakerConfig),
    ("ratelimit", RateLimiterConfig),
    ("reconcile", ReconcileConfig),
    ("http", HttpClientConfig),
    ("quality", QualityConfig),
    ("scoring", ScoringConfig),
    ("storage", StorageConfig),
    ("database", DatabaseConfig),
    ("api", ApiConfig),
    ("jobs", JobsConfig),
    ("safety", SafetyConfig),
    ("platforms", PlatformsConfig),
    ("schedule", ScheduleConfig),
    ("publish", PublishConfig),
    ("presets", PresetsConfig),
    ("occasions", OccasionsConfig),
    ("orders", OrdersConfig),
    ("delivery", DeliveryConfig),
    ("planner", PlannerConfig),
    ("reference", ReferenceConfig),
    ("character", CharacterConfig),
    ("brand", BrandKitConfig),
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
