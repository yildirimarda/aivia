"""Estimate the cost of a format recipe before any provider is called.

A format recipe is an ordered list of stages. Some stages are deterministic and
free (input resolution, rendering, publishing), while others need a paid
provider capability (LLM, TTS, image generation, …). This module turns that
recipe into one :class:`StageEstimate` per stage so a caller can print a table
before starting a run and refuse a run it cannot price.

The estimator never calls a provider and never writes a cost event. It only
reads the static :class:`~aivia.core.models.provider.ProviderMeta` prices and
resolves them through the shared
:class:`~aivia.core.pricing.PricingResolver`. Provider selection is delegated to
the router, so a stage honours the configured capability chain and its
selection strategy (``first_available``, ``cheapest``, ``best_quality``,
``round_robin`` or ``weighted_ab``) instead of always picking the cheapest
provider. Without an explicit strategy the estimator keeps the historical
cheapest-price default, with chain order as the tie breaker.

A stage may override the built-in mapping through its options:

* ``capability`` or ``operation`` names the paid capability explicitly.
* ``units`` (or ``count``, ``quantity``, ``clips``, ``images``, ``seconds``)
  sets the quantity to price. ``seconds`` and ``duration_seconds`` imply the
  ``seconds`` unit.
* ``unit`` or ``unit_kind`` names the requested unit explicitly.
* ``estimate_usd`` or ``cost_estimate`` supplies a fixed amount, for stages
  that cannot be priced from provider metadata.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from aivia.core.config.capability import CapabilityChainConfig, CapabilityStrategy
from aivia.core.config.embeddings import EmbeddingsConfig
from aivia.core.config.formats import FormatRecipeConfig, StageConfig
from aivia.core.errors import ConfigError, ProviderExhaustedError
from aivia.core.models.provider import ProviderMeta
from aivia.core.pricing import PriceSource, PricingResolver
from aivia.core.router import Router
from aivia.providers.base import Provider

__all__ = [
    "EstimateSource",
    "Estimator",
    "RecipeEstimate",
    "StageEstimate",
    "estimate_recipe",
    "render_estimate",
    "stage_capability",
    "stage_units",
]


class EstimateSource(StrEnum):
    """Where a stage's amount came from."""

    METADATA = "metadata"
    OVERRIDE = "override"
    MISSING = "missing"
    FREE = "free"


_FREE_STAGES = frozenset(
    {
        "compose",
        "composition",
        "deliver",
        "delivery",
        "input",
        "input_topic",
        "name_overlay",
        "publish",
        "render",
        "save",
    }
)

_CAPABILITY_BY_STAGE: dict[str, str] = {
    "align": "stt",
    "character_trainer": "character_trainer",
    "embed": "embeddings",
    "embeddings": "embeddings",
    "fact_check": "llm",
    "factcheck": "llm",
    "fetch": "page_fetch",
    "image": "image_gen",
    "image_edit": "image_edit",
    "image_gen": "image_gen",
    "images": "image_gen",
    "lipsync": "lipsync",
    "music": "music_gen",
    "music_gen": "music_gen",
    "page_fetch": "page_fetch",
    "research": "search",
    "review": "llm",
    "rewrite": "llm",
    "search": "search",
    "sfx": "sfx",
    "song": "music_gen",
    "speech": "tts",
    "stock": "stock",
    "stt": "stt",
    "transcribe": "stt",
    "transcription": "stt",
    "tts": "tts",
    "video": "video_gen",
    "video_gen": "video_gen",
    "voice": "tts",
    "write": "llm",
}

_VISUAL_STRATEGIES: dict[str, str] = {
    "ai_image": "image_gen",
    "ai_video": "video_gen",
    "existing": "",
    "image": "image_gen",
    "image_gen": "image_gen",
    "images": "image_gen",
    "none": "",
    "provided": "",
    "stock": "stock",
    "text": "",
    "video": "video_gen",
    "video_gen": "video_gen",
    "videos": "video_gen",
}

_KNOWN_CAPABILITIES = frozenset(
    {
        "character_trainer",
        "delivery",
        "embeddings",
        "image_edit",
        "image_gen",
        "lipsync",
        "llm",
        "music_gen",
        "notify",
        "page_fetch",
        "publish",
        "search",
        "sfx",
        "stock",
        "storage",
        "stt",
        "tts",
        "video_gen",
    }
)

_ESTIMATE_KEYS = ("estimate_usd", "cost_estimate")
_UNIT_KEYS = (
    "units",
    "count",
    "quantity",
    "clips",
    "images",
    "seconds",
    "duration_seconds",
)


def _number_option(
    options: Mapping[str, Any],
    keys: Sequence[str],
    *,
    stage: str,
) -> float | None:
    for key in keys:
        if key not in options:
            continue
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(
                f"stage {stage!r} option {key!r} must be a finite, non-negative number",
                details={"stage": stage, "option": key, "value": value},
            )
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ConfigError(
                f"stage {stage!r} option {key!r} must be a finite, non-negative number",
                details={"stage": stage, "option": key, "value": value},
            )
        return number
    return None


def stage_units(stage: StageConfig) -> tuple[float, str]:
    """Return the quantity and requested unit for *stage*.

    The default is one unit priced in the provider's own unit. An explicit
    ``seconds``/``duration_seconds`` option implies the ``seconds`` unit.
    """
    options = stage.options
    raw_unit = options.get("unit", options.get("unit_kind"))
    if raw_unit is None:
        unit_kind = ""
    elif isinstance(raw_unit, str):
        unit_kind = raw_unit.strip()
    else:
        raise ConfigError(
            f"stage {stage.stage!r} unit must be a string",
            details={"stage": stage.stage, "unit": raw_unit},
        )

    quantity = _number_option(options, _UNIT_KEYS, stage=stage.stage)
    if quantity is None:
        return 1.0, unit_kind
    if not unit_kind and ("seconds" in options or "duration_seconds" in options):
        unit_kind = "seconds"
    return quantity, unit_kind


def stage_capability(
    stage: StageConfig,
    recipe: FormatRecipeConfig | None = None,
) -> str:
    """Return the paid capability *stage* needs, or an empty string when free.

    An explicit ``capability``/``operation`` option wins. The ``visuals``
    stage chooses image, video or stock from its ``strategy`` option, falling
    back to the recipe's declared capabilities. A stage marked skipped or
    disabled needs no paid call. Every other stage uses the built-in name
    mapping; an unknown stage that names a known capability is accepted as-is,
    while any other unknown stage is treated as free.
    """
    options = stage.options
    if (
        options.get("skip") is True
        or options.get("disabled") is True
        or options.get("enabled") is False
    ):
        return ""
    for key in ("capability", "operation"):
        value = options.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    name = "_".join(stage.stage.strip().lower().replace("-", " ").split())
    if name in {"visual", "visuals"}:
        for key in ("strategy", "source", "mode", "visual_strategy"):
            value = options.get(key)
            if isinstance(value, str):
                strategy = "_".join(value.strip().lower().replace("-", " ").split())
                if strategy in _VISUAL_STRATEGIES:
                    return _VISUAL_STRATEGIES[strategy]
        capabilities = set(recipe.capabilities) if recipe is not None else set()
        if "video_gen" in capabilities and "image_gen" not in capabilities:
            return "video_gen"
        return "image_gen"

    if name in _FREE_STAGES:
        return ""
    mapped = _CAPABILITY_BY_STAGE.get(name)
    if mapped is not None:
        return mapped
    if name in _KNOWN_CAPABILITIES:
        return name
    return ""


@dataclass(frozen=True, slots=True)
class StageEstimate:
    """The estimated cost of one stage.

    ``capability`` is empty for a free stage. ``missing`` is true when a paid
    stage could not be priced because no provider serves its capability or no
    provider declares a price for it. ``amount_usd`` is always present so a
    table can show every stage even when the total is incomplete.
    """

    stage: str
    capability: str
    provider: str
    units: float
    unit_kind: str
    amount_usd: float
    source: EstimateSource
    note: str = ""
    missing: bool = False
    free: bool = False

    @property
    def paid(self) -> bool:
        """Whether this stage needs a paid provider call."""
        return not self.free

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for logs and reports."""
        return {
            "stage": self.stage,
            "capability": self.capability,
            "provider": self.provider,
            "units": self.units,
            "unit_kind": self.unit_kind,
            "amount_usd": self.amount_usd,
            "source": self.source.value,
            "note": self.note,
            "missing": self.missing,
            "free": self.free,
        }


@dataclass(frozen=True, slots=True)
class RecipeEstimate:
    """The stage-by-stage estimate for one format recipe."""

    format: str
    stages: tuple[StageEstimate, ...]

    @property
    def total_usd(self) -> float:
        """The sum of every stage estimate in USD."""
        return math.fsum(stage.amount_usd for stage in self.stages)

    @property
    def missing_stages(self) -> tuple[StageEstimate, ...]:
        """Paid stages whose price could not be resolved."""
        return tuple(stage for stage in self.stages if stage.missing)

    @property
    def free_stages(self) -> tuple[StageEstimate, ...]:
        """Stages that do not need a paid call."""
        return tuple(stage for stage in self.stages if stage.free)

    @property
    def paid_stages(self) -> tuple[StageEstimate, ...]:
        """Stages that need a paid provider call."""
        return tuple(stage for stage in self.stages if not stage.free)

    @property
    def complete(self) -> bool:
        """Whether every paid stage has a price."""
        return not self.missing_stages

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for logs and reports."""
        return {
            "format": self.format,
            "stages": [stage.to_dict() for stage in self.stages],
            "total_usd": self.total_usd,
            "complete": self.complete,
        }


class Estimator:
    """Turn a format recipe into stage estimates using static provider prices.

    ``providers`` maps the configured provider name to its metadata. The name
    is used in the report and for pricing overrides; the metadata supplies the
    capabilities and prices. ``overrides`` maps a provider name to a fixed
    price per unit and wins over metadata, matching the pricing resolver.

    ``config`` or ``configs`` supply :class:`CapabilityChainConfig` entries per
    capability, exactly as the router consumes them, so the estimate walks the
    same chain with the same strategy as a real run. ``strategy``, ``chain``,
    ``weights`` and ``seed`` override the default chain for callers that do not
    build a config model. Per-capability pricing overrides win over the
    estimator-wide ``overrides``.
    """

    def __init__(
        self,
        providers: Mapping[str, ProviderMeta] | None = None,
        *,
        overrides: Mapping[str, float] | None = None,
        pricing_version: str = "",
        config: CapabilityChainConfig
        | Mapping[str, CapabilityChainConfig | EmbeddingsConfig]
        | EmbeddingsConfig
        | None = None,
        configs: Mapping[str, CapabilityChainConfig] | None = None,
        embeddings: EmbeddingsConfig | Mapping[str, Any] | None = None,
        strategy: CapabilityStrategy | str | None = None,
        chain: Iterable[str] | None = None,
        weights: Mapping[str, float] | None = None,
        seed: int | None = None,
    ) -> None:
        if isinstance(config, EmbeddingsConfig):
            if embeddings is not None:
                raise TypeError("pass either config or embeddings, not both")
            embeddings = config
            config = None
        if embeddings is not None:
            section = (
                embeddings
                if isinstance(embeddings, EmbeddingsConfig)
                else EmbeddingsConfig.from_mapping(embeddings)
            )
            merged = dict(configs or {})
            merged.setdefault("embeddings", section.to_capability_config())
            configs = merged
        entries: dict[str, ProviderMeta] = {}
        for name, metadata in (providers or {}).items():
            if not isinstance(name, str) or not name.strip():
                raise ConfigError("provider names must be non-empty strings")
            if not isinstance(metadata, ProviderMeta):
                raise TypeError(
                    "provider metadata must be a ProviderMeta, "
                    f"not {type(metadata).__name__}"
                )
            entries[name.strip()] = metadata
        self._providers = dict(entries)
        self._resolver = PricingResolver(
            overrides,
            pricing_version=pricing_version,
        )
        self._default_config, self._configs = self._resolve_configs(
            config,
            configs,
            strategy=strategy,
            chain=chain,
            weights=weights,
        )
        self._router = Router(
            providers={
                name: Provider(metadata, name=name)
                for name, metadata in self._providers.items()
            },
            config=self._default_config,
            configs=self._configs,
            seed=seed,
        )
        self._resolvers: dict[str, PricingResolver] = {}

    def _resolve_configs(
        self,
        config: CapabilityChainConfig
        | Mapping[str, CapabilityChainConfig | EmbeddingsConfig]
        | EmbeddingsConfig
        | None,
        configs: Mapping[str, CapabilityChainConfig] | None,
        *,
        strategy: CapabilityStrategy | str | None,
        chain: Iterable[str] | None,
        weights: Mapping[str, float] | None,
    ) -> tuple[CapabilityChainConfig, dict[str, CapabilityChainConfig]]:
        if config is not None and not isinstance(
            config, (CapabilityChainConfig, Mapping)
        ):
            raise TypeError(
                "config must be a CapabilityChainConfig or a mapping, "
                f"not {type(config).__name__}"
            )
        configured = (
            config is not None
            or configs is not None
            or strategy is not None
            or chain is not None
            or weights is not None
        )
        default_strategy = (
            CapabilityStrategy.FIRST_AVAILABLE
            if configured
            else CapabilityStrategy.CHEAPEST
        )
        base = (
            config
            if isinstance(config, CapabilityChainConfig)
            else CapabilityChainConfig(strategy=default_strategy)
        )
        if strategy is not None or chain is not None or weights is not None:
            values = base.model_dump()
            if strategy is not None:
                values["strategy"] = strategy
            if chain is not None:
                values["chain"] = list(chain)
            if weights is not None:
                values["weights"] = dict(weights)
            base = CapabilityChainConfig.model_validate(values)

        merged: dict[str, CapabilityChainConfig] = {}
        if isinstance(config, Mapping):
            self._add_configs(merged, config)
        if configs is not None:
            self._add_configs(merged, configs)
        return (
            self._with_global_overrides(base),
            {
                capability: self._with_global_overrides(entry)
                for capability, entry in merged.items()
            },
        )

    @staticmethod
    def _add_configs(
        target: dict[str, CapabilityChainConfig],
        entries: Mapping[str, CapabilityChainConfig | EmbeddingsConfig],
    ) -> None:
        for capability, entry in entries.items():
            if isinstance(entry, EmbeddingsConfig):
                entry = entry.to_capability_config()
            if not isinstance(capability, str) or not capability.strip():
                raise ConfigError("capability names must be non-empty strings")
            if not isinstance(entry, CapabilityChainConfig):
                raise TypeError(
                    "capability configs must be CapabilityChainConfig instances"
                )
            target[capability.strip()] = entry

    def _with_global_overrides(
        self, config: CapabilityChainConfig
    ) -> CapabilityChainConfig:
        if not self._resolver.overrides:
            return config
        overrides = dict(self._resolver.overrides)
        overrides.update(config.pricing_overrides)
        return config.model_copy(update={"pricing_overrides": overrides})

    def _resolver_for(self, capability: str) -> PricingResolver:
        try:
            return self._resolvers[capability]
        except KeyError:
            config = self._router.config_for(capability)
            resolver = PricingResolver(
                config.pricing_overrides,
                pricing_version=self._resolver.pricing_version,
            )
            self._resolvers[capability] = resolver
            return resolver

    @property
    def providers(self) -> dict[str, ProviderMeta]:
        """A copy of the provider metadata keyed by configured name."""
        return dict(self._providers)

    @property
    def resolver(self) -> PricingResolver:
        """The pricing resolver shared by every stage."""
        return self._resolver

    def estimate(self, name: str, recipe: FormatRecipeConfig) -> RecipeEstimate:
        """Estimate every stage of *recipe* in its declared order."""
        stages = tuple(self._estimate_stage(stage, recipe) for stage in recipe.stages)
        return RecipeEstimate(format=name, stages=stages)

    def _estimate_stage(
        self,
        stage: StageConfig,
        recipe: FormatRecipeConfig,
    ) -> StageEstimate:
        capability = stage_capability(stage, recipe)
        units, unit_kind = stage_units(stage)
        explicit = _number_option(
            stage.options,
            _ESTIMATE_KEYS,
            stage=stage.stage,
        )
        if explicit is not None:
            return StageEstimate(
                stage=stage.stage,
                capability=capability,
                provider="override",
                units=units,
                unit_kind=unit_kind or "call",
                amount_usd=explicit,
                source=EstimateSource.OVERRIDE,
                note="explicit estimate",
            )

        if not capability:
            return StageEstimate(
                stage=stage.stage,
                capability="",
                provider="",
                units=0.0,
                unit_kind="",
                amount_usd=0.0,
                source=EstimateSource.FREE,
                note="no paid call",
                free=True,
            )

        try:
            selection = self._router.select_with_record(
                capability,
                units=units,
                unit_kind=unit_kind or None,
            )
        except ProviderExhaustedError:
            return StageEstimate(
                stage=stage.stage,
                capability=capability,
                provider="",
                units=units,
                unit_kind=unit_kind or "call",
                amount_usd=0.0,
                source=EstimateSource.MISSING,
                note=f"no provider for {capability}",
                missing=True,
            )

        provider_name = selection.name
        metadata = self._providers[provider_name]
        resolver = self._resolver_for(capability)
        if (
            metadata.pricing_for(capability) is None
            and provider_name not in resolver.overrides
        ):
            return StageEstimate(
                stage=stage.stage,
                capability=capability,
                provider=provider_name,
                units=units,
                unit_kind=unit_kind or "call",
                amount_usd=0.0,
                source=EstimateSource.MISSING,
                note=f"no price for {capability}",
                missing=True,
            )

        try:
            resolved = resolver.resolve(
                metadata,
                capability,
                units,
                unit_kind or None,
                provider_name=provider_name,
            )
        except ConfigError as exc:
            if "from_unit" not in exc.details or "to_unit" not in exc.details:
                raise
            return StageEstimate(
                stage=stage.stage,
                capability=capability,
                provider=provider_name,
                units=units,
                unit_kind=unit_kind or "call",
                amount_usd=0.0,
                source=EstimateSource.MISSING,
                note=(
                    f"cannot convert {exc.details['from_unit']!r} "
                    f"to {exc.details['to_unit']!r}"
                ),
                missing=True,
            )
        return StageEstimate(
            stage=stage.stage,
            capability=capability,
            provider=provider_name,
            units=units,
            unit_kind=resolved.unit_kind,
            amount_usd=resolved.amount_usd,
            source=(
                EstimateSource.OVERRIDE
                if resolved.source is PriceSource.OVERRIDE
                else EstimateSource.METADATA
            ),
        )


def estimate_recipe(
    name: str,
    recipe: FormatRecipeConfig,
    providers: Mapping[str, ProviderMeta] | None = None,
    *,
    overrides: Mapping[str, float] | None = None,
    pricing_version: str = "",
    config: CapabilityChainConfig
    | Mapping[str, CapabilityChainConfig | EmbeddingsConfig]
    | EmbeddingsConfig
    | None = None,
    configs: Mapping[str, CapabilityChainConfig] | None = None,
    embeddings: EmbeddingsConfig | Mapping[str, Any] | None = None,
    strategy: CapabilityStrategy | str | None = None,
    chain: Iterable[str] | None = None,
    weights: Mapping[str, float] | None = None,
    seed: int | None = None,
) -> RecipeEstimate:
    """Estimate *recipe* without constructing a long-lived estimator."""
    return Estimator(
        providers,
        overrides=overrides,
        pricing_version=pricing_version,
        config=config,
        configs=configs,
        embeddings=embeddings,
        strategy=strategy,
        chain=chain,
        weights=weights,
        seed=seed,
    ).estimate(name, recipe)


def render_estimate(estimate: RecipeEstimate) -> str:
    """Render *estimate* as a deterministic tab-separated table."""
    lines = [
        f"Format: {estimate.format}",
        "STAGE\tCAPABILITY\tPROVIDER\tUNITS\tUNIT\tESTIMATE\tNOTE",
    ]
    for stage in estimate.stages:
        lines.append(
            "\t".join(
                (
                    stage.stage,
                    stage.capability or "-",
                    stage.provider or "-",
                    f"{stage.units:g}",
                    stage.unit_kind or "-",
                    f"${stage.amount_usd:.6f}",
                    stage.note,
                )
            )
        )
    lines.append(f"TOTAL\t-\t-\t-\t-\t${estimate.total_usd:.6f}\t")
    return "\n".join(lines) + "\n"
