"""Format recipes: ordered stages, per-stage options and output handling.

A format recipe selects which pipeline stages run and in which order. Each
stage may carry its own options, so the same stage can be configured
differently by two formats without duplicating global defaults. The recipe also
declares the required capabilities, the type of output it produces and how that
output is handled: saved, published or delivered.

Stage contracts are validated later, when the stage registry is available. At
configuration time this module only guarantees a well-formed, ordered recipe
with a known output type and handler.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel
from aivia.core.stage_retry import StageRetryPolicy

__all__ = [
    "FormatConfig",
    "FormatHandler",
    "FormatOutput",
    "FormatOutputType",
    "FormatRecipe",
    "FormatRecipeConfig",
    "FormatStage",
    "FormatStageConfig",
    "Formats",
    "FormatsConfig",
    "Handler",
    "HandlerType",
    "OutputHandler",
    "OutputHandlerType",
    "OutputKind",
    "OutputType",
    "RecipeConfig",
    "Stage",
    "StageConfig",
    "StageOptions",
]


class OutputType(StrEnum):
    """The kind of final artifact a format recipe produces."""

    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    CAROUSEL = "carousel"
    WEB_CARD = "web_card"


FormatOutput = OutputType
FormatOutputType = OutputType
OutputKind = OutputType


class OutputHandler(StrEnum):
    """What happens to the final artifact once a run succeeds."""

    SAVE = "save"
    PUBLISH = "publish"
    DELIVER = "deliver"


FormatHandler = OutputHandler
Handler = OutputHandler
HandlerType = OutputHandler
OutputHandlerType = OutputHandler

_STAGE_KEYS = frozenset(
    {
        "stage",
        "name",
        "id",
        "step",
        "options",
        "params",
        "config",
        "settings",
        "timeout_seconds",
        "timeout",
        "max_attempts",
        "retry_delay_seconds",
        "backoff_multiplier",
        "max_retry_delay_seconds",
        "retry_on_timeout",
        "retry_codes",
        "retry_policy",
        "retry",
    }
)


class StageConfig(ConfigModel):
    """One ordered pipeline stage with its recipe-specific options.

    A stage may be written as a bare string (``write``) or as a mapping with a
    ``stage``/``name``/``id`` field and an optional ``options`` mapping. The
    shorthand ``{write: {mode: facts}}`` is accepted too. A mapping may also
    override the stage class timeout and retry policy through
    ``timeout_seconds`` or a ``retry_policy`` mapping.
    """

    stage: str = Field(
        min_length=1,
        validation_alias=AliasChoices("stage", "name", "id", "step"),
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("options", "params", "config", "settings"),
    )
    timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices("timeout_seconds", "timeout"),
    )
    max_attempts: int | None = Field(default=None, ge=1)
    retry_delay_seconds: float | None = Field(default=None, ge=0)
    backoff_multiplier: float | None = Field(default=None, ge=1)
    max_retry_delay_seconds: float | None = Field(default=None, ge=0)
    retry_on_timeout: bool | None = None
    retry_codes: tuple[str, ...] | None = None
    retry_policy: StageRetryPolicy | None = Field(
        default=None,
        validation_alias=AliasChoices("retry_policy", "retry"),
    )

    @field_validator(
        "timeout_seconds",
        "retry_delay_seconds",
        "backoff_multiplier",
        "max_retry_delay_seconds",
    )
    @classmethod
    def _finite_numbers(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("stage retry values must be finite")
        return value

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"stage": data}
        if not isinstance(data, Mapping):
            return data

        values = dict(data)
        if not values.keys() & _STAGE_KEYS and len(values) == 1:
            name, options = next(iter(values.items()))
            if options is None or isinstance(options, Mapping):
                return {"stage": name, "options": options or {}}
        return values

    @field_validator("stage")
    @classmethod
    def _normalise_stage(cls, value: str) -> str:
        stage = value.strip()
        if not stage:
            raise ValueError("stage must be a non-empty name")
        return stage

    @property
    def name(self) -> str:
        """Alias for :attr:`stage`."""
        return self.stage

    @property
    def id(self) -> str:
        """Alias for :attr:`stage`."""
        return self.stage

    @property
    def params(self) -> dict[str, Any]:
        """Alias for :attr:`options`."""
        return self.options

    @property
    def settings(self) -> dict[str, Any]:
        """Alias for :attr:`options`."""
        return self.options

    @property
    def config(self) -> dict[str, Any]:
        """Alias for :attr:`options`."""
        return self.options

    @property
    def retry_overrides(self) -> dict[str, Any]:
        """The timeout and retry settings this stage overrides, if any."""
        names = (
            "timeout_seconds",
            "max_attempts",
            "retry_delay_seconds",
            "backoff_multiplier",
            "max_retry_delay_seconds",
            "retry_on_timeout",
            "retry_codes",
        )
        overrides = {
            name: getattr(self, name)
            for name in names
            if getattr(self, name) is not None
        }
        if self.retry_policy is not None:
            overrides["retry_policy"] = self.retry_policy
        return overrides

    def to_stage_kwargs(self) -> dict[str, Any]:
        """Return the keyword arguments used to build the stage instance."""
        return {"options": dict(self.options), **self.retry_overrides}


Stage = StageConfig
FormatStage = StageConfig
FormatStageConfig = StageConfig
StageOptions = StageConfig


class FormatRecipeConfig(ConfigModel):
    """An ordered recipe that selects stages, options and output handling.

    ``stages`` keeps its declared order. ``capabilities`` lists the operations
    the recipe needs, ``output_type`` describes the final artifact and
    ``handler`` says whether it is saved, published or delivered.
    """

    description: str = ""
    stages: list[StageConfig] = Field(
        default_factory=list,
        validation_alias=AliasChoices("stages", "steps", "pipeline"),
    )
    capabilities: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "capabilities",
            "required_capabilities",
            "requires",
        ),
    )
    output_type: OutputType = Field(
        default=OutputType.VIDEO,
        validation_alias=AliasChoices(
            "output_type",
            "output",
            "media_type",
            "kind",
        ),
    )
    handler: OutputHandler = Field(
        default=OutputHandler.SAVE,
        validation_alias=AliasChoices(
            "handler",
            "output_handler",
            "on_output",
            "action",
        ),
    )

    @field_validator("stages")
    @classmethod
    def _unique_stages(cls, value: list[StageConfig]) -> list[StageConfig]:
        seen: set[str] = set()
        for item in value:
            if item.stage in seen:
                raise ValueError(f"duplicate stage in recipe: {item.stage}")
            seen.add(item.stage)
        return value

    @field_validator("capabilities")
    @classmethod
    def _normalise_capabilities(cls, value: list[str]) -> list[str]:
        capabilities: list[str] = []
        seen: set[str] = set()
        for entry in value:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError("capabilities must be non-empty names")
            capability = entry.strip()
            if capability in seen:
                raise ValueError(f"duplicate capability: {capability}")
            seen.add(capability)
            capabilities.append(capability)
        return capabilities

    @property
    def steps(self) -> list[StageConfig]:
        """Alias for :attr:`stages`."""
        return self.stages

    @property
    def pipeline(self) -> list[StageConfig]:
        """Alias for :attr:`stages`."""
        return self.stages

    @property
    def required_capabilities(self) -> list[str]:
        """Alias for :attr:`capabilities`."""
        return self.capabilities

    @property
    def requires(self) -> list[str]:
        """Alias for :attr:`capabilities`."""
        return self.capabilities

    @property
    def output(self) -> OutputType:
        """Alias for :attr:`output_type`."""
        return self.output_type

    @property
    def output_handler(self) -> OutputHandler:
        """Alias for :attr:`handler`."""
        return self.handler

    @property
    def kind(self) -> OutputType:
        """Alias for :attr:`output_type`."""
        return self.output_type

    @property
    def media_type(self) -> OutputType:
        """Alias for :attr:`output_type`."""
        return self.output_type

    @property
    def on_output(self) -> OutputHandler:
        """Alias for :attr:`handler`."""
        return self.handler

    @property
    def action(self) -> OutputHandler:
        """Alias for :attr:`handler`."""
        return self.handler


FormatConfig = FormatRecipeConfig
FormatRecipe = FormatRecipeConfig
RecipeConfig = FormatRecipeConfig


class FormatsConfig(ConfigModel):
    """Named format recipes.

    Recipes may be written directly under the section
    (``formats: {narrated_still: {...}}``) or nested under ``formats`` (or
    ``recipes``); every form validates to the same mapping.
    """

    formats: dict[str, FormatRecipeConfig] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_direct_mapping(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        if "formats" in values:
            return values
        if "recipes" in values:
            recipes = values.pop("recipes")
            values.setdefault("formats", recipes)
            return values
        return {"formats": values}

    @field_validator("formats")
    @classmethod
    def _normalise_names(
        cls, value: dict[str, FormatRecipeConfig]
    ) -> dict[str, FormatRecipeConfig]:
        formats: dict[str, FormatRecipeConfig] = {}
        for name, recipe in value.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("format names must be non-empty")
            key = name.strip()
            if key in formats:
                raise ValueError(f"duplicate format: {key}")
            formats[key] = recipe
        return formats

    @property
    def recipes(self) -> dict[str, FormatRecipeConfig]:
        """Alias for :attr:`formats`."""
        return self.formats


Formats = FormatsConfig
