"""Validate an ordered format recipe against the registered stage contracts.

A recipe only names stages in order. The stage registry knows what each stage
consumes and produces, so the recipe can be checked before any paid work
starts: every input a stage declares must already be available from an earlier
stage in the same recipe, or from the caller-supplied initial inputs. A stage
never satisfies its own input and a later stage cannot satisfy an earlier one.

Validation returns every problem it finds instead of stopping at the first.
Each issue names the offending stage so a configuration error points at the
recipe entry to fix.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aivia.core.config.formats import FormatRecipeConfig
from aivia.core.errors import ConfigError
from aivia.formats.registry import (
    ContractType,
    StageRegistration,
    StageRegistry,
    stage_registry,
)

__all__ = [
    "RecipeError",
    "RecipeIssue",
    "RecipeIssues",
    "RecipeValidationError",
    "RecipeValidator",
    "check_recipe",
    "check_recipe_stages",
    "validate_recipe",
    "validate_recipe_or_raise",
    "validate_recipe_stages",
    "validate_stages",
]


def _contract_name(value: ContractType) -> str:
    if isinstance(value, str):
        return value.strip()
    return value.__name__


def _normalise_inputs(values: Iterable[ContractType] | ContractType) -> tuple[str, ...]:
    if isinstance(values, (str, type)):
        values = (values,)
    names: list[str] = []
    for value in values:
        if not isinstance(value, (str, type)):
            raise TypeError("recipe inputs must be contract names or types")
        name = _contract_name(value)
        if not name:
            raise ConfigError("recipe input contract must not be empty")
        if name not in names:
            names.append(name)
    return tuple(names)


def _load_recipe(
    recipe: FormatRecipeConfig | Mapping[str, Any] | Sequence[Any],
) -> FormatRecipeConfig:
    if isinstance(recipe, FormatRecipeConfig):
        return recipe
    if isinstance(recipe, Mapping):
        return FormatRecipeConfig.from_mapping(recipe)
    if isinstance(recipe, (str, bytes)):
        raise TypeError("recipe must be a mapping or a sequence of stages")
    try:
        stages = list(recipe)
    except TypeError:
        raise TypeError("recipe must be a mapping or a sequence of stages") from None
    return FormatRecipeConfig.from_mapping({"stages": stages})


@dataclass(frozen=True, slots=True)
class RecipeIssue:
    """One problem found while validating a recipe."""

    stage: str
    message: str
    code: str = "invalid"
    index: int = -1
    input: str = ""
    missing: tuple[str, ...] = ()
    available: tuple[str, ...] = ()

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Return the issue as a JSON-compatible mapping."""
        payload: dict[str, Any] = {
            "stage": self.stage,
            "message": self.message,
            "code": self.code,
            "index": self.index,
            "missing": list(self.missing),
            "available": list(self.available),
        }
        if self.input:
            payload["input"] = self.input
        return payload


class RecipeIssues(list[RecipeIssue]):
    """Every issue found in one recipe validation pass."""

    @property
    def ok(self) -> bool:
        """Whether the recipe has no issues."""
        return not self

    @property
    def errors(self) -> list[RecipeIssue]:
        """The issues as a plain list."""
        return list(self)

    def raise_for_errors(self) -> None:
        """Raise :class:`RecipeValidationError` when any issue exists."""
        if self:
            raise RecipeValidationError(self)


class RecipeValidationError(ConfigError):
    """A recipe failed one or more stage-contract checks."""

    code = "recipe_invalid"

    def __init__(self, issues: Sequence[RecipeIssue]) -> None:
        self.issues: tuple[RecipeIssue, ...] = tuple(issues)
        message = "recipe is invalid: " + "; ".join(str(issue) for issue in issues)
        details: dict[str, Any] = {
            "errors": [issue.to_dict() for issue in self.issues],
        }
        first = self.issues[0]
        details["stage"] = first.stage
        details["available"] = list(first.available)
        if first.input:
            details["input"] = first.input
        if first.missing:
            details["missing"] = list(first.missing)
        super().__init__(message, details=details)


RecipeError = RecipeValidationError


def _registration(
    registry: StageRegistry,
    stage: str,
    *,
    index: int,
    issues: RecipeIssues,
) -> StageRegistration | None:
    try:
        return registry.registration(stage)
    except ConfigError:
        available = tuple(registry.names())
        issues.append(
            RecipeIssue(
                stage=stage,
                index=index,
                code="unknown_stage",
                message=(
                    f"unknown stage: {stage} "
                    f"(registered: {', '.join(available) or 'none'})"
                ),
                available=available,
            )
        )
        return None


def _recipe_issues(
    recipe: FormatRecipeConfig,
    *,
    registry: StageRegistry,
    inputs: Iterable[ContractType] | ContractType = (),
) -> RecipeIssues:
    produced = set(_normalise_inputs(inputs))
    issues = RecipeIssues()
    for index, stage in enumerate(recipe.stages):
        registration = _registration(
            registry,
            stage.stage,
            index=index,
            issues=issues,
        )
        if registration is None:
            continue
        missing = tuple(
            name for name in registration.input_names if name not in produced
        )
        if missing:
            if len(missing) == 1:
                message = (
                    f"stage {stage.stage!r} requires input {missing[0]!r} "
                    "that no earlier stage produces"
                )
            else:
                message = (
                    f"stage {stage.stage!r} requires inputs that no earlier "
                    f"stage produces: {', '.join(missing)}"
                )
            issues.append(
                RecipeIssue(
                    stage=stage.stage,
                    index=index,
                    input=missing[0],
                    missing=missing,
                    available=tuple(sorted(produced)),
                    code="missing_input",
                    message=message,
                )
            )
        produced.update(registration.output_names)
    return issues


def check_recipe(
    recipe: FormatRecipeConfig | Mapping[str, Any] | Sequence[Any],
    registry: StageRegistry | None = None,
    *,
    inputs: Iterable[ContractType] | ContractType = (),
) -> RecipeIssues:
    """Return every stage-contract problem in *recipe* without raising."""
    loaded = _load_recipe(recipe)
    return _recipe_issues(
        loaded,
        registry=registry or stage_registry,
        inputs=inputs,
    )


def validate_recipe(
    recipe: FormatRecipeConfig | Mapping[str, Any] | Sequence[Any],
    registry: StageRegistry | None = None,
    *,
    inputs: Iterable[ContractType] | ContractType = (),
) -> FormatRecipeConfig:
    """Validate *recipe* and return the parsed recipe.

    *inputs* names the contracts available before the first stage, for example
    a topic supplied by the CLI. Every other input must be produced by an
    earlier stage. Raises :class:`RecipeValidationError` with an issue that
    names the offending stage when the recipe is invalid.
    """
    loaded = _load_recipe(recipe)
    _recipe_issues(
        loaded,
        registry=registry or stage_registry,
        inputs=inputs,
    ).raise_for_errors()
    return loaded


class RecipeValidator:
    """Validate recipes against one registry and set of initial inputs."""

    def __init__(
        self,
        registry: StageRegistry | None = None,
        *,
        inputs: Iterable[ContractType] | ContractType = (),
    ) -> None:
        self.registry = registry or stage_registry
        self.inputs = inputs

    def check(
        self,
        recipe: FormatRecipeConfig | Mapping[str, Any] | Sequence[Any],
    ) -> RecipeIssues:
        """Return every problem in *recipe* without raising."""
        return check_recipe(recipe, self.registry, inputs=self.inputs)

    def validate(
        self,
        recipe: FormatRecipeConfig | Mapping[str, Any] | Sequence[Any],
    ) -> FormatRecipeConfig:
        """Validate *recipe* and return it, raising every issue at once."""
        return validate_recipe(recipe, self.registry, inputs=self.inputs)


validate_recipe_or_raise = validate_recipe
validate_recipe_stages = validate_recipe
validate_stages = validate_recipe
check_recipe_stages = check_recipe
