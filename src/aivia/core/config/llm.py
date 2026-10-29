"""LLM configuration: role enum, module roles, token prices and repair limit.

The core roles are fixed by decision D-012. Modules may add their own roles to
the shared :data:`role_registry`; a role that is neither core nor registered is
rejected with the list of allowed values. Token prices are expressed per
million input and output tokens so callers never need to know a provider's
native unit.

``max_repairs`` is the number of times a structured-output validation failure
may be sent back to the model before the call fails.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel
from aivia.core.errors import ConfigError

__all__ = [
    "CORE_LLM_ROLES",
    "CORE_ROLES",
    "CoreLLMRole",
    "CoreRole",
    "LLMConfig",
    "LLMPricing",
    "LLMRole",
    "LLMRoleEnum",
    "LLMRoleRegistry",
    "LLMTokenPricing",
    "ModelPrice",
    "ModelPricing",
    "PerMillionTokenPricing",
    "RegisteredLLMRole",
    "Role",
    "RoleRegistry",
    "TokenPrice",
    "TokenPricing",
    "llm_role_registry",
    "register_llm_role",
    "register_module_role",
    "register_role",
    "resolve_role",
    "role_registry",
    "roles",
    "validate_role",
]


class LLMRole(StrEnum):
    """The fixed core roles an LLM call may target."""

    DEFAULT = "default"
    WRITING = "writing"
    FACTCHECK = "factcheck"
    LYRICS = "lyrics"
    PLANNER = "planner"
    REFERENCE_ANALYSIS = "reference_analysis"
    VISION_QA = "vision_qa"
    SAFETY = "safety"
    TRANSLATION_CHECK = "translation_check"
    PROMPT_ENHANCE = "prompt_enhance"


CoreLLMRole = LLMRole
CoreRole = LLMRole
LlmRole = LLMRole
LLMRoleEnum = LLMRole
Role = LLMRole

CORE_LLM_ROLES: tuple[str, ...] = tuple(role.value for role in LLMRole)
CORE_ROLES = CORE_LLM_ROLES


@dataclass(frozen=True, slots=True)
class RegisteredLLMRole:
    """One role known to an :class:`LLMRoleRegistry`."""

    name: str
    module: str | None = None
    core: bool = False


def _role_name(role: object) -> str:
    if isinstance(role, LLMRole):
        return role.value
    if not isinstance(role, str) or not role.strip():
        raise ConfigError("LLM role must be a non-empty string")
    return role.strip()


class LLMRoleRegistry:
    """Core roles plus roles registered by optional modules."""

    def __init__(self, roles: Iterable[str | LLMRole] | None = None) -> None:
        self._roles: dict[str, RegisteredLLMRole] = {}
        initial = LLMRole if roles is None else roles
        for role in initial:
            name = _role_name(role)
            self._roles[name] = RegisteredLLMRole(
                name=name,
                core=name in CORE_LLM_ROLES,
            )

    def register(
        self,
        role: str | LLMRole,
        *,
        module: str | None = None,
    ) -> RegisteredLLMRole:
        """Register a module role, rejecting a duplicate name."""
        name = _role_name(role)
        existing = self._roles.get(name)
        if existing is not None:
            raise ConfigError(
                f"LLM role already registered: {name}",
                details={
                    "role": name,
                    "existing_module": existing.module,
                    "module": module,
                },
            )
        registered = RegisteredLLMRole(name=name, module=module)
        self._roles[name] = registered
        return registered

    def register_role(
        self,
        role: str | LLMRole,
        *,
        module: str | None = None,
    ) -> RegisteredLLMRole:
        """Alias for :meth:`register`."""
        return self.register(role, module=module)

    def unregister(self, role: str | LLMRole) -> None:
        """Remove a registered role; core roles cannot be removed."""
        name = _role_name(role)
        existing = self._roles.get(name)
        if existing is None:
            raise ConfigError(f"unknown role: {name} (allowed: {self._allowed()})")
        if existing.core:
            raise ConfigError(f"cannot unregister core LLM role: {name}")
        del self._roles[name]

    def resolve(self, role: str | LLMRole) -> str:
        """Return the canonical role name or fail with the allowed values."""
        name = _role_name(role)
        if name not in self._roles:
            raise ConfigError(
                f"unknown role: {name} (allowed: {self._allowed()})",
                details={"role": name, "allowed": list(self.names())},
            )
        return name

    def get(self, role: str | LLMRole) -> RegisteredLLMRole:
        """Return the registered entry for *role*."""
        return self._roles[self.resolve(role)]

    def names(self) -> tuple[str, ...]:
        """Return every allowed role, core roles first in declaration order."""
        return tuple(self._roles)

    def allowed(self) -> tuple[str, ...]:
        """Alias for :meth:`names`."""
        return self.names()

    def allowed_roles(self) -> tuple[str, ...]:
        """Alias for :meth:`names`."""
        return self.names()

    def known_roles(self) -> tuple[str, ...]:
        """Alias for :meth:`names`."""
        return self.names()

    def validate(self, role: str | LLMRole) -> str:
        """Alias for :meth:`resolve`."""
        return self.resolve(role)

    def _allowed(self) -> str:
        return ", ".join(self.names()) or "none"

    def __contains__(self, role: object) -> bool:
        try:
            name = _role_name(role)
        except ConfigError:
            return False
        return name in self._roles

    def __iter__(self) -> Iterator[RegisteredLLMRole]:
        return iter(self._roles.values())

    def __len__(self) -> int:
        return len(self._roles)


RoleRegistry = LLMRoleRegistry

role_registry = LLMRoleRegistry()
llm_role_registry = role_registry
roles = role_registry


def register_role(
    role: str | LLMRole,
    *,
    module: str | None = None,
) -> RegisteredLLMRole:
    """Register a module role on the shared registry."""
    return role_registry.register(role, module=module)


register_llm_role = register_role
register_module_role = register_role


def resolve_role(role: str | LLMRole) -> str:
    """Resolve *role* against the shared registry or fail with the allowed list."""
    return role_registry.resolve(role)


validate_role = resolve_role


_PRICING_ALIASES: dict[str, tuple[str, ...]] = {
    "input_per_million": (
        "input",
        "prompt",
        "input_price",
        "input_cost",
        "input_per_1m",
        "input_per_mtok",
        "input_per_million_tokens",
        "input_price_per_million",
        "input_usd_per_million",
        "per_million_input",
        "prompt_per_million",
        "prompt_per_million_tokens",
        "prompt_price_per_million",
    ),
    "output_per_million": (
        "output",
        "completion",
        "output_price",
        "output_cost",
        "output_per_1m",
        "output_per_mtok",
        "output_per_million_tokens",
        "output_price_per_million",
        "output_usd_per_million",
        "per_million_output",
        "completion_per_million",
        "completion_per_million_tokens",
        "completion_price_per_million",
    ),
    "cached_input_per_million": (
        "cached",
        "cached_input",
        "cache_read",
        "cached_price",
        "cached_input_price",
        "cached_input_cost",
        "cached_input_per_1m",
        "cached_input_per_mtok",
        "cached_input_per_million_tokens",
        "cached_input_price_per_million",
        "cached_input_usd_per_million",
        "per_million_cached_input",
        "cache_read_per_million",
        "cache_read_input_per_million",
    ),
}


class TokenPricing(ConfigModel):
    """USD prices for one million tokens of a model.

    ``input_per_million`` and ``output_per_million`` are required in spirit but
    default to zero for backward compatibility. ``cached_input_per_million``
    prices prompt tokens a provider served from a cache. Leaving it unset means
    cached input is charged at the ordinary input price. Whether the provider's
    reported input count already contains the cached tokens is recorded on
    :class:`~aivia.providers.protocols.llm.LLMUsage`; the calculator uses that
    flag so cached tokens are never counted twice.
    """

    input_per_million: float = Field(default=0.0, ge=0)
    output_per_million: float = Field(default=0.0, ge=0)
    cached_input_per_million: float | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for canonical, aliases in _PRICING_ALIASES.items():
            for alias in aliases:
                if alias in values:
                    value = values.pop(alias)
                    values.setdefault(canonical, value)
        return values

    @field_validator(
        "input_per_million",
        "output_per_million",
        "cached_input_per_million",
    )
    @classmethod
    def _validate_price(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value):
            raise ValueError("token price must be finite")
        return float(value)

    @property
    def cached_input_price(self) -> float:
        """The effective cached input price, falling back to the input price."""
        if self.cached_input_per_million is None:
            return self.input_per_million
        return self.cached_input_per_million

    @property
    def input(self) -> float:
        """Alias for :attr:`input_per_million`."""
        return self.input_per_million

    @property
    def cached(self) -> float:
        """Alias for :attr:`cached_input_price`."""
        return self.cached_input_price

    @property
    def output(self) -> float:
        """Alias for :attr:`output_per_million`."""
        return self.output_per_million

    @property
    def prompt(self) -> float:
        """Alias for :attr:`input_per_million`."""
        return self.input_per_million

    @property
    def completion(self) -> float:
        """Alias for :attr:`output_per_million`."""
        return self.output_per_million

    @property
    def input_per_million_tokens(self) -> float:
        """Alias for :attr:`input_per_million`."""
        return self.input_per_million

    @property
    def output_per_million_tokens(self) -> float:
        """Alias for :attr:`output_per_million`."""
        return self.output_per_million


LLMPricing = TokenPricing
LLMTokenPricing = TokenPricing
ModelPrice = TokenPricing
ModelPricing = TokenPricing
PerMillionTokenPricing = TokenPricing
TokenPrice = TokenPricing

_CONFIG_ALIASES: dict[str, tuple[str, ...]] = {
    "chain": ("provider_chain", "providers", "order"),
    "roles": ("role_models", "models_by_role", "model_roles"),
    "pricing": ("token_pricing", "model_pricing", "prices"),
    "max_repairs": (
        "repair_limit",
        "max_repair_attempts",
        "repair_attempts",
        "max_repair",
        "repairs",
    ),
}


class LLMConfig(ConfigModel):
    """Provider chain, per-role models, token prices and the repair limit."""

    chain: list[str] = Field(default_factory=list)
    roles: dict[str, str] = Field(default_factory=dict)
    pricing: dict[str, TokenPricing] = Field(default_factory=dict)
    max_repairs: int = Field(default=2, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for canonical, aliases in _CONFIG_ALIASES.items():
            for alias in aliases:
                if alias in values:
                    value = values.pop(alias)
                    values.setdefault(canonical, value)
        return values

    @field_validator("chain")
    @classmethod
    def _validate_chain(cls, value: list[str]) -> list[str]:
        chain: list[str] = []
        seen: set[str] = set()
        for entry in value:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError("chain entries must be non-empty provider names")
            provider = entry.strip()
            if provider in seen:
                raise ValueError(f"duplicate provider in chain: {provider}")
            seen.add(provider)
            chain.append(provider)
        return chain

    @field_validator("roles")
    @classmethod
    def _validate_roles(cls, value: dict[str, str]) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for role, model in value.items():
            try:
                name = role_registry.resolve(role)
            except ConfigError as error:
                raise ConfigError(
                    f"roles.{role}: {error}",
                    details=error.details,
                ) from error
            if not isinstance(model, str) or not model.strip():
                raise ValueError(
                    f"model for LLM role {name} must be a non-empty string"
                )
            resolved[name] = model.strip()
        return resolved

    @field_validator("pricing")
    @classmethod
    def _validate_pricing(
        cls, value: dict[str, TokenPricing]
    ) -> dict[str, TokenPricing]:
        pricing: dict[str, TokenPricing] = {}
        for model, price in value.items():
            if not isinstance(model, str) or not model.strip():
                raise ValueError("pricing model names must be non-empty")
            pricing[model.strip()] = price
        return pricing

    @property
    def repair_limit(self) -> int:
        """Alias for :attr:`max_repairs`."""
        return self.max_repairs

    def model_for(self, role: str | LLMRole) -> str:
        """Return the configured model id for *role*."""
        name = role_registry.resolve(role)
        try:
            return self.roles[name]
        except KeyError:
            raise ConfigError(
                f"no model configured for LLM role: {name}",
                details={"role": name},
            ) from None

    def pricing_for(self, model: str) -> TokenPricing:
        """Return the configured token pricing for *model*."""
        try:
            return self.pricing[model.strip()]
        except KeyError:
            available = ", ".join(sorted(self.pricing)) or "none"
            raise ConfigError(
                f"no pricing configured for LLM model: {model} "
                f"(available: {available})",
                details={"model": model, "available": list(self.pricing)},
            ) from None
