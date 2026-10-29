"""Resolve an LLM role to its configured model id.

Roles are a closed set defined by :class:`~aivia.core.config.llm.LLMRole` plus
any role an optional module registers on the shared registry. Configuration
maps those roles to model ids; this module is the one place that turns a role
into a model, so a caller never has to duplicate the role list or guess a
fallback.

An unknown role fails with the allowed values. A known role without its own
entry uses the ``default`` role's model when one is configured, because most
deployments only override the roles they care about.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aivia.core.config.llm import (
    CORE_LLM_ROLES,
    LLMConfig,
    LLMRole,
    register_role,
    resolve_role,
    role_registry,
)
from aivia.core.errors import ConfigError
from aivia.providers.protocols.llm import LLMRequest

__all__ = [
    "CORE_LLM_ROLES",
    "DEFAULT_ROLE",
    "LLMRole",
    "LLMRoleResolver",
    "ModelResolver",
    "ModelRoleResolver",
    "RoleModelResolver",
    "RoleResolutionError",
    "RoleResolver",
    "UnknownLLMRoleError",
    "UnknownRoleError",
    "apply_role_model",
    "model_for",
    "model_for_request",
    "model_for_role",
    "models_by_role",
    "register_role",
    "resolve_llm_model",
    "resolve_model",
    "resolve_model_for_role",
    "resolve_request_model",
    "resolve_role",
    "role_registry",
]

DEFAULT_ROLE = LLMRole.DEFAULT


class UnknownRoleError(ConfigError):
    """A role is neither a core role nor a registered module role."""

    code = "unknown_role"


UnknownLLMRoleError = UnknownRoleError
RoleResolutionError = UnknownRoleError


def _canonical_role(role: str | LLMRole) -> str:
    if not isinstance(role, (str, LLMRole)):
        raise TypeError("role must be an LLMRole or string")
    try:
        return role_registry.resolve(role)
    except ConfigError as error:
        raise UnknownRoleError(error.message, details=error.details) from error


def _is_config(value: object) -> bool:
    return isinstance(value, (LLMConfig, Mapping)) or isinstance(
        getattr(value, "llm", None),
        (LLMConfig, Mapping),
    )


def _as_llm_config(config: object | None) -> LLMConfig:
    if config is None:
        return LLMConfig()
    if isinstance(config, LLMConfig):
        return config
    if isinstance(config, Mapping):
        section: object = config.get("llm", config)
        if isinstance(section, LLMConfig):
            return section
        if not isinstance(section, Mapping):
            raise ConfigError("llm config must be a mapping")
        return LLMConfig.from_mapping(section)
    section = getattr(config, "llm", None)
    if isinstance(section, LLMConfig):
        return section
    if isinstance(section, Mapping):
        return LLMConfig.from_mapping(section)
    raise TypeError("config must be an LLMConfig, app config or mapping")


class LLMRoleResolver:
    """Turn a role from the config enum into a configured model id."""

    def __init__(
        self,
        config: LLMConfig | Mapping[str, Any] | object | None = None,
        *,
        fallback: bool = True,
        default_role: str | LLMRole = DEFAULT_ROLE,
    ) -> None:
        self._config = _as_llm_config(config)
        self._fallback = bool(fallback)
        self._default_role = _canonical_role(default_role)

    @property
    def config(self) -> LLMConfig:
        """The LLM configuration this resolver reads."""
        return self._config

    @property
    def fallback(self) -> bool:
        """Whether an unconfigured role uses the default role's model."""
        return self._fallback

    @property
    def default_role(self) -> str:
        """The canonical role used as the fallback."""
        return self._default_role

    def allowed_roles(self) -> tuple[str, ...]:
        """Every role the config enum and registered modules know."""
        return role_registry.names()

    def configured_roles(self) -> tuple[str, ...]:
        """Every role that has its own model in the configuration."""
        return tuple(self._config.roles)

    def models(self) -> dict[str, str]:
        """Return a copy of the configured role-to-model mapping."""
        return dict(self._config.roles)

    def resolve(self, role: str | LLMRole) -> str:
        """Return the model id for *role* or fail with the allowed values."""
        name = _canonical_role(role)
        model = self._config.roles.get(name)
        if model is not None:
            return model
        if self._fallback and name != self._default_role:
            fallback = self._config.roles.get(self._default_role)
            if fallback is not None:
                return fallback
        configured = ", ".join(self.configured_roles()) or "none"
        raise ConfigError(
            f"no model configured for LLM role: {name} "
            f"(allowed: {', '.join(self.allowed_roles()) or 'none'}; "
            f"configured: {configured})",
            details={
                "role": name,
                "allowed": list(self.allowed_roles()),
                "configured": list(self.configured_roles()),
            },
        )

    def model_for(self, role: str | LLMRole) -> str:
        """Alias for :meth:`resolve`."""
        return self.resolve(role)

    def resolve_request(self, request: LLMRequest) -> LLMRequest:
        """Return *request* with its role canonicalized and model filled in.

        An explicitly requested model is kept, because a caller that already
        knows the model should not be overridden by configuration.
        """
        if not isinstance(request, LLMRequest):
            raise TypeError("request must be an LLMRequest")
        role = _canonical_role(request.role)
        model = request.model or self.resolve(role)
        return request.model_copy(update={"role": role, "model": model})

    def apply(self, request: LLMRequest) -> LLMRequest:
        """Alias for :meth:`resolve_request`."""
        return self.resolve_request(request)

    def __call__(self, role: str | LLMRole) -> str:
        return self.resolve(role)


RoleModelResolver = LLMRoleResolver
ModelRoleResolver = LLMRoleResolver
RoleResolver = LLMRoleResolver
ModelResolver = LLMRoleResolver


def resolve_model(
    role: str | LLMRole | LLMConfig | Mapping[str, Any] | object,
    config: LLMConfig | Mapping[str, Any] | object | None = None,
    *,
    fallback: bool = True,
    default_role: str | LLMRole = DEFAULT_ROLE,
) -> str:
    """Resolve *role* to a model id, accepting the config in either position."""
    if _is_config(role) and not _is_config(config):
        role, config = config, role
    if not isinstance(role, (str, LLMRole)):
        raise TypeError("role must be an LLMRole or string")
    resolver = LLMRoleResolver(
        config,
        fallback=fallback,
        default_role=default_role,
    )
    return resolver.resolve(role)


resolve_model_for_role = resolve_model
model_for_role = resolve_model
model_for = resolve_model
resolve_llm_model = resolve_model


def models_by_role(
    config: LLMConfig | Mapping[str, Any] | object | None = None,
    *,
    fallback: bool = True,
    default_role: str | LLMRole = DEFAULT_ROLE,
) -> dict[str, str]:
    """Return every allowed role mapped to the model it resolves to."""
    resolver = LLMRoleResolver(
        config,
        fallback=fallback,
        default_role=default_role,
    )
    resolved: dict[str, str] = {}
    for role in resolver.allowed_roles():
        try:
            resolved[role] = resolver.resolve(role)
        except ConfigError:
            continue
    return resolved


def resolve_request_model(
    request: LLMRequest,
    config: LLMConfig | Mapping[str, Any] | object | None = None,
    *,
    fallback: bool = True,
    default_role: str | LLMRole = DEFAULT_ROLE,
) -> LLMRequest:
    """Return *request* with the model for its role filled in."""
    return LLMRoleResolver(
        config,
        fallback=fallback,
        default_role=default_role,
    ).resolve_request(request)


model_for_request = resolve_request_model
apply_role_model = resolve_request_model
