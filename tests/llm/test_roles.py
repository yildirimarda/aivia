"""Resolve LLM roles to configured models using the config enum."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aivia.core.config.llm import CORE_LLM_ROLES, LLMConfig, LLMRole, role_registry
from aivia.core.errors import ConfigError
from aivia.llm import (
    LLMRoleResolver,
    RoleModelResolver,
    UnknownRoleError,
    model_for_role,
    models_by_role,
    register_role,
    resolve_model,
    resolve_request_model,
)
from aivia.providers.protocols.llm import ChatMessage, LLMRequest, MessageRole


def _config(**roles: str) -> LLMConfig:
    return LLMConfig.from_mapping({"roles": roles})


def _request(role: str = "default", model: str | None = None) -> LLMRequest:
    return LLMRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="hello")],
        role=role,
        model=model,
    )


def test_every_core_role_resolves_to_its_configured_model() -> None:
    config = _config(**{role.value: f"model-{role.value}" for role in LLMRole})
    resolver = LLMRoleResolver(config)

    for role in LLMRole:
        assert resolver.resolve(role) == f"model-{role.value}"
        assert resolver.resolve(role.value) == f"model-{role.value}"

    assert resolver.allowed_roles() == CORE_LLM_ROLES
    assert RoleModelResolver is LLMRoleResolver


def test_role_names_are_stripped_before_lookup() -> None:
    resolver = LLMRoleResolver(_config(default="model-a"))

    assert resolver.resolve(" default ") == "model-a"


def test_an_unconfigured_role_falls_back_to_the_default_model() -> None:
    resolver = LLMRoleResolver(_config(default="model-a"))

    assert resolver.resolve("writing") == "model-a"
    assert resolver.resolve(LLMRole.FACTCHECK) == "model-a"


def test_fallback_can_be_disabled_for_strict_lookups() -> None:
    resolver = LLMRoleResolver(_config(default="model-a"), fallback=False)

    with pytest.raises(ConfigError) as caught:
        resolver.resolve("writing")

    message = str(caught.value)
    assert "no model configured for LLM role: writing" in message
    assert "allowed:" in message
    assert "configured: default" in message


def test_a_missing_default_fails_with_the_allowed_roles() -> None:
    resolver = LLMRoleResolver(_config(writing="model-b"))

    with pytest.raises(ConfigError) as caught:
        resolver.resolve("factcheck")

    assert caught.value.details["role"] == "factcheck"
    assert caught.value.details["allowed"] == list(CORE_LLM_ROLES)
    assert caught.value.details["configured"] == ["writing"]


def test_an_unknown_role_lists_every_allowed_value() -> None:
    resolver = LLMRoleResolver(_config(default="model-a"))

    with pytest.raises(UnknownRoleError) as caught:
        resolver.resolve("not_a_role")

    message = str(caught.value)
    assert "unknown role: not_a_role" in message
    assert "allowed:" in message
    for role in CORE_LLM_ROLES:
        assert role in message
    assert caught.value.details["role"] == "not_a_role"
    assert caught.value.details["allowed"] == list(CORE_LLM_ROLES)
    assert isinstance(caught.value, ConfigError)


def test_a_registered_module_role_resolves_and_unregisters_cleanly() -> None:
    role = "test_resolution_role"
    assert role not in role_registry

    try:
        register_role(role, module="tests")
        resolver = LLMRoleResolver(_config(default="model-a", **{role: "model-b"}))

        assert resolver.resolve(role) == "model-b"
        assert role in resolver.allowed_roles()
    finally:
        role_registry.unregister(role)

    with pytest.raises(UnknownRoleError):
        resolver.resolve(role)


def test_resolution_accepts_a_config_mapping_or_a_full_app_config() -> None:
    llm = {"roles": {"default": "model-a"}}
    app = SimpleNamespace(llm=LLMConfig.from_mapping(llm))

    assert resolve_model("writing", llm) == "model-a"
    assert resolve_model("writing", {"llm": llm}) == "model-a"
    assert resolve_model("writing", app) == "model-a"


def test_the_config_may_come_before_or_after_the_role() -> None:
    config = _config(default="model-a")

    assert resolve_model("writing", config) == "model-a"
    assert resolve_model(config, "writing") == "model-a"
    assert model_for_role("writing", config) == "model-a"


def test_models_by_role_covers_every_allowed_role() -> None:
    resolved = models_by_role(_config(default="model-a", writing="model-b"))

    assert resolved["default"] == "model-a"
    assert resolved["writing"] == "model-b"
    assert resolved["factcheck"] == "model-a"
    assert tuple(resolved) == CORE_LLM_ROLES


def test_request_resolution_fills_the_model_and_canonicalizes_the_role() -> None:
    request = _request(" writing ")

    resolved = resolve_request_model(request, _config(default="model-a"))

    assert resolved.role == "writing"
    assert resolved.model == "model-a"
    assert request.model is None


def test_an_explicit_request_model_is_not_overridden() -> None:
    request = _request("writing", model="model-explicit")

    resolved = LLMRoleResolver(_config(writing="model-b")).resolve_request(request)

    assert resolved.model == "model-explicit"


def test_request_resolution_still_rejects_an_unknown_role() -> None:
    with pytest.raises(UnknownRoleError):
        resolve_request_model(_request("not_a_role"), _config(default="model-a"))


def test_an_invalid_role_type_is_rejected() -> None:
    with pytest.raises(TypeError):
        LLMRoleResolver(_config(default="model-a")).resolve(42)  # type: ignore[arg-type]
