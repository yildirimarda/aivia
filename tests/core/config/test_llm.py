"""LLM configuration: core roles, module roles, token prices and repair limit."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    CORE_LLM_ROLES,
    CORE_ROLES,
    CoreLLMRole,
    LLMConfig,
    LLMPricing,
    LLMRole,
    LLMRoleEnum,
    LLMRoleRegistry,
    ModelPricing,
    Role,
    RoleRegistry,
    SectionRegistry,
    TokenPricing,
    register_core_sections,
    register_role,
    resolve_role,
    role_registry,
)
from aivia.core.errors import ConfigError

CORE_ROLE_VALUES = (
    "default",
    "writing",
    "factcheck",
    "lyrics",
    "planner",
    "reference_analysis",
    "vision_qa",
    "safety",
    "translation_check",
    "prompt_enhance",
)


def test_core_role_enum_is_exactly_the_documented_set() -> None:
    assert tuple(role.value for role in LLMRole) == CORE_ROLE_VALUES
    assert CORE_LLM_ROLES == CORE_ROLE_VALUES
    assert CORE_ROLES == CORE_ROLE_VALUES


def test_role_enum_aliases_point_at_the_same_class() -> None:
    assert CoreLLMRole is LLMRole
    assert LLMRoleEnum is LLMRole
    assert Role is LLMRole
    assert RoleRegistry is LLMRoleRegistry


def test_pricing_aliases_point_at_the_same_model() -> None:
    assert LLMPricing is TokenPricing
    assert ModelPricing is TokenPricing


def test_llm_defaults_have_no_chain_roles_or_prices_and_two_repairs() -> None:
    config = LLMConfig.from_mapping({})

    assert config.chain == []
    assert config.roles == {}
    assert config.pricing == {}
    assert config.max_repairs == 2
    assert config.repair_limit == 2


def test_llm_accepts_chain_roles_prices_and_repair_limit() -> None:
    config = LLMConfig.from_mapping(
        {
            "chain": ["primary", "fallback"],
            "roles": {
                "default": "model-a",
                "writing": "model-b",
                "factcheck": "model-c",
            },
            "pricing": {
                "model-a": {"input_per_million": 3, "output_per_million": 15},
                "model-b": {"input": 1.5, "output": 6},
            },
            "max_repairs": 4,
        }
    )

    assert config.chain == ["primary", "fallback"]
    assert config.roles == {
        "default": "model-a",
        "writing": "model-b",
        "factcheck": "model-c",
    }
    assert config.pricing["model-a"].input_per_million == 3.0
    assert config.pricing["model-a"].output_per_million == 15.0
    assert config.pricing["model-b"].input == 1.5
    assert config.pricing["model-b"].output == 6.0
    assert config.max_repairs == 4
    assert config.repair_limit == 4


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("repair_limit", 3),
        ("max_repair_attempts", 3),
        ("repair_attempts", 3),
        ("max_repair", 3),
        ("repairs", 3),
    ],
)
def test_repair_limit_aliases_are_accepted(alias: str, expected: int) -> None:
    config = LLMConfig.from_mapping({alias: expected})

    assert config.max_repairs == expected


def test_repair_limit_accepts_zero_and_rejects_negative_values() -> None:
    assert LLMConfig.from_mapping({"max_repairs": 0}).max_repairs == 0

    with pytest.raises(ConfigError) as caught:
        LLMConfig.from_mapping({"max_repairs": -1})

    assert "max_repairs" in str(caught.value)


def test_llm_accepts_the_long_and_short_field_aliases() -> None:
    config = LLMConfig.from_mapping(
        {
            "provider_chain": ["primary"],
            "role_models": {"default": "model-a"},
            "token_pricing": {"model-a": {"prompt": 2, "completion": 8}},
            "repair_limit": 1,
        }
    )

    assert config.chain == ["primary"]
    assert config.roles == {"default": "model-a"}
    assert config.pricing["model-a"].prompt == 2.0
    assert config.pricing["model-a"].completion == 8.0
    assert config.max_repairs == 1


def test_canonical_fields_win_over_aliases() -> None:
    config = LLMConfig.from_mapping(
        {
            "max_repairs": 5,
            "repair_limit": 1,
            "chain": ["primary"],
            "providers": ["fallback"],
        }
    )

    assert config.max_repairs == 5
    assert config.chain == ["primary"]


@pytest.mark.parametrize("entry", ["", "   "])
def test_chain_entries_must_be_non_empty(entry: str) -> None:
    with pytest.raises(ConfigError) as caught:
        LLMConfig.from_mapping({"chain": [entry]})

    assert "chain" in str(caught.value)


def test_duplicate_chain_entries_are_rejected() -> None:
    with pytest.raises(ConfigError) as caught:
        LLMConfig.from_mapping({"chain": ["primary", "primary"]})

    assert "duplicate provider in chain: primary" in str(caught.value)


def test_every_core_role_is_accepted_and_stored_by_its_canonical_name() -> None:
    config = LLMConfig.from_mapping(
        {"roles": {role.value: f"model-{role.value}" for role in LLMRole}}
    )

    assert config.roles == {role.value: f"model-{role.value}" for role in LLMRole}


def test_unknown_role_fails_with_the_allowed_values() -> None:
    with pytest.raises(ConfigError) as caught:
        LLMConfig.from_mapping({"roles": {"not_a_role": "model-a"}})

    message = str(caught.value)
    assert "unknown role: not_a_role" in message
    assert "allowed:" in message
    assert "default" in message
    assert "prompt_enhance" in message
    assert caught.value.details["role"] == "not_a_role"
    assert "not_a_role" not in caught.value.details["allowed"]


def test_module_roles_are_accepted_only_after_registration() -> None:
    role = "test_module_role"
    assert role not in role_registry

    with pytest.raises(ConfigError) as caught:
        LLMConfig.from_mapping({"roles": {role: "model-a"}})
    assert role in str(caught.value)
    assert role not in caught.value.details["allowed"]

    try:
        registered = register_role(role, module="tests")
        config = LLMConfig.from_mapping({"roles": {role: "model-a"}})
    finally:
        role_registry.unregister(role)

    assert registered.name == role
    assert registered.module == "tests"
    assert registered.core is False
    assert config.roles == {role: "model-a"}


def test_module_role_registration_rejects_a_duplicate() -> None:
    with pytest.raises(ConfigError, match="LLM role already registered: default"):
        role_registry.register("default")

    role = "test_duplicate_role"
    try:
        role_registry.register(role, module="first")
        with pytest.raises(ConfigError) as caught:
            role_registry.register(role, module="second")
    finally:
        role_registry.unregister(role)

    assert caught.value.details["existing_module"] == "first"
    assert caught.value.details["module"] == "second"


def test_core_roles_cannot_be_unregistered() -> None:
    with pytest.raises(ConfigError, match="cannot unregister core LLM role"):
        role_registry.unregister("default")


def test_a_fresh_registry_contains_only_the_core_roles() -> None:
    registry = LLMRoleRegistry()

    assert registry.names() == CORE_ROLE_VALUES
    assert registry.allowed() == CORE_ROLE_VALUES
    assert "scoring" not in registry

    registry.register("scoring", module="scoring")

    assert "scoring" in registry
    assert registry.resolve(" scoring ") == "scoring"
    assert registry.get("scoring").module == "scoring"


def test_resolve_role_uses_the_shared_registry() -> None:
    assert resolve_role("writing") == "writing"

    with pytest.raises(ConfigError, match="unknown role"):
        resolve_role("not_a_role")


def test_roles_are_stripped_and_models_must_not_be_empty() -> None:
    config = LLMConfig.from_mapping({"roles": {" default ": " model-a "}})

    assert config.roles == {"default": "model-a"}

    with pytest.raises(ConfigError) as caught:
        LLMConfig.from_mapping({"roles": {"default": "   "}})

    assert "roles" in str(caught.value)


def test_pricing_accepts_every_per_million_alias() -> None:
    pricing = TokenPricing.from_mapping(
        {
            "input_per_million_tokens": 2,
            "output_per_million_tokens": 10,
        }
    )

    assert pricing.input_per_million == 2.0
    assert pricing.output_per_million == 10.0
    assert pricing.input_per_million_tokens == 2.0
    assert pricing.output_per_million_tokens == 10.0


@pytest.mark.parametrize(
    "value",
    [-0.01, float("inf"), float("-inf"), float("nan")],
)
@pytest.mark.parametrize("field", ["input_per_million", "output_per_million"])
def test_token_prices_must_be_finite_and_non_negative(field: str, value: float) -> None:
    with pytest.raises(ConfigError) as caught:
        TokenPricing.from_mapping({field: value})

    assert field in str(caught.value)


def test_pricing_model_names_are_stripped_and_must_not_be_empty() -> None:
    config = LLMConfig.from_mapping(
        {"pricing": {" model-a ": {"input": 1, "output": 2}}}
    )

    assert list(config.pricing) == ["model-a"]

    with pytest.raises(ConfigError) as caught:
        LLMConfig.from_mapping({"pricing": {" ": {"input": 1, "output": 2}}})

    assert "pricing" in str(caught.value)


def test_model_and_pricing_lookups_use_the_configured_values() -> None:
    config = LLMConfig.from_mapping(
        {
            "roles": {"writing": "model-a"},
            "pricing": {"model-a": {"input": 1, "output": 2}},
        }
    )

    assert config.model_for("writing") == "model-a"
    assert config.pricing_for("model-a").input_per_million == 1.0

    with pytest.raises(ConfigError, match="no model configured for LLM role"):
        config.model_for("default")

    with pytest.raises(ConfigError, match="no pricing configured for LLM model"):
        config.pricing_for("model-b")


def test_registry_reports_the_llm_role_path() -> None:
    sections = register_core_sections(SectionRegistry())

    with pytest.raises(ConfigError) as caught:
        sections.parse({"llm": {"roles": {"not_a_role": "model-a"}}})

    assert "roles.not_a_role" in str(caught.value)
    assert "allowed:" in str(caught.value)


def test_llm_values_are_interpolated_from_the_environment() -> None:
    config = LLMConfig.from_mapping(
        {"roles": {"default": "${AIVA_LLM_MODEL}"}},
        environ={"AIVA_LLM_MODEL": "model-a"},
    )

    assert config.roles == {"default": "model-a"}


def test_defaults_are_independent_between_instances() -> None:
    first = LLMConfig.from_mapping({})
    second = LLMConfig.from_mapping({})

    first.chain.append("primary")
    first.roles["default"] = "model-a"
    first.pricing["model-a"] = TokenPricing()

    assert second.chain == []
    assert second.roles == {}
    assert second.pricing == {}


def test_example_config_declares_the_llm_controls() -> None:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    config = LLMConfig.from_mapping(payload["llm"])

    assert config.chain == []
    assert config.roles == {}
    assert config.pricing == {}
    assert config.max_repairs == 2
