"""Circuit breaker configuration: registration, validation and router wiring."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.circuit import CircuitBreaker
from aivia.core.config import (
    CORE_SECTION_KEYS,
    AppConfig,
    CircuitBreakerConfig,
    CircuitConfig,
    SectionRegistry,
    register_core_sections,
)
from aivia.core.errors import ConfigError
from aivia.core.models import ProviderMeta
from aivia.core.router import Router
from aivia.providers import Provider


class FakeProvider(Provider):
    """A provider that only needs a name and a capability for these tests."""

    def __init__(self, name: str, capabilities: list[str]) -> None:
        super().__init__(
            ProviderMeta.model_validate({"name": name, "capabilities": capabilities})
        )


def make_router(*providers: FakeProvider, **kwargs: Any) -> Router:
    return Router(
        providers={provider.name: provider for provider in providers},
        **kwargs,
    )


def test_circuit_defaults_match_the_router_defaults() -> None:
    config = CircuitBreakerConfig.from_mapping({})

    assert config.failure_threshold == 5
    assert config.cooldown_seconds == 60.0


def test_circuit_config_accepts_a_threshold_and_cooldown() -> None:
    config = CircuitBreakerConfig.from_mapping(
        {"failure_threshold": 2, "cooldown_seconds": 12.5}
    )

    assert config.failure_threshold == 2
    assert config.cooldown_seconds == 12.5


def test_circuit_config_alias_points_at_the_same_model() -> None:
    assert CircuitConfig is CircuitBreakerConfig


@pytest.mark.parametrize(
    "values",
    [
        {"failure_threshold": 0},
        {"cooldown_seconds": -1},
        {"cooldown_seconds": float("nan")},
        {"cooldown_seconds": float("inf")},
        {"unknown": True},
    ],
)
def test_invalid_circuit_settings_are_configuration_errors(
    values: dict[str, Any],
) -> None:
    with pytest.raises(ConfigError):
        CircuitBreakerConfig.from_mapping(values)


def test_circuit_section_is_registered_and_parsed_from_the_root() -> None:
    assert "circuit" in CORE_SECTION_KEYS
    assert "circuit" in register_core_sections(SectionRegistry())

    config = AppConfig.from_mapping(
        {"circuit": {"failure_threshold": 3, "cooldown_seconds": 7.5}}
    )

    assert config.circuit is not None  # type: ignore[attr-defined]
    assert config.circuit.failure_threshold == 3  # type: ignore[attr-defined]
    assert config.circuit.cooldown_seconds == 7.5  # type: ignore[attr-defined]


def test_example_config_declares_the_circuit_section() -> None:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    config = CircuitBreakerConfig.from_mapping(payload["circuit"])

    assert config.failure_threshold == 5
    assert config.cooldown_seconds == 60.0


def test_router_builds_breakers_from_the_circuit_config() -> None:
    image = FakeProvider("image", ["image_gen"])
    router = make_router(
        image,
        circuit_config=CircuitBreakerConfig.from_mapping(
            {"failure_threshold": 2, "cooldown_seconds": 4.5}
        ),
    )

    breaker = router.circuit_for("image")
    assert breaker.failure_threshold == 2
    assert breaker.cooldown_seconds == 4.5


def test_router_uses_the_default_circuit_config_when_none_is_given() -> None:
    image = FakeProvider("image", ["image_gen"])
    router = make_router(image)

    breaker = router.circuit_for("image")
    assert breaker.failure_threshold == 5
    assert breaker.cooldown_seconds == 60.0


def test_router_rejects_a_circuit_config_of_the_wrong_type() -> None:
    image = FakeProvider("image", ["image_gen"])

    with pytest.raises(TypeError, match="CircuitBreakerConfig"):
        make_router(image, circuit_config={"failure_threshold": 2})


def test_injected_breakers_keep_their_own_settings() -> None:
    image = FakeProvider("image", ["image_gen"])
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=9)
    router = make_router(
        image,
        circuit_breakers={"image": breaker},
        circuit_config=CircuitBreakerConfig.from_mapping(
            {"failure_threshold": 7, "cooldown_seconds": 99}
        ),
    )

    assert router.circuit_for("image") is breaker
    assert breaker.failure_threshold == 1
    assert breaker.cooldown_seconds == 9
