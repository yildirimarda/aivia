"""Rate limiter configuration: registration, validation and router wiring."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    CORE_SECTION_KEYS,
    AppConfig,
    OnRateLimit,
    RateLimiterConfig,
    RateLimiterSettings,
    SectionRegistry,
    register_core_sections,
)
from aivia.core.errors import ConfigError
from aivia.core.models import ProviderMeta, RateLimit
from aivia.core.ratelimit import TokenBucket
from aivia.core.router import Router
from aivia.providers import Provider


class FakeProvider(Provider):
    """A provider that only needs a name and a capability for these tests."""

    def __init__(
        self,
        name: str,
        capabilities: list[str],
        rate_limit: RateLimit | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "name": name,
            "capabilities": capabilities,
        }
        if rate_limit is not None:
            payload["rate_limit"] = rate_limit
        super().__init__(ProviderMeta.model_validate(payload))


def make_router(*providers: Provider, **kwargs: Any) -> Router:
    return Router(
        providers={provider.name: provider for provider in providers},
        **kwargs,
    )


def test_ratelimit_defaults_match_the_router_defaults() -> None:
    config = RateLimiterConfig.from_mapping({})

    assert config.requests_per_minute == 60
    assert config.burst == 0
    assert config.on_limit is OnRateLimit.WAIT
    assert config.effective_burst == 60


def test_ratelimit_config_accepts_a_rate_burst_and_on_limit() -> None:
    config = RateLimiterConfig.from_mapping(
        {"requests_per_minute": 120, "burst": 5, "on_limit": "raise"}
    )

    assert config.requests_per_minute == 120
    assert config.burst == 5
    assert config.effective_burst == 5
    assert config.on_limit is OnRateLimit.RAISE


def test_ratelimit_config_alias_points_at_the_same_model() -> None:
    assert RateLimiterSettings is RateLimiterConfig


@pytest.mark.parametrize(
    "values",
    [
        {"requests_per_minute": -1},
        {"burst": -1},
        {"on_limit": "sometimes"},
        {"unknown": True},
    ],
)
def test_invalid_ratelimit_settings_are_configuration_errors(
    values: dict[str, Any],
) -> None:
    with pytest.raises(ConfigError):
        RateLimiterConfig.from_mapping(values)


def test_ratelimit_section_is_registered_and_parsed_from_the_root() -> None:
    assert "ratelimit" in CORE_SECTION_KEYS
    assert "ratelimit" in register_core_sections(SectionRegistry())

    config = AppConfig.from_mapping(
        {
            "ratelimit": {
                "requests_per_minute": 30,
                "burst": 2,
                "on_limit": "raise",
            }
        }
    )

    assert config.ratelimit is not None  # type: ignore[attr-defined]
    assert config.ratelimit.requests_per_minute == 30  # type: ignore[attr-defined]
    assert config.ratelimit.burst == 2  # type: ignore[attr-defined]
    assert config.ratelimit.on_limit is OnRateLimit.RAISE  # type: ignore[attr-defined]


def test_example_config_declares_the_ratelimit_section() -> None:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    config = RateLimiterConfig.from_mapping(payload["ratelimit"])

    assert config.requests_per_minute == 60
    assert config.burst == 0
    assert config.on_limit is OnRateLimit.WAIT


def test_router_builds_limiters_from_the_ratelimit_config() -> None:
    image = FakeProvider("image", ["image_gen"])
    router = make_router(
        image,
        rate_limit_config=RateLimiterConfig.from_mapping({"on_limit": "raise"}),
    )

    limiter = router.limiter_for("image")
    assert limiter.requests_per_minute == 60
    assert limiter.capacity == 60
    assert limiter.on_limit is OnRateLimit.RAISE


def test_router_keeps_the_provider_rate_but_uses_the_configured_on_limit() -> None:
    image = FakeProvider(
        "image",
        ["image_gen"],
        RateLimit(requests_per_minute=240, burst=8),
    )
    router = make_router(
        image,
        rate_limit_config=RateLimiterConfig.from_mapping({"on_limit": "raise"}),
    )

    limiter = router.limiter_for("image")
    assert limiter.requests_per_minute == 240
    assert limiter.capacity == 8
    assert limiter.on_limit is OnRateLimit.RAISE


def test_router_applies_the_configured_rate_and_burst_when_the_provider_has_none() -> (
    None
):
    image = FakeProvider("image", ["image_gen"])
    router = make_router(
        image,
        rate_limit_config=RateLimiterConfig.from_mapping(
            {"requests_per_minute": 120, "burst": 10, "on_limit": "raise"}
        ),
    )

    limiter = router.limiter_for("image")
    assert limiter.requests_per_minute == 120
    assert limiter.capacity == 10
    assert limiter.on_limit is OnRateLimit.RAISE


def test_router_applies_the_configured_burst_when_the_provider_only_sets_a_rate() -> (
    None
):
    image = FakeProvider(
        "image",
        ["image_gen"],
        RateLimit(requests_per_minute=240),
    )
    router = make_router(
        image,
        rate_limit_config=RateLimiterConfig.from_mapping(
            {"requests_per_minute": 120, "burst": 10}
        ),
    )

    limiter = router.limiter_for("image")
    assert limiter.requests_per_minute == 240
    assert limiter.capacity == 10


def test_router_applies_the_configured_rate_when_the_provider_only_sets_a_burst() -> (
    None
):
    image = FakeProvider(
        "image",
        ["image_gen"],
        RateLimit(burst=8),
    )
    router = make_router(
        image,
        rate_limit_config=RateLimiterConfig.from_mapping(
            {"requests_per_minute": 120, "burst": 10}
        ),
    )

    limiter = router.limiter_for("image")
    assert limiter.requests_per_minute == 120
    assert limiter.capacity == 8


def test_router_uses_the_provider_rate_as_burst_when_config_burst_is_zero() -> None:
    image = FakeProvider(
        "image",
        ["image_gen"],
        RateLimit(requests_per_minute=240),
    )
    router = make_router(
        image,
        rate_limit_config=RateLimiterConfig.from_mapping(
            {"requests_per_minute": 120, "burst": 0}
        ),
    )

    limiter = router.limiter_for("image")
    assert limiter.requests_per_minute == 240
    assert limiter.capacity == 240


def test_router_keeps_an_explicit_unlimited_provider_rate() -> None:
    image = FakeProvider(
        "image",
        ["image_gen"],
        RateLimit(requests_per_minute=0),
    )
    router = make_router(
        image,
        rate_limit_config=RateLimiterConfig.from_mapping({"requests_per_minute": 120}),
    )

    limiter = router.limiter_for("image")
    assert limiter.unlimited is True
    assert limiter.requests_per_minute == 0


def test_router_applies_the_config_defaults_after_a_provider_name_override() -> None:
    metadata = ProviderMeta(name="ignored", capabilities=["image_gen"])
    provider = Provider(metadata=metadata, name="image")
    router = make_router(
        provider,
        rate_limit_config=RateLimiterConfig.from_mapping(
            {"requests_per_minute": 120, "burst": 10}
        ),
    )

    limiter = router.limiter_for("image")
    assert limiter.requests_per_minute == 120
    assert limiter.capacity == 10


def test_router_uses_the_default_ratelimit_config_when_none_is_given() -> None:
    image = FakeProvider("image", ["image_gen"])
    router = make_router(image)

    limiter = router.limiter_for("image")
    assert limiter.requests_per_minute == 60
    assert limiter.capacity == 60
    assert limiter.on_limit is OnRateLimit.WAIT


def test_router_rejects_a_ratelimit_config_of_the_wrong_type() -> None:
    image = FakeProvider("image", ["image_gen"])

    with pytest.raises(TypeError, match="RateLimiterConfig"):
        make_router(image, rate_limit_config={"on_limit": "raise"})


def test_injected_limiters_keep_their_own_settings() -> None:
    image = FakeProvider("image", ["image_gen"])
    limiter = TokenBucket(
        requests_per_minute=10,
        burst=1,
        on_limit=OnRateLimit.WAIT,
        name="image",
    )
    router = make_router(
        image,
        rate_limiters={"image": limiter},
        rate_limit_config=RateLimiterConfig.from_mapping(
            {"requests_per_minute": 120, "burst": 4, "on_limit": "raise"}
        ),
    )

    assert router.limiter_for("image") is limiter
    assert limiter.requests_per_minute == 10
    assert limiter.capacity == 1
    assert limiter.on_limit is OnRateLimit.WAIT
