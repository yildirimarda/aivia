"""HTTP client configuration: registration, validation and provider wiring."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    CORE_SECTION_KEYS,
    AppConfig,
    HttpClientConfig,
    HttpConfig,
    SectionRegistry,
    register_core_sections,
)
from aivia.core.errors import ConfigError
from aivia.providers import Provider


def make_provider(**kwargs: Any) -> Provider:
    return Provider(
        {"name": "fake", "capabilities": ["image_gen"]},
        **kwargs,
    )


def test_http_defaults_match_the_client_defaults() -> None:
    config = HttpClientConfig.from_mapping({})

    assert config.timeout_seconds == 30.0
    assert config.max_attempts == 3
    assert config.backoff_base_seconds == 0.5
    assert config.backoff_max_seconds == 30.0
    assert config.jitter_ratio == 0.5
    assert config.retry_statuses == (408, 425, 429, 500, 502, 503, 504)
    assert config.respect_retry_after is True
    assert config.follow_redirects is True


def test_http_config_accepts_a_timeout_and_retry_policy() -> None:
    config = HttpClientConfig.from_mapping(
        {
            "timeout_seconds": 12.5,
            "connect_timeout_seconds": 2.0,
            "read_timeout_seconds": 4.0,
            "write_timeout_seconds": 3.0,
            "pool_timeout_seconds": 1.5,
            "max_attempts": 5,
            "backoff_base_seconds": 0.25,
            "backoff_max_seconds": 8.0,
            "jitter_ratio": 0.25,
            "retry_statuses": [429, 503, 429],
            "respect_retry_after": False,
            "follow_redirects": False,
        }
    )

    assert config.timeout_seconds == 12.5
    assert config.connect_timeout_seconds == 2.0
    assert config.read_timeout_seconds == 4.0
    assert config.write_timeout_seconds == 3.0
    assert config.pool_timeout_seconds == 1.5
    assert config.max_attempts == 5
    assert config.backoff_base_seconds == 0.25
    assert config.backoff_max_seconds == 8.0
    assert config.jitter_ratio == 0.25
    assert config.retry_statuses == (429, 503)
    assert config.respect_retry_after is False
    assert config.follow_redirects is False


def test_http_config_alias_points_at_the_same_model() -> None:
    assert HttpConfig is HttpClientConfig


@pytest.mark.parametrize(
    "values",
    [
        {"timeout_seconds": 0},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": float("inf")},
        {"max_attempts": 0},
        {"backoff_base_seconds": -1},
        {"backoff_base_seconds": 10, "backoff_max_seconds": 1},
        {"jitter_ratio": 1.5},
        {"retry_statuses": (99,)},
        {"unknown": True},
    ],
)
def test_invalid_http_settings_are_configuration_errors(
    values: dict[str, Any],
) -> None:
    with pytest.raises(ConfigError):
        HttpClientConfig.from_mapping(values)


def test_http_section_is_registered_and_parsed_from_the_root() -> None:
    assert "http" in CORE_SECTION_KEYS
    assert "http" in register_core_sections(SectionRegistry())

    config = AppConfig.from_mapping(
        {"http": {"timeout_seconds": 7.5, "max_attempts": 2}}
    )

    assert config.http is not None  # type: ignore[attr-defined]
    assert config.http.timeout_seconds == 7.5  # type: ignore[attr-defined]
    assert config.http.max_attempts == 2  # type: ignore[attr-defined]


def test_example_config_declares_the_http_section() -> None:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    config = HttpClientConfig.from_mapping(payload["http"])

    assert config.timeout_seconds == 30.0
    assert config.max_attempts == 3
    assert config.jitter_ratio == 0.5
    assert config.retry_statuses == (408, 425, 429, 500, 502, 503, 504)


def test_provider_uses_the_http_config_for_its_client() -> None:
    config = HttpClientConfig.from_mapping(
        {"timeout_seconds": 4.5, "max_attempts": 2, "jitter_ratio": 0.0}
    )
    provider = make_provider(http_config=config)

    assert provider.http_config is config
    assert provider.http.config is config
    assert provider.http_client.config is config
    assert provider.create_http_client().config is config


def test_provider_uses_the_default_http_config_when_none_is_given() -> None:
    provider = make_provider()

    assert provider.http_config.timeout_seconds == 30.0
    assert provider.http_config.max_attempts == 3
    assert provider.http.config is provider.http_config


def test_provider_rejects_an_http_config_of_the_wrong_type() -> None:
    with pytest.raises(TypeError, match="HttpClientConfig"):
        make_provider(http_config={"timeout_seconds": 5})
