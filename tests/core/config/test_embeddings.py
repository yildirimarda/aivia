"""Embeddings configuration: chain, dimensions, batching, cache and prices."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    CORE_SECTION_KEYS,
    AppConfig,
    EmbeddingConfig,
    EmbeddingsConfig,
    SectionRegistry,
    register_core_sections,
)
from aivia.core.errors import ConfigError


def test_embeddings_defaults_match_the_provider_defaults() -> None:
    config = EmbeddingsConfig.from_mapping({})

    assert config.chain == []
    assert config.dimensions is None
    assert config.batch_size == 64
    assert config.cache_size == 10_000
    assert config.pricing == {}


def test_embedding_config_alias_points_at_the_same_model() -> None:
    assert EmbeddingConfig is EmbeddingsConfig


def test_embeddings_accepts_chain_dimensions_batching_cache_and_prices() -> None:
    config = EmbeddingsConfig.from_mapping(
        {
            "chain": ["openai", "voyage", "local"],
            "dimensions": 512,
            "batch_size": 32,
            "cache_size": 2_048,
            "pricing": {
                "model-a": 0.02,
                " model-b ": 0.13,
            },
        }
    )

    assert config.chain == ["openai", "voyage", "local"]
    assert config.dimensions == 512
    assert config.batch_size == 32
    assert config.cache_size == 2_048
    assert config.pricing == {"model-a": 0.02, "model-b": 0.13}
    assert config.pricing_for("model-a") == 0.02


def test_embeddings_accepts_the_field_aliases() -> None:
    config = EmbeddingsConfig.from_mapping(
        {
            "provider_chain": ["local"],
            "vector_dimensions": 8,
            "max_batch_size": 4,
            "max_cache_entries": 16,
            "model_prices": {"local-hash": 0.0},
        }
    )

    assert config.chain == ["local"]
    assert config.dimensions == 8
    assert config.batch_size == 4
    assert config.cache_size == 16
    assert config.pricing == {"local-hash": 0.0}


def test_canonical_embedding_fields_win_over_aliases() -> None:
    config = EmbeddingsConfig.from_mapping(
        {
            "chain": ["primary"],
            "providers": ["fallback"],
            "batch_size": 8,
            "batch": 2,
            "cache_size": 64,
            "cache": 4,
        }
    )

    assert config.chain == ["primary"]
    assert config.batch_size == 8
    assert config.cache_size == 64


@pytest.mark.parametrize("entry", ["", "   "])
def test_embedding_chain_entries_must_be_non_empty(entry: str) -> None:
    with pytest.raises(ConfigError) as caught:
        EmbeddingsConfig.from_mapping({"chain": [entry]})

    assert "chain" in str(caught.value)


def test_duplicate_embedding_chain_entries_are_rejected() -> None:
    with pytest.raises(ConfigError) as caught:
        EmbeddingsConfig.from_mapping({"chain": ["local", "local"]})

    assert "duplicate provider in chain: local" in str(caught.value)


@pytest.mark.parametrize("value", [0, -1])
def test_embedding_dimensions_must_be_positive(value: int) -> None:
    with pytest.raises(ConfigError) as caught:
        EmbeddingsConfig.from_mapping({"dimensions": value})

    assert "dimensions" in str(caught.value)


@pytest.mark.parametrize("field", ["batch_size", "cache_size"])
@pytest.mark.parametrize("value", [0, -1])
def test_embedding_batch_and_cache_sizes_must_be_positive(
    field: str, value: int
) -> None:
    with pytest.raises(ConfigError) as caught:
        EmbeddingsConfig.from_mapping({field: value})

    assert field in str(caught.value)


@pytest.mark.parametrize("value", [-0.01, float("inf"), float("-inf"), float("nan")])
def test_embedding_prices_must_be_finite_and_non_negative(value: float) -> None:
    with pytest.raises(ConfigError) as caught:
        EmbeddingsConfig.from_mapping({"pricing": {"model-a": value}})

    assert "model-a" in str(caught.value)


def test_embedding_pricing_model_names_are_stripped_and_must_not_be_empty() -> None:
    config = EmbeddingsConfig.from_mapping(
        {"pricing": {" model-a ": 1.5}}
    )
    assert list(config.pricing) == ["model-a"]

    with pytest.raises(ConfigError) as caught:
        EmbeddingsConfig.from_mapping({"pricing": {" ": 1.5}})

    assert "pricing" in str(caught.value)


def test_embedding_pricing_lookup_lists_the_configured_models() -> None:
    config = EmbeddingsConfig.from_mapping({"pricing": {"model-a": 1.5}})

    with pytest.raises(ConfigError) as caught:
        config.pricing_for("model-b")

    assert "no pricing configured for embeddings model: model-b" in str(
        caught.value
    )
    assert "model-a" in str(caught.value)


def test_embedding_values_are_interpolated_from_the_environment() -> None:
    config = EmbeddingsConfig.from_mapping(
        {
            "dimensions": "${AIVA_EMBEDDING_DIMENSIONS}",
            "pricing": {"model-a": "${AIVA_EMBEDDING_PRICE}"},
        },
        environ={
            "AIVA_EMBEDDING_DIMENSIONS": "256",
            "AIVA_EMBEDDING_PRICE": "0.02",
        },
    )

    assert config.dimensions == 256
    assert config.pricing == {"model-a": 0.02}


def test_embedding_defaults_are_independent_between_instances() -> None:
    first = EmbeddingsConfig.from_mapping({})
    second = EmbeddingsConfig.from_mapping({})

    first.chain.append("local")
    first.pricing["model-a"] = 1.0

    assert second.chain == []
    assert second.pricing == {}


def test_embeddings_section_is_registered_and_parsed_from_the_root() -> None:
    assert "embeddings" in CORE_SECTION_KEYS
    assert "embeddings" in register_core_sections(SectionRegistry())

    config = AppConfig.from_mapping(
        {
            "embeddings": {
                "chain": ["local"],
                "dimensions": 128,
                "batch_size": 16,
                "cache_size": 32,
                "pricing": {"local-hash": 0.0},
            }
        }
    )

    assert config.embeddings is not None  # type: ignore[attr-defined]
    assert config.embeddings.chain == ["local"]  # type: ignore[attr-defined]
    assert config.embeddings.dimensions == 128  # type: ignore[attr-defined]
    assert config.embeddings.batch_size == 16  # type: ignore[attr-defined]
    assert config.embeddings.cache_size == 32  # type: ignore[attr-defined]
    assert config.embeddings.pricing == {"local-hash": 0.0}  # type: ignore[attr-defined]


def test_unknown_embedding_fields_are_rejected_with_the_section_path() -> None:
    with pytest.raises(ConfigError) as caught:
        EmbeddingsConfig.from_mapping({"unknown": True})

    assert "unknown" in str(caught.value)


def test_example_config_declares_the_embeddings_controls() -> None:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    config = EmbeddingsConfig.from_mapping(payload["embeddings"])

    assert config.chain == []
    assert config.dimensions is None
    assert config.batch_size == 64
    assert config.cache_size == 10_000
    assert config.pricing == {}
