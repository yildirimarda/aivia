"""Reference configuration: similarity, gates, regeneration and downloads."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    CORE_SECTION_KEYS,
    DownloadPolicy,
    ReferenceConfig,
    ReferenceDownloadConfig,
    ReferenceDownloadPolicy,
    SectionRegistry,
    SimilarityLevel,
    register_core_sections,
)
from aivia.core.errors import ConfigError


def example_section(name: str) -> dict[str, Any]:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    return cast(dict[str, Any], payload[name])


def test_reference_defaults_bound_similarity_regeneration_and_downloads() -> None:
    reference = ReferenceConfig.from_mapping({})

    assert reference.enabled is True
    assert reference.similarity_level is SimilarityLevel.MEDIUM
    assert reference.similarity is SimilarityLevel.MEDIUM
    assert reference.level is SimilarityLevel.MEDIUM
    assert reference.visual_threshold == 0.9
    assert reference.visual_limit == 0.9
    assert reference.text_threshold == 0.85
    assert reference.text_limit == 0.85
    assert reference.max_regenerations == 2
    assert reference.regeneration_limit == 2
    assert reference.download_policy == ReferenceDownloadConfig()
    assert reference.download is reference.download_policy
    assert reference.downloads_enabled is True
    assert reference.download_policy.mode is DownloadPolicy.ALLOW
    assert reference.download_policy.enabled is True
    assert reference.download_policy.allow is True
    assert reference.download_policy.ask is False
    assert reference.download_policy.max_bytes == 52_428_800
    assert reference.download_policy.timeout_seconds == 30.0
    assert reference.download_policy.allowed_hosts == []
    assert reference.download_policy.follow_redirects is True
    assert reference.download_policy.max_redirects == 5
    assert reference.download_policy.cache is True


def test_reference_accepts_every_documented_field() -> None:
    reference = ReferenceConfig.from_mapping(
        {
            "enabled": False,
            "similarity_level": "high",
            "visual_threshold": 0.95,
            "text_threshold": 0.7,
            "max_regenerations": 0,
            "download_policy": {
                "mode": "ask",
                "max_bytes": 1024,
                "timeout_seconds": 5.5,
                "allowed_hosts": ["Example.Test"],
                "follow_redirects": False,
                "max_redirects": 0,
                "cache": False,
            },
        }
    )

    assert reference.enabled is False
    assert reference.similarity_level is SimilarityLevel.HIGH
    assert reference.visual_threshold == 0.95
    assert reference.text_threshold == 0.7
    assert reference.max_regenerations == 0
    assert reference.download_policy.mode is DownloadPolicy.ASK
    assert reference.download_policy.enabled is True
    assert reference.download_policy.ask is True
    assert reference.download_policy.max_bytes == 1024
    assert reference.download_policy.timeout_seconds == 5.5
    assert reference.download_policy.allowed_hosts == ["example.test"]
    assert reference.download_policy.follow_redirects is False
    assert reference.download_policy.max_redirects == 0
    assert reference.download_policy.cache is False


def test_reference_download_policy_alias_points_at_the_same_class() -> None:
    assert ReferenceDownloadPolicy is ReferenceDownloadConfig


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("low", SimilarityLevel.LOW),
        ("medium", SimilarityLevel.MEDIUM),
        ("high", SimilarityLevel.HIGH),
        ("loose", SimilarityLevel.LOW),
        ("balanced", SimilarityLevel.MEDIUM),
        ("strict", SimilarityLevel.HIGH),
        (" HIGH ", SimilarityLevel.HIGH),
        (SimilarityLevel.LOW, SimilarityLevel.LOW),
    ],
)
def test_reference_accepts_every_similarity_level(
    value: object, expected: SimilarityLevel
) -> None:
    reference = ReferenceConfig.from_mapping({"similarity_level": value})

    assert reference.similarity_level is expected


def test_reference_rejects_an_unknown_similarity_level() -> None:
    with pytest.raises(ConfigError) as caught:
        ReferenceConfig.from_mapping({"similarity_level": "identical"})

    assert "similarity_level" in str(caught.value)


@pytest.mark.parametrize(
    ("values", "field"),
    [
        ({"visual_threshold": -0.01}, "visual_threshold"),
        ({"visual_threshold": 1.01}, "visual_threshold"),
        ({"visual_threshold": float("inf")}, "visual_threshold"),
        ({"text_threshold": -0.01}, "text_threshold"),
        ({"text_threshold": 1.01}, "text_threshold"),
        ({"text_threshold": float("nan")}, "text_threshold"),
    ],
)
def test_reference_thresholds_must_be_finite_fractions(
    values: dict[str, Any], field: str
) -> None:
    with pytest.raises(ConfigError) as caught:
        ReferenceConfig.from_mapping(values)

    assert field in str(caught.value)


def test_reference_regeneration_limit_may_be_zero_but_not_negative() -> None:
    assert ReferenceConfig.from_mapping({"max_regenerations": 0}).max_regenerations == 0

    with pytest.raises(ConfigError) as caught:
        ReferenceConfig.from_mapping({"max_regenerations": -1})

    assert "max_regenerations" in str(caught.value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("allow", DownloadPolicy.ALLOW),
        ("always", DownloadPolicy.ALLOW),
        ("ask", DownloadPolicy.ASK),
        ("confirm", DownloadPolicy.ASK),
        ("deny", DownloadPolicy.DENY),
        ("never", DownloadPolicy.DENY),
        (True, DownloadPolicy.ALLOW),
        (False, DownloadPolicy.DENY),
    ],
)
def test_reference_download_policy_accepts_shorthand(
    value: object, expected: DownloadPolicy
) -> None:
    reference = ReferenceConfig.from_mapping({"download_policy": value})

    assert reference.download_policy.mode is expected


def test_reference_download_policy_accepts_a_nested_mode() -> None:
    reference = ReferenceConfig.from_mapping(
        {"download_policy": {"mode": "on_demand", "max_bytes": 2048}}
    )

    assert reference.download_policy.mode is DownloadPolicy.ASK
    assert reference.download_policy.max_bytes == 2048


def test_reference_accepts_aliases_and_canonical_fields_win() -> None:
    aliased = ReferenceConfig.from_mapping(
        {
            "similarity": "high",
            "visual_similarity_threshold": 0.8,
            "text_limit": 0.6,
            "regeneration_limit": 4,
            "download": {"mode": "block"},
        }
    )
    canonical = ReferenceConfig.from_mapping(
        {
            "similarity_level": "low",
            "similarity": "high",
            "visual_threshold": 0.5,
            "visual_limit": 0.8,
            "max_regenerations": 1,
            "regeneration_limit": 4,
        }
    )

    assert aliased.similarity_level is SimilarityLevel.HIGH
    assert aliased.visual_threshold == 0.8
    assert aliased.text_threshold == 0.6
    assert aliased.max_regenerations == 4
    assert aliased.download_policy.mode is DownloadPolicy.DENY
    assert canonical.similarity_level is SimilarityLevel.LOW
    assert canonical.visual_threshold == 0.5
    assert canonical.max_regenerations == 1


@pytest.mark.parametrize(
    ("values", "field"),
    [
        ({"mode": "sometimes"}, "mode"),
        ({"max_bytes": 0}, "max_bytes"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"timeout_seconds": float("inf")}, "timeout_seconds"),
        ({"max_redirects": -1}, "max_redirects"),
        ({"allowed_hosts": ["a.test", "A.test"]}, "allowed host"),
        ({"allowed_hosts": [""]}, "allowed host"),
        ({"allowed_hosts": ["  "]}, "allowed host"),
    ],
)
def test_invalid_download_policy_fields_are_rejected(
    values: dict[str, Any], field: str
) -> None:
    with pytest.raises(ConfigError) as caught:
        ReferenceDownloadConfig.from_mapping(values)

    assert field in str(caught.value)


def test_reference_nested_download_policy_defaults_are_independent() -> None:
    first = ReferenceConfig.from_mapping({})
    second = ReferenceConfig.from_mapping({})

    first.download_policy.allowed_hosts.append("example.test")

    assert second.download_policy.allowed_hosts == []


def test_reference_is_registered_as_a_core_section() -> None:
    assert "reference" in CORE_SECTION_KEYS

    sections = register_core_sections(SectionRegistry())
    parsed = sections.parse({"reference": {"similarity_level": "high"}})

    assert parsed.reference.similarity_level is SimilarityLevel.HIGH  # type: ignore[attr-defined]


def test_reference_unknown_keys_report_the_full_path() -> None:
    sections = register_core_sections(SectionRegistry())

    with pytest.raises(ConfigError) as caught:
        sections.parse({"reference": {"bogus": True}})

    assert "reference.bogus" in str(caught.value)


def test_the_packaged_example_declares_the_reference_controls() -> None:
    reference = ReferenceConfig.from_mapping(example_section("reference"))

    assert reference.enabled is True
    assert reference.similarity_level is SimilarityLevel.MEDIUM
    assert reference.visual_threshold == 0.9
    assert reference.text_threshold == 0.85
    assert reference.max_regenerations == 2
    assert reference.download_policy.mode is DownloadPolicy.ALLOW
    assert reference.download_policy.max_bytes == 52_428_800
    assert reference.download_policy.timeout_seconds == 30.0
    assert reference.download_policy.allowed_hosts == []
    assert reference.download_policy.follow_redirects is True
    assert reference.download_policy.max_redirects == 5
    assert reference.download_policy.cache is True
