"""Similarity levels map to enforced spec fields with layered precedence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from aivia.core.config import resolve_config
from aivia.core.config.reference import ReferenceConfig, SimilarityLevel
from aivia.core.config.resolve import (
    CLI_ORIGIN,
    PROFILE_ORIGIN_PREFIX,
    USER_ORIGIN,
    ResolvedConfig,
)
from aivia.reference import ENFORCED_SECTIONS, enforced_fields
from aivia.reference.schema import StyleSpec


def write_yaml(path: Path, payload: str | dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else yaml.safe_dump(payload)
    path.write_text(text, encoding="utf-8")
    return path


def test_low_enforces_only_structure() -> None:
    assert enforced_fields(SimilarityLevel.LOW) == frozenset({"structure"})


def test_medium_enforces_structure_and_pacing() -> None:
    assert enforced_fields(SimilarityLevel.MEDIUM) == frozenset({"structure", "pacing"})


def test_high_enforces_every_section() -> None:
    assert enforced_fields(SimilarityLevel.HIGH) == ENFORCED_SECTIONS


def test_enforced_fields_are_real_style_spec_sections() -> None:
    spec = StyleSpec()
    for level in SimilarityLevel:
        for field in enforced_fields(level):
            assert hasattr(spec, field), f"{level}: {field} is not a style spec section"


def test_unknown_similarity_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown similarity level"):
        enforced_fields("identical")  # type: ignore[arg-type]


def _resolve(
    tmp_path: Path,
    *,
    defaults: dict[str, Any] | None = None,
    user: dict[str, Any] | None = None,
    profile: str | None = None,
    set_values: list[str] | None = None,
) -> ResolvedConfig:
    defaults_path = write_yaml(
        tmp_path / "defaults" / "config.example.yaml",
        defaults or {"schema_version": 1, "reference": {"similarity_level": "medium"}},
    )
    user_path = None
    if user is not None:
        user_path = write_yaml(tmp_path / "user" / "config.yaml", user)
    return resolve_config(
        defaults_file=defaults_path,
        config_file=user_path,
        profile=profile,
        set_values=set_values,
    )


def test_defaults_give_medium_similarity_level(tmp_path: Path) -> None:
    resolved = _resolve(tmp_path)

    assert resolved.values["reference"]["similarity_level"] == "medium"
    assert resolved.origins["reference.similarity_level"] == "defaults"


def test_user_config_overrides_the_global_default(tmp_path: Path) -> None:
    resolved = _resolve(tmp_path, user={"reference": {"similarity_level": "low"}})

    assert resolved.values["reference"]["similarity_level"] == "low"
    assert resolved.origins["reference.similarity_level"] == USER_ORIGIN


def test_profile_overrides_the_user_config(tmp_path: Path) -> None:
    resolved = _resolve(
        tmp_path,
        user={
            "reference": {"similarity_level": "low"},
            "profiles": {
                "short": {
                    "overrides": {
                        "reference": {"similarity_level": "high"},
                    }
                }
            },
        },
        profile="short",
    )

    assert resolved.values["reference"]["similarity_level"] == "high"
    assert resolved.origins["reference.similarity_level"] == (
        f"{PROFILE_ORIGIN_PREFIX}short"
    )


def test_cli_set_wins_over_profile_and_global(tmp_path: Path) -> None:
    resolved = _resolve(
        tmp_path,
        defaults={"schema_version": 1, "reference": {"similarity_level": "medium"}},
        user={
            "reference": {"similarity_level": "low"},
            "profiles": {
                "short": {
                    "overrides": {
                        "reference": {"similarity_level": "high"},
                    }
                }
            },
        },
        profile="short",
        set_values=["reference.similarity_level=low"],
    )

    assert resolved.values["reference"]["similarity_level"] == "low"
    assert resolved.origins["reference.similarity_level"] == CLI_ORIGIN


def test_cli_set_wins_without_a_profile(tmp_path: Path) -> None:
    resolved = _resolve(
        tmp_path,
        user={"reference": {"similarity_level": "low"}},
        set_values=["reference.similarity_level=high"],
    )

    assert resolved.values["reference"]["similarity_level"] == "high"
    assert resolved.origins["reference.similarity_level"] == CLI_ORIGIN


def test_enforced_fields_follow_the_effective_level(tmp_path: Path) -> None:
    resolved = _resolve(
        tmp_path,
        defaults={"schema_version": 1, "reference": {"similarity_level": "medium"}},
        user={
            "reference": {"similarity_level": "low"},
            "profiles": {
                "short": {
                    "overrides": {
                        "reference": {"similarity_level": "high"},
                    }
                }
            },
        },
        profile="short",
        set_values=["reference.similarity_level=high"],
    )

    config = ReferenceConfig.model_validate(resolved.values["reference"])
    assert config.similarity_level is SimilarityLevel.HIGH
    assert enforced_fields(config.similarity_level) == ENFORCED_SECTIONS
