"""Research configuration: repeat window and embedding threshold."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    DEFAULT_DEDUPE_WINDOW,
    DEFAULT_EMBEDDING_THRESHOLD,
    AppConfig,
    ConfigModel,
    DedupeWindow,
    ResearchConfig,
    SectionRegistry,
    build_app_config,
    register_core_sections,
    register_research_section,
    research_config,
)
from aivia.core.config.resolve import resolve_config
from aivia.core.errors import ConfigError
from aivia.research.config import ResearchConfig as ModuleResearchConfig
from aivia.research.config import register_research_config


def test_research_defaults_keep_a_full_year_and_the_embedding_threshold() -> None:
    config = ResearchConfig.from_mapping({})

    assert config.dedupe_window == timedelta(days=365)
    assert config.dedupe_window == 365
    assert config.window == timedelta(days=365)
    assert config.window_days == 365.0
    assert config.dedupe_window_days == 365.0
    assert config.dedupe_window_seconds == 365 * 86400
    assert config.embedding_threshold == DEFAULT_EMBEDDING_THRESHOLD
    assert config.threshold == DEFAULT_EMBEDDING_THRESHOLD
    assert config.similarity_threshold == DEFAULT_EMBEDDING_THRESHOLD


def test_the_research_module_reexports_the_same_config_model() -> None:
    assert ModuleResearchConfig is ResearchConfig
    assert register_research_config is register_research_section
    assert timedelta(days=365) == DEFAULT_DEDUPE_WINDOW


@pytest.mark.parametrize(
    ("value", "days"),
    [
        (30, 30.0),
        (2.5, 2.5),
        ("30", 30.0),
        ("30d", 30.0),
        ("2 weeks", 14.0),
        ("12h", 0.5),
        (timedelta(days=3), 3.0),
    ],
)
def test_research_window_accepts_days_and_duration_strings(
    value: object, days: float
) -> None:
    config = ResearchConfig.from_mapping({"dedupe_window": value})

    assert config.dedupe_window == timedelta(days=days)
    assert isinstance(config.dedupe_window, DedupeWindow)


def test_research_window_aliases_are_accepted_and_canonical_wins() -> None:
    aliased = ResearchConfig.from_mapping(
        {"dedupe_window_days": 7, "similarity_threshold": 0.9}
    )
    canonical = ResearchConfig.from_mapping(
        {
            "dedupe_window": 7,
            "window": 90,
            "embedding_threshold": 0.9,
            "threshold": 0.1,
        }
    )

    assert aliased.dedupe_window == timedelta(days=7)
    assert aliased.embedding_threshold == 0.9
    assert canonical.dedupe_window == timedelta(days=7)
    assert canonical.embedding_threshold == 0.9


def test_research_accepts_an_absent_embedding_threshold() -> None:
    config = ResearchConfig.from_mapping({"embedding_threshold": None})

    assert config.embedding_threshold is None
    assert config.dedupe_options() == {
        "window": config.dedupe_window,
        "embedding_threshold": None,
    }


@pytest.mark.parametrize(
    "value",
    [-1, -0.5, "-1d", float("inf"), float("nan"), True, "", "soon", []],
)
def test_invalid_research_windows_are_rejected(value: object) -> None:
    with pytest.raises(ConfigError) as caught:
        ResearchConfig.from_mapping({"dedupe_window": value})

    assert "dedupe_window" in str(caught.value)


@pytest.mark.parametrize(
    "value", [-0.01, 1.01, float("inf"), float("nan"), True, "high"]
)
def test_invalid_embedding_thresholds_are_rejected(value: object) -> None:
    with pytest.raises(ConfigError) as caught:
        ResearchConfig.from_mapping({"embedding_threshold": value})

    assert "embedding_threshold" in str(caught.value)


def test_unknown_research_fields_report_the_section_path() -> None:
    with pytest.raises(ConfigError) as caught:
        ResearchConfig.from_mapping({"bogus": True})

    assert "bogus" in str(caught.value)


def test_research_section_registration_is_explicit_and_idempotent() -> None:
    sections = SectionRegistry()

    assert "research" not in sections
    assert register_research_section(sections) is sections
    assert register_research_section(sections) is sections
    assert sections.get("research").model is ResearchConfig
    assert sections.get("research").module == "research"


def test_a_research_key_is_rejected_until_the_module_registers_it() -> None:
    core_only = build_app_config(register_core_sections(SectionRegistry()))

    with pytest.raises(ConfigError, match="unknown config section: research"):
        core_only.from_mapping({"research": {"dedupe_window": 7}})


def test_research_section_joins_the_root_model_when_registered() -> None:
    sections = register_research_section(register_core_sections(SectionRegistry()))
    model = build_app_config(sections)

    config = model.from_mapping(
        {"research": {"dedupe_window": 7, "embedding_threshold": 0.95}}
    )

    assert config.research is not None  # type: ignore[attr-defined]
    assert config.research.dedupe_window == timedelta(days=7)  # type: ignore[attr-defined]
    assert config.research.embedding_threshold == 0.95  # type: ignore[attr-defined]


def test_registering_research_over_another_model_fails() -> None:
    class OtherResearch(ConfigModel):
        pass

    sections = SectionRegistry()
    sections.register("research", OtherResearch)

    with pytest.raises(ConfigError, match="already registered: research"):
        register_research_section(sections)


def test_profile_overrides_tune_the_research_section(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
schema_version: 1
research:
  dedupe_window_days: 365
  embedding_threshold: 0.85
profiles:
  news:
    overrides:
      research:
        dedupe_window_days: 7
        embedding_threshold: 0.95
""".lstrip(),
        encoding="utf-8",
    )

    resolved = resolve_config(config_file=config_file, profile="news")
    sections = register_research_section(register_core_sections(SectionRegistry()))
    config = resolved.to_app_config(sections=sections)

    assert config.research.dedupe_window == timedelta(days=7)  # type: ignore[attr-defined]
    assert config.research.embedding_threshold == 0.95  # type: ignore[attr-defined]


def test_research_config_helpers_read_a_root_or_resolved_config(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
schema_version: 1
research:
  dedupe_window: 30
""".lstrip(),
        encoding="utf-8",
    )
    resolved = resolve_config(config_file=config_file)

    from_values = ResearchConfig.from_resolved(resolved)
    from_mapping = research_config({"research": {"dedupe_window": 30}})
    missing = research_config({})

    assert from_values.dedupe_window == timedelta(days=30)
    assert from_mapping.dedupe_window == timedelta(days=30)
    assert missing == ResearchConfig()


def test_example_config_has_no_research_section_until_the_module_registers_it() -> None:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))

    assert "research" not in payload
    AppConfig.from_mapping(payload)
