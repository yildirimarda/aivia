"""Per profile source resolution: weights, language filters and unknown names."""

from __future__ import annotations

from typing import Any

import pytest

from aivia.core.config import ContentProfileConfig, ProfilesConfig
from aivia.core.errors import ConfigError
from aivia.providers.sources import (
    MockSourceProvider,
    SourceRegistry,
    WikipediaSourceProvider,
)
from aivia.research import (
    ResolvedSource,
    SourceSelection,
    SourceSelector,
    SourceSpec,
    parse_source_specs,
    resolve_profile_sources,
    resolve_source_selection,
    resolve_sources,
    select_sources,
    source_selection,
)


def test_sources_accept_names_weights_and_language_filters() -> None:
    profile = ContentProfileConfig.from_mapping(
        {
            "languages": ["en", "de"],
            "sources": {
                "wikipedia": {"weight": 3, "languages": ["EN_us", "de"]},
                "reddit": 0.5,
                "mock": None,
            },
        }
    )

    assert profile.sources == ["wikipedia", "reddit", "mock"]
    assert profile.source_weights == {"wikipedia": 3.0, "reddit": 0.5, "mock": 1.0}
    assert profile.source_languages == {
        "wikipedia": ["en-us", "de"],
        "reddit": [],
        "mock": [],
    }
    assert profile.source_options["wikipedia"].name == "wikipedia"
    assert profile.source_configs is profile.source_options
    assert profile.source_settings is profile.source_options

    selection = resolve_sources(profile)

    assert isinstance(selection, SourceSelection)
    assert selection.names == ["wikipedia", "reddit", "mock"]
    assert selection.weights == {"wikipedia": 3.0, "reddit": 0.5, "mock": 1.0}
    assert selection.languages == {
        "wikipedia": ["en", "de"],
        "reddit": ["en", "de"],
        "mock": ["en", "de"],
    }
    assert selection.providers["wikipedia"] is WikipediaSourceProvider
    assert all(isinstance(source, ResolvedSource) for source in selection)


def test_unweighted_sources_default_to_one_and_keep_profile_order() -> None:
    profile = ContentProfileConfig.from_mapping(
        {"sources": ["wikipedia", "reddit", "mock"]}
    )

    selection = resolve_sources(profile)

    assert selection.names == ["wikipedia", "reddit", "mock"]
    assert selection.weights == {"wikipedia": 1.0, "reddit": 1.0, "mock": 1.0}
    assert selection.total_weight == 3.0
    assert selection.normalized_weights() == pytest.approx(
        {"wikipedia": 1 / 3, "reddit": 1 / 3, "mock": 1 / 3}
    )
    assert selection.items() == [("wikipedia", 1.0), ("reddit", 1.0), ("mock", 1.0)]


def test_list_of_source_mappings_is_accepted() -> None:
    profile = ContentProfileConfig.from_mapping(
        {
            "sources": [
                {"name": "wikipedia", "weight": 2, "languages": "de"},
                {"source": "reddit", "weight": 1, "lang": ["en", "de"]},
            ],
            "languages": ["de"],
        }
    )

    assert profile.sources == ["wikipedia", "reddit"]
    assert profile.source_weights == {"wikipedia": 2.0, "reddit": 1.0}
    assert profile.source_languages == {
        "wikipedia": ["de"],
        "reddit": ["en", "de"],
    }


def test_source_weights_and_languages_alias_mappings_are_accepted() -> None:
    profile = ContentProfileConfig.from_mapping(
        {
            "sources": ["wikipedia", "reddit"],
            "source_weights": {"wikipedia": 4},
            "source_languages": {"reddit": ["en"]},
        }
    )

    assert profile.source_weights == {"wikipedia": 4.0, "reddit": 1.0}
    assert profile.source_languages == {"wikipedia": [], "reddit": ["en"]}


def test_language_filters_are_intersected_with_the_requested_languages() -> None:
    profile = ContentProfileConfig.from_mapping(
        {
            "languages": ["en", "de"],
            "sources": {
                "wikipedia": {"languages": ["en"]},
                "reddit": {"languages": ["de"]},
                "mock": {},
            },
        }
    )

    english = resolve_sources(profile, language="EN")
    german = resolve_sources(profile, languages=["de"])
    french = resolve_sources(profile, language="fr")

    assert english.names == ["wikipedia", "mock"]
    assert english.languages == {"wikipedia": ["en"], "mock": ["en"]}
    assert german.names == ["reddit", "mock"]
    assert german.languages == {"reddit": ["de"], "mock": ["de"]}
    assert french.names == ["mock"]
    assert french.languages == {"mock": ["fr"]}


def test_language_filter_is_used_when_the_profile_has_no_languages() -> None:
    profile = ContentProfileConfig.from_mapping(
        {"sources": {"wikipedia": {"languages": ["de"]}, "reddit": {}}}
    )

    selection = resolve_sources(profile)

    assert selection.names == ["wikipedia", "reddit"]
    assert selection.languages == {"wikipedia": ["de"], "reddit": []}


def test_unknown_sources_fail_with_the_name_and_available_sources() -> None:
    profile = ContentProfileConfig.from_mapping(
        {"sources": ["wikipedia", "not-a-source"]}
    )

    with pytest.raises(ConfigError) as caught:
        resolve_sources(profile)

    message = str(caught.value)
    assert "unknown source: not-a-source" in message
    assert "available:" in message
    assert "wikipedia" in message
    assert caught.value.details["source"] == "not-a-source"
    assert "wikipedia" in caught.value.details["available"]


@pytest.mark.parametrize(
    "entry",
    [
        {"weight": 0},
        {"enabled": False},
    ],
)
def test_unknown_sources_fail_even_when_inactive(entry: dict[str, Any]) -> None:
    profile = ContentProfileConfig.from_mapping({"sources": {"not-a-source": entry}})

    with pytest.raises(ConfigError, match="unknown source: not-a-source"):
        resolve_sources(profile)


def test_disabled_and_zero_weight_sources_are_omitted() -> None:
    profile = ContentProfileConfig.from_mapping(
        {
            "sources": {
                "wikipedia": {"weight": 0},
                "reddit": {"enabled": False},
                "mock": {"weight": 2},
            }
        }
    )

    selection = resolve_sources(profile)

    assert selection.names == ["mock"]
    assert selection.weights == {"mock": 2.0}


def test_duplicate_sources_are_rejected() -> None:
    with pytest.raises(ConfigError, match="duplicate source: wikipedia"):
        ContentProfileConfig.from_mapping({"sources": ["wikipedia", "wikipedia"]})

    with pytest.raises(ConfigError, match="duplicate source: wikipedia"):
        ContentProfileConfig.from_mapping(
            {"sources": [{"name": "wikipedia"}, "wikipedia"]}
        )


@pytest.mark.parametrize(
    "entry",
    [
        {"weight": -1},
        {"weight": float("inf")},
        {"weight": "many"},
        {"languages": [" "]},
    ],
)
def test_invalid_source_options_are_rejected(entry: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        ContentProfileConfig.from_mapping({"sources": {"wikipedia": entry}})


def test_custom_registry_resolves_registered_sources() -> None:
    registry = SourceRegistry()
    registry.register_source("custom-source", MockSourceProvider)

    selection = resolve_sources(
        {"sources": {"custom-source": {"weight": 2}}},
        registry=registry,
    )

    assert selection.names == ["custom-source"]
    assert selection.providers["custom-source"].__name__ == "MockSourceProvider"
    assert selection.weights == {"custom-source": 2.0}

    with pytest.raises(ConfigError) as caught:
        resolve_sources({"sources": ["wikipedia"]}, registry=registry)
    assert "unknown source: wikipedia" in str(caught.value)
    assert "custom-source" in str(caught.value)


def test_parse_source_specs_accepts_names_mappings_and_options() -> None:
    specs = parse_source_specs(
        ["wikipedia", {"name": "reddit", "weight": 2}],
        options={"wikipedia": {"languages": "DE"}},
    )

    assert [spec.name for spec in specs] == ["wikipedia", "reddit"]
    assert [spec.weight for spec in specs] == [1.0, 2.0]
    assert specs[0].languages == ["de"]
    assert specs[1].languages == []

    mapping = parse_source_specs({"wikipedia": 3, "reddit": {"languages": ["en"]}})
    assert [spec.weight for spec in mapping] == [3.0, 1.0]
    assert mapping[1].languages == ["en"]


def test_source_spec_is_the_profile_source_model() -> None:
    spec = SourceSpec.from_mapping(
        {"name": "wikipedia", "weight": 2, "languages": ["EN_us"]}
    )

    assert spec.name == "wikipedia"
    assert spec.weight == 2.0
    assert spec.languages == ["en-us"]
    assert spec.enabled is True


def test_selection_helpers_expose_names_weights_and_lookup() -> None:
    profile = ContentProfileConfig.from_mapping(
        {"sources": {"wikipedia": 3, "reddit": 1}}
    )
    selection = resolve_sources(profile)
    wikipedia = selection.by_name("wikipedia")

    assert wikipedia is not None
    assert wikipedia.source == "wikipedia"
    assert wikipedia.provider_class is WikipediaSourceProvider
    assert wikipedia.enabled is True
    assert wikipedia.supports("en")
    assert selection.get("missing") is None
    assert selection.get("missing", wikipedia) is wikipedia
    assert selection.sources == list(selection)


def test_profiles_section_resolves_a_named_profile() -> None:
    profiles = ProfilesConfig.from_mapping(
        {
            "profiles": {
                "short": {"sources": {"wikipedia": 2}},
                "long": {"sources": ["reddit"]},
            }
        }
    )

    selection = resolve_sources(profiles, profile_name="short")
    assert selection.names == ["wikipedia"]
    assert selection.weights == {"wikipedia": 2.0}

    with pytest.raises(ConfigError, match="profile_name is required"):
        resolve_sources(profiles)

    with pytest.raises(ConfigError, match="unknown profile: missing"):
        resolve_sources(profiles, profile_name="missing")


def test_source_selector_binds_the_registry_and_resolves_profiles() -> None:
    registry = SourceRegistry()
    registry.register_source("custom-source", MockSourceProvider)
    selector = SourceSelector(registry)
    profiles = ProfilesConfig.from_mapping(
        {"profiles": {"custom": {"sources": ["custom-source"]}}}
    )

    selection = selector.for_profile(profiles, "custom")

    assert selection.names == ["custom-source"]
    assert selector({"sources": ["custom-source"]}).names == ["custom-source"]


def test_resolution_aliases_share_one_implementation() -> None:
    profile = ContentProfileConfig.from_mapping({"sources": ["wikipedia"]})

    expected = resolve_sources(profile)
    assert resolve_profile_sources(profile) == expected
    assert resolve_source_selection(profile) == expected
    assert select_sources(profile) == expected
    assert source_selection(profile) == expected
