"""Provider instance configuration for router chains."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    CORE_SECTION_KEYS,
    AppConfig,
    ProviderConfig,
    ProvidersConfig,
    SectionRegistry,
    register_core_sections,
)
from aivia.core.errors import ConfigError


def test_providers_default_to_enabled_with_no_entries() -> None:
    config = ProvidersConfig.from_mapping({})

    assert config.enabled is True
    assert config.providers == {}
    assert config.names() == ()
    assert config.enabled_providers() == {}


def test_provider_config_defaults_are_empty_and_independent() -> None:
    first = ProviderConfig.from_mapping({})
    second = ProviderConfig.from_mapping({})

    assert first.type == ""
    assert first.enabled is True
    assert first.options == {}
    assert first.secret_refs == {}

    first.options["queue"] = True
    assert second.options == {}


def test_a_configured_provider_config_still_requires_a_type() -> None:
    with pytest.raises(ConfigError) as caught:
        ProviderConfig.from_mapping({"options": {"queue": True}})

    assert "type" in str(caught.value)


def test_providers_map_names_to_type_options_and_secret_references() -> None:
    config = ProvidersConfig.from_mapping(
        {
            "fal": {
                "type": "fal",
                "options": {"queue": True, "region": "eu"},
                "secret_refs": {"api_key": "AIVA_FAL_API_KEY"},
            },
            "local": {"type": "local"},
        }
    )

    assert config.names() == ("fal", "local")
    assert config.type_for("fal") == "fal"
    assert config.options_for("fal") == {"queue": True, "region": "eu"}
    assert config.secret_refs_for("fal") == {"api_key": "AIVA_FAL_API_KEY"}
    assert config.provider("fal").requires_secrets is True
    assert config.provider("local").requires_secrets is False


def test_providers_accept_entries_directly_or_nested() -> None:
    direct = ProvidersConfig.from_mapping(
        {"fal": {"type": "fal", "options": {"queue": True}}}
    )
    nested = ProvidersConfig.from_mapping(
        {"providers": {"fal": {"type": "fal", "options": {"queue": True}}}}
    )

    assert direct == nested
    assert direct.providers["fal"].options == {"queue": True}


def test_providers_accept_a_bare_type_as_shorthand() -> None:
    config = ProvidersConfig.from_mapping({"fal": "fal"})

    assert config.provider("fal") == ProviderConfig(type="fal")


def test_provider_field_aliases_are_accepted() -> None:
    config = ProvidersConfig.from_mapping(
        {
            "fal": {
                "kind": "fal",
                "settings": {"queue": True},
                "secrets": {"api_key": "AIVA_FAL_API_KEY"},
            }
        }
    )

    provider = config.provider("fal")
    assert provider.type == "fal"
    assert provider.kind == "fal"
    assert provider.provider_type == "fal"
    assert provider.options == {"queue": True}
    assert provider.settings == {"queue": True}
    assert provider.params == {"queue": True}
    assert provider.secret_refs == {"api_key": "AIVA_FAL_API_KEY"}
    assert provider.secrets == {"api_key": "AIVA_FAL_API_KEY"}
    assert provider.secret_references == {"api_key": "AIVA_FAL_API_KEY"}


def test_canonical_provider_fields_win_over_aliases() -> None:
    config = ProvidersConfig.from_mapping(
        {
            "fal": {
                "type": "fal",
                "kind": "other",
                "options": {"queue": True},
                "settings": {"queue": False},
                "secret_refs": {"api_key": "AIVA_FAL_API_KEY"},
                "secrets": {"api_key": "OTHER_KEY"},
            }
        }
    )

    provider = config.provider("fal")
    assert provider.type == "fal"
    assert provider.options == {"queue": True}
    assert provider.secret_refs == {"api_key": "AIVA_FAL_API_KEY"}


def test_secret_references_accept_a_list_or_single_name() -> None:
    listed = ProvidersConfig.from_mapping(
        {"fal": {"type": "fal", "secret_refs": ["AIVA_FAL_API_KEY"]}}
    )
    single = ProvidersConfig.from_mapping(
        {"fal": {"type": "fal", "secret_refs": "AIVA_FAL_API_KEY"}}
    )

    assert listed.provider("fal").secret_refs == {
        "AIVA_FAL_API_KEY": "AIVA_FAL_API_KEY"
    }
    assert single.provider("fal").secret_refs == {
        "AIVA_FAL_API_KEY": "AIVA_FAL_API_KEY"
    }


@pytest.mark.parametrize("name", ["", "   "])
def test_provider_names_must_be_non_empty(name: str) -> None:
    with pytest.raises(ConfigError) as caught:
        ProvidersConfig.from_mapping({name: {"type": "fal"}})

    assert "provider name" in str(caught.value)


def test_provider_names_are_stripped() -> None:
    config = ProvidersConfig.from_mapping({" fal ": {"type": "fal"}})

    assert config.names() == ("fal",)


def test_provider_names_must_not_contain_whitespace() -> None:
    with pytest.raises(ConfigError) as caught:
        ProvidersConfig.from_mapping({"fal image": {"type": "fal"}})

    assert "provider name" in str(caught.value)


def test_every_provider_requires_a_non_empty_type() -> None:
    for entry in ({}, {"type": ""}, {"type": "   "}):
        with pytest.raises(ConfigError) as caught:
            ProvidersConfig.from_mapping({"fal": entry})

        assert "fal" in str(caught.value)
        assert "type" in str(caught.value)


def test_provider_options_must_be_a_mapping() -> None:
    with pytest.raises(ConfigError) as caught:
        ProvidersConfig.from_mapping({"fal": {"type": "fal", "options": ["queue"]}})

    assert "options" in str(caught.value)


def test_provider_option_names_are_stripped_and_unique() -> None:
    config = ProvidersConfig.from_mapping(
        {"fal": {"type": "fal", "options": {" queue ": True}}}
    )

    assert config.provider("fal").options == {"queue": True}

    with pytest.raises(ConfigError) as caught:
        ProvidersConfig.from_mapping(
            {"fal": {"type": "fal", "options": {"queue": True, " queue ": False}}}
        )

    assert "options" in str(caught.value)


@pytest.mark.parametrize("reference", ["", "   ", "AIVA KEY"])
def test_secret_references_must_be_non_empty_names(reference: str) -> None:
    with pytest.raises(ConfigError) as caught:
        ProvidersConfig.from_mapping(
            {"fal": {"type": "fal", "secret_refs": {"api_key": reference}}}
        )

    assert "secret" in str(caught.value)


def test_secret_reference_keys_are_stripped() -> None:
    config = ProvidersConfig.from_mapping(
        {"fal": {"type": "fal", "secret_refs": {" api_key ": " AIVA_FAL_KEY "}}}
    )

    assert config.provider("fal").secret_refs == {"api_key": "AIVA_FAL_KEY"}


def test_unknown_provider_fields_report_the_full_path() -> None:
    with pytest.raises(ConfigError) as caught:
        ProvidersConfig.from_mapping({"fal": {"type": "fal", "bogus": True}})

    assert "fal.bogus" in str(caught.value)


def test_enabled_providers_can_be_filtered() -> None:
    config = ProvidersConfig.from_mapping(
        {
            "fal": {"type": "fal"},
            "local": {"type": "local", "enabled": False},
        }
    )

    assert list(config.enabled_providers()) == ["fal"]
    assert "fal" in config
    assert "missing" not in config
    assert len(config) == 2
    assert list(config.providers) == ["fal", "local"]


def test_an_unknown_provider_names_the_configured_entries() -> None:
    config = ProvidersConfig.from_mapping({"fal": {"type": "fal"}})

    assert config.get("missing") is None
    with pytest.raises(ConfigError) as caught:
        config.provider("missing")

    assert "missing" in str(caught.value)
    assert "fal" in str(caught.value)


def test_provider_values_are_interpolated_from_the_environment() -> None:
    config = ProvidersConfig.from_mapping(
        {
            "fal": {
                "type": "${AIVA_FAL_TYPE}",
                "options": {"region": "${AIVA_FAL_REGION}"},
                "secret_refs": {"api_key": "${AIVA_FAL_KEY_REF}"},
            }
        },
        environ={
            "AIVA_FAL_TYPE": "fal",
            "AIVA_FAL_REGION": "eu",
            "AIVA_FAL_KEY_REF": "AIVA_FAL_API_KEY",
        },
    )

    provider = config.provider("fal")
    assert provider.type == "fal"
    assert provider.options == {"region": "eu"}
    assert provider.secret_refs == {"api_key": "AIVA_FAL_API_KEY"}


def test_provider_defaults_are_independent_between_instances() -> None:
    first = ProvidersConfig.from_mapping({"fal": {"type": "fal"}})
    second = ProvidersConfig.from_mapping({"fal": {"type": "fal"}})

    first.provider("fal").options["queue"] = True
    first.provider("fal").secret_refs["api_key"] = "AIVA_FAL_KEY"

    assert second.provider("fal").options == {}
    assert second.provider("fal").secret_refs == {}


def test_providers_section_is_registered_and_parsed_from_the_root() -> None:
    assert "providers" in CORE_SECTION_KEYS
    assert "providers" in register_core_sections(SectionRegistry())

    config = AppConfig.from_mapping(
        {"providers": {"fal": {"type": "fal", "options": {"queue": True}}}}
    )

    assert config.providers is not None  # type: ignore[attr-defined]
    assert config.providers.provider("fal").type == "fal"  # type: ignore[attr-defined]


def test_example_config_declares_the_providers_section() -> None:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    config = ProvidersConfig.from_mapping(payload["providers"])

    assert config.enabled is True
    assert config.providers == {}
