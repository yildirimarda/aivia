"""Occasion, order and delivery configuration."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    CORE_SECTION_KEYS,
    DeliveryConfig,
    DeliveryProviderType,
    EmailDeliveryConfig,
    LinkDeliveryConfig,
    LocalDeliveryConfig,
    NameVerifyConfig,
    OccasionConfig,
    OccasionsConfig,
    OccasionVariableConfig,
    OrderPricingConfig,
    OrdersConfig,
    PersonalizationLevelConfig,
    SectionRegistry,
    VariableType,
    WebhookDeliveryConfig,
    register_core_sections,
)
from aivia.core.errors import ConfigError


def example_section(name: str) -> dict[str, Any]:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    return cast(dict[str, Any], payload[name])


def test_occasion_order_and_delivery_sections_are_registered() -> None:
    assert {"occasions", "orders", "delivery"} <= set(CORE_SECTION_KEYS)

    sections = register_core_sections(SectionRegistry())
    parsed = sections.parse(
        {
            "occasions": {
                "default": "birthday",
                "birthday": {"variables": ["name"]},
            },
            "orders": {"name_verify_retries": 2},
            "delivery": {"folder": {"path": "/tmp/deliveries"}},
        }
    )

    assert parsed.occasions.default == "birthday"  # type: ignore[attr-defined]
    assert parsed.orders.name_verify_retries == 2  # type: ignore[attr-defined]
    assert parsed.delivery.folder.path == "/tmp/deliveries"  # type: ignore[attr-defined]


def test_empty_sections_are_valid_and_defaults_are_independent() -> None:
    first = OrdersConfig.from_mapping({})
    second = OrdersConfig.from_mapping({})

    assert first.enabled is True
    assert first.variables == {}
    assert first.personalization_levels == {}
    assert first.default_level == ""
    assert first.name_verify_retries == 3
    assert first.name_verify == NameVerifyConfig()
    assert first.pricing == OrderPricingConfig()
    assert first.timezone == "UTC"
    assert first.max_batch_size == 100
    assert first.max_name_length == 80
    assert first.max_message_length == 1000
    assert first.allow_custom_occasion is True
    assert first.require_age is False
    assert first.require_message is False

    first.personalization_levels["basic"] = PersonalizationLevelConfig()
    first.variables["name"] = OccasionVariableConfig(name="name")

    assert second.personalization_levels == {}
    assert second.variables == {}


def test_occasions_accept_direct_and_nested_mappings() -> None:
    direct = OccasionsConfig.from_mapping(
        {
            "default": "birthday",
            "birthday": {"label": "Birthday"},
            "anniversary": {"label": "Anniversary"},
        }
    )
    nested = OccasionsConfig.from_mapping(
        {
            "default": "birthday",
            "occasions": {
                "birthday": {"label": "Birthday"},
                "anniversary": {"label": "Anniversary"},
            },
        }
    )

    assert set(direct.occasions) == {"birthday", "anniversary"}
    assert direct.occasions["birthday"].label == "Birthday"
    assert direct.occasions["birthday"].name == "birthday"
    assert nested == direct


def test_occasions_reject_an_unknown_default() -> None:
    with pytest.raises(ConfigError) as caught:
        OccasionsConfig.from_mapping(
            {"default": "wedding", "birthday": {"label": "Birthday"}}
        )

    assert "wedding" in str(caught.value)


def test_occasion_variables_accept_names_mappings_and_aliases() -> None:
    occasion = OccasionConfig.from_mapping(
        {
            "variables": {
                "name": {"required": True, "max": 60},
                "age": {"kind": "int", "required": False},
                "style": {
                    "type": "enum",
                    "choices": ["warm", "funny"],
                },
            },
            "levels": ["basic", "premium"],
            "templates": ["birthday.txt"],
        }
    )

    assert occasion.variable_names == ["name", "age", "style"]
    assert occasion.variables["name"].required is True
    assert occasion.variables["name"].max_length == 60
    assert occasion.variables["age"].type is VariableType.INTEGER
    assert occasion.variables["style"].type is VariableType.CHOICE
    assert occasion.variables["style"].options == ["warm", "funny"]
    assert occasion.personalization_levels == ["basic", "premium"]
    assert occasion.levels == ["basic", "premium"]
    assert occasion.files == ["birthday.txt"]


def test_occasion_variables_accept_a_list_form() -> None:
    occasion = OccasionConfig.from_mapping(
        {
            "variables": [
                "name",
                {"key": "age", "type": "number"},
            ]
        }
    )

    assert list(occasion.variables) == ["name", "age"]
    assert occasion.variables["age"].type is VariableType.NUMBER


@pytest.mark.parametrize(
    ("values", "field"),
    [
        ({"variables": {"name": {"max": 5, "min": 10}}}, "min_length"),
        ({"variables": {"name": {"type": "nonsense"}}}, "type"),
        ({"variables": {"name": {"options": ["a", "a"]}}}, "option"),
        ({"levels": ["basic", "basic"]}, "level"),
    ],
)
def test_invalid_occasion_variables_are_rejected(
    values: dict[str, Any], field: str
) -> None:
    with pytest.raises(ConfigError) as caught:
        OccasionConfig.from_mapping(values)

    assert field in str(caught.value)


def test_personalization_levels_accept_mappings_lists_and_aliases() -> None:
    orders = OrdersConfig.from_mapping(
        {
            "default_level": "standard",
            "levels": {
                "basic": {"steps": ["name_overlay"], "outputs": ["png"]},
                "standard": {
                    "stages": ["write", "music", "render"],
                    "amount": 19,
                    "include_age": True,
                },
                "premium": ["write", "music", "voice"],
            },
        }
    )

    assert set(orders.personalization_levels) == {
        "basic",
        "standard",
        "premium",
    }
    basic = orders.personalization_levels["basic"]
    assert basic.name == "basic"
    assert basic.stages == ["name_overlay"]
    assert basic.outputs == ["png"]
    assert orders.personalization_levels["standard"].price == 19
    assert orders.personalization_levels["standard"].include_age is True
    assert orders.personalization_levels["premium"].stages == [
        "write",
        "music",
        "voice",
    ]
    assert orders.levels is orders.personalization_levels
    assert orders.default_personalization == "standard"


def test_orders_reject_an_unknown_default_level() -> None:
    with pytest.raises(ConfigError) as caught:
        OrdersConfig.from_mapping(
            {"default_level": "platinum", "levels": {"basic": {}}}
        )

    assert "platinum" in str(caught.value)


def test_name_verification_retries_are_kept_in_sync() -> None:
    direct = OrdersConfig.from_mapping({"name_verify_retries": 5})
    nested = OrdersConfig.from_mapping({"name_verify": {"retries": 4}})
    aliased = OrdersConfig.from_mapping({"verify_retries": 2})
    disabled = OrdersConfig.from_mapping({"name_verification": False})

    assert direct.name_verify_retries == 5
    assert direct.name_verify.retries == 5
    assert nested.name_verify_retries == 4
    assert nested.name_verify.retries == 4
    assert aliased.name_verify.retries == 2
    assert disabled.name_verify.enabled is False
    assert disabled.name_verify.retries == 3


def test_name_verification_accepts_a_retry_count_as_shorthand() -> None:
    verify = NameVerifyConfig.model_validate(6)

    assert verify.retries == 6
    assert verify.max_retries == 6
    assert verify.max_attempts == 6
    assert verify.verify_retries == 6
    assert verify.threshold == 0.8


@pytest.mark.parametrize(
    ("values", "field"),
    [
        ({"name_verify_retries": -1}, "name_verify_retries"),
        ({"name_verify": {"threshold": 1.5}}, "threshold"),
    ],
)
def test_invalid_name_verification_is_rejected(
    values: dict[str, Any], field: str
) -> None:
    with pytest.raises(ConfigError) as caught:
        OrdersConfig.from_mapping(values)

    assert field in str(caught.value)


def test_pricing_normalises_currency_and_accepts_aliases() -> None:
    pricing = OrderPricingConfig.from_mapping(
        {
            "currency_code": "eur",
            "price": 19.5,
            "tax_percent": 0.19,
            "taxes_included": True,
            "per_output": {"mp4": 19.5, "png": 0},
        }
    )

    assert pricing.currency == "EUR"
    assert pricing.currency_code == "EUR"
    assert pricing.amount == 19.5
    assert pricing.price == 19.5
    assert pricing.tax_rate == 0.19
    assert pricing.flat_tax_rate == 0.19
    assert pricing.tax_percent == pytest.approx(19)
    assert pricing.tax_included is True
    assert pricing.per_output == {"mp4": 19.5, "png": 0}


@pytest.mark.parametrize(
    ("values", "field"),
    [
        ({"currency": "US"}, "currency"),
        ({"currency": "US1"}, "currency"),
        ({"amount": -1}, "amount"),
        ({"tax_rate": 1.5}, "tax_rate"),
        ({"per_output": {"mp4": -1}}, "per_output"),
    ],
)
def test_invalid_pricing_is_rejected(values: dict[str, Any], field: str) -> None:
    with pytest.raises(ConfigError) as caught:
        OrderPricingConfig.from_mapping(values)

    assert field in str(caught.value)


def test_delivery_defaults_use_a_local_folder() -> None:
    delivery = DeliveryConfig.from_mapping({})

    assert delivery.enabled is True
    assert delivery.providers == {}
    assert delivery.chain == []
    assert delivery.default == "folder"
    assert delivery.retry_seconds == 60
    assert delivery.max_attempts == 3
    assert delivery.notify_on_failure is True


def test_delivery_accepts_providers_directly_or_nested() -> None:
    direct = DeliveryConfig.from_mapping(
        {
            "default": "folder",
            "chain": ["folder", "webhook"],
            "folder": "/tmp/deliveries",
            "webhook": {
                "url": "https://hooks.example.test/deliver",
                "secret_ref": "AIVA_WEBHOOK_SECRET",
                "allowlist": ["Hooks.Example.Test"],
            },
        }
    )
    nested = DeliveryConfig.from_mapping(
        {
            "default": "folder",
            "chain": ["folder", "webhook"],
            "providers": {
                "folder": {"path": "/tmp/deliveries"},
                "webhook": {
                    "type": "http",
                    "url": "https://hooks.example.test/deliver",
                    "secret_ref": "AIVA_WEBHOOK_SECRET",
                    "allowlist": ["Hooks.Example.Test"],
                },
            },
        }
    )

    assert direct == nested
    assert isinstance(direct.folder, LocalDeliveryConfig)
    assert direct.folder.path == "/tmp/deliveries"
    assert direct.local is direct.folder
    assert isinstance(direct.webhook, WebhookDeliveryConfig)
    assert direct.webhook.type == "webhook"
    assert direct.webhook.secret_ref == "AIVA_WEBHOOK_SECRET"
    assert direct.webhook.allowlist == ["hooks.example.test"]
    assert direct.chain == ["folder", "webhook"]


def test_delivery_provider_types_are_normalised_and_aliased() -> None:
    delivery = DeliveryConfig.from_mapping(
        {
            "default": "folder",
            "chain": ["folder", "link", "email", "webhook"],
            "providers": {
                "folder": {"directory": "/tmp/out"},
                "link": {"ttl": 3600},
                "email": {
                    "sender": "sender@example.test",
                    "subject": "Your card",
                },
                "webhook": {
                    "kind": "http",
                    "callback_url": "https://hooks.example.test/cb",
                    "allowed_hosts": ["hooks.example.test"],
                },
            },
        }
    )

    assert delivery.provider("folder") is delivery.folder
    assert delivery.provider("local") is delivery.folder
    assert delivery.provider("storage_link") is delivery.link
    assert delivery.provider("mail") is delivery.email
    assert isinstance(delivery.link, LinkDeliveryConfig)
    assert delivery.link.expires_seconds == 3600
    assert delivery.link.ttl_seconds == 3600
    assert isinstance(delivery.email, EmailDeliveryConfig)
    assert delivery.email.from_address == "sender@example.test"
    assert delivery.email.sender == "sender@example.test"
    assert delivery.email.subject == "Your card"
    assert isinstance(delivery.webhook, WebhookDeliveryConfig)
    assert delivery.webhook.type == "webhook"
    assert delivery.webhook.url == "https://hooks.example.test/cb"
    assert delivery.webhook.allowed_hosts == ["hooks.example.test"]


def test_delivery_chain_and_default_must_reference_a_provider() -> None:
    with pytest.raises(ConfigError) as caught:
        DeliveryConfig.from_mapping(
            {"chain": ["folder", "carrier_pigeon"], "folder": {}}
        )
    assert "carrier_pigeon" in str(caught.value)

    with pytest.raises(ConfigError) as caught:
        DeliveryConfig.from_mapping({"default": "email", "folder": {}})
    assert "email" in str(caught.value)


def test_delivery_chain_aliases_resolve_to_the_configured_key() -> None:
    delivery = DeliveryConfig.from_mapping(
        {
            "default": "local",
            "chain": ["local", "storage_link"],
            "providers": {"local": {"path": "/tmp/out"}, "link": {}},
        }
    )

    assert delivery.default == "local"
    assert delivery.chain == ["local", "link"]
    assert delivery.folder is delivery.providers["local"]


@pytest.mark.parametrize(
    ("values", "field"),
    [
        ({"providers": {"webhook": {"url": "ftp://example.test"}}}, "url"),
        ({"providers": {"link": {"expires_seconds": 0}}}, "expires_seconds"),
        ({"providers": {"folder": {"timeout_seconds": 0}}}, "timeout_seconds"),
        (
            {"providers": {"webhook": {"allowlist": ["a.test", "A.test"]}}},
            "allowlist",
        ),
        ({"providers": {"folder": {"retries": 0}}}, "max_attempts"),
    ],
)
def test_invalid_delivery_provider_fields_are_rejected(
    values: dict[str, Any], field: str
) -> None:
    with pytest.raises(ConfigError) as caught:
        DeliveryConfig.from_mapping(values)

    assert field in str(caught.value)


def test_delivery_provider_type_enum_is_the_documented_set() -> None:
    assert tuple(item.value for item in DeliveryProviderType) == (
        "folder",
        "link",
        "email",
        "webhook",
    )


def test_the_packaged_example_declares_the_order_sections() -> None:
    occasions = OccasionsConfig.from_mapping(example_section("occasions"))
    orders = OrdersConfig.from_mapping(example_section("orders"))
    delivery = DeliveryConfig.from_mapping(example_section("delivery"))

    assert occasions.default == "birthday"
    assert occasions.occasions["birthday"].variables["name"].required is True
    assert orders.default_level == "standard"
    assert orders.personalization_levels["premium"].include_interests is True
    assert orders.pricing.currency == "USD"
    assert orders.name_verify.retries == orders.name_verify_retries
    assert delivery.default == "folder"
    assert delivery.chain == ["folder", "link", "email", "webhook"]
    assert delivery.folder is not None
    assert delivery.folder.path == "deliveries"
