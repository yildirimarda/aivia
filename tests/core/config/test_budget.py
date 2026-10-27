"""Budget configuration: caps, provider limits, alerts and overrun policy."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    BudgetConfig,
    BudgetExceedBehavior,
    BudgetExceedBehaviour,
    OnExceed,
    OnExceedBehavior,
    OnExceedBehaviour,
    SectionRegistry,
    register_core_sections,
)
from aivia.core.errors import ConfigError


def test_budget_defaults_have_no_caps_no_alerts_and_pause_policy() -> None:
    budget = BudgetConfig.from_mapping({})

    assert budget.per_video == 0.0
    assert budget.per_day == 0.0
    assert budget.per_month == 0.0
    assert budget.per_order == 0.0
    assert budget.per_instruction == 0.0
    assert budget.per_run == 0.0
    assert budget.per_call == 0.0
    assert budget.per_provider == {}
    assert budget.alerts == []
    assert budget.on_exceed == OnExceed.PAUSE
    assert budget.on_exceed.value == "pause"


def test_budget_accepts_every_cap() -> None:
    budget = BudgetConfig.from_mapping(
        {
            "per_video": 2.5,
            "per_day": 10,
            "per_month": 100,
            "per_order": 5,
            "per_instruction": 1.25,
            "per_run": 3,
            "per_call": 0.75,
        }
    )

    assert budget.per_video == 2.5
    assert budget.per_day == 10.0
    assert budget.per_month == 100.0
    assert budget.per_order == 5.0
    assert budget.per_instruction == 1.25
    assert budget.per_run == 3.0
    assert budget.per_call == 0.75


@pytest.mark.parametrize(
    "field",
    [
        "per_video",
        "per_day",
        "per_month",
        "per_order",
        "per_instruction",
        "per_run",
        "per_call",
    ],
)
def test_caps_must_not_be_negative(field: str) -> None:
    with pytest.raises(ConfigError) as caught:
        BudgetConfig.from_mapping({field: -0.01})

    assert field in str(caught.value)


def test_provider_monthly_limits_are_keyed_by_provider() -> None:
    budget = BudgetConfig.from_mapping(
        {"per_provider": {"openai": 25, "stability": 10.5}}
    )

    assert budget.per_provider == {"openai": 25.0, "stability": 10.5}
    assert budget.provider_monthly == {"openai": 25.0, "stability": 10.5}
    assert budget.provider_monthly_limits == {"openai": 25.0, "stability": 10.5}


def test_provider_monthly_alias_is_accepted() -> None:
    budget = BudgetConfig.from_mapping({"provider_monthly": {"openai": 25}})

    assert budget.per_provider == {"openai": 25.0}


def test_provider_limits_accept_the_nested_monthly_shape() -> None:
    budget = BudgetConfig.from_mapping({"per_provider": {"openai": {"monthly": 25}}})

    assert budget.per_provider == {"openai": 25.0}


def test_provider_monthly_limits_must_not_be_negative() -> None:
    with pytest.raises(ConfigError) as caught:
        BudgetConfig.from_mapping({"per_provider": {"openai": -1}})

    assert "per_provider" in str(caught.value)


def test_provider_names_must_not_be_empty() -> None:
    with pytest.raises(ConfigError) as caught:
        BudgetConfig.from_mapping({"per_provider": {" ": 10}})

    assert "per_provider" in str(caught.value)


def test_alert_thresholds_are_sorted_ascending() -> None:
    budget = BudgetConfig.from_mapping({"alerts": [0.9, 0.5, 0.75]})

    assert budget.alerts == [0.5, 0.75, 0.9]
    assert budget.alert_thresholds == [0.5, 0.75, 0.9]


def test_alert_thresholds_are_sorted_again_on_assignment() -> None:
    budget = BudgetConfig.from_mapping({"alerts": [0.5, 0.9]})

    budget.alerts = [1.0, 0.25, 0.5]

    assert budget.alerts == [0.25, 0.5, 1.0]


def test_alert_thresholds_must_not_be_negative() -> None:
    with pytest.raises(ConfigError) as caught:
        BudgetConfig.from_mapping({"alerts": [0.5, -0.1]})

    assert "alerts" in str(caught.value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("pause", OnExceed.PAUSE),
        ("downgrade", OnExceed.DOWNGRADE),
        ("notify", OnExceed.NOTIFY),
        (OnExceed.PAUSE, OnExceed.PAUSE),
        (OnExceed.DOWNGRADE, OnExceed.DOWNGRADE),
        (OnExceed.NOTIFY, OnExceed.NOTIFY),
    ],
)
def test_on_exceed_accepts_every_behaviour(
    value: str | OnExceed, expected: OnExceed
) -> None:
    budget = BudgetConfig.from_mapping({"on_exceed": value})

    assert budget.on_exceed == expected


def test_on_exceed_rejects_an_unknown_behaviour() -> None:
    with pytest.raises(ConfigError) as caught:
        BudgetConfig.from_mapping({"on_exceed": "explode"})

    assert "on_exceed" in str(caught.value)


def test_on_exceed_behaviour_enum_aliases_point_at_the_same_class() -> None:
    assert OnExceedBehavior is OnExceed
    assert OnExceedBehaviour is OnExceed
    assert BudgetExceedBehavior is OnExceed
    assert BudgetExceedBehaviour is OnExceed


def test_registry_reports_the_full_budget_path() -> None:
    sections = register_core_sections(SectionRegistry())

    with pytest.raises(ConfigError) as caught:
        sections.parse({"budget": {"alerts": [-1]}})

    assert "budget.alerts" in str(caught.value)


def test_example_config_declares_the_budget_controls() -> None:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    budget = BudgetConfig.from_mapping(payload["budget"])

    assert budget.per_run == 0.0
    assert budget.per_call == 0.0
    assert budget.per_provider == {}
    assert budget.alerts == []
    assert budget.on_exceed == OnExceed.PAUSE
