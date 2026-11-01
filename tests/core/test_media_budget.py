"""Per-clip budget policy: suggest a cheaper option or raise before submit."""

from __future__ import annotations

from typing import Any, cast

import pytest

from aivia.core.config import BudgetConfig
from aivia.core.errors import BudgetError, ConfigError
from aivia.core.media_budget import (
    CheaperHint,
    ClipBudget,
    ClipBudgetAction,
    ClipBudgetCheck,
    ClipBudgetDecision,
    ClipBudgetPolicy,
    ClipOption,
    MediaBudgetPolicy,
    PerClipBudget,
    PerClipBudgetPolicy,
    apply_clip_budget,
    check_clip_budget,
    decide_clip_budget,
    enforce_clip_budget,
    ensure_clip_budget,
    evaluate_clip_budget,
    resolve_clip_budget,
)


def test_a_clip_inside_the_cap_proceeds_without_a_hint() -> None:
    policy = PerClipBudgetPolicy(BudgetConfig(per_video=0.50))

    decision = policy.evaluate(0.25, provider="acme")

    assert decision.action is ClipBudgetAction.PROCEED
    assert decision.allowed is True
    assert decision.blocked is False
    assert decision.exceeded is False
    assert decision.can_submit is True
    assert decision.cheaper is False
    assert decision.hint is None
    assert decision.cap_usd == pytest.approx(0.50)
    assert decision.estimate_usd == pytest.approx(0.25)
    assert "within the per-clip budget" in decision.message


def test_the_per_video_cap_is_used_by_default() -> None:
    policy = PerClipBudgetPolicy(BudgetConfig(per_video=0.10))

    assert policy.cap == pytest.approx(0.10)
    assert policy.per_clip == pytest.approx(0.10)
    assert policy.per_video == pytest.approx(0.10)
    assert policy.check(0.10).allowed is True

    with pytest.raises(BudgetError) as caught:
        policy.check(0.11)

    error = caught.value
    assert error.code == "budget"
    assert error.retryable is False
    assert error.details["scope"] == "per_clip"
    assert error.details["cap"] == pytest.approx(0.10)
    assert error.details["amount_usd"] == pytest.approx(0.11)
    assert "per-clip cap" in str(error)


def test_a_cheaper_candidate_returns_a_hint_before_submit() -> None:
    policy = PerClipBudgetPolicy(per_clip=0.10)

    decision = policy.check(
        0.20,
        provider="pricey",
        candidates={
            "pricey": 0.20,
            "mid": 0.08,
            "cheap": 0.04,
            "over": 0.50,
        },
    )

    assert decision.action is ClipBudgetAction.DOWNGRADE
    assert decision.allowed is True
    assert decision.blocked is False
    assert decision.exceeded is True
    assert decision.downgraded is True
    assert decision.cheaper is True
    assert decision.can_submit is False
    assert decision.hint is not None
    assert decision.hint.provider == "cheap"
    assert decision.hint.estimate_usd == pytest.approx(0.04)
    assert decision.hint.savings_usd == pytest.approx(0.16)
    assert "cheap" in decision.message
    assert "per-clip cap" in decision.message
    assert policy.hint(0.20, candidates={"cheap": 0.04}) is not None


def test_the_cheapest_affordable_candidate_wins_with_order_as_tie_breaker() -> None:
    policy = PerClipBudgetPolicy(per_clip=0.10)

    decision = policy.check(
        0.20,
        candidates=[("mid", 0.08), ("cheap", 0.04), ("same", 0.04)],
    )

    assert decision.hint is not None
    assert decision.hint.name == "cheap"


def test_a_candidate_above_the_cap_is_not_a_hint() -> None:
    policy = PerClipBudgetPolicy(per_clip=0.10)

    blocked = policy.evaluate(0.20, candidates={"still_over": 0.15})

    assert blocked.action is ClipBudgetAction.BLOCKED
    assert blocked.blocked is True
    assert blocked.allowed is False
    assert blocked.hint is None
    assert "no cheaper option fits" in blocked.message
    with pytest.raises(BudgetError):
        policy.check(0.20, candidates={"still_over": 0.15})


def test_a_candidate_that_is_not_cheaper_is_not_a_hint() -> None:
    policy = PerClipBudgetPolicy(per_clip=0.10)

    with pytest.raises(BudgetError):
        policy.check(0.20, candidates={"equal": 0.20, "higher": 0.25})


def test_a_cheaper_model_from_the_same_provider_is_a_hint() -> None:
    policy = PerClipBudgetPolicy(per_clip=0.10)
    options = [
        ClipOption(provider="acme", model="lite", estimate_usd=0.03),
        ClipOption(provider="other", model="pro", estimate_usd=0.09),
    ]

    decision = policy.check(
        0.20,
        provider="acme",
        model="pro",
        options=options,
    )

    assert decision.hint is not None
    assert decision.hint.provider == "acme"
    assert decision.hint.model == "lite"
    assert decision.hint.name == "acme:lite"


def test_an_explicit_cap_overrides_the_config() -> None:
    policy = PerClipBudgetPolicy(BudgetConfig(per_video=0.50), per_clip=0.10)

    assert policy.cap == pytest.approx(0.10)
    assert policy.check(0.20, candidates={"cheap": 0.08}).downgraded is True
    with pytest.raises(BudgetError):
        policy.check(0.20)


def test_a_number_config_is_the_cap() -> None:
    policy = PerClipBudgetPolicy(0.10)

    assert policy.cap == pytest.approx(0.10)
    assert policy.check(0.10).allowed is True
    with pytest.raises(BudgetError):
        policy.check(0.11)


def test_a_mapping_config_uses_the_per_video_alias() -> None:
    policy = PerClipBudgetPolicy({"per_video": 0.10})

    assert policy.cap == pytest.approx(0.10)
    assert isinstance(policy.config, BudgetConfig)


def test_a_zero_cap_disables_the_policy() -> None:
    policy = PerClipBudgetPolicy(BudgetConfig())

    decision = policy.check(1_000.0, provider="acme")

    assert decision.cap_usd == 0.0
    assert decision.action is ClipBudgetAction.PROCEED
    assert decision.allowed is True
    assert decision.hint is None


def test_evaluate_reports_a_blocked_clip_without_raising() -> None:
    policy = PerClipBudgetPolicy(per_clip=0.10)

    decision = policy.evaluate(0.20, provider="acme", model="pro")

    assert isinstance(decision, ClipBudgetDecision)
    assert decision.action is ClipBudgetAction.BLOCKED
    assert decision.allowed is False
    assert decision.blocked is True
    assert decision.exceeded is True
    assert decision.can_submit is False
    assert decision.provider == "acme"
    assert decision.model == "pro"
    assert decision.to_dict()["action"] == "blocked"


def test_decisions_and_hints_serialize() -> None:
    policy = PerClipBudgetPolicy(per_clip=0.10)

    decision = policy.check(0.20, provider="pricey", candidates={"cheap": 0.04})

    payload = decision.to_dict()
    assert payload["action"] == "downgrade"
    assert payload["allowed"] is True
    assert payload["cap_usd"] == pytest.approx(0.10)
    hint_payload = cast(dict[str, Any], payload["hint"])
    assert hint_payload["provider"] == "cheap"
    assert hint_payload["savings_usd"] == pytest.approx(0.16)
    assert decision.hint is not None
    assert hint_payload["message"] == decision.hint.message


def test_invalid_amounts_caps_and_candidates_are_rejected() -> None:
    with pytest.raises(ConfigError):
        PerClipBudgetPolicy(per_clip=-0.01)
    with pytest.raises(ConfigError):
        PerClipBudgetPolicy(per_clip=float("nan"))
    with pytest.raises(TypeError):
        PerClipBudgetPolicy(per_clip=0.1, cap=0.2)

    policy = PerClipBudgetPolicy(per_clip=0.10)
    with pytest.raises(ConfigError):
        policy.check(-0.01)
    with pytest.raises(ConfigError):
        policy.check(float("inf"))
    with pytest.raises(ConfigError):
        policy.check(0.20, candidates={"cheap": -0.01})
    with pytest.raises(ConfigError):
        policy.check(0.20, candidates={"cheap": float("nan")})
    with pytest.raises(ConfigError):
        policy.check(0.20, candidates={"": 0.01})
    with pytest.raises(ConfigError):
        policy.check(0.20, candidates=cast(Any, [("cheap", 0.01, 0.02)]))
    with pytest.raises(ConfigError):
        policy.check(0.20, candidates=cast(Any, 5))


def test_policy_aliases_and_module_helpers() -> None:
    assert ClipBudgetPolicy is PerClipBudgetPolicy
    assert ClipBudget is PerClipBudgetPolicy
    assert MediaBudgetPolicy is PerClipBudgetPolicy
    assert PerClipBudget is PerClipBudgetPolicy
    assert ClipBudgetCheck is ClipBudgetDecision

    config = BudgetConfig(per_video=0.10)
    for check in (
        check_clip_budget,
        decide_clip_budget,
        apply_clip_budget,
        resolve_clip_budget,
        enforce_clip_budget,
        ensure_clip_budget,
    ):
        decision = check(0.20, config=config, candidates={"cheap": 0.04})
        assert isinstance(decision, ClipBudgetDecision)
        assert decision.action is ClipBudgetAction.DOWNGRADE

    evaluated = evaluate_clip_budget(0.20, config=config)
    assert evaluated.blocked is True


def test_the_policy_is_a_cheaper_hint_and_not_a_paid_call() -> None:
    policy = PerClipBudgetPolicy(per_clip=0.10)

    hint = policy.hint(0.20, candidates={"cheap": 0.04})

    assert isinstance(hint, CheaperHint)
    assert hint.option.estimate_usd == pytest.approx(0.04)
    assert hint.to_dict()["estimate_usd"] == pytest.approx(0.04)
