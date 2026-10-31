"""Prompt and content screening gates every paid provider call."""

from __future__ import annotations

import pytest

from aivia.core.config.sections import SafetyConfig
from aivia.core.errors import ContentPolicyError, ValidationError
from aivia.media.screening import (
    DEFAULT_CONTENT_KIND,
    DEFAULT_PROMPT_KIND,
    SCREENING_TOKEN_PREFIX,
    Screener,
    ScreeningConfig,
    ScreeningToken,
    build_screening_token,
    check_content,
    check_prompt,
    configured_terms,
    find_blocked_terms,
    is_screened,
    parse_screening_token,
    require_screened,
    screen_content,
    screen_prompt,
    screening_token,
    validate_token,
)


def test_a_safe_prompt_returns_a_token_bound_to_its_text() -> None:
    token = screen_prompt(
        "A lighthouse at sunset",
        SafetyConfig(blocked_terms=["violence"]),
    )

    assert isinstance(token, ScreeningToken)
    assert isinstance(token, str)
    assert token.startswith(f"{SCREENING_TOKEN_PREFIX}:{DEFAULT_PROMPT_KIND}:")
    assert token.kind == DEFAULT_PROMPT_KIND
    assert token.prompt == "A lighthouse at sunset"
    assert token.token == str(token)
    assert token.allowed is True
    assert token.blocked_terms == ()
    assert token.categories == ()
    assert token.as_mapping()["text"] == "A lighthouse at sunset"


def test_a_configured_term_blocks_before_a_provider_can_be_called() -> None:
    with pytest.raises(ContentPolicyError) as caught:
        screen_prompt(
            "A graphic violence scene",
            SafetyConfig(blocked_terms=["violence"]),
        )

    error = caught.value
    assert error.details["source"] == "screening"
    assert error.details["blocked_terms"] == ["violence"]
    assert error.details["kind"] == DEFAULT_PROMPT_KIND
    assert error.retryable is False


def test_matching_is_case_insensitive_and_whole_word_only() -> None:
    assert find_blocked_terms("a class act", ["ass"]) == ()
    assert find_blocked_terms("ASS", ["ass"]) == ("ass",)
    assert find_blocked_terms("SELF-HARM", ["self-harm"]) == ("self-harm",)
    assert find_blocked_terms("self harm", ["self-harm"]) == ()


def test_phrases_match_across_whitespace_and_unicode_normalisation() -> None:
    assert find_blocked_terms("graphic   VIOLENCE here", ["graphic violence"]) == (
        "graphic violence",
    )
    assert find_blocked_terms("STRASSE", ["straße"]) == ("straße",)
    fullwidth = "\uff41\uff44\uff55\uff4c\uff54 content"
    assert find_blocked_terms(fullwidth, ["adult"]) == ("adult",)


def test_configured_categories_are_literal_terms_too() -> None:
    config = SafetyConfig(block_categories=["violence"])

    assert configured_terms(config) == ("violence",)
    with pytest.raises(ContentPolicyError):
        screen_prompt("violence", config)


def test_terms_can_be_extended_per_call_and_are_deduplicated() -> None:
    config = SafetyConfig(blocked_terms=["secret"])

    assert configured_terms(config) == ("secret",)
    with pytest.raises(ContentPolicyError) as caught:
        screen_prompt("a weapon", config, terms=["weapon", "Weapon"])
    assert caught.value.details["blocked_terms"] == ["weapon"]

    assert find_blocked_terms("weapon", ["weapon", "WEAPON"]) == ("weapon",)


def test_the_configuration_can_be_a_mapping_or_a_root_config() -> None:
    nested = {"safety": {"blocked_terms": ["forbidden"]}}
    direct = {"blocked_terms": ["forbidden"]}
    aliased = {"safety": {"terms": ["forbidden"]}}

    for config in (nested, direct, aliased):
        with pytest.raises(ContentPolicyError):
            screen_prompt("forbidden", config)

    class Root:
        safety = SafetyConfig(blocked_terms=["forbidden"])

    with pytest.raises(ContentPolicyError):
        screen_prompt("forbidden", Root())


def test_disabling_safety_or_the_prompt_screen_still_returns_a_token() -> None:
    disabled = SafetyConfig(enabled=False, blocked_terms=["violence"])
    prompt_off = SafetyConfig(screen_prompts=False, blocked_terms=["violence"])

    assert screen_prompt("violence", disabled).allowed is True
    assert screen_prompt("violence", prompt_off).allowed is True


def test_content_screening_honours_the_media_switch() -> None:
    config = SafetyConfig(screen_media=False, blocked_terms=["violence"])

    token = screen_content("violence", config)
    assert token.kind == DEFAULT_CONTENT_KIND
    assert token.allowed is True

    with pytest.raises(ContentPolicyError):
        screen_content("violence", SafetyConfig(blocked_terms=["violence"]))


def test_a_paid_service_refuses_a_missing_malformed_or_mismatched_token() -> None:
    token = screen_prompt("safe prompt")

    with pytest.raises(ContentPolicyError) as missing:
        require_screened(None, "safe prompt")
    assert missing.value.details["reason"] == "missing_token"

    with pytest.raises(ContentPolicyError) as malformed:
        require_screened("not-a-token", "safe prompt")
    assert malformed.value.details["reason"] == "invalid_token"

    with pytest.raises(ContentPolicyError) as mismatched:
        require_screened(token, "a different prompt")
    assert mismatched.value.details["reason"] == "text_mismatch"

    with pytest.raises(ContentPolicyError) as wrong_kind:
        require_screened(token, "safe prompt", kind="content")
    assert wrong_kind.value.details["reason"] == "kind_mismatch"


def test_a_valid_token_is_accepted_by_the_paid_gate() -> None:
    token = screen_prompt("safe prompt")

    assert require_screened(token, "safe prompt") == token
    assert validate_token(token, "safe prompt") is True
    assert is_screened(str(token), "safe prompt") is True
    assert validate_token(token, "safe prompt", kind=DEFAULT_PROMPT_KIND) is True
    assert validate_token(token, "safe prompt", kind=DEFAULT_CONTENT_KIND) is False


def test_a_token_cannot_bypass_a_newly_configured_term() -> None:
    token = screen_prompt("weapon")

    with pytest.raises(ContentPolicyError):
        require_screened(token, "weapon", config=SafetyConfig(blocked_terms=["weapon"]))


def test_tokens_are_self_describing_and_parse_back() -> None:
    token = screening_token("safe", kind="image")
    parsed = parse_screening_token(str(token))

    assert parsed is not None
    assert parsed.kind == "image"
    assert parsed.prompt_hash == token.prompt_hash
    assert parse_screening_token("aivia-screened:prompt:nothex") is None
    assert parse_screening_token(123) is None


def test_build_screening_token_does_not_run_the_filter() -> None:
    token = build_screening_token("violence", blocked_terms=("violence",))

    assert token.blocked_terms == ("violence",)
    assert token.allowed is True


def test_non_raising_checks_report_the_blocked_terms() -> None:
    blocked = check_prompt("violence", SafetyConfig(blocked_terms=["violence"]))
    allowed = check_prompt("a calm field")

    assert blocked.allowed is False
    assert blocked.blocked is True
    assert blocked.blocked_terms == ("violence",)
    assert blocked.token is None
    assert allowed.allowed is True
    assert allowed.token is not None

    content = check_content("violence", SafetyConfig(blocked_terms=["violence"]))
    assert content.allowed is False
    assert content.kind == DEFAULT_CONTENT_KIND


def test_the_screener_class_binds_configuration_and_terms() -> None:
    screener = Screener(SafetyConfig(), terms=["secret"])

    assert screener.blocked_terms == ("secret",)
    with pytest.raises(ContentPolicyError):
        screener.screen_prompt("a secret plan")

    token = screener.screen_prompt("a safe plan")
    assert screener.require(token, "a safe plan") == token
    assert screener.validate(token, "another plan") is False


def test_standalone_screening_config_accepts_aliases_and_string_terms() -> None:
    config = ScreeningConfig.from_mapping({"terms": "forbidden"})
    assert config.blocked_terms == ["forbidden"]

    with pytest.raises(ContentPolicyError):
        screen_prompt("forbidden", config)


def test_bad_term_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        screen_prompt(object())  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        SafetyConfig(blocked_terms=[""])
