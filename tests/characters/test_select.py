"""Character selection: CLI flag, then order or instruction field, then profile."""

from __future__ import annotations

from aivia.characters import select_character


def test_cli_flag_wins_over_order_and_profile() -> None:
    assert select_character("mascot", "order_char", "profile_char") == "mascot"
    assert select_character("mascot", None, "profile_char") == "mascot"


def test_order_field_wins_over_profile() -> None:
    assert select_character(None, "order_char", "profile_char") == "order_char"


def test_profile_used_when_cli_and_order_absent() -> None:
    assert select_character(None, None, "profile_char") == "profile_char"


def test_none_when_nothing_selected() -> None:
    assert select_character(None, None, None) is None


def test_blank_values_are_treated_as_absent() -> None:
    assert select_character("", None, "profile_char") == "profile_char"
    assert select_character(None, "", "profile_char") == "profile_char"
    assert select_character("", "", None) is None


def test_whitespace_is_stripped() -> None:
    assert select_character("  mascot  ", None, None) == "mascot"
    assert select_character(None, "  order_char  ", None) == "order_char"
    assert select_character(None, None, "  profile_char  ") == "profile_char"
