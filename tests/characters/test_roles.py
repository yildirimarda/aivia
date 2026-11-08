"""Character roles: several characters per run with host, sidekick and narrator.

A run may cast several characters, one per role. The roles are fixed:
host, sidekick and narrator. Selection resolves each role independently:
CLI flag, then the order or instruction field, then the profile.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from aivia.characters import (
    CHARACTER_ROLES,
    CharacterCast,
    select_characters,
)


def test_character_roles_are_host_sidekick_and_narrator() -> None:
    assert CHARACTER_ROLES == ("host", "sidekick", "narrator")


def test_character_cast_holds_one_character_per_role() -> None:
    cast = CharacterCast(
        host="alice",
        sidekick="bob",
        narrator="carol",
    )

    assert cast.host == "alice"
    assert cast.sidekick == "bob"
    assert cast.narrator == "carol"
    assert cast.items() == [
        ("host", "alice"),
        ("sidekick", "bob"),
        ("narrator", "carol"),
    ]
    assert cast.names() == ("alice", "bob", "carol")
    assert cast.get("host") == "alice"
    assert cast.get("sidekick") == "bob"
    assert cast.get("narrator") == "carol"
    assert "host" in cast
    assert "sidekick" in cast
    assert "narrator" in cast


def test_character_cast_defaults_are_uncast() -> None:
    cast = CharacterCast()

    assert cast.host is None
    assert cast.sidekick is None
    assert cast.narrator is None
    assert cast.items() == []
    assert cast.names() == ()
    assert cast.get("host") is None
    assert "host" not in cast


def test_character_cast_blank_names_are_uncast() -> None:
    cast = CharacterCast(host="  ", sidekick="bob", narrator="  ")

    assert cast.host is None
    assert cast.sidekick == "bob"
    assert cast.narrator is None
    assert cast.items() == [("sidekick", "bob")]
    assert cast.names() == ("bob",)


def test_character_cast_strips_whitespace() -> None:
    cast = CharacterCast(host="  alice  ", sidekick="\tbob\n", narrator=" carol ")

    assert cast.host == "alice"
    assert cast.sidekick == "bob"
    assert cast.narrator == "carol"


def test_character_cast_rejects_unknown_fields() -> None:
    with pytest.raises(PydanticValidationError):
        CharacterCast(host="alice", villain="eve")  # type: ignore[call-arg]


def test_character_cast_get_unknown_role_returns_none() -> None:
    cast = CharacterCast(host="alice")

    assert cast.get("villain") is None
    assert "villain" not in cast
    assert 123 not in cast


def test_select_characters_each_role_resolved_independently() -> None:
    cast = select_characters(
        cli={"host": "alice", "narrator": "carol"},
        order_or_instruction={"sidekick": "bob"},
        profile={"host": "profile_host", "narrator": "profile_narrator"},
    )

    assert cast.host == "alice"
    assert cast.sidekick == "bob"
    assert cast.narrator == "carol"


def test_select_characters_cli_wins_per_role() -> None:
    cast = select_characters(
        cli={"host": "cli_host", "sidekick": "cli_sidekick"},
        order_or_instruction={"host": "order_host", "sidekick": "order_sidekick"},
        profile={"host": "profile_host", "sidekick": "profile_sidekick"},
    )

    assert cast.host == "cli_host"
    assert cast.sidekick == "cli_sidekick"


def test_select_characters_order_wins_over_profile_per_role() -> None:
    cast = select_characters(
        cli={},
        order_or_instruction={"host": "order_host"},
        profile={"host": "profile_host"},
    )

    assert cast.host == "order_host"


def test_select_characters_profile_used_when_cli_and_order_absent() -> None:
    cast = select_characters(
        cli=None,
        order_or_instruction=None,
        profile={"narrator": "profile_narrator"},
    )

    assert cast.narrator == "profile_narrator"


def test_select_characters_none_when_nothing_selected() -> None:
    cast = select_characters()

    assert cast.host is None
    assert cast.sidekick is None
    assert cast.narrator is None
    assert cast.names() == ()


def test_select_characters_blank_values_are_absent() -> None:
    cast = select_characters(
        cli={"host": "  ", "sidekick": ""},
        order_or_instruction={"host": "order_host"},
        profile={"narrator": "  "},
    )

    assert cast.host == "order_host"
    assert cast.sidekick is None
    assert cast.narrator is None


def test_select_characters_unknown_role_keys_are_ignored() -> None:
    cast = select_characters(
        cli={"villain": "eve", "host": "alice"},
    )

    assert cast.host == "alice"
    assert cast.names() == ("alice",)
