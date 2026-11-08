"""Character roles: several characters per run, each with a role.

A run may cast several characters, one per role. The roles are fixed:
``host``, ``sidekick`` and ``narrator``. A :class:`CharacterCast` maps each
role to a character name; blank or absent entries mean that role is uncast.
Selection resolves each role independently: CLI flag, then the order or
instruction field, then the profile. When nothing is selected the cast is
empty and callers run no character code.

This module is plain validation only: no AI call and no network request
happens here. Loading the cast characters from disk is left to
:class:`~aivia.characters.loader.CharacterLibrary`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
)

__all__ = [
    "CHARACTER_ROLES",
    "CharacterCast",
    "select_characters",
]

CHARACTER_ROLES: tuple[str, ...] = ("host", "sidekick", "narrator")
"""The roles a run may assign to characters, in declaration order."""


class CharacterCast(BaseModel):
    """Several characters assigned to roles for one run.

    Each role holds a character name or ``None`` when that role is uncast.
    Unknown roles cannot be set.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    host: str | None = None
    sidekick: str | None = None
    narrator: str | None = None

    @field_validator("host", "sidekick", "narrator")
    @classmethod
    def _normalise(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text or None

    def items(self) -> list[tuple[str, str]]:
        """Return ``(role, name)`` for every cast role, in role order."""
        result: list[tuple[str, str]] = []
        for role in CHARACTER_ROLES:
            name = getattr(self, role)
            if name:
                result.append((role, name))
        return result

    def names(self) -> tuple[str, ...]:
        """Return every cast character name, in role order."""
        return tuple(name for _, name in self.items())

    def get(self, role: str) -> str | None:
        """Return the character name for *role*, or ``None`` when uncast."""
        if role not in CHARACTER_ROLES:
            return None
        return cast("str | None", getattr(self, role))

    def __contains__(self, role: object) -> bool:
        if not isinstance(role, str):
            return False
        return role in CHARACTER_ROLES and bool(getattr(self, role))


def _role_value(
    sources: Sequence[Mapping[str, str | None] | None],
    role: str,
) -> str | None:
    for source in sources:
        if not source:
            continue
        value: object = source.get(role)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return None


def select_characters(
    cli: Mapping[str, str | None] | None = None,
    order_or_instruction: Mapping[str, str | None] | None = None,
    profile: Mapping[str, str | None] | None = None,
) -> CharacterCast:
    """Resolve each role from CLI, then order/instruction, then profile.

    Each role is resolved independently against the sources in precedence
    order. Blank values are treated as absent. A role with no value in any
    source is left uncast.
    """
    sources = (cli, order_or_instruction, profile)
    return CharacterCast(
        host=_role_value(sources, "host"),
        sidekick=_role_value(sources, "sidekick"),
        narrator=_role_value(sources, "narrator"),
    )
