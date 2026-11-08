"""Character selection: resolve which character a run opts into.

Characters are strictly opt-in. The selected character is resolved in this
precedence:

1. the CLI flag (``--character``)
2. the order or instruction field
3. the profile's ``character``

When nothing opts in, the result is ``None`` and no character code runs.
"""

from __future__ import annotations

__all__ = ["select_character"]


def select_character(
    cli: str | None,
    order_or_instruction: str | None,
    profile: str | None,
) -> str | None:
    """Return the character name to use, or ``None`` when none is selected.

    Precedence is CLI flag, then the order or instruction field, then the
    profile. Blank values are treated as absent. When the result is ``None``,
    callers should run no character code.
    """
    for candidate in (cli, order_or_instruction, profile):
        text = candidate.strip() if isinstance(candidate, str) else ""
        if text:
            return text
    return None
