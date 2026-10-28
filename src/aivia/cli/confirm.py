"""Confirmation gate for a run whose estimate exceeds the configured threshold.

A run below the threshold proceeds without asking. Above it, an interactive
terminal is prompted for a yes/no answer, while a non-interactive caller must
pass ``--yes`` explicitly; otherwise the run is refused instead of hanging or
silently spending money.
"""

from __future__ import annotations

import sys
from typing import TextIO

__all__ = [
    "confirm_action",
    "confirm_spend",
    "confirmation_required",
]

_YES_ANSWERS = frozenset({"y", "yes"})


def confirmation_required(amount_usd: float, confirm_above_usd: float) -> bool:
    """Return whether *amount_usd* is above the configured confirmation floor.

    A floor of ``0`` disables confirmation, matching the budget caps where
    ``0`` means "not configured". The comparison is strict: spending exactly the
    floor does not need confirmation.
    """
    if confirm_above_usd <= 0:
        return False
    return amount_usd > confirm_above_usd


def _is_interactive(stream: TextIO) -> bool:
    isatty = getattr(stream, "isatty", None)
    return bool(isatty()) if callable(isatty) else False


def confirm_spend(
    amount_usd: float,
    confirm_above_usd: float,
    *,
    assume_yes: bool = False,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> bool:
    """Return whether a run estimated at *amount_usd* may proceed.

    ``assume_yes`` is the non-interactive confirmation flag. Without it, a
    non-interactive stream is refused with an actionable error. An interactive
    stream is asked once; anything other than ``y`` or ``yes`` is a refusal.
    """
    if not confirmation_required(amount_usd, confirm_above_usd):
        return True
    if assume_yes:
        return True

    source = sys.stdin if stdin is None else stdin
    target = sys.stdout if stdout is None else stdout
    if not _is_interactive(source):
        print(
            f"error: estimated cost ${amount_usd:.6f} exceeds the confirmation "
            f"threshold ${confirm_above_usd:.6f}; non-interactive runs require --yes",
            file=sys.stderr,
        )
        return False

    target.write(
        f"Estimated cost ${amount_usd:.6f} exceeds the confirmation threshold "
        f"${confirm_above_usd:.6f}. Continue? [y/N] "
    )
    target.flush()
    answer = source.readline().strip().lower()
    if answer in _YES_ANSWERS:
        return True

    print("error: confirmation declined; run cancelled", file=sys.stderr)
    return False


def confirm_action(
    message: str,
    *,
    assume_yes: bool = False,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> bool:
    """Return whether a consequential action may proceed.

    ``assume_yes`` is the non-interactive confirmation flag. Without it, a
    non-interactive stream is refused with an actionable error. An interactive
    stream is asked once; anything other than ``y`` or ``yes`` is a refusal.
    """
    if assume_yes:
        return True

    source = sys.stdin if stdin is None else stdin
    target = sys.stdout if stdout is None else stdout
    if not _is_interactive(source):
        print(
            f"error: {message}; non-interactive runs require --yes",
            file=sys.stderr,
        )
        return False

    target.write(f"{message} [y/N] ")
    target.flush()
    answer = source.readline().strip().lower()
    if answer in _YES_ANSWERS:
        return True

    print("error: confirmation declined; action cancelled", file=sys.stderr)
    return False
