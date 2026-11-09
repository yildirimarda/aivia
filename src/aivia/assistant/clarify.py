"""Clarifier: ask one short question interactively, or apply defaults in batch.

When the instruction is ambiguous the assistant can ask exactly one short
clarifying question on an interactive stream and incorporate the answer. In
batch mode (a non-interactive stream) no question is asked; instead the
configured defaults are applied to the instruction and a note is recorded so
the caller can surface why the plan differs from an interactive answer.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TextIO

from aivia.core.config.planner import PlannerDefaultsConfig

__all__ = ["Clarification", "Clarifier"]


@dataclass(frozen=True, slots=True)
class Clarification:
    """The result of clarifying an instruction.

    ``instruction`` is the (possibly refined) instruction. ``notes`` records
    assumptions, including the batch-mode note when no question was asked.
    ``asked`` is whether a clarifying question was actually posed.
    """

    instruction: str
    notes: list[str]
    asked: bool


def _is_interactive(stream: TextIO) -> bool:
    isatty = getattr(stream, "isatty", None)
    return bool(isatty()) if callable(isatty) else False


class Clarifier:
    """Resolve ambiguity in an instruction using a single question or defaults.

    ``defaults`` holds the values applied in batch mode when the instruction
    leaves a setting unspecified (see :class:`PlannerDefaultsConfig`).
    """

    def __init__(self, *, defaults: PlannerDefaultsConfig | None = None) -> None:
        self._defaults = defaults or PlannerDefaultsConfig()

    def clarify(
        self,
        instruction: str,
        *,
        question: str = "",
        interactive: bool | None = None,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> Clarification:
        """Return a :class:`Clarification` for *instruction*.

        When interactive (or *interactive* is true), *question* is written to
        *stdout* and one line is read from *stdin*; a non-empty answer is
        appended to the instruction and ``asked`` is true.

        When not interactive, nothing is read or written. The configured
        defaults are applied to the instruction and a note is recorded.
        """
        source = sys.stdin if stdin is None else stdin
        target = sys.stdout if stdout is None else stdout
        if interactive is None:
            interactive = _is_interactive(source)

        notes: list[str] = []
        asked = False
        result = instruction

        if interactive:
            if question:
                target.write(question if question.endswith("\n") else question + " ")
                target.flush()
                answer = source.readline().strip()
                if answer:
                    result = f"{instruction} {answer}"
                asked = True
        else:
            applied = {
                key: value
                for key, value in self._defaults.values.items()
                if value not in (None, "", [], {})
            }
            if applied:
                pairs = [f"{key}={value}" for key, value in sorted(applied.items())]
                suffix = " ".join(pairs)
                result = f"{instruction} {suffix}" if instruction else suffix
            notes.append(
                "clarify: batch mode; clarifying question skipped; "
                f"defaults applied: {dict(sorted(applied.items()))}"
            )

        return Clarification(instruction=result, notes=notes, asked=asked)
