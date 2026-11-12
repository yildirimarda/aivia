"""Compare alternative hooks in comparable slots and record the winner.

An A/B test places alternative hooks into comparable slots — the same opening
position across videos — and compares their measured views. Within each slot the
alternative with the most views is the winner, and a significance note records
how decisive the win is. The computation is deterministic and offline: it never
calls a provider.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ABResult", "Alternative", "compare_alternatives"]


class Alternative(BaseModel):
    """One hook alternative deployed to a slot with measured views."""

    model_config = ConfigDict(frozen=True)

    slot: str
    """The slot the hook occupies. Only alternatives in the same slot compare."""

    hook: str
    """The hook text."""

    views: int = Field(ge=0)
    """Measured views for this alternative."""


class ABResult(BaseModel):
    """The outcome of comparing the alternatives in one slot."""

    model_config = ConfigDict(frozen=True)

    slot: str
    winner_index: int
    winner_hook: str
    winner_views: int
    runner_up_index: int | None = None
    runner_up_views: int | None = None
    margin_views: int = 0
    margin_percent: float = 0.0
    significance_note: str = ""


def compare_alternatives(alternatives: Iterable[Alternative]) -> list[ABResult]:
    """Compare alternatives grouped by slot and record each slot's winner.

    Results are returned in first-seen slot order. Within a slot the alternative
    with the most views wins; ties resolve to the first alternative in input
    order. When a slot has a single alternative there is no winner to record.
    """
    grouped: dict[str, list[Alternative]] = {}
    for alternative in alternatives:
        grouped.setdefault(alternative.slot, []).append(alternative)

    results: list[ABResult] = []
    for slot, members in grouped.items():
        best_index = max(range(len(members)), key=lambda i: members[i].views)
        best = members[best_index]

        runner_up_index: int | None = None
        runner_up_views: int | None = None
        margin_views = 0
        margin_percent = 0.0
        if len(members) >= 2:
            runner_up_index = max(
                (index for index in range(len(members)) if index != best_index),
                key=lambda index: members[index].views,
            )
            runner_up_views = members[runner_up_index].views
            margin_views = best.views - runner_up_views
            if runner_up_views > 0:
                margin_percent = round(margin_views / runner_up_views * 100.0, 1)

        results.append(
            ABResult(
                slot=slot,
                winner_index=best_index,
                winner_hook=best.hook,
                winner_views=best.views,
                runner_up_index=runner_up_index,
                runner_up_views=runner_up_views,
                margin_views=margin_views,
                margin_percent=margin_percent,
                significance_note=_significance_note(
                    best.views, runner_up_views, margin_views
                ),
            )
        )
    return results


def _significance_note(
    best_views: int, runner_up_views: int | None, margin_views: int
) -> str:
    """Return a plain-English note describing how decisive the win is."""
    if runner_up_views is None:
        return "no alternative to compare"
    if margin_views == 0:
        return f"inconclusive — tied with runner-up on {best_views} views"
    return f"beat runner-up by {margin_views} views"
