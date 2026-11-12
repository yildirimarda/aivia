"""Comparing alternative hooks in comparable slots records the winner.

An A/B test places alternative hooks into comparable slots — the same opening
position across videos — and compares their measured views. Within each slot the
most-viewed alternative wins and a significance note records how decisive the
win is. The computation is deterministic and never calls a provider.
"""

from __future__ import annotations

from aivia.analytics import Alternative, compare_alternatives


def _alt(slot: str, hook: str, views: int) -> Alternative:
    return Alternative(slot=slot, hook=hook, views=views)


def test_compares_within_a_slot_and_picks_the_most_viewed() -> None:
    alternatives = [
        _alt("open", "Question hook.", 120),
        _alt("open", "Statistic hook.", 450),
        _alt("open", "Story hook.", 300),
    ]

    results = compare_alternatives(alternatives)

    assert len(results) == 1
    result = results[0]
    assert result.slot == "open"
    assert result.winner_index == 1
    assert result.winner_hook == "Statistic hook."
    assert result.winner_views == 450
    assert result.runner_up_index == 2
    assert result.runner_up_views == 300
    assert result.margin_views == 150
    assert result.significance_note == "beat runner-up by 150 views"


def test_alternatives_in_different_slots_are_compared_separately() -> None:
    alternatives = [
        _alt("open", "A.", 100),
        _alt("close", "B.", 900),
        _alt("open", "C.", 50),
    ]

    results = compare_alternatives(alternatives)

    assert [result.slot for result in results] == ["open", "close"]
    open_result = results[0]
    assert open_result.winner_hook == "A."
    assert open_result.winner_views == 100
    close_result = results[1]
    assert close_result.winner_hook == "B."
    assert close_result.winner_views == 900
    assert close_result.margin_views == 0


def test_single_alternative_has_no_runner_up() -> None:
    alternatives = [_alt("open", "Only hook.", 42)]

    results = compare_alternatives(alternatives)

    assert len(results) == 1
    result = results[0]
    assert result.winner_index == 0
    assert result.runner_up_index is None
    assert result.runner_up_views is None
    assert result.margin_views == 0
    assert result.significance_note == "no alternative to compare"


def test_tie_is_inconclusive() -> None:
    alternatives = [
        _alt("open", "First hook.", 200),
        _alt("open", "Second hook.", 200),
    ]

    results = compare_alternatives(alternatives)

    assert len(results) == 1
    result = results[0]
    assert result.winner_index == 0
    assert result.margin_views == 0
    assert result.significance_note == "inconclusive — tied with runner-up on 200 views"


def test_margin_percent_is_zero_without_a_runner_up() -> None:
    alternatives = [_alt("open", "Only hook.", 7)]

    results = compare_alternatives(alternatives)

    assert results[0].margin_percent == 0.0


def test_empty_input_returns_empty() -> None:
    assert compare_alternatives([]) == []
