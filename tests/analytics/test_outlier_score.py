"""Ranking videos by views relative to their own channel median.

A small channel's breakout video must outrank a large channel's mundane upload
because the score is views divided by the channel's median views, not absolute
views. The ordering is deterministic and never calls a provider.
"""

from __future__ import annotations

from aivia.analytics import (
    ChannelBaseline,
    OutlierScore,
    VideoEntry,
    channel_baselines,
    channel_medians,
    outlier_scores,
    rank_outliers,
)


def _entry(video_id: str, channel: str, views: int) -> VideoEntry:
    return VideoEntry(video_id=video_id, channel=channel, views=views)


def test_channel_medians_groups_by_channel() -> None:
    entries = [
        _entry("a", "big", 1000),
        _entry("b", "big", 100),
        _entry("c", "big", 100),
        _entry("d", "small", 500),
        _entry("e", "small", 1),
        _entry("f", "small", 1),
    ]

    assert channel_medians(entries) == {"big": 100.0, "small": 1.0}


def test_channel_medians_even_count_averages_middle_two() -> None:
    entries = [
        _entry("a", "chan", 10),
        _entry("b", "chan", 20),
        _entry("c", "chan", 30),
        _entry("d", "chan", 40),
    ]

    assert channel_medians(entries) == {"chan": 25.0}


def test_channel_medians_empty_input_is_empty() -> None:
    assert channel_medians([]) == {}


def test_channel_baselines_ordered_by_channel() -> None:
    entries = [
        _entry("a", "zeta", 5),
        _entry("b", "alpha", 5),
    ]

    baselines = channel_baselines(entries)
    assert isinstance(baselines, list)
    assert [b.channel for b in baselines] == ["alpha", "zeta"]
    assert all(isinstance(b, ChannelBaseline) for b in baselines)
    assert baselines[0].video_count == 1
    assert baselines[0].median_views == 5.0


def test_outlier_score_is_views_divided_by_channel_median() -> None:
    entries = [
        _entry("breakout", "small", 1000),
        _entry("mundane", "small", 100),
        _entry("also_mundane", "small", 100),
    ]

    scores = outlier_scores(entries)
    by_id = {score.video_id: score for score in scores}
    assert by_id["breakout"].score == 10.0
    assert by_id["breakout"].median_views == 100.0
    assert by_id["mundane"].score == 1.0


def test_outlier_score_zero_median_is_zero_not_infinite() -> None:
    entries = [
        _entry("a", "dead", 0),
        _entry("b", "dead", 0),
    ]

    scores = outlier_scores(entries)
    assert all(score.score == 0.0 for score in scores)
    assert all(score.median_views == 0.0 for score in scores)


def test_outlier_scores_returns_one_per_input_video() -> None:
    entries = [_entry("a", "c", 1), _entry("b", "c", 2), _entry("c", "c", 3)]
    assert len(outlier_scores(entries)) == 3


def test_single_video_per_channel_scores_one_point_zero() -> None:
    entries = [_entry("only", "chan", 42)]
    scores = outlier_scores(entries)
    assert scores[0].score == 1.0
    assert scores[0].median_views == 42.0


def test_rank_outliers_puts_relative_overperformer_first() -> None:
    """A small channel's breakout beats a large channel's mundane upload."""
    entries = [
        _entry("mundane_big", "big", 5000),
        _entry("breakout_small", "small", 1000),
        _entry("normal_big", "big", 5000),
        _entry("normal_small", "small", 100),
    ]

    ranked = rank_outliers(entries)
    assert [score.video_id for score in ranked] == [
        "breakout_small",  # 1000 / 100 = 10
        "mundane_big",  # 5000 / 5000 = 1
        "normal_big",  # 5000 / 5000 = 1
        "normal_small",  # 100 / 100 = 1
    ]


def test_rank_outliers_absolute_views_alone_would_rank_wrong() -> None:
    """By absolute views the two 5000-view videos would win; relatively they lose."""
    entries = [
        _entry("tiny", "small", 100),
        _entry("huge", "big", 5000),
    ]

    ranked = rank_outliers(entries)
    # small channel median is 100 -> score 1; big channel median is 5000 -> score 1.
    # Tie breaks by channel then video id: "big" < "small", so huge comes first here.
    assert [score.video_id for score in ranked] == ["huge", "tiny"]


def test_rank_outliers_breaks_ties_deterministically() -> None:
    entries = [
        _entry("z", "chan", 100),
        _entry("a", "chan", 100),
    ]

    ranked = rank_outliers(entries)
    assert [score.video_id for score in ranked] == ["a", "z"]


def test_rank_outliers_empty_input_is_empty() -> None:
    assert rank_outliers([]) == []


def test_rank_outliers_score_is_float() -> None:
    entries = [_entry("a", "c", 3), _entry("b", "c", 1)]
    ranked = rank_outliers(entries)
    assert all(isinstance(score.score, float) for score in ranked)
    assert all(isinstance(score, OutlierScore) for score in ranked)
