"""Rank videos by views relative to their own channel median.

Absolute view counts favour large channels. A video's *outlier score* is its
views divided by its channel's median views, so a small channel's breakout
video ranks above a large channel's mundane upload. The stage is deterministic
and offline: it never calls a provider.
"""

from __future__ import annotations

from collections.abc import Iterable
from statistics import median

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ChannelBaseline",
    "OutlierScore",
    "VideoEntry",
    "channel_baselines",
    "channel_medians",
    "outlier_scores",
    "rank_outliers",
]


class VideoEntry(BaseModel):
    """A single video with its channel and view count."""

    model_config = ConfigDict(frozen=True)

    video_id: str | None = None
    channel: str
    views: int = Field(ge=0)


class ChannelBaseline(BaseModel):
    """A channel's median view count baseline and how many videos define it."""

    model_config = ConfigDict(frozen=True)

    channel: str
    median_views: float
    video_count: int


class OutlierScore(BaseModel):
    """A video's views relative to its channel median.

    ``score`` is ``views / median_views``. When a channel's median is zero the
    score is ``0.0`` to avoid division by zero.
    """

    model_config = ConfigDict(frozen=True)

    video_id: str | None
    channel: str
    views: int
    median_views: float
    score: float


def channel_medians(entries: Iterable[VideoEntry]) -> dict[str, float]:
    """Return each channel's median view count."""
    grouped: dict[str, list[int]] = {}
    for entry in entries:
        grouped.setdefault(entry.channel, []).append(entry.views)
    return {channel: float(median(views)) for channel, views in grouped.items()}


def channel_baselines(entries: Iterable[VideoEntry]) -> list[ChannelBaseline]:
    """Return each channel's baseline, ordered by channel name."""
    medians = channel_medians(entries)
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.channel] = counts.get(entry.channel, 0) + 1
    return [
        ChannelBaseline(
            channel=channel,
            median_views=median_views,
            video_count=counts[channel],
        )
        for channel, median_views in sorted(medians.items())
    ]


def outlier_scores(entries: Iterable[VideoEntry]) -> list[OutlierScore]:
    """Score every video by views relative to its channel median."""
    medians = channel_medians(entries)
    scores: list[OutlierScore] = []
    for entry in entries:
        median_views = medians[entry.channel]
        score = entry.views / median_views if median_views > 0 else 0.0
        scores.append(
            OutlierScore(
                video_id=entry.video_id,
                channel=entry.channel,
                views=entry.views,
                median_views=median_views,
                score=score,
            )
        )
    return scores


def rank_outliers(
    entries: Iterable[VideoEntry],
) -> list[OutlierScore]:
    """Return videos ranked by views relative to channel median, highest first.

    Ties are broken deterministically by channel then video id so the same
    input always produces the same order.
    """
    scores = outlier_scores(list(entries))
    return sorted(
        scores,
        key=lambda score: (-score.score, score.channel, score.video_id or ""),
    )
