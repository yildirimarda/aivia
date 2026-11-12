"""Platform metrics collection and reporting.

The analytics module is optional: the core never imports it. It normalizes
raw platform metrics into a canonical shape so downstream storage and
reporting can treat every platform the same way, and scores videos by how
their views compare to their own channel's median.
"""

from aivia.analytics.collect import CANONICAL_METRICS, collect
from aivia.analytics.outlier import (
    ChannelBaseline,
    OutlierScore,
    VideoEntry,
    channel_baselines,
    channel_medians,
    outlier_scores,
    rank_outliers,
)

__all__ = [
    "CANONICAL_METRICS",
    "ChannelBaseline",
    "OutlierScore",
    "VideoEntry",
    "channel_baselines",
    "channel_medians",
    "collect",
    "outlier_scores",
    "rank_outliers",
]
