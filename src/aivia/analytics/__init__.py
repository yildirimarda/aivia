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
from aivia.analytics.reports import (
    PerformanceBucket,
    PerformanceReport,
    build_performance_report,
    performance_by,
)
from aivia.analytics.retention import (
    ClosingSlide,
    HookLeak,
    RetentionExport,
    RetentionPoint,
    SteepestDrop,
    closing_slide,
    hook_leak,
    import_retention_export,
    steepest_drops,
)

__all__ = [
    "CANONICAL_METRICS",
    "ChannelBaseline",
    "ClosingSlide",
    "HookLeak",
    "OutlierScore",
    "PerformanceBucket",
    "PerformanceReport",
    "RetentionExport",
    "RetentionPoint",
    "SteepestDrop",
    "VideoEntry",
    "build_performance_report",
    "channel_baselines",
    "channel_medians",
    "closing_slide",
    "collect",
    "hook_leak",
    "import_retention_export",
    "outlier_scores",
    "performance_by",
    "rank_outliers",
    "steepest_drops",
]
