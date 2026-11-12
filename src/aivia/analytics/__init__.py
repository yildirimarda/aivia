"""Platform metrics collection and reporting.

The analytics module is optional: the core never imports it. It normalizes
raw platform metrics into a canonical shape so downstream storage and
reporting can treat every platform the same way, and scores videos by how
their views compare to their own channel's median.
"""

from aivia.analytics.ab import ABResult, Alternative, compare_alternatives
from aivia.analytics.collect import CANONICAL_METRICS, collect
from aivia.analytics.learning import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT,
    MAX_WEIGHT,
    MIN_WEIGHT,
    CategoryWeight,
    CategoryWeightChange,
    CategoryWeights,
    apply_performance_to_category_weights,
    apply_performance_to_report,
    category_weights_path,
    load_category_weights,
    save_category_weights,
)
from aivia.analytics.outlier import (
    ChannelBaseline,
    OutlierScore,
    VideoEntry,
    channel_baselines,
    channel_medians,
    outlier_scores,
    rank_outliers,
)
from aivia.analytics.ratings import all_ratings, capture_rating, rating_for
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
from aivia.analytics.weekly import (
    TOP_PERFORMER_LIMIT,
    WeeklyProviderComparison,
    WeeklyReport,
    WeeklyTopPerformer,
    build_weekly_report,
    deliver_weekly_report,
    weekly_report_message,
)

__all__ = [
    "CANONICAL_METRICS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_WEIGHT",
    "MAX_WEIGHT",
    "MIN_WEIGHT",
    "TOP_PERFORMER_LIMIT",
    "ABResult",
    "Alternative",
    "CategoryWeight",
    "CategoryWeightChange",
    "CategoryWeights",
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
    "WeeklyProviderComparison",
    "WeeklyReport",
    "WeeklyTopPerformer",
    "all_ratings",
    "apply_performance_to_category_weights",
    "apply_performance_to_report",
    "build_performance_report",
    "build_weekly_report",
    "capture_rating",
    "category_weights_path",
    "channel_baselines",
    "channel_medians",
    "closing_slide",
    "collect",
    "compare_alternatives",
    "deliver_weekly_report",
    "hook_leak",
    "import_retention_export",
    "load_category_weights",
    "outlier_scores",
    "performance_by",
    "rank_outliers",
    "rating_for",
    "save_category_weights",
    "steepest_drops",
    "weekly_report_message",
]
