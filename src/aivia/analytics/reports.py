"""Performance reporting by content and platform dimension plus cost per thousand views.

The analytics module is optional: the core never imports it. The computation
lives in :mod:`aivia.core.performance` over the shared cost ledger and platform
metric tables; this module re-exports it so the analytics package owns the
reporting surface while the static dashboard (in core) can consume the same
numbers without importing analytics.
"""

from __future__ import annotations

from aivia.core.performance import (
    PerformanceBucket,
    PerformanceReport,
    build_performance_report,
    performance_by,
)

__all__ = [
    "PerformanceBucket",
    "PerformanceReport",
    "build_performance_report",
    "performance_by",
]
