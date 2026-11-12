"""Platform metrics collection and reporting.

The analytics module is optional: the core never imports it. It normalizes
raw platform metrics into a canonical shape so downstream storage and
reporting can treat every platform the same way.
"""

from aivia.analytics.collect import CANONICAL_METRICS, collect

__all__ = ["CANONICAL_METRICS", "collect"]
