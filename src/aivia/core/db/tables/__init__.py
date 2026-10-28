"""Table definitions attached to the shared database metadata."""

from aivia.core.db.tables.core import (
    CLAIM_INDEX_NAME,
    IDEMPOTENCY_INDEX_NAME,
    jobs,
    runs,
    stage_runs,
)

__all__ = [
    "CLAIM_INDEX_NAME",
    "IDEMPOTENCY_INDEX_NAME",
    "jobs",
    "runs",
    "stage_runs",
]
