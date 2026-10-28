"""Table definitions attached to the shared database metadata."""

from aivia.core.db.tables.assets import (
    ASSET_CREATED_AT_INDEX_NAME,
    ASSET_PINNED_INDEX_NAME,
    ASSET_PROMPT_HASH_INDEX_NAME,
    ASSET_PROVIDER_INDEX_NAME,
    ASSET_RUN_INDEX_NAME,
    COMPOSITION_CREATED_AT_INDEX_NAME,
    COMPOSITION_PINNED_INDEX_NAME,
    COMPOSITION_RUN_INDEX_NAME,
    COST_EVENT_CREATED_AT_INDEX_NAME,
    COST_EVENT_PROVIDER_INDEX_NAME,
    COST_EVENT_RUN_INDEX_NAME,
    assets,
    compositions,
    cost_events,
)
from aivia.core.db.tables.core import (
    CLAIM_INDEX_NAME,
    IDEMPOTENCY_INDEX_NAME,
    jobs,
    runs,
    stage_runs,
)

__all__ = [
    "ASSET_CREATED_AT_INDEX_NAME",
    "ASSET_PINNED_INDEX_NAME",
    "ASSET_PROMPT_HASH_INDEX_NAME",
    "ASSET_PROVIDER_INDEX_NAME",
    "ASSET_RUN_INDEX_NAME",
    "CLAIM_INDEX_NAME",
    "COMPOSITION_CREATED_AT_INDEX_NAME",
    "COMPOSITION_PINNED_INDEX_NAME",
    "COMPOSITION_RUN_INDEX_NAME",
    "COST_EVENT_CREATED_AT_INDEX_NAME",
    "COST_EVENT_PROVIDER_INDEX_NAME",
    "COST_EVENT_RUN_INDEX_NAME",
    "IDEMPOTENCY_INDEX_NAME",
    "assets",
    "compositions",
    "cost_events",
    "jobs",
    "runs",
    "stage_runs",
]
