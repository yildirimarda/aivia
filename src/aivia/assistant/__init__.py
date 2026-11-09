"""Asset registry for one session or run.

The registry holds files with labels, content hashes and analysis. Originals
are copied into the registry directory and marked read-only, so they can never
be modified in place. Duplicate labels are rejected.
"""

from aivia.assistant.registry import (
    AssetRegistry,
    DuplicateLabelError,
    UnknownLabelError,
    resolve_labels,
)

__all__ = [
    "AssetRegistry",
    "DuplicateLabelError",
    "UnknownLabelError",
    "resolve_labels",
]
