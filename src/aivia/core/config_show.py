"""Rendering for ``aivia config show``: merged values, origins and masking.

The layered merge itself lives in :mod:`aivia.core.config.resolve`; this module
adds the presentation concerns. It masks values whose key looks secret and
renders deterministic YAML, optionally with the origin of every leaf.
"""

from __future__ import annotations

import yaml

from aivia.core.config.masking import mask_secrets
from aivia.core.config.resolve import (
    DEFAULT_ORIGIN,
    PROFILE_ORIGIN_PREFIX,
    USER_ORIGIN,
    Layer,
    ResolvedConfig,
    merge_layers,
    profile_overrides,
    resolve_config,
)

__all__ = [
    "DEFAULT_ORIGIN",
    "PROFILE_ORIGIN_PREFIX",
    "USER_ORIGIN",
    "Layer",
    "ResolvedConfig",
    "dump_config",
    "mask_secrets",
    "merge_layers",
    "profile_overrides",
    "resolve_config",
]


def dump_config(
    resolved: ResolvedConfig,
    *,
    include_origins: bool = False,
) -> str:
    """Render a resolved configuration as deterministic, masked YAML."""
    lines: list[str] = []
    if resolved.profile is not None:
        lines.append(f"# profile: {resolved.profile}")
    body = yaml.safe_dump(
        mask_secrets(resolved.values),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    ).rstrip("\n")
    lines.append(body)
    if include_origins:
        lines.append("# origins")
        for path in sorted(resolved.origins):
            lines.append(f"# {path}: {resolved.origins[path]}")
    return "\n".join(lines) + "\n"
