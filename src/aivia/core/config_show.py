"""Rendering for ``aivia config show``: merged values, origins and masking.

The layered merge itself lives in :mod:`aivia.core.config.resolve`; this module
adds the presentation concerns. It masks values whose key looks secret and
renders deterministic YAML, optionally with the origin of every leaf.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import yaml

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
from aivia.core.redact import REDACTED, redact

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

_SECRET_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "auth_token",
        "bearer_token",
        "client_secret",
        "credential",
        "credentials",
        "key",
        "keys",
        "passwd",
        "password",
        "private_key",
        "pwd",
        "refresh_token",
        "secret",
        "secret_key",
        "token",
        "access_token",
    }
)
_SECRET_SUFFIXES = _SECRET_NAMES - {"key", "keys"}
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def _normalise_key(key: object) -> str:
    return _NON_ALPHANUMERIC.sub("_", str(key).lower()).strip("_")


def _is_secret_key(key: object) -> bool:
    normalised = _normalise_key(key)
    if normalised in _SECRET_NAMES:
        return True
    return any(normalised.endswith(f"_{name}") for name in _SECRET_SUFFIXES)


def mask_secrets(value: Any) -> Any:
    """Return *value* with secret-looking keys and token-like strings masked."""
    if isinstance(value, Mapping):
        masked: dict[Any, Any] = {}
        for key, item in value.items():
            masked[key] = REDACTED if _is_secret_key(key) else mask_secrets(item)
        return masked
    if isinstance(value, list):
        return [mask_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(mask_secrets(item) for item in value)
    if isinstance(value, str):
        return redact(value)
    return value


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
