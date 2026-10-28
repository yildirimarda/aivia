"""Masking for artifacts that leave the process or are written to disk.

Resolved configuration, run snapshots and sidecars are reproducibility records,
not working state. They must never contain credential values, and prompt text
must follow the ``privacy`` settings: omit it when ``store_prompt_text`` is
false, replace it entirely when ``redact_prompts`` is true, and otherwise keep
it with known secrets masked.

Secret values are registered with :mod:`aivia.core.redact` when the secrets
loader resolves them. :func:`mask_secrets` additionally catches token-like
strings by pattern and masks values whose key names a credential.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from aivia.core.config.resolve import ResolvedConfig
from aivia.core.config.sections import PrivacyConfig
from aivia.core.redact import REDACTED, redact

__all__ = [
    "mask_artifact",
    "mask_prompt",
    "mask_resolved_config",
    "mask_secrets",
    "mask_sidecar",
    "mask_snapshot",
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
_PROMPT_KEYS = frozenset(
    {
        "prompt",
        "prompt_content",
        "prompt_text",
        "prompts",
        "negative_prompt",
        "system_prompt",
        "user_prompt",
    }
)
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def _normalise_key(key: object) -> str:
    return _NON_ALPHANUMERIC.sub("_", str(key).lower()).strip("_")


def _is_secret_key(key: object) -> bool:
    normalised = _normalise_key(key)
    if normalised in _SECRET_NAMES:
        return True
    return any(normalised.endswith(f"_{name}") for name in _SECRET_SUFFIXES)


def _is_prompt_key(key: object) -> bool:
    return _normalise_key(key) in _PROMPT_KEYS


def _mask(value: Any, *, privacy: PrivacyConfig | None, prompts: bool) -> Any:
    if isinstance(value, Mapping):
        masked: dict[Any, Any] = {}
        for key, item in value.items():
            safe_key = redact(key) if isinstance(key, str) else key
            if _is_secret_key(key):
                masked[safe_key] = REDACTED
            elif prompts and _is_prompt_key(key):
                masked[safe_key] = mask_prompt(item, privacy)
            else:
                masked[safe_key] = _mask(item, privacy=privacy, prompts=prompts)
        return masked
    if isinstance(value, list):
        return [_mask(item, privacy=privacy, prompts=prompts) for item in value]
    if isinstance(value, tuple):
        return tuple(_mask(item, privacy=privacy, prompts=prompts) for item in value)
    if isinstance(value, set):
        return {_mask(item, privacy=privacy, prompts=prompts) for item in value}
    if isinstance(value, frozenset):
        return frozenset(
            _mask(item, privacy=privacy, prompts=prompts) for item in value
        )
    if isinstance(value, str):
        return redact(value)
    return value


def mask_secrets(value: Any) -> Any:
    """Return *value* with credential keys and token-like strings masked.

    Prompt text is treated like any other string; use :func:`mask_artifact`
    when the value may contain prompts that must follow the privacy settings.
    """
    return _mask(value, privacy=None, prompts=False)


def mask_prompt(value: Any, privacy: PrivacyConfig | None = None) -> Any:
    """Return prompt *value* according to the privacy settings.

    Returns ``None`` when prompt text must not be stored, :data:`REDACTED`
    when prompts are redacted, and otherwise the prompt with secrets masked.
    """
    policy = privacy or PrivacyConfig()
    if not policy.store_prompt_text:
        return None
    if policy.redact_prompts:
        return REDACTED
    return mask_secrets(value)


def mask_artifact(
    value: Any,
    *,
    privacy: PrivacyConfig | None = None,
) -> Any:
    """Mask a run snapshot or sidecar, including prompt-aware redaction."""
    return _mask(value, privacy=privacy, prompts=True)


def mask_snapshot(
    value: Any,
    *,
    privacy: PrivacyConfig | None = None,
) -> Any:
    """Return a run snapshot with secrets and prompt content masked."""
    return mask_artifact(value, privacy=privacy)


def mask_sidecar(
    value: Any,
    *,
    privacy: PrivacyConfig | None = None,
) -> Any:
    """Return a sidecar with secrets and prompt content masked."""
    return mask_artifact(value, privacy=privacy)


def mask_resolved_config(resolved: ResolvedConfig) -> ResolvedConfig:
    """Return a copy of *resolved* whose values are safe to persist or print.

    The original object is not mutated: in-memory configuration still carries
    the real values the runtime needs.
    """
    return replace(
        resolved,
        values=mask_secrets(resolved.values),
        origins=dict(resolved.origins),
        sources=tuple(resolved.sources),
        origin_files=dict(resolved.origin_files),
    )
