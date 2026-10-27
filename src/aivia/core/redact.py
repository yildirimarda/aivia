"""Redaction of registered secrets and token-like strings.

The helper is deliberately process-wide: secrets are registered once, when they
are resolved, and every log record or formatted error can then be passed through
:func:`redact` without threading a configuration object through the call stack.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from threading import RLock
from typing import Any

__all__ = [
    "REDACTED",
    "clear_secrets",
    "redact",
    "redact_data",
    "redact_exception",
    "register_secret",
    "register_secrets",
    "registered_secrets",
    "unregister_secret",
]

REDACTED = "[REDACTED]"

_secrets: set[str] = set()
_secrets_lock = RLock()


def _token_patterns() -> tuple[tuple[re.Pattern[str], str], ...]:
    return (
        (
            re.compile(
                r"""
                (?P<keyquote>["']?)
                \b(?P<key>
                    api[_-]?key | secret | token | password | passwd | pwd |
                    access[_-]?token | refresh[_-]?token | client[_-]?secret |
                    auth[_-]?token | bearer[_-]?token
                )\b
                (?P=keyquote)
                (?P<sep>\s*[:=]\s*)
                (?P<quote>["']?)
                (?P<value>(?-i:(?![A-Z][A-Z0-9_]*\b))[^\s"'&,;]{4,})
                (?P=quote)
                """,
                re.IGNORECASE | re.VERBOSE,
            ),
            r"\g<keyquote>\g<key>\g<keyquote>\g<sep>\g<quote>"
            + REDACTED
            + r"\g<quote>",
        ),
        (
            re.compile(r"\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
            r"\1 " + REDACTED,
        ),
        (
            re.compile(r"\b(https?://)([^/\s:@]+):([^/\s@]+)@", re.IGNORECASE),
            r"\1\2:" + REDACTED + "@",
        ),
        (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"), REDACTED),
        (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), REDACTED),
        (
            re.compile(
                r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
            ),
            REDACTED,
        ),
        (re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b"), REDACTED),
        (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), REDACTED),
        (
            re.compile(
                r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
            ),
            REDACTED,
        ),
        (
            re.compile(
                r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?"
                r"-----END [A-Z ]*PRIVATE KEY-----",
                re.DOTALL,
            ),
            REDACTED,
        ),
    )


_TOKEN_PATTERNS = _token_patterns()


def register_secret(value: str) -> None:
    """Remember *value* so it is masked in every later redaction.

    Empty and whitespace-only values are ignored, as is the redaction marker
    itself so that repeated redaction stays idempotent.
    """
    if not isinstance(value, str):
        raise TypeError("secret must be a string")
    if not value.strip() or value == REDACTED:
        return
    with _secrets_lock:
        _secrets.add(value)


def register_secrets(*values: str) -> None:
    """Register several secret values at once."""
    for value in values:
        register_secret(value)


def unregister_secret(value: str) -> None:
    """Forget a previously registered secret value."""
    with _secrets_lock:
        _secrets.discard(value)


def clear_secrets() -> None:
    """Forget every registered secret value."""
    with _secrets_lock:
        _secrets.clear()


def registered_secrets() -> tuple[str, ...]:
    """Return the registered secret values in a stable order."""
    with _secrets_lock:
        return tuple(sorted(_secrets))


def redact(text: str) -> str:
    """Return *text* with known secrets and token-like strings masked."""
    if not text:
        return text

    with _secrets_lock:
        secrets = tuple(sorted(_secrets, key=len, reverse=True))

    redacted = text
    for secret in secrets:
        redacted = redacted.replace(secret, REDACTED)
    for pattern, replacement in _TOKEN_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def redact_data(value: Any) -> Any:
    """Recursively redact strings inside mappings, sequences and sets."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Mapping):
        return {redact_data(key): redact_data(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(redact_data(item) for item in value)
    if isinstance(value, list):
        return [redact_data(item) for item in value]
    if isinstance(value, frozenset):
        return frozenset(redact_data(item) for item in value)
    if isinstance(value, set):
        return {redact_data(item) for item in value}
    return value


def redact_exception(exc: BaseException) -> str:
    """Return the exception and its cause chain as one redacted line."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current)
        name = type(current).__name__
        parts.append(f"{name}: {message}" if message else name)
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__suppress_context__:
            current = None
        else:
            current = current.__context__
    return redact(" <- ".join(parts))
