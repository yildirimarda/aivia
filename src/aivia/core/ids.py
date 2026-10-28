"""Time-sortable ULIDs with prefixes for the core record kinds.

A ULID is a 128-bit identifier: a 48-bit millisecond timestamp followed by 80
bits of randomness, encoded as 26 Crockford base32 characters. Lexicographic
order therefore matches creation order. Within one millisecond the generator
increments the random component instead of drawing a new one, so even IDs made
in the same tick stay sortable and unique.
"""

from __future__ import annotations

import os
import threading
import time
from enum import StrEnum

__all__ = [
    "CROCKFORD_ALPHABET",
    "ID_PREFIXES",
    "ULID_LENGTH",
    "IdKind",
    "encode_ulid",
    "is_ulid",
    "new_asset_id",
    "new_composition_id",
    "new_cost_id",
    "new_delivery_id",
    "new_id",
    "new_job_id",
    "new_order_id",
    "new_plan_id",
    "new_post_id",
    "new_run_id",
    "new_step_id",
    "ulid",
]

CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
ULID_LENGTH = 26
_ENTROPY_BYTES = 10
_MAX_TIMESTAMP_MS = (1 << 48) - 1
_MAX_ENTROPY = (1 << 80) - 1


class IdKind(StrEnum):
    """The record kinds that own a prefixed identifier."""

    RUN = "run"
    JOB = "job"
    ASSET = "asset"
    ORDER = "order"
    COST = "cost"


ID_PREFIXES: tuple[str, ...] = tuple(kind.value for kind in IdKind)

_lock = threading.Lock()
_last_timestamp_ms = -1
_last_entropy = 0


def _encode(value: int, length: int) -> str:
    chars: list[str] = []
    for _ in range(length):
        chars.append(CROCKFORD_ALPHABET[value & 0x1F])
        value >>= 5
    chars.reverse()
    return "".join(chars)


def encode_ulid(timestamp_ms: int, entropy: bytes) -> str:
    """Encode a millisecond timestamp and 10 bytes of entropy as a ULID.

    This is the pure half of the generator. It is useful for replaying or
    comparing identifiers without touching the clock or the random source.
    """
    if not 0 <= timestamp_ms <= _MAX_TIMESTAMP_MS:
        raise ValueError(f"timestamp_ms must be between 0 and {_MAX_TIMESTAMP_MS}")
    if len(entropy) != _ENTROPY_BYTES:
        raise ValueError(f"entropy must be exactly {_ENTROPY_BYTES} bytes")

    value = (timestamp_ms << 80) | int.from_bytes(entropy, "big")
    return _encode(value, ULID_LENGTH)


def ulid() -> str:
    """Return a new time-sortable ULID.

    The generator is thread safe. When the wall clock does not advance, or moves
    backwards, it keeps the last timestamp and increments the random component
    so identifiers remain strictly increasing.
    """
    global _last_entropy, _last_timestamp_ms

    timestamp_ms = time.time_ns() // 1_000_000
    with _lock:
        if timestamp_ms > _last_timestamp_ms:
            _last_timestamp_ms = timestamp_ms
            _last_entropy = int.from_bytes(os.urandom(_ENTROPY_BYTES), "big")
        else:
            timestamp_ms = _last_timestamp_ms
            _last_entropy += 1
            if _last_entropy > _MAX_ENTROPY:
                _last_timestamp_ms += 1
                timestamp_ms = _last_timestamp_ms
                _last_entropy = int.from_bytes(os.urandom(_ENTROPY_BYTES), "big")
        entropy = _last_entropy.to_bytes(_ENTROPY_BYTES, "big")

    return encode_ulid(timestamp_ms, entropy)


def is_ulid(value: str) -> bool:
    """Return whether *value* is a canonical, non-overflowing ULID."""
    if len(value) != ULID_LENGTH or value[0] > "7":
        return False
    return all(char in CROCKFORD_ALPHABET for char in value)


def new_id(kind: IdKind | str) -> str:
    """Return a prefixed identifier for *kind*, such as ``run_01...``."""
    try:
        resolved = IdKind(kind)
    except ValueError as exc:
        raise ValueError(f"unknown id kind: {kind}") from exc
    return f"{resolved.value}_{ulid()}"


def new_run_id() -> str:
    """Return a new run identifier."""
    return new_id(IdKind.RUN)


def new_job_id() -> str:
    """Return a new job identifier."""
    return new_id(IdKind.JOB)


def new_asset_id() -> str:
    """Return a new asset identifier."""
    return new_id(IdKind.ASSET)


def new_order_id() -> str:
    """Return a new order identifier."""
    return new_id(IdKind.ORDER)


def new_cost_id() -> str:
    """Return a new cost event identifier."""
    return new_id(IdKind.COST)


def new_post_id() -> str:
    """Return a new post identifier."""
    return f"post_{ulid()}"


def new_delivery_id() -> str:
    """Return a new delivery identifier."""
    return f"delivery_{ulid()}"


def new_plan_id() -> str:
    """Return a new plan identifier."""
    return f"plan_{ulid()}"


def new_step_id() -> str:
    """Return a new plan step identifier."""
    return f"step_{ulid()}"


def new_composition_id() -> str:
    """Return a new composition reference identifier."""
    return f"composition_{ulid()}"
