"""Idempotent run creation keyed by a caller supplied request key.

A run may carry an ``idempotency_key``. The first request with a key creates a
run and stores a digest of the request payload on it. A later request with the
same key returns that run when its payload digest matches, and raises
:class:`IdempotencyConflict` when the caller reused the key for different input.
The unique index on ``runs.idempotency_key`` is the final guard against two
requests creating a run at the same instant; the loser re-reads the winner and
applies the same payload check.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Connection
from sqlalchemy.exc import IntegrityError

from aivia.core.db.repos import RunRepository
from aivia.core.errors import AiviaError
from aivia.core.models import Run
from aivia.core.models.base import canonical_json

__all__ = [
    "IdempotencyConflict",
    "IdempotentRun",
    "create_run_once",
    "run_payload",
    "run_payload_digest",
]


class IdempotencyConflict(AiviaError):
    """A run idempotency key was reused with a different request payload."""

    code = "idempotency_conflict"


@dataclass(frozen=True, slots=True)
class IdempotentRun:
    """The run for a request and whether this call created it."""

    run: Run
    created: bool


def run_payload(run: Run) -> dict[str, Any]:
    """Return the request-defining fields of *run*.

    Provenance and lifecycle fields are deliberately excluded so repeating a
    request after the first run executed still matches the original payload.
    """
    return {
        "kind": run.kind.value,
        "profile": run.profile,
        "format": run.format,
        "config_snapshot": run.config_snapshot,
    }


def run_payload_digest(run: Run) -> str:
    """Return the stable SHA-256 digest of :func:`run_payload` for *run*."""
    encoded = canonical_json(run_payload(run)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_run_once(
    repository: RunRepository,
    run: Run,
    *,
    prepare: Callable[[Run], None] | None = None,
    connection: Connection | None = None,
) -> IdempotentRun:
    """Create *run* unless its idempotency key already names an equal request.

    ``prepare`` runs after the existing-key check and before the insert, so a
    caller can create the run directory without leaving one behind for a
    repeated key. Pass ``connection`` to join an outer transaction; the key
    lookup and insert then commit or roll back with the caller's work.
    """
    digest = run_payload_digest(run)
    key = run.idempotency_key
    if key:
        existing = _find_by_key(repository, key, connection)
        if existing is not None:
            return _match(existing, run, digest, key)

    if prepare is not None:
        prepare(run)

    candidate = run.model_copy(update={"idempotency_payload_hash": digest})
    try:
        stored = (
            repository.create(candidate)
            if connection is None
            else repository.create_in(connection, candidate)
        )
    except IntegrityError:
        if not key:
            raise
        existing = _find_by_key(repository, key, connection)
        if existing is None:
            raise
        return _match(existing, run, digest, key)
    return IdempotentRun(stored, True)


def _find_by_key(
    repository: RunRepository,
    key: str,
    connection: Connection | None,
) -> Run | None:
    if connection is None:
        return repository.find_by_idempotency_key(key)
    return repository.find_by_idempotency_key_in(connection, key)


def _match(existing: Run, requested: Run, digest: str, key: str) -> IdempotentRun:
    same = existing.idempotency_payload_hash == digest
    if not existing.idempotency_payload_hash:
        same = (
            existing.kind == requested.kind
            and existing.profile == requested.profile
            and existing.format == requested.format
        )
    if same:
        return IdempotentRun(existing, False)
    raise IdempotencyConflict(
        f"idempotency key {key!r} was already used for run {existing.id} "
        "with a different payload",
        details={
            "idempotency_key": key,
            "run_id": existing.id,
            "existing_run_id": existing.id,
        },
    )
