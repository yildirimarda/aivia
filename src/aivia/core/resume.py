"""Stable hashes that decide whether a resumed run can skip a stage.

A stage is skipped only when its previous result succeeded and the inputs it
would receive now hash to the same value it recorded. The hash deliberately
covers the stage name, its recipe options, the resolved configuration and the
outputs of every earlier stage, and deliberately excludes run ids, timestamps
and ordering so the same recipe over the same inputs produces the same digest
on a fresh process.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from aivia.core.models.base import canonical_json
from aivia.core.models.run import StageResult, StageState
from aivia.core.run_dirs import RunDirectory
from aivia.core.stages import Stage

__all__ = [
    "artifacts_intact",
    "compute_input_hash",
    "compute_output_hash",
    "input_hash",
    "output_hash",
    "should_skip_stage",
    "stable_value",
    "verify_artifacts",
]


def stable_value(value: Any) -> Any:
    """Convert *value* into deterministic JSON-compatible data.

    Pydantic models, dataclasses, enums, mappings, sequences and sets are
    normalised recursively. Values with no stable representation fall back to
    their ``repr`` so hashing never raises for an exotic config field.
    """
    if isinstance(value, BaseModel):
        return stable_value(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return stable_value(value.value)
    if isinstance(value, Mapping):
        return {
            str(key): stable_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [stable_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [stable_value(item) for item in value]
        return sorted(items, key=canonical_json)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: stable_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _stage_implementation_version(stage: Stage) -> Any:
    value = getattr(stage, "implementation_version", None)
    if value is None or value == "":
        value = getattr(stage, "version", "")
    return stable_value(value)


def _stage_code_fingerprint(stage: Stage) -> str:
    target = type(stage)
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        method = target.__dict__.get("run")
        try:
            source = inspect.getsource(method) if method is not None else ""
        except (OSError, TypeError):
            source = ""
        source = f"{target.__module__}.{target.__qualname__}:{source}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _input_path(declared: str, base_dir: Path | None) -> Path:
    candidate = Path(declared).expanduser()
    if candidate.is_absolute():
        return candidate
    if base_dir is not None:
        rooted = Path(base_dir).expanduser() / candidate
        if rooted.is_file():
            return rooted
    return candidate


def _input_file_digest(declared: str, base_dir: Path | None) -> dict[str, Any]:
    path = _input_path(declared, base_dir)
    row: dict[str, Any] = {
        "path": declared,
        "sha256": "",
        "size_bytes": 0,
        "missing": True,
    }
    try:
        if not path.is_file():
            return row
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError:
        return row
    return {
        "path": declared,
        "sha256": digest.hexdigest(),
        "size_bytes": size,
        "missing": False,
    }


def _external_inputs(stage: Stage, base_dir: Path | None) -> list[dict[str, Any]]:
    declared = getattr(stage, "inputs", ())
    if declared is None:
        return []
    if isinstance(declared, (str, Path)):
        declared = (declared,)
    rows = [_input_file_digest(str(path), base_dir) for path in declared]
    rows.sort(key=lambda row: row["path"])
    return rows


def compute_input_hash(
    stage: Stage,
    *,
    config: Any = None,
    upstream: Iterable[StageResult] = (),
    base_dir: str | Path | None = None,
) -> str:
    """Return the stable input digest for *stage*.

    *upstream* is the ordered list of earlier stage results in this pass. Only
    their name, status, output hash and registered artifact paths contribute;
    their timestamps and costs do not, so a resumed run hashes the same inputs
    as the original. The stage's implementation version and source also
    contribute, so bumping a version or changing its code re-runs the stage.
    The content digest of every declared external input file contributes too,
    so editing a source file re-runs the stage. Relative input paths are
    resolved against *base_dir* first and then the current directory.
    """
    root = Path(base_dir).expanduser() if base_dir is not None else None
    payload = {
        "stage": stage.name,
        "version": _stage_implementation_version(stage),
        "implementation": _stage_code_fingerprint(stage),
        "options": stable_value(stage.options),
        "config": stable_value(config),
        "declared_artifacts": sorted(stage.artifacts),
        "declared_inputs": _external_inputs(stage, root),
        "upstream": [
            {
                "stage": result.stage,
                "status": result.status.value,
                "output_hash": result.output_hash,
                "artifacts": sorted(result.artifacts),
            }
            for result in upstream
        ],
    }
    return _digest(payload)


input_hash = compute_input_hash
"""Alias for :func:`compute_input_hash`."""


def compute_output_hash(artifacts: Iterable[Any]) -> str:
    """Return the digest of the registered artifacts for one stage.

    *artifacts* may be :class:`~aivia.core.run_dirs.ArtifactRecord` objects or
    mappings with ``path``, ``sha256`` and ``size_bytes`` keys. An empty list
    hashes to an empty string so a stage without artifacts stays distinguishable
    from one whose files happen to hash to a constant.
    """
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        if isinstance(artifact, Mapping):
            path = str(artifact.get("path", ""))
            digest = str(artifact.get("sha256", ""))
            size = int(artifact.get("size_bytes", 0) or 0)
        else:
            path = str(getattr(artifact, "path", ""))
            digest = str(getattr(artifact, "sha256", ""))
            size = int(getattr(artifact, "size_bytes", 0) or 0)
        rows.append({"path": path, "sha256": digest, "size_bytes": size})
    if not rows:
        return ""
    rows.sort(key=lambda row: row["path"])
    return _digest(rows)


output_hash = compute_output_hash
"""Alias for :func:`compute_output_hash`."""


def artifacts_intact(
    previous: StageResult | None,
    directory: RunDirectory,
    *,
    superseded: Iterable[str] = (),
) -> bool:
    """Whether the artifacts registered for *previous* still match on disk.

    The set of paths recorded on the previous result must match the stage's
    manifest entries, and every file must still exist with the registered size
    and digest. A result without artifacts is intact only when nothing was
    registered for its stage, so a stage whose files vanished or whose manifest
    was lost is never skipped.

    *superseded* lists paths a later stage registered again. Those files belong
    to the later stage now, so their current contents are not evidence that
    this stage's output changed.
    """
    if previous is None:
        return False
    ignored = {str(path) for path in superseded}
    registered = directory.artifacts(previous.stage)
    expected_paths = {
        str(path) for path in previous.artifacts if str(path) not in ignored
    }
    registered_paths = {
        record.path for record in registered if record.path not in ignored
    }
    if expected_paths != registered_paths:
        return False
    if not expected_paths:
        return True
    owned = tuple(record for record in registered if record.path not in ignored)
    return not directory.verify(previous.stage, owned)


verify_artifacts = artifacts_intact
"""Alias for :func:`artifacts_intact`."""


def should_skip_stage(
    previous: StageResult | None,
    input_hash: str,
    *,
    directory: RunDirectory | None = None,
    superseded: Iterable[str] = (),
) -> bool:
    """Whether *previous* can stand in for a stage with *input_hash*.

    Only a succeeded result with a recorded hash is reusable. A failed,
    skipped, paused or legacy result without a hash is executed again. When
    *directory* is given its registered artifact hashes are also verified, so a
    changed or vanished output forces the stage to run again.
    """
    if not (
        previous is not None
        and previous.status is StageState.SUCCEEDED
        and bool(previous.input_hash)
        and previous.input_hash == input_hash
    ):
        return False
    return not (
        directory is not None
        and not artifacts_intact(previous, directory, superseded=superseded)
    )
