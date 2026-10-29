"""Run directories and the per-stage artifact registry.

Every run owns one directory below the user data directory. Stages write their
outputs there, and after a stage finishes the orchestrator registers the files
it produced. Registration validates that each path stays inside the run
directory, hashes the file and records it under the stage that wrote it, so a
resume or a replay can see exactly what a stage left behind.

The registry is a small JSON manifest (``artifacts.json``) in the run
directory. It is written atomically so a crash never leaves a half-written
manifest, and registering the same stage again replaces that stage's entries
instead of duplicating them. Artifact paths in the manifest are relative and
use forward slashes, which keeps the manifest portable and safe to compare
between machines.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aivia.core.models.base import canonical_json
from aivia.core.models.run import Run, StageResult

__all__ = [
    "ARTIFACT_MANIFEST_NAME",
    "RUNS_DIR_NAME",
    "ArtifactRecord",
    "RunDirectories",
    "RunDirectory",
    "create_run_directory",
    "register_stage_artifacts",
]

RUNS_DIR_NAME = "runs"
"""The directory below the user data directory that holds every run."""

ARTIFACT_MANIFEST_NAME = "artifacts.json"
"""The file in a run directory that records artifacts per stage."""

_MANIFEST_VERSION = 1
_READ_CHUNK = 1024 * 1024


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser()


def _normalise_component(value: str, *, what: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string")
    component = value.strip()
    if not component:
        raise ValueError(f"{what} must be a non-empty string")
    if component in {".", ".."} or "/" in component or "\\" in component:
        raise ValueError(f"{what} must not contain a path separator")
    if "\x00" in component:
        raise ValueError(f"{what} must not contain a NUL byte")
    return component


def _normalise_stage(value: str) -> str:
    return _normalise_component(value, what="stage name")


def _normalise_artifacts(
    values: Iterable[str | Path] | str | Path,
) -> tuple[str | Path, ...]:
    if isinstance(values, (str, Path)):
        return (values,)
    return tuple(values)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """One file registered under the stage that produced it.

    ``path`` is relative to the run directory and always uses ``/`` so the
    manifest does not depend on the platform that wrote it. ``sha256`` is the
    content digest and ``size_bytes`` the file size at registration time.
    """

    stage: str
    path: str
    sha256: str
    size_bytes: int

    @property
    def content_hash(self) -> str:
        """Alias for :attr:`sha256`."""
        return self.sha256

    @property
    def size(self) -> int:
        """Alias for :attr:`size_bytes`."""
        return self.size_bytes

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON representation stored in the manifest."""
        return {
            "stage": self.stage,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArtifactRecord:
        """Build a record from a manifest entry."""
        if not isinstance(data, Mapping):
            raise ValueError("artifact entry must be an object")
        try:
            raw_stage = data["stage"]
            raw_path = data["path"]
            raw_digest = data["sha256"]
            raw_size = data["size_bytes"]
        except KeyError as exc:
            raise ValueError("artifact entry is missing a required field") from exc
        stage = _normalise_stage(raw_stage)
        path = str(raw_path).strip()
        if not path:
            raise ValueError("artifact path must not be empty")
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact path must be relative to the run directory")
        digest = str(raw_digest).lower()
        if digest and (
            len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("artifact sha256 must be a 64 character hex digest")
        try:
            size = int(raw_size)
        except (TypeError, ValueError) as exc:
            raise ValueError("artifact size must be an integer") from exc
        if size < 0:
            raise ValueError("artifact size must not be negative")
        return cls(stage=stage, path=path, sha256=digest, size_bytes=size)


class RunDirectory:
    """One run directory and the artifacts registered inside it.

    The directory is created lazily by :meth:`create`; opening an existing run
    for a resume uses the same object without touching the filesystem. Every
    path that enters the registry is resolved and checked to be inside
    :attr:`path`, so a stage cannot register a file outside its run.
    """

    def __init__(self, path: str | Path, *, run_id: str = "") -> None:
        self.path = _as_path(path).resolve(strict=False)
        self.run_id = run_id.strip()

    def __repr__(self) -> str:
        return f"RunDirectory(path={str(self.path)!r}, run_id={self.run_id!r})"

    def __fspath__(self) -> str:
        return str(self.path)

    @property
    def run_dir(self) -> Path:
        """Alias for :attr:`path`."""
        return self.path

    @property
    def directory(self) -> Path:
        """Alias for :attr:`path`."""
        return self.path

    @property
    def root(self) -> Path:
        """Alias for :attr:`path`."""
        return self.path

    @property
    def manifest_path(self) -> Path:
        """The artifact manifest inside this run directory."""
        return self.path / ARTIFACT_MANIFEST_NAME

    @property
    def snapshot_path(self) -> Path:
        """The replay snapshot inside this run directory."""
        from aivia.core.snapshot import snapshot_path

        return snapshot_path(self.path)

    def write_snapshot(
        self, snapshot: Mapping[str, Any] | None = None, **options: Any
    ) -> Path:
        """Write the replay snapshot into this run directory."""
        from aivia.core.snapshot import write_config_snapshot

        return write_config_snapshot(self.path, snapshot, **options)

    def read_snapshot(self, **options: Any) -> Any:
        """Read the replay snapshot from this run directory."""
        from aivia.core.snapshot import read_config_snapshot

        return read_config_snapshot(self.path, **options)

    @property
    def exists(self) -> bool:
        """Whether the run directory exists."""
        return self.path.is_dir()

    def create(self, *, exist_ok: bool = True) -> RunDirectory:
        """Create the run directory and return ``self`` for chaining.

        ``exist_ok`` lets a resumed run reuse its directory. When it is false
        an existing directory raises :class:`FileExistsError`. A path that is
        occupied by a file is always an error.
        """
        if self.path.exists() and not self.path.is_dir():
            raise NotADirectoryError(f"run path is not a directory: {self.path}")
        self.path.mkdir(parents=True, exist_ok=exist_ok)
        return self

    def stage_dir(self, stage: str, *, create: bool = True) -> Path:
        """Return a private directory for one stage.

        Stages that produce several intermediate files may use this directory
        instead of writing into the run root. The path stays below the run
        directory and is not itself registered as an artifact.
        """
        name = _normalise_stage(stage)
        path = self.path / name
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def resolve(self, artifact: str | Path) -> Path:
        """Resolve *artifact* to an absolute path inside the run directory.

        A relative path is joined to :attr:`path`; an absolute path is accepted
        only when it already points inside the run directory. ``..`` segments
        and symlinks that escape the run are rejected.
        """
        raw = str(artifact) if isinstance(artifact, Path) else artifact
        if not raw.strip():
            raise ValueError("artifact path must not be empty")
        candidate = _as_path(artifact)
        full = candidate if candidate.is_absolute() else self.path / candidate
        resolved = full.resolve(strict=False)
        if resolved != self.path and not resolved.is_relative_to(self.path):
            raise ValueError(f"artifact path escapes the run directory: {artifact}")
        return resolved

    def relative_path(self, artifact: str | Path) -> str:
        """Return *artifact* relative to the run directory as a POSIX path."""
        resolved = self.resolve(artifact)
        return resolved.relative_to(self.path).as_posix()

    def register(
        self,
        stage: str,
        artifacts: Iterable[str | Path] | str | Path | None = None,
        *,
        result: StageResult | None = None,
        require_exists: bool = True,
    ) -> tuple[ArtifactRecord, ...]:
        """Register the artifacts one stage produced.

        *artifacts* may be a single path or an iterable of paths. When omitted,
        ``result.artifacts`` is used. Files must exist and be regular files
        unless ``require_exists`` is false. The stage's previous entries are
        replaced, so re-running a stage does not accumulate stale paths.

        When *result* is supplied its ``artifacts`` list is replaced with the
        registered relative paths, which is what the orchestrator persists in
        the stage result.
        """
        name = _normalise_stage(stage)
        if result is not None:
            if result.stage != name:
                raise ValueError(
                    f"result stage {result.stage!r} does not match {name!r}"
                )
            if artifacts is None:
                artifacts = result.artifacts
        if artifacts is None:
            artifacts = ()

        records: list[ArtifactRecord] = []
        seen: set[str] = set()
        for artifact in _normalise_artifacts(artifacts):
            resolved = self.resolve(artifact)
            relative = resolved.relative_to(self.path).as_posix()
            if relative in seen:
                continue
            if require_exists and not resolved.exists():
                raise FileNotFoundError(f"artifact does not exist: {relative}")
            if resolved.exists() and not resolved.is_file():
                raise IsADirectoryError(f"artifact is not a file: {relative}")
            if resolved.is_file():
                digest = _sha256_file(resolved)
                size = resolved.stat().st_size
            else:
                digest = ""
                size = 0
            records.append(
                ArtifactRecord(
                    stage=name,
                    path=relative,
                    sha256=digest,
                    size_bytes=size,
                )
            )
            seen.add(relative)

        self._write_stage(name, records)
        if result is not None:
            result.artifacts = [record.path for record in records]
        return tuple(records)

    def artifacts(self, stage: str | None = None) -> tuple[ArtifactRecord, ...]:
        """Return registered artifacts, optionally limited to one stage."""
        manifest = self._read_manifest()
        stages: Mapping[str, Any] = manifest.get("stages", {})
        if stage is not None:
            name = _normalise_stage(stage)
            return tuple(
                ArtifactRecord.from_dict(entry) for entry in stages.get(name, ())
            )
        records: list[ArtifactRecord] = []
        for entries in stages.values():
            records.extend(ArtifactRecord.from_dict(entry) for entry in entries)
        return tuple(records)

    def verify(
        self,
        stage: str,
        records: Iterable[ArtifactRecord] | None = None,
    ) -> tuple[ArtifactRecord, ...]:
        """Return registered artifacts that no longer match their manifest entry.

        A record is stale when its file vanished, stopped being a regular file,
        changed size or hashes to a different digest. *records* overrides the
        stage's current manifest entries, which lets a resume verify the set a
        previous stage result actually registered.
        """
        name = _normalise_stage(stage)
        expected = tuple(self.artifacts(name) if records is None else records)
        stale: list[ArtifactRecord] = []
        for record in expected:
            if record.stage != name:
                stale.append(record)
                continue
            try:
                path = self.resolve(record.path)
            except ValueError:
                stale.append(record)
                continue
            if not path.is_file():
                stale.append(record)
                continue
            try:
                if path.stat().st_size != record.size_bytes:
                    stale.append(record)
                    continue
                if _sha256_file(path) != record.sha256:
                    stale.append(record)
            except OSError:
                stale.append(record)
        return tuple(stale)

    def stages(self) -> tuple[str, ...]:
        """Return the stages that have registered artifacts, sorted by name."""
        manifest = self._read_manifest()
        return tuple(sorted(manifest.get("stages", {})))

    def _read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid artifact manifest: {self.manifest_path}"
            ) from exc
        if not isinstance(data, dict):
            raise ValueError(f"invalid artifact manifest: {self.manifest_path}")
        return data

    def _write_stage(self, stage: str, records: Iterable[ArtifactRecord]) -> None:
        manifest = self._read_manifest()
        raw_stages = manifest.get("stages")
        stages: dict[str, Any] = (
            dict(raw_stages) if isinstance(raw_stages, dict) else {}
        )
        stages[stage] = [record.to_dict() for record in records]
        payload = {
            "version": _MANIFEST_VERSION,
            "run_id": self.run_id,
            "stages": stages,
        }
        self.path.mkdir(parents=True, exist_ok=True)
        self._write_atomic(self.manifest_path, canonical_json(payload))

    @staticmethod
    def _write_atomic(path: Path, text: str) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise


class RunDirectories:
    """The runs root below one user data directory.

    A run directory is ``<data_dir>/runs/<run id>`` unless the run already
    carries an explicit ``run_dir``. Creating a run directory records its
    absolute path on the run record so later stages and a resume use the same
    location.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = _as_path(data_dir).resolve(strict=False)

    def __repr__(self) -> str:
        return f"RunDirectories(data_dir={str(self.data_dir)!r})"

    @property
    def runs_dir(self) -> Path:
        """The directory that contains every run."""
        return self.data_dir / RUNS_DIR_NAME

    def path_for(self, run: Run | str) -> Path:
        """Return the directory a run uses without creating it."""
        if isinstance(run, Run):
            if run.run_dir:
                return _as_path(run.run_dir).resolve(strict=False)
            run_id = run.id
        else:
            run_id = run
        return (self.runs_dir / _normalise_component(run_id, what="run id")).resolve(
            strict=False
        )

    def create(self, run: Run, *, exist_ok: bool = True) -> RunDirectory:
        """Create the directory for *run* and record it on the run."""
        directory = RunDirectory(self.path_for(run), run_id=run.id)
        directory.create(exist_ok=exist_ok)
        run.run_dir = str(directory.path)
        return directory

    def open(self, run: Run | str) -> RunDirectory:
        """Open an existing run directory without creating it."""
        run_id = run.id if isinstance(run, Run) else run
        return RunDirectory(self.path_for(run), run_id=run_id)

    def exists(self, run: Run | str) -> bool:
        """Whether the directory for *run* exists."""
        return self.open(run).exists


def create_run_directory(
    data_dir: str | Path,
    run: Run,
    *,
    exist_ok: bool = True,
) -> RunDirectory:
    """Create the run directory for *run* below *data_dir*."""
    return RunDirectories(data_dir).create(run, exist_ok=exist_ok)


def register_stage_artifacts(
    run_dir: str | Path | RunDirectory,
    stage: str,
    artifacts: Iterable[str | Path] | str | Path | None = None,
    *,
    result: StageResult | None = None,
    require_exists: bool = True,
) -> tuple[ArtifactRecord, ...]:
    """Register the artifacts one stage produced in *run_dir*."""
    directory = run_dir if isinstance(run_dir, RunDirectory) else RunDirectory(run_dir)
    return directory.register(
        stage,
        artifacts,
        result=result,
        require_exists=require_exists,
    )
