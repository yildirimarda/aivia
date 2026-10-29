"""Cross-process locks that keep scheduled runs from overlapping unsafely.

A scheduled run takes two locks for its whole lifetime: one per content profile
and one of a fixed number of global slots. The profile lock stops two runs of
the same profile from touching the same output files, while the global slots
enforce the configured parallel run limit across every profile and process.
Locks are advisory file locks, so the operating system releases them when a
process exits or crashes; no heartbeat or stale timeout is needed.

The lock files live under ``locks/`` in the user data directory. Profile names
are hashed into the file name, so a name cannot escape the directory and two
names that sanitise the same way still get distinct locks. The owner metadata
next to each lock is best-effort information for diagnostics and error details;
correctness comes from the lock itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO

from aivia.core import paths
from aivia.core.config.ops import JobsConfig
from aivia.core.errors import AiviaError, ConfigError
from aivia.core.models.base import utcnow

__all__ = [
    "DEFAULT_LOCK_DIRNAME",
    "GlobalRunLimitError",
    "LockBusyError",
    "LockError",
    "LockOwner",
    "ParallelRunLimitError",
    "ProfileLockError",
    "RunLimitError",
    "RunLock",
    "RunLockManager",
    "RunLockStore",
    "RunLocks",
    "run_locks_dir",
]

DEFAULT_LOCK_DIRNAME = "locks"
"""The directory under the user data directory that holds run locks."""

_METADATA_VERSION = 1
_FLOOR = 0.0
_PROFILE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class LockError(AiviaError):
    """Base class for run lock failures."""

    code = "lock"


class LockBusyError(LockError):
    """A lock is held by another run and was not acquired."""

    code = "lock_busy"
    retryable = True


class ProfileLockError(LockBusyError):
    """Another run of the same profile is still active."""

    code = "profile_locked"


class ParallelRunLimitError(LockBusyError):
    """The global parallel run limit has no free slot."""

    code = "parallel_run_limit"


GlobalRunLimitError = ParallelRunLimitError
"""Alias for :class:`ParallelRunLimitError`."""

RunLimitError = ParallelRunLimitError
"""Alias for :class:`ParallelRunLimitError`."""


def run_locks_dir(
    data_dir: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    dirname: str = DEFAULT_LOCK_DIRNAME,
) -> Path:
    """Return the default directory that holds run lock files.

    The directory lives under the user data directory, which is resolved
    without consulting the current working directory. Pass *data_dir* to
    override it.
    """
    root = paths.data_dir(environ) if data_dir is None else Path(data_dir)
    return root.expanduser() / dirname


def _coerce_limit(value: object, *, label: str = "max_parallel") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"{label} must be a non-negative integer",
            details={label: value},
        )
    if value < 0:
        raise ConfigError(
            f"{label} must be a non-negative integer",
            details={label: value},
        )
    return value


def _coerce_profile(profile: object) -> str:
    if not isinstance(profile, str):
        raise ConfigError("profile must be a string")
    text = profile.strip()
    if not text:
        raise ConfigError("profile must be a non-empty name")
    return text


def _coerce_timeout(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("lock timeout must be a number")
    timeout = float(value)
    if timeout < _FLOOR:
        raise ConfigError("lock timeout must be non-negative")
    return timeout


def _profile_filename(profile: str) -> str:
    slug = _PROFILE_NAME.sub("_", profile).strip("._-")[:64] or "profile"
    digest = hashlib.sha256(profile.encode("utf-8")).hexdigest()[:16]
    return f"{slug}-{digest}"


def _lock_handle(handle: BinaryIO, *, unlock: bool) -> None:
    """Take or release an exclusive advisory lock on *handle*."""
    if os.name == "nt":
        import importlib

        msvcrt: Any = importlib.import_module("msvcrt")
        handle.seek(0)
        msvcrt.locking(
            handle.fileno(),
            msvcrt.LK_UNLCK if unlock else msvcrt.LK_NBLCK,
            1,
        )
    else:
        import fcntl

        fcntl.flock(
            handle.fileno(),
            fcntl.LOCK_UN if unlock else fcntl.LOCK_EX | fcntl.LOCK_NB,
        )


def _open_lock(path: Path) -> BinaryIO | None:
    """Open *path* and take its lock without waiting, or return ``None``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
        _lock_handle(handle, unlock=False)
        return handle
    except OSError:
        handle.close()
        return None


def _close_lock(handle: BinaryIO) -> None:
    """Release *handle* and close it, tolerating an already-broken lock."""
    try:
        _lock_handle(handle, unlock=True)
    except OSError:
        pass
    finally:
        handle.close()


def _is_locked(path: Path) -> bool:
    """Return whether another owner currently holds *path*."""
    handle = _open_lock(path)
    if handle is None:
        return True
    _close_lock(handle)
    return False


@dataclass(frozen=True, slots=True)
class LockOwner:
    """The run that holds a lock, as recorded beside the lock file."""

    profile: str
    run_id: str = ""
    pid: int = 0
    host: str = ""
    acquired_at: datetime | None = None
    global_slot: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for logs and errors."""
        payload: dict[str, object] = {
            "profile": self.profile,
            "run_id": self.run_id,
            "pid": self.pid,
            "host": self.host,
        }
        if self.acquired_at is not None:
            payload["acquired_at"] = self.acquired_at.isoformat()
        if self.global_slot is not None:
            payload["global_slot"] = self.global_slot
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LockOwner:
        """Build an owner from a metadata mapping, ignoring unknown keys."""
        acquired = value.get("acquired_at")
        acquired_at: datetime | None = None
        if isinstance(acquired, str) and acquired.strip():
            try:
                acquired_at = datetime.fromisoformat(acquired)
            except ValueError:
                acquired_at = None
        slot = value.get("global_slot")
        return cls(
            profile=str(value.get("profile", "")),
            run_id=str(value.get("run_id", "")),
            pid=int(value.get("pid", 0) or 0),
            host=str(value.get("host", "")),
            acquired_at=acquired_at,
            global_slot=int(slot) if isinstance(slot, int) else None,
        )


def _read_owner(path: Path) -> LockOwner | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    owner = LockOwner.from_dict(payload)
    return owner if owner.profile else None


def _write_owner(path: Path, owner: LockOwner) -> None:
    """Atomically replace *path* with *owner* as canonical JSON."""
    payload = {"version": _METADATA_VERSION, **owner.to_dict()}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
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


class RunLock:
    """One held profile lock plus its optional global slot.

    A lock is released explicitly with :meth:`release` or by leaving its
    context manager. Releasing twice is harmless. If the process exits without
    releasing, the operating system drops the file locks, so a crashed run
    cannot block later schedules.
    """

    def __init__(
        self,
        manager: RunLockManager,
        profile: str,
        owner: LockOwner,
        handles: tuple[BinaryIO, ...],
        *,
        lock_path: Path,
        metadata_path: Path,
        global_slot: int | None,
    ) -> None:
        self._manager = manager
        self._profile = profile
        self._owner = owner
        self._handles = handles
        self._lock_path = lock_path
        self._metadata_path = metadata_path
        self._global_slot = global_slot
        self._released = False

    @property
    def manager(self) -> RunLockManager:
        """The manager that created this lock."""
        return self._manager

    @property
    def profile(self) -> str:
        """The content profile this lock protects."""
        return self._profile

    @property
    def run_id(self) -> str:
        """The run id recorded in the owner metadata."""
        return self._owner.run_id

    @property
    def owner(self) -> LockOwner:
        """The metadata describing the lock holder."""
        return self._owner

    @property
    def acquired_at(self) -> datetime | None:
        """When the lock was acquired."""
        return self._owner.acquired_at

    @property
    def global_slot(self) -> int | None:
        """The global slot index, or ``None`` when no limit is configured."""
        return self._global_slot

    @property
    def path(self) -> Path:
        """The profile lock file."""
        return self._lock_path

    @property
    def metadata_path(self) -> Path:
        """The owner metadata file written beside the lock."""
        return self._metadata_path

    @property
    def released(self) -> bool:
        """Whether this lock has already been released."""
        return self._released

    def release(self) -> None:
        """Release the profile lock and its global slot."""
        if self._released:
            return
        self._released = True
        try:
            self._metadata_path.unlink(missing_ok=True)
        finally:
            for handle in self._handles:
                _close_lock(handle)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot for logs and errors."""
        payload = self._owner.to_dict()
        payload.update(
            {
                "path": str(self._lock_path),
                "released": self._released,
            }
        )
        return payload

    def __enter__(self) -> RunLock:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


class RunLockManager:
    """Acquire per-profile and global run locks without overlapping runs.

    ``max_parallel`` is the number of runs that may hold a lock at once across
    every profile and process. ``0`` disables the global limit; the default of
    ``1`` serialises scheduled runs. ``lock_dir`` overrides where the lock
    files live; otherwise they sit under the user data directory. ``per_profile``
    may be disabled for callers that only need the global limit.

    :meth:`acquire` raises :class:`ProfileLockError` or
    :class:`ParallelRunLimitError` when a lock is busy. Pass ``wait=True`` to
    poll instead, optionally with a ``timeout``. :meth:`try_acquire` returns
    ``None`` instead of raising.
    """

    def __init__(
        self,
        max_parallel: int = 1,
        *,
        limit: int | None = None,
        max_parallel_runs: int | None = None,
        lock_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
        per_profile: bool = True,
        poll_interval: float = 0.05,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if limit is not None and max_parallel_runs is not None:
            raise ConfigError("pass either limit or max_parallel_runs, not both")
        if lock_dir is not None and data_dir is not None:
            raise ConfigError("pass either lock_dir or data_dir, not both")
        if max_parallel_runs is not None:
            max_parallel = max_parallel_runs
        elif limit is not None:
            max_parallel = limit
        if not isinstance(per_profile, bool):
            raise TypeError("per_profile must be a boolean")
        if not isinstance(poll_interval, (int, float)) or isinstance(
            poll_interval, bool
        ):
            raise ConfigError("lock poll interval must be a number")
        if poll_interval <= 0:
            raise ConfigError("lock poll interval must be positive")
        if clock is not None and not callable(clock):
            raise ConfigError("lock clock must be callable")
        if sleep is not None and not callable(sleep):
            raise ConfigError("lock sleep must be callable")

        self._max_parallel = _coerce_limit(max_parallel)
        self._per_profile = per_profile
        self._lock_dir = (
            Path(lock_dir).expanduser()
            if lock_dir is not None
            else run_locks_dir(data_dir, environ=environ)
        )
        self._poll_interval = float(poll_interval)
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep

    @classmethod
    def from_config(
        cls,
        config: JobsConfig | Mapping[str, Any] | None = None,
        *,
        lock_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> RunLockManager:
        """Build a manager whose limit comes from a jobs config section."""
        if config is None:
            max_parallel = 1
        elif isinstance(config, JobsConfig):
            max_parallel = config.max_parallel_runs
        elif isinstance(config, Mapping):
            max_parallel = config.get(
                "max_parallel_runs",
                config.get("max_parallel", 1),
            )
        else:
            max_parallel = getattr(
                config,
                "max_parallel_runs",
                getattr(config, "max_parallel", 1),
            )
        return cls(
            max_parallel,
            lock_dir=lock_dir,
            data_dir=data_dir,
            environ=environ,
            **kwargs,
        )

    @property
    def max_parallel(self) -> int:
        """The global run limit, where ``0`` means unlimited."""
        return self._max_parallel

    @property
    def max_parallel_runs(self) -> int:
        """Alias for :attr:`max_parallel`."""
        return self._max_parallel

    @property
    def lock_dir(self) -> Path:
        """The directory holding every lock and metadata file."""
        return self._lock_dir

    @property
    def per_profile(self) -> bool:
        """Whether a per-profile lock is taken before a global slot."""
        return self._per_profile

    @property
    def profiles_dir(self) -> Path:
        """The directory holding per-profile lock files."""
        return self._lock_dir / "profiles"

    @property
    def global_dir(self) -> Path:
        """The directory holding the global slot lock files."""
        return self._lock_dir / "global"

    def profile_lock_path(self, profile: str) -> Path:
        """Return the lock file for *profile* without creating it."""
        return self.profiles_dir / f"{_profile_filename(_coerce_profile(profile))}.lock"

    def global_lock_path(self, slot: int) -> Path:
        """Return the lock file for one global slot."""
        return self.global_dir / f"slot-{_coerce_limit(slot, label='slot')}.lock"

    def acquire(
        self,
        profile: str,
        *,
        run_id: str = "",
        wait: bool = False,
        timeout: float | None = None,
        owner: LockOwner | Mapping[str, Any] | None = None,
    ) -> RunLock:
        """Take the profile lock and a global slot, or raise when busy.

        With ``wait=True`` the call polls until both are available. ``timeout``
        bounds that wait in seconds; ``None`` waits indefinitely.
        """
        name = _coerce_profile(profile)
        if not isinstance(wait, bool):
            raise TypeError("wait must be a boolean")
        deadline = _coerce_timeout(timeout)
        started = self._clock()
        while True:
            lock, busy = self._attempt(name, run_id=run_id, owner=owner)
            if lock is not None:
                return lock
            if not wait:
                if busy is None:
                    raise LockError("lock was not acquired")
                raise busy
            elapsed = self._clock() - started
            if deadline is not None and elapsed >= deadline:
                if busy is None:
                    raise LockError("lock was not acquired")
                busy.details["waited_seconds"] = elapsed
                raise busy
            delay = self._poll_interval
            if deadline is not None:
                delay = min(delay, max(deadline - elapsed, 0.0))
            self._sleep(delay)

    def acquire_nowait(
        self,
        profile: str,
        *,
        run_id: str = "",
        owner: LockOwner | Mapping[str, Any] | None = None,
    ) -> RunLock:
        """Alias for :meth:`acquire` with ``wait=False``."""
        return self.acquire(profile, run_id=run_id, wait=False, owner=owner)

    def try_acquire(
        self,
        profile: str,
        *,
        run_id: str = "",
        owner: LockOwner | Mapping[str, Any] | None = None,
    ) -> RunLock | None:
        """Try once and return ``None`` instead of raising when busy."""
        name = _coerce_profile(profile)
        lock, _ = self._attempt(name, run_id=run_id, owner=owner)
        return lock

    def is_locked(self, profile: str) -> bool:
        """Return whether another owner holds the lock for *profile*."""
        return _is_locked(self.profile_lock_path(profile))

    def owner(self, profile: str) -> LockOwner | None:
        """Return the current owner metadata for *profile*, if any."""
        path = self.profile_lock_path(profile)
        if not _is_locked(path):
            return None
        return _read_owner(path.with_suffix(".json"))

    def active(self) -> tuple[LockOwner, ...]:
        """Return metadata for every lock currently held."""
        directory = self.profiles_dir
        if not directory.is_dir():
            return ()
        owners: list[LockOwner] = []
        for metadata in sorted(directory.glob("*.json")):
            owner = _read_owner(metadata)
            if owner is None:
                continue
            if _is_locked(metadata.with_suffix(".lock")):
                owners.append(owner)
        return tuple(sorted(owners, key=lambda item: (item.profile, item.run_id)))

    def active_profiles(self) -> tuple[str, ...]:
        """Return the profiles whose locks are currently held."""
        return tuple(owner.profile for owner in self.active())

    @property
    def active_count(self) -> int:
        """The number of locks currently held."""
        return len(self.active())

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot for logs and status output."""
        return {
            "max_parallel": self._max_parallel,
            "per_profile": self._per_profile,
            "lock_dir": str(self._lock_dir),
            "active": [owner.to_dict() for owner in self.active()],
        }

    def _attempt(
        self,
        profile: str,
        *,
        run_id: str,
        owner: LockOwner | Mapping[str, Any] | None,
    ) -> tuple[RunLock | None, LockBusyError | None]:
        profile_path = self.profile_lock_path(profile)
        profile_handle = _open_lock(profile_path) if self._per_profile else None
        if self._per_profile and profile_handle is None:
            existing = _read_owner(profile_path.with_suffix(".json"))
            return None, ProfileLockError(
                f"profile {profile!r} is already running",
                details={
                    "profile": profile,
                    "owner": existing.to_dict() if existing is not None else None,
                },
            )

        slot: int | None = None
        global_handle: BinaryIO | None = None
        if self._max_parallel:
            for index in range(self._max_parallel):
                candidate = _open_lock(self.global_lock_path(index))
                if candidate is not None:
                    slot = index
                    global_handle = candidate
                    break
            if global_handle is None:
                if profile_handle is not None:
                    _close_lock(profile_handle)
                return None, ParallelRunLimitError(
                    f"global parallel run limit of {self._max_parallel} reached",
                    details={
                        "profile": profile,
                        "max_parallel": self._max_parallel,
                        "active": self.active_count,
                    },
                )

        record = self._owner(profile, run_id=run_id, owner=owner, slot=slot)
        metadata_path = profile_path.with_suffix(".json")
        handles: list[BinaryIO] = []
        if global_handle is not None:
            handles.append(global_handle)
        if profile_handle is not None:
            handles.append(profile_handle)
        try:
            if self._per_profile:
                _write_owner(metadata_path, record)
        except BaseException:
            for handle in handles:
                _close_lock(handle)
            raise
        return (
            RunLock(
                self,
                profile,
                record,
                tuple(handles),
                lock_path=profile_path,
                metadata_path=metadata_path,
                global_slot=slot,
            ),
            None,
        )

    def _owner(
        self,
        profile: str,
        *,
        run_id: str,
        owner: LockOwner | Mapping[str, Any] | None,
        slot: int | None,
    ) -> LockOwner:
        if isinstance(owner, Mapping):
            record = LockOwner.from_dict(owner)
        elif isinstance(owner, LockOwner):
            record = owner
        else:
            record = LockOwner(
                profile=profile,
                run_id=run_id,
                pid=os.getpid(),
                host=socket.gethostname(),
                acquired_at=utcnow(),
                global_slot=slot,
            )
        return LockOwner(
            profile=profile,
            run_id=record.run_id or run_id,
            pid=record.pid or os.getpid(),
            host=record.host or socket.gethostname(),
            acquired_at=record.acquired_at or utcnow(),
            global_slot=slot if slot is not None else record.global_slot,
        )


RunLockStore = RunLockManager
"""Alias for :class:`RunLockManager`."""

RunLocks = RunLockManager
"""Alias for :class:`RunLockManager`."""
