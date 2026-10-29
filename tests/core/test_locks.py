"""Per-profile and global run locks that keep scheduled runs from overlapping."""

from __future__ import annotations

import json
import multiprocessing
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from aivia.core.config import JobsConfig
from aivia.core.errors import ConfigError
from aivia.core.locks import (
    DEFAULT_LOCK_DIRNAME,
    GlobalRunLimitError,
    LockBusyError,
    LockError,
    LockOwner,
    ParallelRunLimitError,
    ProfileLockError,
    RunLimitError,
    RunLock,
    RunLockManager,
    RunLocks,
    RunLockStore,
    run_locks_dir,
)


class FakeClock:
    """A manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeSleep:
    """Advance a fake clock instead of sleeping."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)
        self.clock.advance(seconds)


def _hold_lock_in_child(
    lock_dir: str,
    profile: str,
    acquired: Any,
    release: Any,
) -> None:
    manager = RunLockManager(max_parallel=0, lock_dir=lock_dir)
    lock = manager.acquire(profile, run_id="child-run")
    acquired.set()
    release.wait(10)
    lock.release()


def _acquire_and_exit(lock_dir: str, profile: str, acquired: Any) -> None:
    manager = RunLockManager(max_parallel=0, lock_dir=lock_dir)
    manager.acquire(profile, run_id="crashed-run")
    acquired.put("acquired")


def _context() -> Any:
    methods = multiprocessing.get_all_start_methods()
    return multiprocessing.get_context("fork" if "fork" in methods else "spawn")


def test_acquire_records_the_owner_and_release_frees_both_locks(
    tmp_path: Path,
) -> None:
    manager = RunLockManager(lock_dir=tmp_path / "locks")

    lock = manager.acquire("alpha", run_id="run_1")

    assert isinstance(lock, RunLock)
    assert lock.profile == "alpha"
    assert lock.run_id == "run_1"
    assert lock.global_slot == 0
    assert lock.owner.pid > 0
    assert lock.owner.host
    assert lock.owner.acquired_at is not None
    assert lock.path == manager.profile_lock_path("alpha")
    assert lock.path.is_file()
    assert lock.metadata_path.is_file()
    assert manager.is_locked("alpha") is True
    assert manager.owner("alpha") == lock.owner
    assert manager.active_profiles() == ("alpha",)
    assert manager.active_count == 1
    assert manager.to_dict()["active"] == [lock.owner.to_dict()]

    lock.release()

    assert lock.released is True
    assert manager.is_locked("alpha") is False
    assert manager.owner("alpha") is None
    assert manager.active() == ()
    assert lock.metadata_path.exists() is False
    lock.release()


def test_the_same_profile_cannot_run_twice(tmp_path: Path) -> None:
    first = RunLockManager(lock_dir=tmp_path / "locks")
    second = RunLockManager(lock_dir=tmp_path / "locks")
    held = first.acquire("alpha", run_id="run_1")

    with pytest.raises(ProfileLockError) as caught:
        second.acquire("alpha", run_id="run_2")

    error = caught.value
    assert error.code == "profile_locked"
    assert error.retryable is True
    assert error.details["profile"] == "alpha"
    assert error.details["owner"] == held.owner.to_dict()
    assert second.try_acquire("alpha") is None
    assert second.is_locked("alpha") is True

    held.release()
    assert second.try_acquire("alpha") is not None


def test_the_global_limit_applies_across_profiles(tmp_path: Path) -> None:
    manager = RunLockManager(max_parallel=2, lock_dir=tmp_path / "locks")
    first = manager.acquire("alpha")
    second = manager.acquire("beta")

    with pytest.raises(ParallelRunLimitError) as caught:
        manager.acquire("gamma")

    error = caught.value
    assert error.code == "parallel_run_limit"
    assert error.details["max_parallel"] == 2
    assert error.details["active"] == 2
    assert {first.global_slot, second.global_slot} == {0, 1}
    assert manager.active_profiles() == ("alpha", "beta")

    first.release()
    third = manager.acquire("gamma")

    assert third.global_slot == first.global_slot


def test_zero_disables_the_global_limit_but_keeps_profile_locks(
    tmp_path: Path,
) -> None:
    manager = RunLockManager(max_parallel=0, lock_dir=tmp_path / "locks")

    locks = [manager.acquire(name) for name in ("alpha", "beta", "gamma")]

    assert all(lock.global_slot is None for lock in locks)
    assert manager.max_parallel == 0
    with pytest.raises(ProfileLockError):
        manager.acquire("alpha")
    for lock in locks:
        lock.release()


def test_context_manager_releases_and_try_acquire_never_raises(
    tmp_path: Path,
) -> None:
    manager = RunLockManager(max_parallel=2, lock_dir=tmp_path / "locks")

    with manager.acquire("alpha") as lock:
        assert manager.try_acquire("alpha") is None
        other = manager.acquire_nowait("beta")
        assert other.profile == "beta"
        other.release()

    assert lock.released is True
    assert manager.is_locked("alpha") is False


def test_wait_polls_until_the_holder_releases(tmp_path: Path) -> None:
    manager = RunLockManager(lock_dir=tmp_path / "locks")
    held = manager.acquire("alpha", run_id="first")
    started = threading.Event()
    result: list[RunLock] = []

    def wait_for_lock() -> None:
        started.set()
        result.append(manager.acquire("alpha", run_id="second", wait=True))

    worker = threading.Thread(target=wait_for_lock)
    worker.start()
    assert started.wait(5)
    held.release()
    worker.join(5)

    assert worker.is_alive() is False
    assert len(result) == 1
    assert result[0].run_id == "second"
    result[0].release()


def test_wait_timeout_raises_the_original_busy_error(tmp_path: Path) -> None:
    clock = FakeClock()
    sleep = FakeSleep(clock)
    manager = RunLockManager(
        max_parallel=1,
        lock_dir=tmp_path / "locks",
        clock=clock,
        sleep=sleep,
    )
    held = manager.acquire("alpha")

    with pytest.raises(ProfileLockError) as caught:
        manager.acquire("alpha", wait=True, timeout=1)

    assert caught.value.details["waited_seconds"] >= 1
    assert sleep.delays
    held.release()


def test_profile_names_cannot_escape_the_lock_directory(tmp_path: Path) -> None:
    manager = RunLockManager(max_parallel=0, lock_dir=tmp_path / "locks")
    traversal = manager.profile_lock_path("../outside")
    colliding = manager.profile_lock_path(".._outside")

    assert traversal.parent == manager.profiles_dir
    assert colliding.parent == manager.profiles_dir
    assert traversal != colliding

    first = manager.acquire("../outside")
    second = manager.acquire(".._outside")

    assert first.path != second.path
    assert manager.is_locked("../outside")
    assert manager.is_locked(".._outside")
    first.release()
    second.release()


def test_a_lock_is_visible_to_another_process(tmp_path: Path) -> None:
    lock_dir = str(tmp_path / "locks")
    ctx = _context()
    acquired = ctx.Event()
    release = ctx.Event()
    process = ctx.Process(
        target=_hold_lock_in_child,
        args=(lock_dir, "alpha", acquired, release),
    )
    process.start()
    try:
        assert acquired.wait(10)
        manager = RunLockManager(lock_dir=lock_dir)
        assert manager.is_locked("alpha") is True
        assert manager.owner("alpha") is not None
        assert manager.owner("alpha").run_id == "child-run"  # type: ignore[union-attr]
        with pytest.raises(ProfileLockError):
            manager.acquire("alpha")
    finally:
        release.set()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)

    assert process.exitcode == 0
    manager = RunLockManager(lock_dir=lock_dir)
    assert manager.is_locked("alpha") is False


def test_a_crashed_process_does_not_leave_the_profile_locked(tmp_path: Path) -> None:
    lock_dir = str(tmp_path / "locks")
    ctx = _context()
    acquired = ctx.Queue()
    process = ctx.Process(
        target=_acquire_and_exit,
        args=(lock_dir, "alpha", acquired),
    )
    process.start()
    try:
        assert acquired.get(timeout=10) == "acquired"
    finally:
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)

    assert process.exitcode == 0
    manager = RunLockManager(lock_dir=lock_dir)
    lock = manager.acquire("alpha", run_id="after-crash")
    assert lock.run_id == "after-crash"
    lock.release()


def test_stale_metadata_without_a_lock_is_not_reported_active(
    tmp_path: Path,
) -> None:
    manager = RunLockManager(lock_dir=tmp_path / "locks")
    metadata = manager.profile_lock_path("alpha").with_suffix(".json")
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps(
            {
                "profile": "alpha",
                "run_id": "gone",
                "pid": 1,
                "host": "host",
                "acquired_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    assert manager.active() == ()
    assert manager.is_locked("alpha") is False


def test_from_config_reads_the_jobs_run_limit(tmp_path: Path) -> None:
    manager = RunLockManager.from_config(
        JobsConfig(max_parallel_runs=3),
        lock_dir=tmp_path / "locks",
    )

    assert manager.max_parallel == 3
    assert manager.max_parallel_runs == 3
    assert RunLockManager.from_config({"max_parallel_runs": 2}).max_parallel == 2
    assert RunLockManager.from_config(None).max_parallel == 1


def test_aliases_and_owner_serialization() -> None:
    assert RunLockStore is RunLockManager
    assert RunLocks is RunLockManager
    assert GlobalRunLimitError is ParallelRunLimitError
    assert RunLimitError is ParallelRunLimitError
    assert issubclass(ProfileLockError, LockBusyError)
    assert issubclass(LockBusyError, LockError)

    owner = LockOwner(
        profile="alpha",
        run_id="run_1",
        pid=42,
        host="host",
        acquired_at=datetime(2026, 1, 2, 3, 4, tzinfo=UTC),
        global_slot=1,
    )
    payload = owner.to_dict()

    assert payload["acquired_at"] == "2026-01-02T03:04:00+00:00"
    assert LockOwner.from_dict(payload) == owner


def test_the_default_lock_directory_lives_under_the_data_directory(
    tmp_path: Path,
) -> None:
    assert run_locks_dir(tmp_path) == tmp_path / DEFAULT_LOCK_DIRNAME
    assert run_locks_dir(environ={"XDG_CONFIG_HOME": str(tmp_path)}) == (
        tmp_path / "aivia" / "data" / DEFAULT_LOCK_DIRNAME
    )


def test_invalid_lock_inputs_fail_clearly(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        RunLockManager(max_parallel=-1)
    with pytest.raises(ConfigError):
        RunLockManager(max_parallel=cast(int, True))
    with pytest.raises(ConfigError):
        RunLockManager(limit=1, max_parallel_runs=1)
    with pytest.raises(ConfigError):
        RunLockManager(lock_dir=tmp_path, data_dir=tmp_path)
    with pytest.raises(ConfigError):
        RunLockManager(poll_interval=0)
    with pytest.raises(ConfigError):
        RunLockManager().acquire("")
    with pytest.raises(ConfigError):
        RunLockManager().acquire(cast(str, None))
    with pytest.raises(ConfigError):
        RunLockManager().acquire("alpha", timeout=-1)
    with pytest.raises(TypeError):
        RunLockManager().acquire("alpha", wait=cast(bool, "yes"))
    with pytest.raises(TypeError):
        RunLockManager(per_profile=cast(bool, "yes"))
