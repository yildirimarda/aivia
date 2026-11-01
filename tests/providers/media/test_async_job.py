"""The async media helper submits, polls, downloads, times out and cancels."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import Engine
from sqlalchemy.exc import IntegrityError

from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import InvalidRequestError, ValidationError
from aivia.core.http import HttpClient
from aivia.core.idempotency import IdempotencyConflict
from aivia.providers.media import (
    AsyncJob,
    AsyncJobState,
    AsyncJobTimeoutError,
    AsyncMediaJobHelper,
    DatabaseAsyncJobStore,
)


class FakeClock:
    """A monotonic clock that advances only when the helper sleeps."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    async def sleep(self, seconds: float) -> None:
        self.value += seconds


class FakeBackend:
    """A provider queue double returning scripted poll responses."""

    name = "fake-media"

    def __init__(
        self,
        statuses: list[Any] | None = None,
        *,
        submit_response: Any = None,
    ) -> None:
        self.statuses = list(statuses or [])
        self.submit_response = (
            {"job_id": "remote-1"} if submit_response is None else submit_response
        )
        self.submitted: list[Any] = []
        self.polls: list[str] = []
        self.cancels: list[str] = []

    async def submit(self, request: Any) -> Any:
        self.submitted.append(request)
        return self.submit_response

    async def poll(self, remote_id: str) -> Any:
        self.polls.append(remote_id)
        if self.statuses:
            return self.statuses.pop(0)
        return {"status": "running"}

    async def cancel(self, remote_id: str) -> None:
        self.cancels.append(remote_id)


class ForgetfulStore(DatabaseAsyncJobStore):
    """A store whose first key lookup misses, like a racing process."""

    def __init__(self, engine: Engine, *, kind: str = "media_async") -> None:
        super().__init__(engine, kind=kind)
        self._miss_once = True

    def find_by_idempotency_key(self, key: str) -> AsyncJob | None:
        if self._miss_once:
            self._miss_once = False
            return None
        return super().find_by_idempotency_key(key)


class WaitingStore(ForgetfulStore):
    """A racing store that signals when the helper starts waiting."""

    def __init__(self, engine: Engine, *, kind: str = "media_async") -> None:
        super().__init__(engine, kind=kind)
        self.first_get = asyncio.Event()

    def get(self, job_id: str) -> AsyncJob | None:
        self.first_get.set()
        return super().get(job_id)


class YieldingClock(FakeClock):
    """A fake clock whose sleep still yields once to the event loop."""

    async def sleep(self, seconds: float) -> None:
        self.value += seconds
        await asyncio.sleep(0)


class StaleReadStore(DatabaseAsyncJobStore):
    """A store that keeps returning the crashed row both processes read.

    A second process can read the pending row before the first one claims it.
    Returning the same snapshot reproduces that interleaving deterministically
    and records every recovery claim attempt.
    """

    def __init__(
        self,
        engine: Engine,
        snapshot: AsyncJob,
        *,
        kind: str = "media_async",
    ) -> None:
        super().__init__(engine, kind=kind)
        self.snapshot = snapshot
        self.claim_attempts = 0
        self.claim_attempted = asyncio.Event()

    def find_by_idempotency_key(self, key: str) -> AsyncJob | None:
        if key and key == self.snapshot.idempotency_key:
            return self.snapshot
        return super().find_by_idempotency_key(key)

    def claim_recovery(
        self, job: AsyncJob, *, owner: str, started: datetime
    ) -> AsyncJob | None:
        self.claim_attempts += 1
        try:
            return super().claim_recovery(job, owner=owner, started=started)
        finally:
            self.claim_attempted.set()


class BlockingBackend(FakeBackend):
    """A backend whose submit stays in flight until the test releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def submit(self, request: Any) -> Any:
        self.started.set()
        await self.release.wait()
        return await super().submit(request)


class CrashingBackend(FakeBackend):
    """A backend whose first submit hangs until its task is cancelled."""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.calls = 0

    async def submit(self, request: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            await asyncio.Event().wait()
        return await super().submit(request)


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


def make_helper(
    backend: FakeBackend,
    *,
    engine: Engine | None = None,
    clock: FakeClock | None = None,
    http: HttpClient | None = None,
    **kwargs: Any,
) -> AsyncMediaJobHelper:
    active = clock or FakeClock()
    options: dict[str, Any] = {
        "timeout_seconds": 5.0,
        "poll_interval_seconds": 1.0,
        **kwargs,
    }
    return AsyncMediaJobHelper(
        backend,
        engine=engine,
        provider="fake-media",
        clock=active,
        sleep=active.sleep,
        http=http,
        **options,
    )


def test_submit_persists_the_remote_id_and_reuses_an_idempotency_key(
    engine: Engine,
) -> None:
    backend = FakeBackend()
    helper = make_helper(backend, engine=engine)

    async def scenario() -> tuple[AsyncJob, AsyncJob]:
        first = await helper.submit({"prompt": "a wave"}, idempotency_key="request-1")
        second = await helper.submit({"prompt": "a wave"}, idempotency_key="request-1")
        return first, second

    first, second = asyncio.run(scenario())

    assert first.id.startswith("job_")
    assert first.remote_id == "remote-1"
    assert first.provider == "fake-media"
    assert first.state is AsyncJobState.RUNNING
    assert first.idempotency_key == "request-1"
    assert second.id == first.id
    assert second.remote_id == first.remote_id
    assert backend.submitted == [{"prompt": "a wave"}]
    assert helper.get(first.id) is not None

    with pytest.raises(IdempotencyConflict):
        asyncio.run(
            helper.submit({"prompt": "a different wave"}, idempotency_key="request-1")
        )
    assert len(backend.submitted) == 1


def test_a_repeated_key_from_another_process_cannot_submit_twice(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    winner_engine = create_database_engine(database)
    racer_engine = create_database_engine(database)
    try:
        winner_backend = FakeBackend()
        winner = AsyncMediaJobHelper(
            winner_backend,
            engine=winner_engine,
            provider="fake-media",
            timeout_seconds=5.0,
            poll_interval_seconds=1.0,
            clock=FakeClock(),
        )
        racer_backend = FakeBackend()
        racer = AsyncMediaJobHelper(
            racer_backend,
            store=ForgetfulStore(racer_engine),
            provider="fake-media",
            timeout_seconds=5.0,
            poll_interval_seconds=1.0,
            clock=FakeClock(),
        )

        first = asyncio.run(
            winner.submit({"prompt": "a wave"}, idempotency_key="shared-key")
        )
        second = asyncio.run(
            racer.submit({"prompt": "a wave"}, idempotency_key="shared-key")
        )

        assert second.id == first.id
        assert second.remote_id == first.remote_id == "remote-1"
        assert winner_backend.submitted == [{"prompt": "a wave"}]
        assert racer_backend.submitted == []

        with pytest.raises(IdempotencyConflict):
            asyncio.run(
                racer.submit(
                    {"prompt": "a different wave"},
                    idempotency_key="shared-key",
                )
            )
        assert racer_backend.submitted == []
    finally:
        winner_engine.dispose()
        racer_engine.dispose()


def test_a_repeated_key_waits_for_an_in_flight_submission(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    winner_engine = create_database_engine(database)
    racer_engine = create_database_engine(database)
    try:
        winner_backend = BlockingBackend()
        winner = AsyncMediaJobHelper(
            winner_backend,
            engine=winner_engine,
            provider="fake-media",
            timeout_seconds=5.0,
            poll_interval_seconds=1.0,
            clock=FakeClock(),
        )
        racer_backend = FakeBackend()
        racer_store = WaitingStore(racer_engine)
        racer = make_helper(
            racer_backend,
            store=racer_store,
            clock=YieldingClock(),
            timeout_seconds=30.0,
            poll_interval_seconds=1.0,
        )

        async def scenario() -> tuple[AsyncJob, AsyncJob, bool]:
            winner_task = asyncio.create_task(
                winner.submit({"prompt": "a wave"}, idempotency_key="shared-key")
            )
            await winner_backend.started.wait()
            racer_task = asyncio.create_task(
                racer.submit({"prompt": "a wave"}, idempotency_key="shared-key")
            )
            await asyncio.wait_for(racer_store.first_get.wait(), timeout=1.0)
            waited = not racer_task.done()
            winner_backend.release.set()
            first = await winner_task
            second = await racer_task
            return first, second, waited

        first, second, waited = asyncio.run(scenario())

        assert waited
        assert second.id == first.id
        assert second.remote_id == first.remote_id == "remote-1"
        assert winner_backend.submitted == [{"prompt": "a wave"}]
        assert racer_backend.submitted == []
    finally:
        winner_engine.dispose()
        racer_engine.dispose()


def test_a_repeated_key_resumes_a_job_left_pending_by_a_crash(
    engine: Engine,
) -> None:
    backend = CrashingBackend()
    helper = make_helper(backend, engine=engine)

    async def scenario() -> tuple[AsyncJob, AsyncJob]:
        crashed = asyncio.create_task(
            helper.submit({"prompt": "a wave"}, idempotency_key="crash-1")
        )
        await backend.started.wait()
        crashed.cancel()
        with pytest.raises(asyncio.CancelledError):
            await crashed
        pending = helper.store.find_by_idempotency_key("crash-1")
        assert pending is not None
        assert pending.remote_id == ""
        assert pending.state is AsyncJobState.PENDING
        recovered = await helper.submit({"prompt": "a wave"}, idempotency_key="crash-1")
        return pending, recovered

    pending, recovered = asyncio.run(scenario())

    assert recovered.id == pending.id
    assert recovered.remote_id == "remote-1"
    assert recovered.state is AsyncJobState.RUNNING
    assert backend.calls == 2
    assert backend.submitted == [{"prompt": "a wave"}]
    stored = helper.get(pending.id)
    assert stored is not None
    assert stored.remote_id == "remote-1"
    assert stored.state is AsyncJobState.RUNNING


def test_claim_recovery_lets_only_one_process_take_a_crashed_job(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    setup_engine = create_database_engine(database)
    winner_engine = create_database_engine(database)
    loser_engine = create_database_engine(database)
    try:
        crashed = DatabaseAsyncJobStore(setup_engine).save(
            AsyncJob(
                id="job_crashed",
                provider="fake-media",
                idempotency_key="crash-claim",
                submission_owner="1:dead",
                submission_started_at=datetime(2020, 1, 1, tzinfo=UTC),
            )
        )
        winner_store = DatabaseAsyncJobStore(winner_engine)
        loser_store = DatabaseAsyncJobStore(loser_engine)
        started = datetime.now(UTC)

        winner = winner_store.claim_recovery(crashed, owner="2:winner", started=started)
        loser = loser_store.claim_recovery(crashed, owner="3:loser", started=started)

        assert winner is not None
        assert winner.submission_owner == "2:winner"
        assert loser is None
        stored = winner_store.get(crashed.id)
        assert stored is not None
        assert stored.submission_owner == "2:winner"
        assert stored.state is AsyncJobState.PENDING
    finally:
        setup_engine.dispose()
        winner_engine.dispose()
        loser_engine.dispose()


def test_two_processes_cannot_resubmit_the_same_crashed_job(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    setup_engine = create_database_engine(database)
    try:
        crashing = CrashingBackend()
        seed = make_helper(crashing, engine=setup_engine)

        async def crash() -> AsyncJob:
            task = asyncio.create_task(
                seed.submit({"prompt": "a wave"}, idempotency_key="crash-race")
            )
            await crashing.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            pending = seed.store.find_by_idempotency_key("crash-race")
            assert pending is not None
            return pending

        crashed = asyncio.run(crash())
    finally:
        setup_engine.dispose()

    winner_engine = create_database_engine(database)
    loser_engine = create_database_engine(database)
    try:
        winner_backend = BlockingBackend()
        winner_store = StaleReadStore(winner_engine, crashed)
        winner = make_helper(
            winner_backend,
            store=winner_store,
            clock=YieldingClock(),
            timeout_seconds=30.0,
            poll_interval_seconds=1.0,
        )
        loser_backend = FakeBackend()
        loser_store = StaleReadStore(loser_engine, crashed)
        loser = make_helper(
            loser_backend,
            store=loser_store,
            clock=YieldingClock(),
            timeout_seconds=30.0,
            poll_interval_seconds=1.0,
        )

        async def scenario() -> tuple[AsyncJob, AsyncJob]:
            winner_task = asyncio.create_task(
                winner.submit({"prompt": "a wave"}, idempotency_key="crash-race")
            )
            await winner_backend.started.wait()
            loser_task = asyncio.create_task(
                loser.submit({"prompt": "a wave"}, idempotency_key="crash-race")
            )
            await asyncio.wait_for(loser_store.claim_attempted.wait(), timeout=1.0)
            assert not loser_task.done()
            assert loser_backend.submitted == []
            winner_backend.release.set()
            return await winner_task, await loser_task

        first, second = asyncio.run(scenario())

        assert first.id == crashed.id == second.id
        assert first.remote_id == second.remote_id == "remote-1"
        assert winner_backend.submitted == [{"prompt": "a wave"}]
        assert loser_backend.submitted == []
        assert winner_store.claim_attempts == 1
        assert loser_store.claim_attempts == 1
        stored = winner_store.get(crashed.id)
        assert stored is not None
        assert stored.remote_id == "remote-1"
        assert stored.submission_owner == ""
    finally:
        winner_engine.dispose()
        loser_engine.dispose()


def test_the_jobs_table_rejects_a_duplicate_async_idempotency_key(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "aivia.db"
    upgrade_database(database)
    engine = create_database_engine(database)
    try:
        store = DatabaseAsyncJobStore(engine)
        store.save(
            AsyncJob(id="job_first", provider="fake-media", idempotency_key="shared")
        )

        with pytest.raises(IntegrityError):
            store.save(
                AsyncJob(
                    id="job_second",
                    provider="fake-media",
                    idempotency_key="shared",
                )
            )

        store.save(AsyncJob(id="job_empty_a", provider="fake-media"))
        store.save(AsyncJob(id="job_empty_b", provider="fake-media"))

        other_kind = DatabaseAsyncJobStore(engine, kind="other_kind")
        other_kind.save(
            AsyncJob(id="job_other", provider="fake-media", idempotency_key="shared")
        )
    finally:
        engine.dispose()


def test_a_restarted_helper_resumes_the_persisted_job_without_resubmitting(
    engine: Engine,
) -> None:
    backend = FakeBackend()
    first_helper = make_helper(backend, engine=engine)
    job = asyncio.run(
        first_helper.submit({"prompt": "a wave"}, idempotency_key="restart-1")
    )

    restarted = make_helper(backend, engine=engine)
    same = asyncio.run(
        restarted.submit({"prompt": "a wave"}, idempotency_key="restart-1")
    )
    assert same.id == job.id
    assert same.remote_id == "remote-1"
    assert len(backend.submitted) == 1

    backend.statuses = [
        {"status": "running", "progress": 50},
        {
            "status": "succeeded",
            "output": {"url": "https://cdn.example.com/result.png"},
        },
    ]
    done = asyncio.run(restarted.poll(job.id))

    assert done.state is AsyncJobState.SUCCEEDED
    assert done.result_urls == ["https://cdn.example.com/result.png"]
    assert done.progress == 0.5
    assert backend.polls == ["remote-1", "remote-1"]


def test_poll_records_progress_and_terminal_success(engine: Engine) -> None:
    backend = FakeBackend(
        [
            {"status": "running", "progress": 0.25},
            {"status": "running", "progress": 0.75},
            {
                "status": "succeeded",
                "result_urls": ["https://cdn.example.com/one.png"],
            },
        ]
    )
    helper = make_helper(backend, engine=engine)
    job = asyncio.run(helper.submit({"prompt": "a wave"}))

    done = asyncio.run(helper.poll(job))

    assert done.state is AsyncJobState.SUCCEEDED
    assert done.progress == 0.75
    assert done.result_urls == ["https://cdn.example.com/one.png"]
    assert backend.polls == ["remote-1", "remote-1", "remote-1"]
    stored = helper.get(job.id)
    assert stored is not None
    assert stored.state is AsyncJobState.SUCCEEDED
    assert stored.result_urls == done.result_urls


def test_poll_records_a_provider_failure(engine: Engine) -> None:
    backend = FakeBackend(
        [
            {
                "status": "failed",
                "error": {"code": "content_policy", "message": "blocked"},
            }
        ]
    )
    helper = make_helper(backend, engine=engine)
    job = asyncio.run(helper.submit({"prompt": "a wave"}))

    done = asyncio.run(helper.poll(job))

    assert done.state is AsyncJobState.FAILED
    assert done.error_code == "content_policy"
    assert done.error_message == "blocked"


def test_poll_times_out_and_persists_the_timed_out_state(engine: Engine) -> None:
    clock = FakeClock()
    backend = FakeBackend()
    helper = make_helper(
        backend,
        engine=engine,
        clock=clock,
        timeout_seconds=2.5,
        poll_interval_seconds=1.0,
    )
    job = asyncio.run(helper.submit({"prompt": "a slow wave"}))

    with pytest.raises(AsyncJobTimeoutError) as raised:
        asyncio.run(helper.poll(job))

    assert raised.value.job is not None
    assert raised.value.job.state is AsyncJobState.TIMED_OUT
    stored = helper.get(job.id)
    assert stored is not None
    assert stored.state is AsyncJobState.TIMED_OUT
    assert stored.error_code == "timeout"
    assert backend.cancels == []

    cancelled = asyncio.run(helper.cancel(stored))
    assert cancelled.state is AsyncJobState.CANCELLED
    assert backend.cancels == ["remote-1"]


def test_cancel_calls_the_provider_once_and_is_idempotent(engine: Engine) -> None:
    backend = FakeBackend()
    helper = make_helper(backend, engine=engine)
    job = asyncio.run(helper.submit({"prompt": "a wave"}))

    cancelled = asyncio.run(helper.cancel(job))
    again = asyncio.run(helper.cancel(job))

    assert cancelled.state is AsyncJobState.CANCELLED
    assert again.state is AsyncJobState.CANCELLED
    assert backend.cancels == ["remote-1"]
    stored = helper.get(job.id)
    assert stored is not None
    assert stored.state is AsyncJobState.CANCELLED


def test_download_fetches_a_result_and_enforces_the_size_cap(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://cdn.example.com/result.png"
        return httpx.Response(
            200,
            content=b"image-bytes",
            headers={"content-length": str(len(b"image-bytes"))},
        )

    client = HttpClient(transport=httpx.MockTransport(handler))
    helper = make_helper(FakeBackend(), http=client)
    job = AsyncJob(
        id="job_download",
        state=AsyncJobState.SUCCEEDED,
        result_urls=["https://cdn.example.com/result.png"],
    )

    data = asyncio.run(helper.download(job))
    assert data == b"image-bytes"

    destination = tmp_path / "nested" / "result.png"
    asyncio.run(helper.download(job, destination=destination))
    assert destination.read_bytes() == b"image-bytes"

    with pytest.raises(ValidationError):
        asyncio.run(helper.download(job, max_bytes=4))


def test_download_rejects_a_job_without_a_result(engine: Engine) -> None:
    backend = FakeBackend()
    helper = make_helper(backend, engine=engine)
    job = asyncio.run(helper.submit({"prompt": "a wave"}))

    with pytest.raises(InvalidRequestError):
        asyncio.run(helper.download(job))
