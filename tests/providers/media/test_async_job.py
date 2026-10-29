"""The async media helper submits, polls, downloads, times out and cancels."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import Engine

from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import InvalidRequestError, ValidationError
from aivia.core.http import HttpClient
from aivia.core.idempotency import IdempotencyConflict
from aivia.providers.media import (
    AsyncJob,
    AsyncJobState,
    AsyncJobTimeoutError,
    AsyncMediaJobHelper,
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
