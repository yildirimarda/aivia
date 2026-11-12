"""Inject failures into the paid path and assert no double charge.

Covers provider outages and fallback, malformed responses, partial downloads,
expired publish tokens and a full job queue. A failed call never leaves a
partial or duplicated cost event behind.
"""

from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import Callable, Coroutine, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import Engine

from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import (
    ProviderExhaustedError,
    QuotaError,
    ValidationError,
)
from aivia.core.errors import TimeoutError as AiviaTimeoutError
from aivia.core.jobs import JobQueue
from aivia.core.media_probe import MediaDownloader
from aivia.core.models import ProviderMeta, ProviderPrice, RateLimit
from aivia.core.router import Router
from aivia.core.secrets import SecretsLoader
from aivia.core.token_store import TokenError, TokenStore
from aivia.media.image_service import IMAGE_OPERATION, IMAGE_UNIT, ImageService
from aivia.media.screening import screen_prompt
from aivia.providers import Provider
from aivia.providers.mock import MockImageProvider
from aivia.providers.protocols import (
    CAPABILITY_IMAGE_GEN,
    ImageFormat,
    ImageRequest,
    ImageResponse,
    ImageResult,
)
from aivia.providers.storage_local import LocalStorageProvider

PROMPT = "A lighthouse at sunset"
PRICE = 0.02

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)
# A download cut short: valid PNG magic prefix, incomplete header.
TRUNCATED_PNG = PNG_1X1[:20]


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coro)


def _token(prompt: str = PROMPT) -> str:
    return str(screen_prompt(prompt))


class _Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def tracker(engine: Engine) -> CostTracker:
    return CostTracker(engine)


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorageProvider:
    return LocalStorageProvider(tmp_path / "media")


def _loader(*, key: str = "store-key-1") -> SecretsLoader:
    return SecretsLoader(environ={"AIVA_TOKEN_STORE_KEY": key})


def _store(*, clock: Callable[[], float] | None = None) -> TokenStore:
    return TokenStore(_loader(), clock=clock or time.time)


def _router(*providers: Provider) -> Router:
    return Router(
        {provider.name: provider for provider in providers},
        config=CapabilityChainConfig(chain=[provider.name for provider in providers]),
    )


def _service(
    router: Router,
    tracker: CostTracker,
    storage: LocalStorageProvider,
    **values: Any,
) -> ImageService:
    return ImageService(router, tracker=tracker, storage=storage, **values)


class BadImageProvider(Provider):
    """A provider whose bytes are not an image at all."""

    def __init__(self) -> None:
        super().__init__(
            ProviderMeta(
                name="bad",
                capabilities=[CAPABILITY_IMAGE_GEN],
                pricing={CAPABILITY_IMAGE_GEN: ProviderPrice(amount_usd=PRICE)},
                rate_limit=RateLimit(requests_per_minute=0),
            )
        )
        self.calls = 0

    async def generate(self, request: ImageRequest) -> ImageResponse:
        self.calls += 1
        return ImageResponse(
            images=[ImageResult(data=b"not an image", format=ImageFormat.PNG)],
            model="bad-image",
            provider=self.name,
        )


class UrlImageProvider(Provider):
    """A provider that returns a remote URL instead of inline bytes."""

    def __init__(self, url: str) -> None:
        super().__init__(
            ProviderMeta(
                name="url",
                capabilities=[CAPABILITY_IMAGE_GEN],
                pricing={CAPABILITY_IMAGE_GEN: ProviderPrice(amount_usd=PRICE)},
                rate_limit=RateLimit(requests_per_minute=0),
            )
        )
        self.url = url
        self.calls = 0

    async def generate(self, request: ImageRequest) -> ImageResponse:
        self.calls += 1
        return ImageResponse(
            images=[
                ImageResult(url=self.url, format=ImageFormat.PNG, width=1, height=1)
            ],
            model="url-image",
            provider=self.name,
        )


# ---------------------------------------------------------------------------
# Provider outages and fallback
# ---------------------------------------------------------------------------


def test_provider_outage_falls_back_and_charges_once(
    tracker: CostTracker, storage: LocalStorageProvider
) -> None:
    failing = MockImageProvider(
        name="failing", error=AiviaTimeoutError("slow"), cost_usd=0.02
    )
    serving = MockImageProvider(name="serving", cost_usd=0.03)
    service = _service(_router(failing, serving), tracker, storage)

    outcome = _run(service.generate(ImageRequest(prompt=PROMPT), token=_token()))

    assert len(failing.calls) == 0
    assert len(serving.calls) == 1
    assert outcome.provider == "serving"
    assert outcome.selection is not None
    assert outcome.selection.name == "serving"
    assert outcome.event is not None
    assert outcome.event.provider == "serving"
    assert outcome.event.amount_usd == pytest.approx(0.03)
    # One call, one charge: the failed provider is never billed and nothing is
    # recorded twice.
    assert tracker.count() == 1
    assert tracker.total() == pytest.approx(0.03)


def test_total_outage_raises_provider_exhausted_and_charges_nothing(
    tracker: CostTracker, storage: LocalStorageProvider
) -> None:
    failing = MockImageProvider(name="failing", error=AiviaTimeoutError("slow"))
    service = _service(_router(failing), tracker, storage)

    with pytest.raises(ProviderExhaustedError):
        _run(service.generate(ImageRequest(prompt=PROMPT), token=_token()))

    assert len(failing.calls) == 0
    assert tracker.count() == 0
    assert tracker.total() == 0.0


def test_circuit_opens_after_repeated_outages_and_skips_provider(
    tracker: CostTracker, storage: LocalStorageProvider
) -> None:
    failing = MockImageProvider(name="failing", error=AiviaTimeoutError("slow"))
    service = _service(_router(failing), tracker, storage)
    breaker = service.router.breaker_for("failing")

    for _ in range(5):
        with pytest.raises(ProviderExhaustedError):
            _run(service.generate(ImageRequest(prompt=PROMPT), token=_token()))

    assert breaker.failures == 5
    assert breaker.is_open
    # The open circuit is skipped without another provider call and without a
    # charge.
    with pytest.raises(ProviderExhaustedError):
        _run(service.generate(ImageRequest(prompt=PROMPT), token=_token()))
    assert len(failing.calls) == 0
    assert tracker.count() == 0


# ---------------------------------------------------------------------------
# Malformed responses
# ---------------------------------------------------------------------------


def test_malformed_response_is_rejected_before_storage_or_cost(
    tracker: CostTracker, storage: LocalStorageProvider
) -> None:
    provider = BadImageProvider()
    service = _service(_router(provider), tracker, storage)

    with pytest.raises(ValidationError):
        _run(service.generate(ImageRequest(prompt=PROMPT), token=_token()))

    assert provider.calls == 1
    assert tracker.count() == 0
    assert tracker.total() == 0.0
    assert list(storage.root.rglob("*")) == []


# ---------------------------------------------------------------------------
# Partial downloads
# ---------------------------------------------------------------------------


def test_partial_download_is_rejected_before_storage_or_cost(
    tracker: CostTracker, storage: LocalStorageProvider
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=TRUNCATED_PNG,
        )

    downloader = MediaDownloader(transport=httpx.MockTransport(handler))
    provider = UrlImageProvider("https://example.com/image.png")
    service = _service(_router(provider), tracker, storage, downloader=downloader)

    with pytest.raises(ValidationError):
        _run(service.generate(ImageRequest(prompt=PROMPT), token=_token()))

    assert provider.calls == 1
    assert tracker.count() == 0
    assert tracker.total() == 0.0
    assert list(storage.root.rglob("*")) == []


# ---------------------------------------------------------------------------
# Expired tokens
# ---------------------------------------------------------------------------


def test_expired_token_without_refresh_raises() -> None:
    clock = _Clock(0.0)
    store = _store(clock=clock)
    store.store("youtube", "acct-1", "token", expires_in_seconds=10)
    clock.value = 11.0

    with pytest.raises(TokenError):
        _run(store.get("youtube", "acct-1"))


def test_expired_token_recovers_via_refresh() -> None:
    clock = _Clock(0.0)
    refreshed: list[tuple[str, str]] = []

    async def refresh(platform: str, account: str) -> str:
        refreshed.append((platform, account))
        return "fresh-token"

    store = TokenStore(_loader(), refresh=refresh, clock=clock)
    store.store("youtube", "acct-1", "stale-token", expires_in_seconds=10)
    clock.value = 11.0

    token = _run(store.get("youtube", "acct-1"))
    assert token == "fresh-token"
    assert refreshed == [("youtube", "acct-1")]
    # The refreshed token is stored and reused without another refresh call.
    assert _run(store.get("youtube", "acct-1")) == "fresh-token"
    assert refreshed == [("youtube", "acct-1")]


# ---------------------------------------------------------------------------
# Full queue
# ---------------------------------------------------------------------------


def test_full_queue_rejects_submission_and_writes_nothing(engine: Engine) -> None:
    queue = JobQueue(engine, max_queue_depth=2)

    queue.enqueue(kind="media_async", payload={"stage": "render"})
    queue.enqueue(kind="media_async", payload={"stage": "render"})

    assert queue.depth() == 2

    with pytest.raises(QuotaError) as caught:
        queue.enqueue(kind="media_async", payload={"stage": "render"})

    details = caught.value.details
    assert details["depth"] == 2
    assert details["max_queue_depth"] == 2
    # Nothing extra was inserted: no orphan job, no duplicate work.
    assert queue.depth() == 2


# ---------------------------------------------------------------------------
# No double charge across the paid path
# ---------------------------------------------------------------------------


def test_failed_call_records_no_cost_event(
    tracker: CostTracker, storage: LocalStorageProvider
) -> None:
    failing = MockImageProvider(name="failing", error=AiviaTimeoutError("slow"))
    serving = MockImageProvider(name="serving", cost_usd=0.03)
    service = _service(_router(failing, serving), tracker, storage)

    _run(service.generate(ImageRequest(prompt=PROMPT), token=_token()))

    # Only the provider that actually served the call is billed, exactly once.
    events = tracker.events()
    assert len(events) == 1
    assert events[0].provider == "serving"
    assert events[0].amount_usd == pytest.approx(0.03)
    assert events[0].operation == IMAGE_OPERATION
    assert events[0].unit_kind == IMAGE_UNIT
    assert tracker.total() == pytest.approx(0.03)
    assert tracker.actual_total() == pytest.approx(0.03)
