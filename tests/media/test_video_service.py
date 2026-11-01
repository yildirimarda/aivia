"""The video service is the single screened, cached, budgeted paid path."""

from __future__ import annotations

import asyncio
import json
import struct
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import pytest
from sqlalchemy import Engine

from aivia.core.budget import BudgetGuard
from aivia.core.config.budget import BudgetConfig
from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import (
    BudgetError,
    ContentPolicyError,
    InvalidRequestError,
    ProviderExhaustedError,
    ValidationError,
)
from aivia.core.errors import TimeoutError as AiviaTimeoutError
from aivia.core.models import AssetKind, ProviderMeta, ProviderPrice, RateLimit
from aivia.core.router import Router
from aivia.media.screening import screen_prompt
from aivia.media.video_cache import VIDEO_OPERATION, VideoCache
from aivia.media.video_pricing import VideoCostEstimate, VideoPricing
from aivia.media.video_service import VideoOutcome, VideoService
from aivia.providers import Provider
from aivia.providers.protocols import (
    CAPABILITY_VIDEO_FIRST_FRAME,
    CAPABILITY_VIDEO_GEN,
    CAPABILITY_VIDEO_INPUT,
    MediaInput,
    VideoFormat,
    VideoRequest,
    VideoResponse,
    VideoResult,
)
from aivia.providers.storage_local import LocalStorageProvider

PROMPT = "A lighthouse in a storm"
PRICE = 0.05


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


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


def mp4() -> bytes:
    return struct.pack(">I", 24) + b"ftypisom" + b"\x00" * 12


def probe_json(
    *,
    width: int = 1280,
    height: int = 720,
    duration: str = "5.0",
    audio: bool = True,
) -> str:
    streams: list[dict[str, Any]] = [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "width": width,
            "height": height,
            "duration": duration,
            "avg_frame_rate": "30/1",
        }
    ]
    if audio:
        streams.append(
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "duration": duration,
                "sample_rate": "48000",
                "channels": 2,
            }
        )
    return json.dumps(
        {
            "streams": streams,
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "format_long_name": "QuickTime / MOV",
                "duration": duration,
                "size": "12345",
            },
        }
    )


def runner_for(
    stdout: str,
    *,
    calls: list[list[str]] | None = None,
) -> Any:
    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append(list(argv))
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout=stdout,
            stderr="",
        )

    return runner


def _path(asset_path: str) -> Path:
    parsed = urlsplit(asset_path)
    assert parsed.scheme == "file"
    return Path(unquote(parsed.path))


class FakeVideoProvider(Provider):
    """A video provider returning validated-looking inline MP4 bytes."""

    def __init__(
        self,
        name: str = "acme",
        *,
        price: float = PRICE,
        unit: str = "clip",
        model: str = "video-1",
        error: BaseException | None = None,
        capabilities: Sequence[str] = (CAPABILITY_VIDEO_GEN,),
    ) -> None:
        super().__init__(
            ProviderMeta(
                name=name,
                capabilities=list(capabilities),
                pricing={
                    CAPABILITY_VIDEO_GEN: ProviderPrice(
                        amount_usd=price,
                        unit=unit,
                    )
                },
                rate_limit=RateLimit(requests_per_minute=0),
            )
        )
        self.model = model
        self.error = error
        self.calls: list[VideoRequest] = []

    async def generate(self, request: VideoRequest) -> VideoResponse:
        if self.error is not None:
            raise self.error
        self.calls.append(request)
        return VideoResponse(
            videos=[
                VideoResult(
                    data=mp4(),
                    format=VideoFormat.MP4,
                    fps=request.fps,
                    seed=request.seed,
                )
                for _ in range(request.count)
            ],
            model=self.model,
            provider=self.name,
        )


def _router(*providers: Provider) -> Router:
    return Router(
        {provider.name: provider for provider in providers},
        config=CapabilityChainConfig(chain=[provider.name for provider in providers]),
    )


def _service(
    provider: Provider,
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
    **values: Any,
) -> VideoService:
    probe_calls: list[list[str]] = []
    cache = VideoCache(
        engine=engine,
        storage=storage,
        ffprobe_runner=runner_for(probe_json(), calls=probe_calls),
    )
    return VideoService(
        _router(provider),
        tracker=tracker,
        cache=cache,
        storage=storage,
        **values,
    )


def _token(prompt: str = PROMPT) -> str:
    return str(screen_prompt(prompt))


def test_a_missing_or_mismatched_token_blocks_before_any_paid_step(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    provider = FakeVideoProvider()
    service = _service(provider, tracker, storage, engine)

    with pytest.raises(ContentPolicyError) as missing:
        _run(service.generate(VideoRequest(prompt=PROMPT)))
    assert missing.value.details["reason"] == "missing_token"

    with pytest.raises(ContentPolicyError) as mismatched:
        _run(
            service.generate(
                VideoRequest(prompt=PROMPT),
                token=_token("A different prompt"),
            )
        )
    assert mismatched.value.details["reason"] == "text_mismatch"

    assert provider.calls == []
    assert tracker.count() == 0
    assert service.cache.cache is not None
    assert service.cache.cache.repository.list() == []
    assert list(storage.root.rglob("*")) == []


def test_a_non_video_source_is_rejected_before_any_paid_step(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    provider = FakeVideoProvider(
        capabilities=(CAPABILITY_VIDEO_GEN, CAPABILITY_VIDEO_INPUT),
    )
    service = _service(provider, tracker, storage, engine)
    request = VideoRequest(
        prompt=PROMPT,
        source_video=MediaInput(
            url="https://example.com/media/source.png",
            media_type="image/png",
        ),
    )

    with pytest.raises(InvalidRequestError) as caught:
        _run(service.generate(request, token=_token()))

    assert caught.value.details["field"] == "source_video"
    assert caught.value.details["media_type"] == "image/png"
    assert provider.calls == []
    assert tracker.count() == 0
    assert list(storage.root.rglob("*")) == []


def test_a_screened_request_routes_validates_stores_prices_and_writes_a_sidecar(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    provider = FakeVideoProvider()
    service = _service(provider, tracker, storage, engine)
    request = VideoRequest(
        prompt=PROMPT,
        duration_seconds=5.0,
        width=1280,
        height=720,
        fps=30,
    )

    outcome = _run(
        service.generate(
            request,
            token=_token(),
            run_id="run_1",
            metadata={"stage": "visuals"},
        )
    )

    assert isinstance(outcome, VideoOutcome)
    assert outcome.cache_hit is False
    assert outcome.response.count == 1
    assert outcome.provider == "acme"
    assert outcome.model == "video-1"
    assert len(provider.calls) == 1

    assert len(outcome.assets) == 1
    asset = outcome.asset
    assert asset is not None
    assert asset.kind is AssetKind.VIDEO
    assert asset.provider == "acme"
    assert asset.model == "video-1"
    assert asset.prompt == PROMPT
    assert asset.run_id == "run_1"
    assert asset.pinned is True
    assert asset.content_hash == outcome.probes[0].sha256
    assert asset.analysis.width == 1280
    assert asset.analysis.height == 720
    assert asset.analysis.duration_seconds == pytest.approx(5.0)

    path = _path(asset.path)
    assert path.is_file()
    assert path.read_bytes() == outcome.probes[0].data

    event = outcome.event
    assert event is not None
    assert event.provider == "acme"
    assert event.operation == VIDEO_OPERATION
    assert event.unit_kind == "clip"
    assert event.units == 1.0
    assert event.amount_usd == pytest.approx(PRICE)
    assert event.run_id == "run_1"
    assert event.estimated is False
    assert event.metadata["stage"] == "visuals"
    assert event.metadata["video_count"] == 1
    assert event.metadata["duration_seconds"] == pytest.approx(5.0)
    assert asset.cost_links[0].cost_event_id == event.id
    assert tracker.total() == pytest.approx(PRICE)

    assert outcome.price is not None
    assert outcome.price.amount_usd == pytest.approx(PRICE)
    assert outcome.selection is not None
    assert outcome.selection.name == "acme"

    sidecar = outcome.sidecar
    assert sidecar is not None
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text("utf-8"))
    assert payload["output"] == str(path)
    assert payload["run_id"] == "run_1"
    assert payload["totals"]["total_usd"] == pytest.approx(PRICE)
    assert payload["items"][0]["cost_event_id"] == event.id
    assert payload["providers"] == ["acme"]

    assert service.cache.cache is not None
    entries = service.cache.cache.repository.list()
    assert len(entries) == 1
    assert entries[0].provider == ""
    assert entries[0].model == ""
    assert entries[0].prompt_hash == asset.prompt_hash


def test_an_identical_request_is_served_from_cache_without_a_provider_call(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    provider = FakeVideoProvider()
    service = _service(provider, tracker, storage, engine)
    request = VideoRequest(
        prompt=PROMPT,
        duration_seconds=5.0,
        width=1280,
        height=720,
    )

    first = _run(service.generate(request, token=_token(), run_id="run_1"))
    second = _run(service.generate(request, token=_token(), run_id="run_2"))

    assert len(provider.calls) == 1
    assert second.cache_hit is True
    assert second.asset is not None
    assert first.asset is not None
    assert second.asset.id == first.asset.id
    assert second.response.count == 1
    assert second.event is not None
    assert second.event.amount_usd == 0.0
    assert second.event.metadata["cache"] == "hit"
    assert second.event.run_id == "run_2"
    assert second.amount_usd == 0.0
    assert tracker.total() == pytest.approx(PRICE)
    assert second.sidecar is not None
    assert second.sidecar.exists()

    assert service.cache.cache is not None
    entry = service.cache.cache.entry(service.cache_key(request))
    assert entry is not None
    assert entry.hit_count == 1


def test_an_over_budget_call_never_reaches_the_provider(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    provider = FakeVideoProvider(name="pricey", price=0.10, unit="call")
    service = _service(provider, tracker, storage, engine)

    with pytest.raises(ProviderExhaustedError) as caught:
        _run(
            service.generate(
                VideoRequest(prompt=PROMPT),
                token=_token(),
                remaining_budget=0.05,
            )
        )

    assert provider.calls == []
    assert caught.value.details["skipped"] == [
        {"provider": "pricey", "reason": "budget_exceeded"}
    ]
    assert tracker.count() == 0


def test_a_budget_guard_supplies_the_remaining_budget(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    provider = FakeVideoProvider(name="pricey", price=0.10, unit="call")
    guard = BudgetGuard(BudgetConfig(per_call=0.05), tracker)
    service = _service(provider, tracker, storage, engine, budget=guard)

    with pytest.raises(ProviderExhaustedError):
        _run(service.generate(VideoRequest(prompt=PROMPT), token=_token()))

    assert provider.calls == []
    assert tracker.count() == 0


def test_a_pricing_table_checks_the_budget_before_submit(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    provider = FakeVideoProvider(price=0.0)
    service = _service(
        provider,
        tracker,
        storage,
        engine,
        pricing=VideoPricing(per_second=0.10),
    )

    with pytest.raises(BudgetError) as caught:
        _run(
            service.generate(
                VideoRequest(prompt=PROMPT, duration_seconds=5.0),
                token=_token(),
                remaining_budget=0.20,
            )
        )

    assert caught.value.code == "budget"
    assert provider.calls == []
    assert tracker.count() == 0
    assert service.failures[0].stage == "budget"


def test_retryable_failure_falls_back_and_records_the_failed_attempt(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    failing = FakeVideoProvider(
        name="failing",
        error=AiviaTimeoutError("slow"),
    )
    serving = FakeVideoProvider(name="serving", price=0.03, model="video-2")
    probe_calls: list[list[str]] = []
    cache = VideoCache(
        engine=engine,
        storage=storage,
        ffprobe_runner=runner_for(probe_json(), calls=probe_calls),
    )
    service = VideoService(
        _router(failing, serving),
        tracker=tracker,
        cache=cache,
        storage=storage,
    )

    outcome = _run(service.generate(VideoRequest(prompt=PROMPT), token=_token()))

    assert failing.calls == []
    assert len(serving.calls) == 1
    assert outcome.provider == "serving"
    assert outcome.selection is not None
    assert outcome.selection.name == "serving"
    assert outcome.event is not None
    assert outcome.event.provider == "serving"
    assert outcome.event.amount_usd == pytest.approx(0.03)
    assert tracker.total() == pytest.approx(0.03)
    assert len(outcome.failures) == 1
    assert outcome.failures[0].provider == "failing"
    assert outcome.failures[0].error == "timeout"
    assert service.failures == outcome.failures


def test_exhausted_failures_are_recorded_before_the_error_is_raised(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    failing = FakeVideoProvider(
        name="failing",
        error=AiviaTimeoutError("slow"),
    )
    service = _service(failing, tracker, storage, engine)

    with pytest.raises(ProviderExhaustedError):
        _run(service.generate(VideoRequest(prompt=PROMPT), token=_token()))

    assert service.failures
    assert service.failures[0].provider == "failing"
    assert service.failures[0].error == "timeout"
    assert tracker.count() == 0


def test_invalid_video_bytes_are_rejected_before_storage_or_cost(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    class BadVideoProvider(FakeVideoProvider):
        async def generate(self, request: VideoRequest) -> VideoResponse:
            self.calls.append(request)
            return VideoResponse(
                videos=[VideoResult(data=b"not a video", format=VideoFormat.MP4)],
                model=self.model,
                provider=self.name,
            )

    provider = BadVideoProvider(name="bad")
    service = _service(provider, tracker, storage, engine)

    with pytest.raises(ValidationError):
        _run(service.generate(VideoRequest(prompt=PROMPT), token=_token()))

    assert len(provider.calls) == 1
    assert tracker.count() == 0
    assert service.cache.cache is not None
    assert service.cache.cache.repository.list() == []
    assert list(storage.root.rglob("*")) == []
    assert service.failures
    assert service.failures[0].stage == "validation"


def test_a_pricing_table_prices_the_validated_duration_on_completion(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    provider = FakeVideoProvider(price=0.0)
    service = _service(
        provider,
        tracker,
        storage,
        engine,
        pricing=VideoPricing(per_second=0.10),
    )

    outcome = _run(
        service.generate(
            VideoRequest(prompt=PROMPT, duration_seconds=2.0),
            token=_token(),
            run_id="run_1",
        )
    )

    assert outcome.event is not None
    assert outcome.event.amount_usd == pytest.approx(0.50)
    assert outcome.event.units == pytest.approx(5.0)
    assert outcome.event.unit_kind == "second"
    assert isinstance(outcome.price, VideoCostEstimate)
    assert outcome.price.billable_seconds == pytest.approx(5.0)
    assert tracker.total() == pytest.approx(0.50)


def test_count_stores_one_asset_and_cache_entry_per_output(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    provider = FakeVideoProvider()
    service = _service(provider, tracker, storage, engine)
    request = VideoRequest(
        prompt=PROMPT,
        duration_seconds=2.0,
        width=640,
        height=360,
        count=2,
    )

    first = _run(service.generate(request, token=_token(), run_id="run_1"))

    assert len(provider.calls) == 1
    assert len(first.assets) == 2
    assert len(first.storage_objects) == 2
    assert len({asset.id for asset in first.assets}) == 2
    assert first.event is not None
    assert first.event.units == 2.0
    assert first.event.amount_usd == pytest.approx(PRICE * 2)
    assert service.cache.cache is not None
    assert len(service.cache.cache.repository.list()) == 2

    second = _run(service.generate(request, token=_token(), run_id="run_2"))

    assert len(provider.calls) == 1
    assert second.cache_hit is True
    assert second.response.count == 2
    assert [asset.id for asset in second.assets] == [asset.id for asset in first.assets]
    assert len(second.events) == 2
    assert second.amount_usd == 0.0
    assert tracker.total() == pytest.approx(PRICE * 2)


def test_no_cache_forces_every_request_through_the_provider(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    provider = FakeVideoProvider()
    service = _service(provider, tracker, storage, engine, no_cache=True)
    request = VideoRequest(prompt=PROMPT)

    _run(service.generate(request, token=_token()))
    _run(service.generate(request, token=_token()))

    assert len(provider.calls) == 2
    assert service.cache.cache is not None
    assert service.cache.cache.repository.list() == []
    assert tracker.total() == pytest.approx(PRICE * 2)


def test_a_first_frame_requires_a_provider_that_declares_the_capability(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    provider = FakeVideoProvider()
    service = _service(provider, tracker, storage, engine)
    request = VideoRequest(
        prompt=PROMPT,
        first_frame=MediaInput(data=b"frame", media_type="image/png"),
    )

    with pytest.raises(ProviderExhaustedError):
        _run(service.generate(request, token=_token()))

    assert provider.calls == []
    assert tracker.count() == 0


def test_a_capable_provider_accepts_a_first_frame(
    tracker: CostTracker,
    storage: LocalStorageProvider,
    engine: Engine,
) -> None:
    provider = FakeVideoProvider(
        capabilities=(CAPABILITY_VIDEO_GEN, CAPABILITY_VIDEO_FIRST_FRAME),
    )
    service = _service(provider, tracker, storage, engine)
    request = VideoRequest(
        prompt=PROMPT,
        first_frame=MediaInput(data=b"frame", media_type="image/png"),
    )

    outcome = _run(service.generate(request, token=_token()))

    assert len(provider.calls) == 1
    assert outcome.asset is not None
    assert tracker.total() == pytest.approx(PRICE)
