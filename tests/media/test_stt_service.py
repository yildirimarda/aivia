"""The speech to text service caches by audio hash, budgets and prices by duration."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import subprocess
import wave
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import httpx
import pytest
from sqlalchemy import Engine

from aivia.core.config.budget import BudgetConfig
from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import BudgetError, ProviderExhaustedError
from aivia.core.models import ProviderMeta, ProviderPrice, RateLimit
from aivia.core.router import Router
from aivia.media.stt_service import (
    MINUTE_UNIT,
    STT_OPERATION,
    STTService,
    stt_cache_key,
)
from aivia.providers import Provider
from aivia.providers.protocols import (
    CAPABILITY_STT,
    STTRequest,
    STTResponse,
    TranscriptSegment,
    WordTimestamp,
)
from aivia.providers.storage_local import LocalStorageProvider

PRICE = 0.60
DURATION = 2.0
TEXT = "hello brave world"


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


def wav_bytes(duration: float = DURATION) -> bytes:
    sample_rate = 8000
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * int(sample_rate * duration))
    return buffer.getvalue()


def probe_json(duration: float) -> str:
    return json.dumps(
        {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "audio",
                    "codec_name": "pcm_s16le",
                    "duration": f"{duration:.3f}",
                    "sample_rate": "8000",
                    "channels": 1,
                }
            ],
            "format": {
                "format_name": "wav",
                "format_long_name": "WAV / WAVE",
                "duration": f"{duration:.3f}",
                "size": "12345",
            },
        }
    )


def probe_runner(durations: Sequence[float]) -> Any:
    remaining = list(durations)

    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        duration = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout=probe_json(duration),
            stderr="",
        )

    return runner


def failing_probe_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        list(argv),
        1,
        stdout="",
        stderr="ffprobe rejected the media",
    )


class FakeSTTProvider(Provider):
    """A speech to text provider returning inline transcripts with timing."""

    def __init__(
        self,
        name: str = "acme-stt",
        *,
        price: float = PRICE,
        duration: float | None = DURATION,
        text: str = TEXT,
        error: BaseException | None = None,
    ) -> None:
        super().__init__(
            ProviderMeta(
                name=name,
                capabilities=[CAPABILITY_STT],
                pricing={
                    CAPABILITY_STT: ProviderPrice(
                        amount_usd=price,
                        unit=MINUTE_UNIT,
                    )
                },
                rate_limit=RateLimit(requests_per_minute=0),
            )
        )
        self.calls: list[STTRequest] = []
        self.duration = duration
        self.text = text
        self.error = error

    async def transcribe(self, request: STTRequest) -> STTResponse:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        words = self.text.split()
        timestamps = [
            WordTimestamp(
                word=word,
                start_seconds=index * 0.25,
                end_seconds=(index + 1) * 0.25,
            )
            for index, word in enumerate(words)
        ]
        segments = (
            [
                TranscriptSegment(
                    text=self.text,
                    start_seconds=0.0,
                    end_seconds=(
                        self.duration
                        if self.duration is not None
                        else len(words) * 0.25
                    ),
                    words=timestamps,
                )
            ]
            if self.text
            else []
        )
        return STTResponse(
            text=self.text,
            language=request.language or "en",
            duration_seconds=self.duration,
            segments=segments,
            words=timestamps,
            model="stt-1",
            provider=self.name,
            request_id="req-1",
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
    *,
    durations: Sequence[float] = (DURATION,),
    **values: Any,
) -> STTService:
    return STTService(
        _router(provider),
        tracker=tracker,
        storage=storage,
        ffprobe_runner=probe_runner(durations),
        **values,
    )


def _path(asset_path: str) -> Path:
    parsed = urlsplit(asset_path)
    assert parsed.scheme == "file"
    return Path(unquote(parsed.path))


def _remote_transport(
    payload: bytes,
    downloads: list[str] | None = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if downloads is not None:
            downloads.append(str(request.url))
        return httpx.Response(
            200,
            content=payload,
            headers={"content-type": "audio/wav"},
        )

    return httpx.MockTransport(handler)


def test_a_transcript_is_stored_priced_and_writes_a_cost_sidecar(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeSTTProvider()
    service = _service(provider, tracker, storage)
    audio = wav_bytes()

    outcome = _run(
        service.generate(
            STTRequest(audio=audio, language="en"),
            run_id="run-1",
            metadata={"stage": "stt"},
        )
    )

    assert outcome.cache_hit is False
    assert outcome.cached is False
    assert len(provider.calls) == 1
    assert outcome.text == TEXT
    assert outcome.provider == "acme-stt"
    assert outcome.model == "stt-1"
    assert [word.word for word in outcome.words] == TEXT.split()

    asset = outcome.asset
    assert asset is not None
    assert asset.kind.value == "text"
    assert asset.provider == "acme-stt"
    assert asset.model == "stt-1"
    assert asset.run_id == "run-1"
    assert asset.pinned is True
    assert (
        asset.content_hash == hashlib.sha256(_path(asset.path).read_bytes()).hexdigest()
    )
    stored = json.loads(_path(asset.path).read_text("utf-8"))
    assert stored["text"] == TEXT
    assert stored["duration_seconds"] == DURATION
    assert stored["provider"] == "acme-stt"

    event = outcome.event
    assert event is not None
    assert event.provider == "acme-stt"
    assert event.operation == STT_OPERATION
    assert event.unit_kind == MINUTE_UNIT
    assert event.units == pytest.approx(DURATION / 60.0)
    assert event.amount_usd == pytest.approx(PRICE * DURATION / 60.0)
    assert event.metadata["stage"] == "stt"
    assert asset.cost_links[0].cost_event_id == event.id
    assert tracker.total() == pytest.approx(PRICE * DURATION / 60.0)

    sidecar = outcome.sidecar
    assert sidecar is not None and sidecar.exists()
    payload = json.loads(sidecar.read_text("utf-8"))
    assert payload["output"] == str(_path(asset.path))
    assert payload["totals"]["total_usd"] == pytest.approx(PRICE * DURATION / 60.0)

    assert service.cache is not None
    entry = service.cache.entry(
        service.cache_key(STTRequest(audio=audio, language="en"))
    )
    assert entry is not None
    assert entry.provider == ""
    assert entry.reference_hashes == [f"audio:{hashlib.sha256(audio).hexdigest()}"]


def test_identical_audio_is_served_from_cache_without_another_provider_call(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeSTTProvider()
    service = _service(provider, tracker, storage)
    audio = wav_bytes()

    first = _run(
        service.generate(
            STTRequest(audio=audio, filename="first.wav"),
            run_id="run-1",
        )
    )
    second = _run(
        service.generate(
            STTRequest(audio=audio, filename="renamed.wav"),
            run_id="run-2",
        )
    )

    assert len(provider.calls) == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.text == first.text
    assert [word.word for word in second.words] == [word.word for word in first.words]
    assert second.asset is not None and first.asset is not None
    assert second.asset.id == first.asset.id
    assert second.event is not None
    assert second.event.amount_usd == 0.0
    assert second.event.metadata["cache"] == "hit"
    assert second.event.run_id == "run-2"
    assert second.amount_usd == 0.0
    assert tracker.total() == pytest.approx(PRICE * DURATION / 60.0)
    assert second.sidecar is not None and second.sidecar.exists()

    assert service.cache is not None
    entry = service.cache.entry(stt_cache_key(STTRequest(audio=audio)))
    assert entry is not None
    assert entry.hit_count == 1


def test_different_audio_is_not_served_from_cache(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeSTTProvider()
    service = _service(provider, tracker, storage)

    _run(service.generate(STTRequest(audio=wav_bytes(1.0)), run_id="run-1"))
    _run(service.generate(STTRequest(audio=wav_bytes(2.0)), run_id="run-2"))

    assert len(provider.calls) == 2
    assert tracker.total() == pytest.approx(PRICE * 2 * DURATION / 60.0)


def test_the_budget_is_checked_against_the_audio_duration(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeSTTProvider(price=PRICE)
    service = _service(provider, tracker, storage, durations=(DURATION,))

    with pytest.raises(ProviderExhaustedError) as caught:
        _run(
            service.generate(
                STTRequest(audio=wav_bytes()),
                remaining_budget=PRICE * DURATION / 60.0 / 2.0,
            )
        )

    assert provider.calls == []
    assert caught.value.details["skipped"] == [
        {"provider": "acme-stt", "reason": "budget_exceeded"}
    ]
    assert tracker.count() == 0


def test_a_duration_hint_avoids_a_probe_and_allows_the_call(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeSTTProvider(price=PRICE)
    service = _service(provider, tracker, storage, durations=(DURATION,))

    outcome = _run(
        service.generate(
            STTRequest(audio=wav_bytes()),
            remaining_budget=PRICE * DURATION / 60.0,
            duration_seconds=DURATION,
        )
    )

    assert len(provider.calls) == 1
    assert outcome.amount_usd == pytest.approx(PRICE * DURATION / 60.0)


def test_a_duration_hint_for_remote_audio_avoids_downloading_it(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeSTTProvider(price=PRICE)
    downloads: list[str] = []
    service = _service(
        provider,
        tracker,
        storage,
        durations=(DURATION,),
        transport=_remote_transport(wav_bytes(), downloads),
    )
    url = "https://cdn.example.com/audio/clip.wav"

    outcome = _run(
        service.generate(
            STTRequest(audio_url=url),
            remaining_budget=PRICE * DURATION / 60.0,
            duration_seconds=DURATION,
        )
    )

    assert downloads == []
    assert len(provider.calls) == 1
    assert outcome.amount_usd == pytest.approx(PRICE * DURATION / 60.0)


def test_a_metadata_duration_hint_for_remote_audio_guards_the_budget(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeSTTProvider(price=PRICE)
    downloads: list[str] = []
    service = _service(
        provider,
        tracker,
        storage,
        durations=(DURATION,),
        transport=_remote_transport(wav_bytes(), downloads),
    )

    with pytest.raises(ProviderExhaustedError) as caught:
        _run(
            service.generate(
                STTRequest(
                    audio_url="https://cdn.example.com/audio/clip.wav",
                    metadata={"duration_seconds": DURATION},
                ),
                remaining_budget=PRICE * DURATION / 60.0 / 2.0,
            )
        )

    assert downloads == []
    assert provider.calls == []
    assert caught.value.details["skipped"] == [
        {"provider": "acme-stt", "reason": "budget_exceeded"}
    ]


def test_remote_audio_is_probed_before_the_budget_check(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeSTTProvider(price=PRICE)
    downloads: list[str] = []
    service = _service(
        provider,
        tracker,
        storage,
        durations=(DURATION,),
        transport=_remote_transport(wav_bytes(), downloads),
    )
    url = "https://cdn.example.com/audio/clip.wav"

    with pytest.raises(ProviderExhaustedError) as caught:
        _run(
            service.generate(
                STTRequest(audio_url=url),
                remaining_budget=PRICE * DURATION / 60.0 / 2.0,
            )
        )

    assert downloads == [url]
    assert provider.calls == []
    assert caught.value.details["skipped"] == [
        {"provider": "acme-stt", "reason": "budget_exceeded"}
    ]
    assert tracker.count() == 0


def test_remote_audio_is_probed_and_allowed_when_the_budget_covers_it(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeSTTProvider(price=PRICE)
    downloads: list[str] = []
    service = _service(
        provider,
        tracker,
        storage,
        durations=(DURATION,),
        transport=_remote_transport(wav_bytes(), downloads),
    )
    url = "https://cdn.example.com/audio/clip.wav"

    outcome = _run(
        service.generate(
            STTRequest(audio_url=url),
            remaining_budget=PRICE * DURATION / 60.0,
        )
    )

    assert downloads == [url]
    assert len(provider.calls) == 1
    event = outcome.event
    assert event is not None
    assert event.units == pytest.approx(DURATION / 60.0)
    assert outcome.amount_usd == pytest.approx(PRICE * DURATION / 60.0)


def test_a_budget_guard_uses_the_configured_cap(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeSTTProvider(price=PRICE)
    service = _service(
        provider,
        tracker,
        storage,
        durations=(DURATION,),
        budget=BudgetConfig(per_call=PRICE * DURATION / 60.0 / 2.0),
    )

    with pytest.raises(ProviderExhaustedError):
        _run(service.generate(STTRequest(audio=wav_bytes()), run_id="run-1"))

    assert provider.calls == []


def test_a_configured_budget_refuses_a_call_with_an_unknown_duration(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeSTTProvider(price=PRICE)
    service = STTService(
        _router(provider),
        tracker=tracker,
        storage=storage,
        budget=BudgetConfig(per_call=PRICE),
        ffprobe_runner=failing_probe_runner,
    )

    with pytest.raises(BudgetError) as caught:
        _run(service.generate(STTRequest(audio=wav_bytes()), run_id="run-1"))

    assert provider.calls == []
    assert tracker.count() == 0
    assert caught.value.details["run_id"] == "run-1"
    assert caught.value.details["operation"] == STT_OPERATION
    assert caught.value.details["remaining_budget"] == PRICE


def test_a_remaining_budget_refuses_a_call_with_an_unknown_duration(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeSTTProvider(price=PRICE)
    service = STTService(
        _router(provider),
        tracker=tracker,
        storage=storage,
        ffprobe_runner=failing_probe_runner,
    )

    with pytest.raises(BudgetError) as caught:
        _run(
            service.generate(
                STTRequest(audio=wav_bytes()),
                remaining_budget=PRICE,
            )
        )

    assert provider.calls == []
    assert tracker.count() == 0
    assert caught.value.details["remaining_budget"] == PRICE


def test_an_unknown_duration_is_allowed_when_no_budget_is_configured(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeSTTProvider(price=PRICE)
    service = STTService(
        _router(provider),
        tracker=tracker,
        storage=storage,
        ffprobe_runner=failing_probe_runner,
    )

    outcome = _run(service.generate(STTRequest(audio=wav_bytes()), run_id="run-1"))

    assert len(provider.calls) == 1
    assert outcome.amount_usd == pytest.approx(PRICE * DURATION / 60.0)


def test_cost_falls_back_to_timing_when_the_provider_reports_no_duration(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    text = " ".join(f"word{index}" for index in range(12))
    provider = FakeSTTProvider(duration=None, text=text)
    service = _service(provider, tracker, storage)

    outcome = _run(service.generate(STTRequest(audio=wav_bytes())))

    assert outcome.response.duration_seconds == pytest.approx(3.0)
    event = outcome.event
    assert event is not None
    assert event.units == pytest.approx(3.0 / 60.0)
    assert event.amount_usd == pytest.approx(PRICE * 3.0 / 60.0)


def test_a_cache_hit_still_writes_a_zero_cost_sidecar(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeSTTProvider()
    service = _service(provider, tracker, storage)

    _run(service.generate(STTRequest(audio=wav_bytes()), run_id="run-1"))
    second = _run(service.generate(STTRequest(audio=wav_bytes()), run_id="run-2"))

    assert second.event is not None
    assert second.event.amount_usd == 0.0
    sidecar = second.sidecar
    assert sidecar is not None and sidecar.exists()
    payload = json.loads(sidecar.read_text("utf-8"))
    assert payload["totals"]["total_usd"] == 0.0
    assert payload["totals"]["count"] == 1
