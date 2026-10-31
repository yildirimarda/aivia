"""The speech service caches, measures, trims, fades and normalizes audio."""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import pytest
from sqlalchemy import Engine

from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import ContentPolicyError
from aivia.core.models import AssetKind, ProviderMeta, ProviderPrice, RateLimit
from aivia.core.router import Router
from aivia.media.audio_utils import (
    AudioSettings,
    fade_audio,
    measure_audio_duration,
    normalize_audio,
    process_audio,
    trim_audio,
)
from aivia.media.screening import screen_prompt
from aivia.media.tts_service import (
    TTS_OPERATION,
    SpeechOutcome,
    SpeechService,
    speech_cache_key,
)
from aivia.providers import Provider
from aivia.providers.protocols import (
    CAPABILITY_TTS,
    AudioFormat,
    TTSRequest,
    TTSResponse,
)
from aivia.providers.storage_local import LocalStorageProvider

TEXT = "A lighthouse in a storm."
VOICE = "voice-narrator"
PRICE = 0.0001
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00source-audio"
PROCESSED = b"ID3\x04\x00\x00\x00\x00\x00\x00processed-audio"


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


def probe_json(duration: float) -> str:
    return json.dumps(
        {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "audio",
                    "codec_name": "mp3",
                    "duration": f"{duration:.3f}",
                    "sample_rate": "44100",
                    "channels": 1,
                }
            ],
            "format": {
                "format_name": "mp3",
                "format_long_name": "MP2/3 (MPEG audio layer 2/3)",
                "duration": f"{duration:.3f}",
                "size": "12345",
            },
        }
    )


def probe_runner(
    durations: Sequence[float],
    *,
    calls: list[list[str]] | None = None,
) -> Any:
    remaining = list(durations)

    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append(list(argv))
        duration = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout=probe_json(duration),
            stderr="",
        )

    return runner


def ffmpeg_writer(
    *,
    calls: list[list[str]] | None = None,
    payload: bytes = PROCESSED,
) -> Any:
    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
        if calls is not None:
            calls.append(list(argv))
        Path(argv[-1]).write_bytes(payload)
        return subprocess.CompletedProcess(list(argv), 0, stdout=b"", stderr=b"")

    return runner


def option(argv: Sequence[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


class FakeTTSProvider(Provider):
    """A text to speech provider returning inline MP3 bytes."""

    def __init__(
        self,
        name: str = "acme-tts",
        *,
        price: float = PRICE,
        unit: str = "character",
    ) -> None:
        super().__init__(
            ProviderMeta(
                name=name,
                capabilities=[CAPABILITY_TTS],
                pricing={
                    CAPABILITY_TTS: ProviderPrice(amount_usd=price, unit=unit),
                },
                rate_limit=RateLimit(requests_per_minute=0),
            )
        )
        self.calls: list[TTSRequest] = []

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        self.calls.append(request)
        return TTSResponse(
            audio=MP3,
            format=AudioFormat.MP3,
            sample_rate=44100,
            voice=request.voice,
            model=request.model or "tts-1",
            provider=self.name,
            metadata={"characters": len(request.text)},
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
    durations: Sequence[float] = (3.0,),
    ffmpeg_calls: list[list[str]] | None = None,
    settings: AudioSettings | None = None,
) -> SpeechService:
    return SpeechService(
        _router(provider),
        tracker=tracker,
        storage=storage,
        settings=settings,
        ffprobe_runner=probe_runner(durations),
        ffmpeg_runner=ffmpeg_writer(calls=ffmpeg_calls),
    )


def _path(asset_path: str) -> Path:
    parsed = urlsplit(asset_path)
    assert parsed.scheme == "file"
    return Path(unquote(parsed.path))


def test_a_request_is_stored_priced_and_measured(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeTTSProvider()
    service = _service(provider, tracker, storage, durations=(2.75,))
    request = TTSRequest(text=TEXT, voice=VOICE)

    outcome = _run(service.synthesize(request, run_id="run-1"))

    assert isinstance(outcome, SpeechOutcome)
    assert outcome.cache_hit is False
    assert outcome.provider == "acme-tts"
    assert outcome.model == "tts-1"
    assert outcome.voice == VOICE
    assert outcome.audio == MP3
    assert outcome.format is AudioFormat.MP3
    assert outcome.duration_seconds == pytest.approx(2.75)
    assert outcome.processed is False
    assert provider.calls == [request]

    asset = outcome.asset
    assert asset is not None
    assert asset.kind is AssetKind.VOICE
    assert asset.provider == "acme-tts"
    assert asset.prompt == TEXT
    assert asset.analysis.duration_seconds == pytest.approx(2.75)
    assert asset.content_hash == outcome.probes[0].sha256
    path = _path(asset.path)
    assert path.read_bytes() == MP3

    event = outcome.event
    assert event is not None
    assert event.operation == TTS_OPERATION
    assert event.unit_kind == "character"
    assert event.units == float(len(TEXT))
    assert event.amount_usd == pytest.approx(PRICE * len(TEXT))
    assert event.run_id == "run-1"
    assert tracker.total() == pytest.approx(PRICE * len(TEXT))
    assert outcome.amount_usd == pytest.approx(PRICE * len(TEXT))


def test_identical_text_voice_and_settings_are_served_from_cache(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeTTSProvider()
    service = _service(provider, tracker, storage, durations=(2.75,))
    request = TTSRequest(text=TEXT, voice=VOICE)

    first = _run(service.synthesize(request))
    spent = tracker.total()
    second = _run(service.synthesize(request))

    assert len(provider.calls) == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.audio == first.audio
    assert second.duration_seconds == first.duration_seconds
    assert second.event is not None
    assert second.event.amount_usd == 0.0
    assert second.event.metadata["cache"] == "hit"
    assert tracker.total() == pytest.approx(spent)


def test_text_voice_and_settings_all_take_part_in_the_cache_key(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeTTSProvider()
    service = _service(provider, tracker, storage, durations=(3.0,))
    base = TTSRequest(text=TEXT, voice=VOICE)

    _run(service.synthesize(base))
    _run(service.synthesize(base.model_copy(update={"text": "Different text."})))
    _run(service.synthesize(base.model_copy(update={"voice": "voice-other"})))
    _run(
        service.synthesize(
            base,
            settings=AudioSettings(normalize_loudness=True),
        )
    )

    assert len(provider.calls) == 4
    assert speech_cache_key(base, AudioSettings()) != speech_cache_key(
        base,
        AudioSettings(normalize_loudness=True),
    )
    assert speech_cache_key(base, AudioSettings()) != speech_cache_key(
        base.model_copy(update={"voice": "voice-other"}),
        AudioSettings(),
    )


def test_loudness_normalization_runs_ffmpeg_with_the_target(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeTTSProvider()
    calls: list[list[str]] = []
    service = _service(
        provider,
        tracker,
        storage,
        durations=(2.5,),
        ffmpeg_calls=calls,
        settings=AudioSettings(
            normalize_loudness=True,
            target_lufs=-14.0,
            true_peak_db=-2.0,
            loudness_range=9.0,
        ),
    )

    outcome = _run(service.synthesize(TTSRequest(text=TEXT, voice=VOICE)))

    assert outcome.processed is True
    assert outcome.audio == PROCESSED
    assert outcome.duration_seconds == pytest.approx(2.5)
    assert len(calls) == 1
    filters = option(calls[0], "-af")
    assert "loudnorm=I=-14.000000:TP=-2.000000:LRA=9.000000" in filters


def test_trimming_and_fades_are_applied_and_the_duration_is_remeasured(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeTTSProvider()
    calls: list[list[str]] = []
    service = _service(
        provider,
        tracker,
        storage,
        durations=(2.0,),
        ffmpeg_calls=calls,
    )

    outcome = _run(
        service.synthesize(
            TTSRequest(text=TEXT, voice=VOICE),
            settings=AudioSettings(
                trim_start_seconds=0.5,
                trim_end_seconds=2.5,
                fade_in_seconds=0.25,
                fade_out_seconds=0.5,
            ),
        )
    )

    filters = option(calls[0], "-af")
    assert "atrim=start=0.500000:end=2.500000" in filters
    assert "afade=t=in:st=0:d=0.250000" in filters
    assert "afade=t=out:st=1.500000:d=0.500000" in filters
    assert outcome.processed is True
    assert outcome.duration_seconds == pytest.approx(2.0)
    assert outcome.settings.trim_start_seconds == 0.5
    assert outcome.settings.fade_out_seconds == 0.5


def test_settings_reject_an_inverted_trim_range() -> None:
    with pytest.raises(ValueError, match="trim_end_seconds"):
        AudioSettings(trim_start_seconds=2.0, trim_end_seconds=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        AudioSettings(fade_in_seconds=-1.0)


def test_a_mismatched_screening_token_blocks_before_the_provider(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeTTSProvider()
    service = _service(provider, tracker, storage)

    with pytest.raises(ContentPolicyError):
        _run(
            service.synthesize(
                TTSRequest(text=TEXT, voice=VOICE),
                token=str(screen_prompt("A different sentence.")),
            )
        )

    assert provider.calls == []
    assert tracker.total() == 0.0


def test_audio_helpers_measure_trim_fade_and_normalize(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp3"
    source.write_bytes(MP3)
    probe_calls: list[list[str]] = []
    ffmpeg_calls: list[list[str]] = []
    probe = probe_runner([4.0], calls=probe_calls)
    ffmpeg = ffmpeg_writer(calls=ffmpeg_calls)

    assert measure_audio_duration(source, ffprobe_runner=probe) == pytest.approx(4.0)

    trim_audio(
        source,
        tmp_path / "trimmed.mp3",
        start_seconds=1.0,
        end_seconds=3.0,
        ffmpeg_runner=ffmpeg,
    )
    assert "atrim=start=1.000000:end=3.000000" in option(ffmpeg_calls[-1], "-af")

    fade_audio(
        source,
        tmp_path / "faded.mp3",
        fade_in_seconds=0.5,
        fade_out_seconds=1.0,
        duration_seconds=4.0,
        ffmpeg_runner=ffmpeg,
    )
    assert "afade=t=out:st=3.000000:d=1.000000" in option(ffmpeg_calls[-1], "-af")

    normalize_audio(
        source,
        tmp_path / "normalized.mp3",
        target_lufs=-16.0,
        ffmpeg_runner=ffmpeg,
    )
    assert "loudnorm=I=-16.000000" in option(ffmpeg_calls[-1], "-af")

    process_audio(
        source,
        tmp_path / "processed.mp3",
        AudioSettings(
            trim_start_seconds=0.5,
            fade_in_seconds=0.25,
            normalize_loudness=True,
        ),
        duration_seconds=4.0,
        ffmpeg_runner=ffmpeg,
    )
    filters = option(ffmpeg_calls[-1], "-af")
    assert "atrim=start=0.500000" in filters
    assert "afade=t=in:st=0:d=0.250000" in filters
    assert "loudnorm=" in filters
