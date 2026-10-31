"""The music service caches, prices and licenses every generated track."""

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
from aivia.core.errors import ContentPolicyError, ProviderExhaustedError
from aivia.core.models import (
    AssetKind,
    AssetLicense,
    ProviderMeta,
    ProviderPrice,
    RateLimit,
)
from aivia.core.router import Router
from aivia.media.music_service import (
    MUSIC_OPERATION,
    MUSIC_UNIT,
    LicenseDecision,
    MusicLicenseError,
    MusicLicenseGate,
    MusicOutcome,
    MusicService,
    check_commercial_license,
    require_commercial_license,
)
from aivia.media.screening import screen_prompt
from aivia.providers import Provider
from aivia.providers.protocols import (
    CAPABILITY_MUSIC_GEN,
    AudioFormat,
    MusicRequest,
    MusicResponse,
    MusicTrack,
)
from aivia.providers.storage_local import LocalStorageProvider

PROMPT = "A warm acoustic theme"
PRICE = 0.02
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00music-track"

COMMERCIAL = AssetLicense(
    name="Test Commercial License",
    url="https://example.com/commercial",
    commercial_use=True,
)
NON_COMMERCIAL = AssetLicense(
    name="Test Non Commercial License",
    url="https://example.com/non-commercial",
    commercial_use=False,
)


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


class FakeMusicProvider(Provider):
    """A music provider returning inline MP3 tracks with configurable rights."""

    def __init__(
        self,
        name: str = "acme-music",
        *,
        price: float = PRICE,
        unit: str = "second",
        license: AssetLicense = COMMERCIAL,
        track_metadata: dict[str, Any] | None = None,
        response_metadata: dict[str, Any] | None = None,
        duration: float = 2.0,
        error: BaseException | None = None,
    ) -> None:
        super().__init__(
            ProviderMeta(
                name=name,
                capabilities=[CAPABILITY_MUSIC_GEN],
                pricing={
                    CAPABILITY_MUSIC_GEN: ProviderPrice(amount_usd=price, unit=unit),
                },
                rate_limit=RateLimit(requests_per_minute=0),
                license=license,
            )
        )
        self.calls: list[MusicRequest] = []
        self.track_metadata = track_metadata if track_metadata is not None else {}
        self.response_metadata = (
            response_metadata if response_metadata is not None else {}
        )
        self.duration = duration
        self.error = error

    async def generate(self, request: MusicRequest) -> MusicResponse:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        tracks = [
            MusicTrack(
                data=MP3,
                format=AudioFormat.MP3,
                duration_seconds=self.duration,
                sample_rate=44100,
                channels=1,
                seed=None if request.seed is None else request.seed + index,
                title=request.prompt,
                lyrics=None if request.instrumental else request.lyrics,
                metadata=dict(self.track_metadata),
            )
            for index in range(request.count)
        ]
        return MusicResponse(
            tracks=tracks,
            model="music-1",
            provider=self.name,
            request_id="req-1",
            metadata=dict(self.response_metadata),
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
    durations: Sequence[float] = (2.0,),
    **values: Any,
) -> MusicService:
    return MusicService(
        _router(provider),
        tracker=tracker,
        storage=storage,
        ffprobe_runner=probe_runner(durations),
        **values,
    )


def _token(prompt: str = PROMPT) -> str:
    return str(screen_prompt(prompt))


def _path(asset_path: str) -> Path:
    parsed = urlsplit(asset_path)
    assert parsed.scheme == "file"
    return Path(unquote(parsed.path))


def test_a_request_is_validated_stored_priced_and_records_its_license(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeMusicProvider(duration=2.0)
    service = _service(provider, tracker, storage, durations=(2.0,))

    outcome = _run(
        service.generate(
            MusicRequest(prompt=PROMPT, duration_seconds=2.0),
            token=_token(),
            run_id="run-1",
            metadata={"stage": "music"},
        )
    )

    assert isinstance(outcome, MusicOutcome)
    assert outcome.cache_hit is False
    assert len(provider.calls) == 1
    assert outcome.provider == "acme-music"
    assert outcome.model == "music-1"
    assert outcome.duration_seconds == pytest.approx(2.0)

    asset = outcome.asset
    assert asset is not None
    assert asset.kind is AssetKind.MUSIC
    assert asset.provider == "acme-music"
    assert asset.model == "music-1"
    assert asset.prompt == PROMPT
    assert asset.run_id == "run-1"
    assert asset.pinned is True
    assert asset.content_hash == outcome.probes[0].sha256
    assert asset.analysis.duration_seconds == pytest.approx(2.0)
    assert _path(asset.path).read_bytes() == MP3
    assert asset.license.name == COMMERCIAL.name
    assert asset.license.url == COMMERCIAL.url
    assert asset.license.commercial_use is True

    event = outcome.event
    assert event is not None
    assert event.provider == "acme-music"
    assert event.operation == MUSIC_OPERATION
    assert event.unit_kind == MUSIC_UNIT
    assert event.units == pytest.approx(2.0)
    assert event.amount_usd == pytest.approx(PRICE * 2)
    assert event.run_id == "run-1"
    assert event.metadata["stage"] == "music"
    assert event.metadata["track_count"] == 1
    assert asset.cost_links[0].cost_event_id == event.id
    assert tracker.total() == pytest.approx(PRICE * 2)

    assert outcome.licenses[0].name == COMMERCIAL.name
    assert outcome.is_commercial is True
    assert outcome.is_commercially_cleared is True
    decision = outcome.require_commercial_license("publish")
    assert isinstance(decision, LicenseDecision)
    assert decision.allowed is True

    sidecar = outcome.sidecar
    assert sidecar is not None and sidecar.exists()
    payload = json.loads(sidecar.read_text("utf-8"))
    assert payload["output"] == str(_path(asset.path))
    assert payload["totals"]["total_usd"] == pytest.approx(PRICE * 2)

    assert service.cache is not None
    entries = service.cache.repository.list()
    assert len(entries) == 1
    assert entries[0].provider == ""
    assert entries[0].model == ""


def test_an_identical_request_is_served_from_cache_without_a_provider_call(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeMusicProvider()
    service = _service(provider, tracker, storage)

    first = _run(
        service.generate(
            MusicRequest(prompt=PROMPT, duration_seconds=2.0),
            token=_token(),
            run_id="run-1",
        )
    )
    second = _run(
        service.generate(
            MusicRequest(prompt=PROMPT, duration_seconds=2.0),
            token=_token(),
            run_id="run-2",
        )
    )

    assert len(provider.calls) == 1
    assert second.cache_hit is True
    assert first.asset is not None and second.asset is not None
    assert second.asset.id == first.asset.id
    assert second.event is not None
    assert second.event.amount_usd == 0.0
    assert second.event.metadata["cache"] == "hit"
    assert second.event.run_id == "run-2"
    assert second.amount_usd == 0.0
    assert tracker.total() == pytest.approx(PRICE * 2)
    assert second.licenses[0].name == COMMERCIAL.name
    assert second.is_commercially_cleared is True
    assert second.sidecar is not None and second.sidecar.exists()

    assert service.cache is not None
    entry = service.cache.entry(
        service.cache_key(MusicRequest(prompt=PROMPT, duration_seconds=2.0))
    )
    assert entry is not None
    assert entry.hit_count == 1


def test_a_non_commercial_track_blocks_publish_and_deliver(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeMusicProvider(license=NON_COMMERCIAL)
    service = _service(provider, tracker, storage)

    outcome = _run(service.generate(MusicRequest(prompt=PROMPT), token=_token()))

    assert outcome.is_commercial is False
    assert outcome.is_commercially_cleared is False
    decision = outcome.check_commercial_license(action="publish")
    assert decision.allowed is False
    assert "commercial_use_not_allowed:0" in decision.blockers

    with pytest.raises(MusicLicenseError) as publish:
        outcome.require_commercial_license("publish")
    assert publish.value.action == "publish"
    assert publish.value.code == "music_license"
    assert publish.value.details["blockers"] == ["commercial_use_not_allowed:0"]

    with pytest.raises(MusicLicenseError) as delivery:
        outcome.require_for_delivery()
    assert delivery.value.action == "deliver"

    with pytest.raises(MusicLicenseError):
        require_commercial_license(outcome, action="deliver")


def test_missing_or_unstated_license_metadata_is_never_treated_as_cleared(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeMusicProvider(license=AssetLicense())
    service = _service(provider, tracker, storage)

    outcome = _run(service.generate(MusicRequest(prompt=PROMPT), token=_token()))

    assert outcome.license is not None
    assert outcome.license.commercial_use is None
    decision = check_commercial_license(outcome)
    assert decision.allowed is False
    assert "license_missing:0" in decision.blockers
    assert "commercial_use_unconfirmed:0" in decision.blockers
    with pytest.raises(MusicLicenseError):
        outcome.require_for_publish()


def test_a_track_license_overrides_the_provider_and_requires_attribution(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    track_license = {
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "commercial_use": True,
        "attribution_required": True,
        "attribution": "Music by Example",
    }
    provider = FakeMusicProvider(track_metadata=track_license)
    service = _service(provider, tracker, storage)

    outcome = _run(service.generate(MusicRequest(prompt=PROMPT), token=_token()))

    assert outcome.license is not None
    assert outcome.license.name == "CC BY 4.0"
    assert outcome.license.commercial_use is True
    assert outcome.requires_attribution is True
    assert outcome.attribution == ("Music by Example",)
    decision = outcome.require_for_publish()
    assert decision.allowed is True
    assert decision.attribution == ("Music by Example",)

    strict = MusicLicenseGate(require_attribution=True)
    assert strict.check(outcome).allowed is True

    missing_credit = FakeMusicProvider(
        track_metadata={
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "commercial_use": True,
            "attribution_required": True,
        }
    )
    blocked = _run(
        _service(missing_credit, tracker, storage).generate(
            MusicRequest(prompt="A different theme"),
            token=_token("A different theme"),
        )
    )
    assert strict.check(blocked).allowed is False
    assert "attribution_missing:0" in strict.check(blocked).blockers


def test_count_stores_one_asset_and_cache_entry_per_track(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeMusicProvider(price=0.05, unit="track")
    service = _service(provider, tracker, storage, durations=(3.0, 3.0))
    request = MusicRequest(prompt=PROMPT, count=2, duration_seconds=3.0)

    first = _run(service.generate(request, token=_token(), run_id="run-1"))

    assert len(provider.calls) == 1
    assert len(first.assets) == 2
    assert len(first.storage_objects) == 2
    assert len({asset.id for asset in first.assets}) == 2
    assert first.event is not None
    assert first.event.unit_kind == "track"
    assert first.event.units == pytest.approx(2.0)
    assert first.event.amount_usd == pytest.approx(0.10)
    assert first.licenses and all(item.is_commercial for item in first.licenses)
    assert service.cache is not None
    assert len(service.cache.repository.list()) == 2

    second = _run(service.generate(request, token=_token(), run_id="run-2"))

    assert len(provider.calls) == 1
    assert second.cache_hit is True
    assert second.response.count == 2
    assert [asset.id for asset in second.assets] == [asset.id for asset in first.assets]
    assert len(second.events) == 2
    assert second.amount_usd == 0.0
    assert tracker.total() == pytest.approx(0.10)


def test_a_missing_screening_token_blocks_before_any_paid_step(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeMusicProvider()
    service = _service(provider, tracker, storage)

    with pytest.raises(ContentPolicyError) as missing:
        _run(service.generate(MusicRequest(prompt=PROMPT)))

    assert missing.value.details["reason"] == "missing_token"
    assert provider.calls == []
    assert tracker.count() == 0
    assert service.cache is not None
    assert service.cache.repository.list() == []


def test_an_over_budget_call_never_reaches_the_provider(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeMusicProvider(price=1.0, unit="second")
    service = _service(provider, tracker, storage)

    with pytest.raises(ProviderExhaustedError) as caught:
        _run(
            service.generate(
                MusicRequest(prompt=PROMPT, duration_seconds=10.0),
                token=_token(),
                remaining_budget=0.5,
            )
        )

    assert provider.calls == []
    assert caught.value.details["skipped"] == [
        {"provider": "acme-music", "reason": "budget_exceeded"}
    ]
    assert tracker.count() == 0


def test_no_cache_forces_every_request_through_the_provider(
    tracker: CostTracker,
    storage: LocalStorageProvider,
) -> None:
    provider = FakeMusicProvider()
    service = _service(provider, tracker, storage, no_cache=True)

    _run(service.generate(MusicRequest(prompt=PROMPT), token=_token()))
    _run(service.generate(MusicRequest(prompt=PROMPT), token=_token()))

    assert len(provider.calls) == 2
    assert service.cache is not None
    assert service.cache.repository.list() == []
    assert tracker.total() == pytest.approx(PRICE * 4)
