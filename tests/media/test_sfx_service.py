"""The sound effect service stores every cost sidecar, locally or remotely."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import pytest
from sqlalchemy import Engine

from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.models import ProviderMeta, ProviderPrice, RateLimit
from aivia.core.router import Router
from aivia.core.sidecar import SIDECAR_SUFFIX
from aivia.media.screening import screen_prompt
from aivia.media.sfx_service import (
    SFX_OPERATION,
    SFX_UNIT,
    SFXOutcome,
    SFXService,
)
from aivia.providers import Provider
from aivia.providers.protocols import (
    CAPABILITY_SFX,
    SFXClip,
    SFXRequest,
    SFXResponse,
)
from aivia.providers.protocols.storage import (
    CAPABILITY_STORAGE,
    StorageObject,
    StorageProvider,
    StoragePutRequest,
)
from aivia.providers.storage_local import LocalStorageProvider

PROMPT = "A heavy wooden door creaks open"
PRICE = 0.04
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00sound-effect"


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
def local_storage(tmp_path: Path) -> LocalStorageProvider:
    return LocalStorageProvider(tmp_path / "media")


@pytest.fixture(autouse=True)
def fake_ffprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    from aivia.core.media_probe import FFprobeInfo, FFprobeStream

    def run_ffprobe(
        path: str | Path,
        *,
        executable: str = "ffprobe",
        timeout_seconds: float = 30.0,
        runner: Any = None,
    ) -> FFprobeInfo:
        return FFprobeInfo(
            format_name="mp3",
            format_long_name="MP2/3 (MPEG audio layer 2/3)",
            duration_seconds=1.5,
            size_bytes=Path(path).stat().st_size,
            streams=(
                FFprobeStream(
                    index=0,
                    codec_type="audio",
                    codec_name="mp3",
                    duration_seconds=1.5,
                    sample_rate=44_100,
                    channels=1,
                ),
            ),
        )

    monkeypatch.setattr("aivia.core.media_probe.run_ffprobe", run_ffprobe)


class FakeRemoteStorage(Provider):
    """An in-memory storage provider that returns HTTP URLs like a remote."""

    default_metadata = ProviderMeta(
        name="remote-storage",
        capabilities=[CAPABILITY_STORAGE],
    )

    def __init__(self) -> None:
        super().__init__()
        self.objects: dict[str, bytes] = {}
        self.puts: list[StoragePutRequest] = []

    async def put(self, request: StoragePutRequest) -> StorageObject:
        self.puts.append(request)
        digest = hashlib.sha256(request.data).hexdigest()
        key = request.key or f"objects/{digest[:2]}/{digest}"
        existing = self.objects.get(key)
        if existing is not None and existing != request.data and not request.overwrite:
            raise ValueError(f"storage key already exists: {key}")
        self.objects[key] = request.data
        return StorageObject(
            key=key,
            size_bytes=len(request.data),
            content_type=request.content_type,
            sha256=digest,
            url=self.url(key),
            created_at=datetime.now(UTC),
        )

    async def get(self, key: str) -> bytes:
        return self.objects[key]

    async def stat(self, key: str) -> StorageObject | None:
        if key not in self.objects:
            return None
        return StorageObject(
            key=key,
            size_bytes=len(self.objects[key]),
            url=self.url(key),
        )

    async def delete(self, key: str) -> bool:
        return self.objects.pop(key, None) is not None

    def url(self, key: str, *, expires_seconds: int | None = None) -> str:
        base = f"https://storage.example.com/{key}"
        if expires_seconds is not None:
            return f"{base}?expires={expires_seconds}"
        return base


class FakeSFXProvider(Provider):
    """A sound effect provider returning distinct inline MP3 clips."""

    def __init__(
        self,
        name: str = "acme-sfx",
        *,
        price: float = PRICE,
        unit: str = "clip",
        duration: float = 1.5,
        error: BaseException | None = None,
    ) -> None:
        super().__init__(
            ProviderMeta(
                name=name,
                capabilities=[CAPABILITY_SFX],
                pricing={
                    CAPABILITY_SFX: ProviderPrice(amount_usd=price, unit=unit),
                },
                rate_limit=RateLimit(requests_per_minute=0),
            )
        )
        self.calls: list[SFXRequest] = []
        self.duration = duration
        self.error = error

    async def generate(self, request: SFXRequest) -> SFXResponse:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        clips = [
            SFXClip(
                data=MP3 + bytes([index]),
                format=request.format,
                duration_seconds=self.duration,
                sample_rate=44_100,
                loop=request.loop,
                seed=request.seed,
            )
            for index in range(request.count)
        ]
        return SFXResponse(
            clips=clips,
            model="sfx-1",
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
    storage: StorageProvider | None,
    **values: Any,
) -> SFXService:
    return SFXService(
        _router(provider),
        tracker=tracker,
        storage=storage,
        **values,
    )


def _token(prompt: str = PROMPT) -> str:
    return str(screen_prompt(prompt))


def _path(asset_path: str) -> Path:
    parsed = urlsplit(asset_path)
    assert parsed.scheme == "file"
    return Path(unquote(parsed.path))


def test_a_remote_output_stores_its_cost_sidecar_through_storage(
    tracker: CostTracker,
) -> None:
    provider = FakeSFXProvider()
    storage = FakeRemoteStorage()
    service = _service(provider, tracker, storage)

    outcome = _run(
        service.generate(
            SFXRequest(prompt=PROMPT, duration_seconds=1.5),
            token=_token(),
            run_id="run-1",
        )
    )

    assert isinstance(outcome, SFXOutcome)
    assert outcome.sidecars == ()
    asset = outcome.asset
    assert asset is not None
    assert urlsplit(asset.path).scheme == "https"
    stored = outcome.storage_objects[0]
    sidecar_key = f"{stored.key}{SIDECAR_SUFFIX}"
    assert sidecar_key in storage.objects

    sidecar_puts = [
        put
        for put in storage.puts
        if put.key is not None and put.key.endswith(SIDECAR_SUFFIX)
    ]
    assert len(sidecar_puts) == 1
    assert sidecar_puts[0].key == sidecar_key
    assert sidecar_puts[0].content_type == "application/json"
    assert sidecar_puts[0].overwrite is True

    payload = json.loads(storage.objects[sidecar_key].decode("utf-8"))
    assert payload["output"] == asset.path
    assert payload["run_id"] == "run-1"
    assert payload["totals"]["total_usd"] == pytest.approx(PRICE)
    assert payload["totals"]["count"] == 1
    assert payload["items"][0]["operation"] == SFX_OPERATION
    assert payload["items"][0]["unit_kind"] == SFX_UNIT
    assert payload["items"][0]["amount_usd"] == pytest.approx(PRICE)
    assert payload["items"][0]["provider"] == "acme-sfx"
    assert payload["items"][0]["model"] == "sfx-1"


def test_every_remote_clip_gets_its_own_cost_sidecar(tracker: CostTracker) -> None:
    provider = FakeSFXProvider()
    storage = FakeRemoteStorage()
    service = _service(provider, tracker, storage)

    outcome = _run(
        service.generate(
            SFXRequest(prompt=PROMPT, duration_seconds=1.5, count=2),
            token=_token(),
            run_id="run-2",
        )
    )

    assert len(outcome.assets) == 2
    assert outcome.amount_usd == pytest.approx(PRICE * 2)
    keys = {f"{stored.key}{SIDECAR_SUFFIX}" for stored in outcome.storage_objects}
    assert len(keys) == 2
    assert keys <= storage.objects.keys()
    for key in keys:
        payload = json.loads(storage.objects[key].decode("utf-8"))
        assert payload["output"].startswith("https://storage.example.com/")
        assert payload["run_id"] == "run-2"
        assert payload["totals"]["total_usd"] == pytest.approx(PRICE * 2)


def test_a_local_output_still_writes_the_cost_sidecar_next_to_the_file(
    tracker: CostTracker,
    local_storage: LocalStorageProvider,
) -> None:
    provider = FakeSFXProvider()
    service = _service(provider, tracker, local_storage)

    outcome = _run(
        service.generate(
            SFXRequest(prompt=PROMPT, duration_seconds=1.5),
            token=_token(),
            run_id="run-3",
        )
    )

    asset = outcome.asset
    assert asset is not None
    path = _path(asset.path)
    sidecar = path.with_name(path.name + SIDECAR_SUFFIX)
    assert outcome.sidecar == sidecar
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text("utf-8"))
    assert payload["output"] == str(path)
    assert payload["run_id"] == "run-3"
    assert payload["totals"]["total_usd"] == pytest.approx(PRICE)
    assert payload["items"][0]["operation"] == SFX_OPERATION
