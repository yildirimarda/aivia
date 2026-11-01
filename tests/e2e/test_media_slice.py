"""An end-to-end media slice reuses every cached artifact on a repeat run.

The slice drives the facade the CLI, API and pipeline stages share. It creates
real assets, cache entries, cost events and sidecars in a file database and a
local storage directory, then repeats the same requests under a second run id.
The second run must call no provider, record zero spend and leave every total
(result, ledger, sidecars and metrics) unchanged.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy import Engine

from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.cost import CostTracker
from aivia.core.db import RunRepository, create_database_engine, upgrade_database
from aivia.core.models import Run, RunKind
from aivia.core.router import Router
from aivia.media import MediaResult, MediaServiceFacade
from aivia.providers.mock import MockProvider
from aivia.providers.protocols.image_edit import ImageEditOperation
from aivia.providers.storage_local import LocalStorageProvider

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def mp4() -> bytes:
    return b"\x00\x00\x00\x18ftypisom" + b"\x00" * 12


def probe_json(*, video: bool, duration: float = 1.0) -> str:
    if video:
        streams: list[dict[str, Any]] = [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 512,
                "height": 512,
                "duration": f"{duration:.3f}",
                "avg_frame_rate": "24/1",
            }
        ]
        format_name = "mov,mp4,m4a,3gp,3g2,mj2"
    else:
        streams = [
            {
                "index": 0,
                "codec_type": "audio",
                "codec_name": "pcm_s16le",
                "duration": f"{duration:.3f}",
                "sample_rate": "8000",
                "channels": 1,
            }
        ]
        format_name = "wav"
    return json.dumps(
        {
            "streams": streams,
            "format": {
                "format_name": format_name,
                "format_long_name": format_name,
                "duration": f"{duration:.3f}",
                "size": "12345",
            },
        }
    )


def probe_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    target = Path(argv[-1])
    video = target.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv", ".gif"}
    return subprocess.CompletedProcess(
        list(argv),
        0,
        stdout=probe_json(video=video),
        stderr="",
    )


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    created = create_database_engine(path=tmp_path / "data" / "aivia.db")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def provider() -> MockProvider:
    return MockProvider(name="mock", ffmpeg_runner=lambda _argv: mp4())


@pytest.fixture
def facade(
    provider: MockProvider,
    engine: Engine,
    tmp_path: Path,
) -> MediaServiceFacade:
    router = Router(
        {provider.name: provider},
        config=CapabilityChainConfig(chain=[provider.name]),
    )
    return MediaServiceFacade(
        router,
        tracker=CostTracker(engine),
        engine=engine,
        storage=LocalStorageProvider(tmp_path / "media"),
        metrics_path=tmp_path / "data" / "media-metrics.json",
        ffprobe_runner=probe_runner,
    )


def _media_slice(facade: MediaServiceFacade, run_id: str) -> list[MediaResult]:
    return [
        _run(facade.image("A lighthouse at sunset", run_id=run_id)),
        _run(
            facade.video(
                "A storm over the sea",
                duration_seconds=1.0,
                run_id=run_id,
            )
        ),
        _run(
            facade.speech(
                "A lighthouse in a storm",
                voice="narrator",
                format="wav",
                run_id=run_id,
            )
        ),
        _run(
            facade.music(
                "Warm ambient piano",
                instrumental=True,
                duration_seconds=1.0,
                format="wav",
                run_id=run_id,
            )
        ),
        _run(
            facade.edit(
                PNG_1X1,
                operation=ImageEditOperation.UPSCALE,
                scale=2.0,
                run_id=run_id,
            )
        ),
    ]


def _sidecar_total(result: MediaResult) -> float:
    totals: list[float] = []
    for path in result.sidecars:
        payload = json.loads(path.read_text(encoding="utf-8"))
        totals.append(float(payload["totals"]["total_usd"]))
    return math.fsum(totals)


def _metrics(facade: MediaServiceFacade) -> dict[str, Any]:
    assert facade.metrics_path is not None
    return cast(
        dict[str, Any],
        json.loads(facade.metrics_path.read_text(encoding="utf-8")),
    )


def _create_run(engine: Engine, run_id: str) -> None:
    RunRepository(engine).create(
        Run(id=run_id, kind=RunKind.MEDIA, profile="media-slice")
    )


def test_a_second_identical_media_run_costs_nothing_and_totals_match(
    facade: MediaServiceFacade,
    provider: MockProvider,
    engine: Engine,
) -> None:
    first_run = "run_media_slice_first"
    second_run = "run_media_slice_second"
    _create_run(engine, first_run)
    _create_run(engine, second_run)
    tracker = CostTracker(engine)

    first = _media_slice(facade, first_run)
    first_total = math.fsum(result.cost_usd for result in first)
    calls_after_first = len(provider.calls)

    assert all(result.cached is False for result in first)
    assert all(result.cost_usd > 0 for result in first)
    assert first_total == pytest.approx(tracker.total(run_id=first_run))
    assert first_total == pytest.approx(math.fsum(map(_sidecar_total, first)))
    assert _metrics(facade)["cost_usd"] == pytest.approx(first_total)

    second = _media_slice(facade, second_run)

    assert len(provider.calls) == calls_after_first
    assert all(result.cached is True for result in second)
    assert all(result.cost_usd == pytest.approx(0.0) for result in second)
    assert [result.paths for result in second] == [result.paths for result in first]
    for result in second:
        assert result.events
        assert all(event.amount_usd == 0.0 for event in result.events)
        assert all(event.metadata.get("cache") == "hit" for event in result.events)
        assert _sidecar_total(result) == pytest.approx(0.0)

    assert tracker.total(run_id=second_run) == pytest.approx(0.0)
    assert tracker.totals(run_id=second_run)["count"] == len(second)
    assert tracker.total(run_id=first_run) == pytest.approx(first_total)
    assert tracker.total() == pytest.approx(first_total)
    assert tracker.total() == pytest.approx(
        math.fsum(result.cost_usd for result in first)
    )

    document = _metrics(facade)
    assert document["cost_usd"] == pytest.approx(first_total)
    assert document["calls"] == len(first) + len(second)
    assert document["cache_hits"] == len(second)
    assert document["kinds"]["image"]["cost_usd"] == pytest.approx(first[0].cost_usd)
    assert document["kinds"]["video"]["cost_usd"] == pytest.approx(first[1].cost_usd)
    assert document["kinds"]["speech"]["cost_usd"] == pytest.approx(first[2].cost_usd)
    assert document["kinds"]["music"]["cost_usd"] == pytest.approx(first[3].cost_usd)
    assert document["kinds"]["edit"]["cost_usd"] == pytest.approx(first[4].cost_usd)
