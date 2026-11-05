"""Image beats resolve through the media service with their provenance kept."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.models.asset import Asset, AssetKind, compute_prompt_hash
from aivia.core.models.script import MediaKind, VisualType
from aivia.core.router import Router
from aivia.core.stages import StageContext
from aivia.providers.mock import MockImageProvider
from aivia.providers.protocols.image import ImageRequest
from aivia.visuals import (
    DEFAULT_IMAGE_FILENAME,
    ImageStage,
    ImageStageResult,
    resolve_image_request,
)
from aivia.visuals.strategy import VisualRequest


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _request(
    prompt: str,
    *,
    index: int = 0,
    visual_type: VisualType = VisualType.IMAGE,
    metadata: dict[str, Any] | None = None,
    **values: Any,
) -> VisualRequest:
    payload: dict[str, Any] = {
        "index": index,
        "visual_type": visual_type,
        "kind": (
            MediaKind.IMAGE if visual_type is VisualType.IMAGE else MediaKind.STOCK
        ),
        "prompt": prompt,
        "query": prompt if visual_type is VisualType.STOCK else "",
        "metadata": metadata or {},
        **values,
    }
    return VisualRequest.model_validate(payload)


class FakeImageService:
    """Return one stored asset per request and remember every typed request."""

    def __init__(self) -> None:
        self.requests: list[ImageRequest] = []
        self.runs: list[tuple[str, dict[str, Any] | None]] = []

    async def generate(
        self,
        request: ImageRequest,
        *,
        run_id: str = "",
        metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        self.requests.append(request)
        self.runs.append((run_id, metadata))
        asset = Asset(
            kind=AssetKind.IMAGE,
            prompt=request.prompt,
            provider="fake-image",
            model=request.model or "fake-model",
            path=f"file:///tmp/{request.prompt.replace(' ', '-')}.png",
            run_id=run_id,
            seed=request.seed,
        )
        return SimpleNamespace(
            asset=asset,
            assets=(asset,),
            provider="fake-image",
            model=asset.model,
            cost_usd=0.04,
            cached=False,
        )


def test_an_image_beat_is_sent_through_the_service_with_its_prompt_options() -> None:
    service = FakeImageService()
    request = _request(
        "A lighthouse at sunset",
        index=3,
        negative_prompt="no text",
        style="documentary",
        aspect_ratio="9:16",
        seed=7,
        metadata={"model": "beat-model"},
    )

    result = _run(
        resolve_image_request(
            request,
            service=service,
            run_id="run_1",
            metadata={"position": 3},
        )
    )

    assert len(service.requests) == 1
    sent = service.requests[0]
    assert sent.prompt == "A lighthouse at sunset"
    assert sent.negative_prompt == "no text"
    assert sent.style == "documentary"
    assert sent.aspect_ratio == "9:16"
    assert sent.seed == 7
    assert sent.model == "beat-model"
    assert sent.count == 1
    assert sent.metadata["stage"] == "image"
    assert sent.metadata["beat_index"] == 3
    assert service.runs[0][0] == "run_1"

    assert result.resolved is True
    assert result.asset is not None
    assert result.asset.prompt == "A lighthouse at sunset"
    assert result.asset.prompt_hash == compute_prompt_hash("A lighthouse at sunset")
    assert result.asset.provider == "fake-image"
    assert result.asset.model == "beat-model"
    assert result.prompt_hash == result.asset.prompt_hash
    assert result.provider == "fake-image"
    assert result.model == "beat-model"
    assert result.cost_usd == 0.04


def test_the_stage_resolves_only_image_beats_and_writes_the_manifest(
    tmp_path: Path,
) -> None:
    service = FakeImageService()
    requests = [
        _request("A generated lighthouse", index=0, aspect_ratio="16:9"),
        _request("a stock query", index=1, visual_type=VisualType.STOCK),
        _request("A generated bee", index=2, aspect_ratio="16:9"),
    ]
    stage = ImageStage(requests=requests, service=service)
    context = StageContext(run_dir=tmp_path, config={})

    result = _run(stage.run(context))

    assert isinstance(result, ImageStageResult)
    assert [beat.index for beat in result.beats] == [0, 2]
    assert [request.prompt for request in service.requests] == [
        "A generated lighthouse",
        "A generated bee",
    ]
    assert [asset.prompt_hash for asset in result.assets] == [
        compute_prompt_hash("A generated lighthouse"),
        compute_prompt_hash("A generated bee"),
    ]
    assert result.metadata["beats"] == 2
    assert result.metadata["resolved"] == 2
    assert result.metadata["unresolved"] == 0
    assert result.metadata["providers"] == ["fake-image"]
    assert result.metadata["models"] == ["fake-model"]
    assert result.cost_usd == 0.08

    manifest = tmp_path / DEFAULT_IMAGE_FILENAME
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["stage"] == "image"
    assert [beat["index"] for beat in payload["beats"]] == [0, 2]
    first = payload["beats"][0]
    assert first["prompt_hash"] == compute_prompt_hash("A generated lighthouse")
    assert first["provider"] == "fake-image"
    assert first["model"] == "fake-model"
    assert first["asset"]["prompt_hash"] == first["prompt_hash"]
    assert first["asset"]["provider"] == "fake-image"
    assert first["asset"]["model"] == "fake-model"


def test_an_empty_prompt_is_unresolved_without_calling_the_service(
    tmp_path: Path,
) -> None:
    service = FakeImageService()
    stage = ImageStage(
        requests=[_request("", index=0)],
        service=service,
    )
    context = StageContext(run_dir=tmp_path, config={})

    result = _run(stage.run(context))

    assert isinstance(result, ImageStageResult)
    assert service.requests == []
    assert result.beats[0].resolved is False
    assert result.beats[0].error == ""
    assert result.beats[0].metadata["reason"] == "no prompt"
    assert result.unresolved == result.beats
    assert result.metadata["unresolved"] == 1
    assert (tmp_path / DEFAULT_IMAGE_FILENAME).exists()


def test_the_default_path_uses_the_stage_context_media_service(
    tmp_path: Path,
) -> None:
    provider = MockImageProvider(name="mock-image")
    router = Router(
        {provider.name: provider},
        config=CapabilityChainConfig(chain=[provider.name]),
    )
    stage = ImageStage(requests=[_request("A lighthouse at sunset")])
    context = StageContext(run_dir=tmp_path, config={}, router=router)

    result = _run(stage.run(context))

    assert isinstance(result, ImageStageResult)
    assert result.beats[0].resolved is True
    assert result.assets
    asset = result.assets[0]
    assert asset.prompt_hash == compute_prompt_hash("A lighthouse at sunset")
    assert asset.provider == "mock-image"
    assert asset.model == "mock-image"
    assert result.beats[0].provider == "mock-image"
    assert result.beats[0].model == "mock-image"
    assert result.cost_usd > 0.0
