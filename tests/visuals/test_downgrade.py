"""Visual beats downgrade video -> image -> stock without exceeding the cap."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aivia.core.errors import BudgetError
from aivia.core.models.asset import Asset, AssetKind
from aivia.core.models.script import MediaKind, VisualType
from aivia.core.stages import StageContext
from aivia.providers.protocols.stock import (
    StockAsset,
    StockMediaKind,
    StockRequest,
    StockResponse,
)
from aivia.visuals import (
    DEFAULT_DOWNGRADE_FILENAME,
    DEFAULT_VISUAL_DOWNGRADE_FILENAME,
    VisualDowngradePolicy,
    VisualDowngradeStage,
    VisualDowngradeStageResult,
    resolve_visual_downgrade,
    resolve_visual_downgrade_requests,
)
from aivia.visuals.strategy import VisualRequest


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _request(
    prompt: str,
    *,
    index: int = 0,
    visual_type: VisualType = VisualType.VIDEO,
) -> VisualRequest:
    return VisualRequest(
        index=index,
        visual_type=visual_type,
        kind=(
            MediaKind.VIDEO
            if visual_type is VisualType.VIDEO
            else MediaKind.IMAGE
            if visual_type is VisualType.IMAGE
            else MediaKind.STOCK
        ),
        prompt=prompt,
        query=prompt if visual_type is VisualType.STOCK else "",
        text=prompt,
    )


class FakeVideoService:
    """Return one video asset, or raise the configured budget error."""

    def __init__(self, *, cost: float = 0.5, error: bool = False) -> None:
        self.cost = cost
        self.error = error
        self.requests: list[Any] = []

    async def generate(self, request: Any, **_: Any) -> SimpleNamespace:
        self.requests.append(request)
        if self.error:
            raise BudgetError("video estimate exceeds the remaining budget")
        asset = Asset(
            kind=AssetKind.VIDEO,
            prompt=request.prompt,
            provider="fake-video",
            model="fake-video-model",
            path="file:///tmp/video.mp4",
        )
        return SimpleNamespace(
            asset=asset,
            assets=(asset,),
            provider="fake-video",
            model="fake-video-model",
            cost_usd=self.cost,
            cached=False,
        )


class FakeImageService:
    """Return one image asset and remember every request."""

    def __init__(self, *, cost: float = 0.04) -> None:
        self.cost = cost
        self.requests: list[Any] = []

    async def generate(self, request: Any, **_: Any) -> SimpleNamespace:
        self.requests.append(request)
        asset = Asset(
            kind=AssetKind.IMAGE,
            prompt=request.prompt,
            provider="fake-image",
            model="fake-image-model",
            path="file:///tmp/image.png",
        )
        return SimpleNamespace(
            asset=asset,
            assets=(asset,),
            provider="fake-image",
            model="fake-image-model",
            cost_usd=self.cost,
            cached=False,
        )


class FakeStockSearch:
    """Return one stock image per query and remember every request."""

    def __init__(self, *, cost: float = 0.0) -> None:
        self.cost = cost
        self.requests: list[StockRequest] = []

    async def search(self, request: StockRequest) -> StockResponse:
        self.requests.append(request)
        asset = StockAsset(
            id="stock-1",
            provider="fake-stock",
            kind=StockMediaKind.IMAGE,
            url="https://cdn.example.com/stock-1.jpg",
            width=1920,
            height=1080,
            creator="Synthetic Creator",
            license="CC0",
            attribution="Synthetic Creator / example.com",
        )
        return StockResponse(
            query=request.query,
            assets=[asset],
            provider="fake-stock",
            metadata={"cost_usd": self.cost},
        )


class FakeNotifier:
    """Record every alert instead of printing it."""

    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def send(self, message: Any) -> SimpleNamespace:
        self.messages.append(message)
        return SimpleNamespace(delivered=True)


def test_a_video_beat_downgrades_to_image_when_video_does_not_fit() -> None:
    video = FakeVideoService()
    image = FakeImageService()
    stock = FakeStockSearch()
    notifier = FakeNotifier()

    result = _run(
        resolve_visual_downgrade(
            _request("A lighthouse at sunset"),
            video=video,
            image=image,
            stock=stock,
            cap_usd=0.10,
            estimates={"video": 0.5, "image": 0.04, "stock": 0.0},
            notifier=notifier,
        )
    )

    assert video.requests == []
    assert len(image.requests) == 1
    assert stock.requests == []
    assert result.requested_type is VisualType.VIDEO
    assert result.chosen_type is VisualType.IMAGE
    assert result.downgraded is True
    assert result.resolved is True
    assert result.cost_usd == 0.04
    assert result.decision is not None
    assert result.decision.cap_usd == 0.10
    assert len(result.alerts) == 1
    assert "video -> image" in result.alerts[0].title
    assert len(notifier.messages) == 1
    assert notifier.messages[0].metadata["cap_usd"] == 0.10


def test_a_video_beat_downgrades_to_stock_when_image_does_not_fit() -> None:
    video = FakeVideoService()
    image = FakeImageService()
    stock = FakeStockSearch()
    notifier = FakeNotifier()

    result = _run(
        resolve_visual_downgrade(
            _request("A lighthouse at sunset"),
            video=video,
            image=image,
            stock=stock,
            cap_usd=0.02,
            estimates={"video": 0.5, "image": 0.04, "stock": 0.0},
            notifier=notifier,
        )
    )

    assert video.requests == []
    assert image.requests == []
    assert len(stock.requests) == 1
    assert result.chosen_type is VisualType.STOCK
    assert result.downgraded is True
    assert result.cost_usd == 0.0
    assert result.asset is not None
    assert result.asset.id == "stock-1"
    assert len(notifier.messages) == 1
    assert "video -> stock" in notifier.messages[0].title


def test_a_video_beat_stays_video_when_it_fits_and_emits_no_alert() -> None:
    video = FakeVideoService()
    image = FakeImageService()
    stock = FakeStockSearch()
    notifier = FakeNotifier()

    result = _run(
        resolve_visual_downgrade(
            _request("A lighthouse at sunset"),
            video=video,
            image=image,
            stock=stock,
            cap_usd=1.0,
            estimates={"video": 0.5, "image": 0.04, "stock": 0.0},
            notifier=notifier,
        )
    )

    assert len(video.requests) == 1
    assert image.requests == []
    assert stock.requests == []
    assert result.chosen_type is VisualType.VIDEO
    assert result.downgraded is False
    assert result.cost_usd == 0.5
    assert result.alerts == []
    assert notifier.messages == []


def test_the_stage_shares_one_cap_across_beats_and_never_exceeds_it(
    tmp_path: Path,
) -> None:
    video = FakeVideoService()
    image = FakeImageService()
    stock = FakeStockSearch()
    notifier = FakeNotifier()
    requests = [
        _request("first beat", index=0),
        _request("second beat", index=1),
        _request("third beat", index=2),
    ]
    stage = VisualDowngradeStage(
        requests=requests,
        video=video,
        image=image,
        stock=stock,
        cap_usd=0.06,
        estimates={"video": 0.5, "image": 0.04, "stock": 0.0},
        notifier=notifier,
    )

    result = _run(stage.run(StageContext(run_dir=tmp_path, config={})))

    assert isinstance(result, VisualDowngradeStageResult)
    assert [beat.chosen_type for beat in result.beats] == [
        VisualType.IMAGE,
        VisualType.STOCK,
        VisualType.STOCK,
    ]
    assert len(image.requests) == 1
    assert len(stock.requests) == 2
    assert result.cost_usd == 0.04
    assert result.cost_usd <= 0.06
    assert result.metadata["downgraded"] == 3
    assert len(result.alerts) == 3
    assert len(notifier.messages) == 3

    manifest = tmp_path / DEFAULT_DOWNGRADE_FILENAME
    alias = tmp_path / DEFAULT_VISUAL_DOWNGRADE_FILENAME
    assert manifest.exists()
    assert alias.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["stage"] == "downgrade"
    assert [beat["chosen_type"] for beat in payload["beats"]] == [
        "image",
        "stock",
        "stock",
    ]
    assert payload["metadata"]["cost_usd"] == 0.04
    assert payload["metadata"]["cap_usd"] == 0.06
    assert len(payload["alerts"]) == 3


def test_a_service_budget_error_falls_back_to_the_next_tier() -> None:
    video = FakeVideoService(error=True)
    image = FakeImageService()
    stock = FakeStockSearch()
    notifier = FakeNotifier()

    result = _run(
        resolve_visual_downgrade(
            _request("A lighthouse at sunset"),
            video=video,
            image=image,
            stock=stock,
            cap_usd=0.10,
            estimates={"video": 0.01, "image": 0.04, "stock": 0.0},
            notifier=notifier,
        )
    )

    assert len(video.requests) == 1
    assert len(image.requests) == 1
    assert result.chosen_type is VisualType.IMAGE
    assert result.downgraded is True
    assert result.cost_usd == 0.04
    assert result.attempts[0]["tier"] == "video"
    assert result.attempts[0]["selected"] is False
    assert "budget error" in result.attempts[0]["reason"]
    assert len(notifier.messages) == 1


def test_no_tier_fitting_leaves_the_beat_unresolved_and_alerts() -> None:
    video = FakeVideoService()
    image = FakeImageService()
    stock = FakeStockSearch()
    notifier = FakeNotifier()

    result = _run(
        resolve_visual_downgrade(
            _request("A lighthouse at sunset"),
            video=video,
            image=image,
            stock=stock,
            cap_usd=0.01,
            estimates={"video": 0.5, "image": 0.04, "stock": 0.02},
            notifier=notifier,
        )
    )

    assert video.requests == []
    assert image.requests == []
    assert stock.requests == []
    assert result.chosen_type is None
    assert result.resolved is False
    assert result.cost_usd == 0.0
    assert result.error == "no visual tier could be resolved within the budget"
    assert len(result.alerts) == 1
    assert "none" in result.alerts[0].title


def test_an_image_beat_only_falls_back_to_stock_and_is_never_upgraded() -> None:
    video = FakeVideoService()
    image = FakeImageService()
    stock = FakeStockSearch()
    notifier = FakeNotifier()

    result = _run(
        resolve_visual_downgrade(
            _request("A still frame", visual_type=VisualType.IMAGE),
            video=video,
            image=image,
            stock=stock,
            cap_usd=0.01,
            estimates={"video": 0.5, "image": 0.04, "stock": 0.0},
            notifier=notifier,
        )
    )

    assert video.requests == []
    assert image.requests == []
    assert len(stock.requests) == 1
    assert result.requested_type is VisualType.IMAGE
    assert result.chosen_type is VisualType.STOCK
    assert result.downgraded is True


def test_a_stock_beat_is_never_upgraded() -> None:
    video = FakeVideoService()
    image = FakeImageService()
    stock = FakeStockSearch()
    notifier = FakeNotifier()

    result = _run(
        resolve_visual_downgrade(
            _request("A stock lighthouse", visual_type=VisualType.STOCK),
            video=video,
            image=image,
            stock=stock,
            cap_usd=1.0,
            estimates={"video": 0.5, "image": 0.04, "stock": 0.0},
            notifier=notifier,
        )
    )

    assert video.requests == []
    assert image.requests == []
    assert len(stock.requests) == 1
    assert result.requested_type is VisualType.STOCK
    assert result.chosen_type is VisualType.STOCK
    assert result.downgraded is False
    assert result.alerts == []


def test_the_policy_explains_every_candidate_and_the_chosen_tier() -> None:
    policy = VisualDowngradePolicy(
        cap_usd=0.05,
        estimates={"video": 0.5, "image": 0.04, "stock": 0.0},
    )

    decision = policy.decide(_request("A lighthouse"))

    assert decision.requested_type is VisualType.VIDEO
    assert decision.chosen_type is VisualType.IMAGE
    assert decision.estimate_usd == 0.04
    assert decision.remaining_usd == 0.05
    assert decision.downgraded is True
    assert [item["tier"] for item in decision.candidates] == [
        "video",
        "image",
        "stock",
    ]
    assert [item["affordable"] for item in decision.candidates] == [
        False,
        True,
        True,
    ]
    assert "video" in decision.reason
    assert "image" in decision.reason


def test_the_requests_helper_keeps_one_budget_for_the_sequence() -> None:
    video = FakeVideoService()
    image = FakeImageService()
    stock = FakeStockSearch()
    notifier = FakeNotifier()

    results = _run(
        resolve_visual_downgrade_requests(
            [_request("first", index=0), _request("second", index=1)],
            video=video,
            image=image,
            stock=stock,
            cap_usd=0.06,
            estimates={"video": 0.5, "image": 0.04, "stock": 0.0},
            notifier=notifier,
        )
    )

    assert [result.chosen_type for result in results] == [
        VisualType.IMAGE,
        VisualType.STOCK,
    ]
    assert sum(result.cost_usd for result in results) <= 0.06
    assert len(stock.requests) == 1
