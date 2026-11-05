"""Visuals are generated once per run and shared by every language."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from aivia.core.models.asset import Asset, AssetKind
from aivia.core.models.script import (
    Beat,
    MediaKind,
    MediaPrompt,
    Script,
    VisualType,
)
from aivia.core.stages import StageContext
from aivia.providers.protocols.stock import (
    StockAsset,
    StockMediaKind,
    StockRequest,
    StockResponse,
)
from aivia.visuals import (
    DEFAULT_SHARED_VISUALS_FILENAME,
    DEFAULT_VISUAL_REUSE_FILENAME,
    SharedVisuals,
    VisualReuseStage,
    VisualReuseStageResult,
    VisualReuseStore,
    resolve_visual_reuse,
    share_visual_requests,
    shared_visual_requests,
)
from aivia.visuals.strategy import VisualRequest


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _script(language: str, prompts: list[str]) -> Script:
    return Script(
        language=language,
        title=f"{language} title",
        beats=[
            Beat(
                index=index,
                text=f"{language} beat {index}",
                media_prompts=[MediaPrompt(kind=MediaKind.IMAGE, prompt=prompt)],
            )
            for index, prompt in enumerate(prompts)
        ],
    )


def _request(
    prompt: str,
    *,
    index: int = 0,
    visual_type: VisualType = VisualType.IMAGE,
) -> VisualRequest:
    return VisualRequest(
        index=index,
        visual_type=visual_type,
        kind=(
            MediaKind.VIDEO
            if visual_type is VisualType.VIDEO
            else MediaKind.STOCK
            if visual_type is VisualType.STOCK
            else MediaKind.IMAGE
        ),
        prompt=prompt,
        query=prompt if visual_type is VisualType.STOCK else "",
        text=prompt,
    )


class FakeImageService:
    """Return one image asset per request and count every call."""

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
            path=f"file:///tmp/{len(self.requests)}.png",
        )
        return SimpleNamespace(
            asset=asset,
            assets=(asset,),
            provider="fake-image",
            model="fake-image-model",
            cost_usd=self.cost,
            cached=False,
        )


class FakeVideoService:
    """Return one video asset per request and count every call."""

    def __init__(self, *, cost: float = 0.5) -> None:
        self.cost = cost
        self.requests: list[Any] = []

    async def generate(self, request: Any, **_: Any) -> SimpleNamespace:
        self.requests.append(request)
        asset = Asset(
            kind=AssetKind.VIDEO,
            prompt=request.prompt,
            provider="fake-video",
            model="fake-video-model",
            path=f"file:///tmp/{len(self.requests)}.mp4",
        )
        return SimpleNamespace(
            asset=asset,
            assets=(asset,),
            provider="fake-video",
            model="fake-video-model",
            cost_usd=self.cost,
            cached=False,
        )


class FakeStockSearch:
    """Return one stock asset per query and count every call."""

    def __init__(self) -> None:
        self.requests: list[StockRequest] = []

    async def search(self, request: StockRequest) -> StockResponse:
        self.requests.append(request)
        asset = StockAsset(
            id=f"stock-{len(self.requests)}",
            provider="fake-stock",
            kind=StockMediaKind.IMAGE,
            url="https://cdn.example.com/stock.jpg",
            width=1920,
            height=1080,
            creator="Synthetic Creator",
            license="CC0",
            attribution="Synthetic Creator / example.com",
        )
        return StockResponse(query=request.query, assets=[asset], provider="fake-stock")


def test_shared_requests_come_from_the_source_language_only() -> None:
    scripts = {
        "en": _script("en", ["English lighthouse", "English harbour"]),
        "de": _script("de", ["Deutscher Leuchtturm", "Deutscher Hafen"]),
    }

    requests = shared_visual_requests(
        scripts,
        source_language="en",
        strategy="image",
    )
    shared = share_visual_requests(
        scripts,
        source_language="en",
        strategy="image",
    )

    assert [request.prompt for request in requests] == [
        "English lighthouse",
        "English harbour",
    ]
    assert shared["en"] == requests
    assert shared["de"] == requests
    assert shared["de"][0] is shared["en"][0]


def test_the_stage_generates_each_beat_once_for_every_language(
    tmp_path: Path,
) -> None:
    service = FakeImageService()
    stage = VisualReuseStage(
        scripts={
            "en": _script("en", ["English lighthouse", "English harbour"]),
            "de": _script("de", ["Deutscher Leuchtturm", "Deutscher Hafen"]),
        },
        source_language="en",
        strategy="image",
        image=service,
    )
    context = StageContext(run_dir=tmp_path, config={})

    result = _run(stage.run(context))

    assert isinstance(result, VisualReuseStageResult)
    assert [request.prompt for request in service.requests] == [
        "English lighthouse",
        "English harbour",
    ]
    assert result.languages == ["en", "de"]
    assert result.source_language == "en"
    assert result.total_cost_usd == 0.08
    assert result.cost_usd == 0.08
    assert result.metadata["reused_across"] == 2
    assert [asset.prompt for asset in result.assets] == [
        "English lighthouse",
        "English harbour",
    ]
    assert result.by_language["de"] == result.beats
    assert result.by_language["en"] == result.beats

    manifest = tmp_path / DEFAULT_VISUAL_REUSE_FILENAME
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["source_language"] == "en"
    assert payload["languages"] == ["en", "de"]
    assert [beat["request"]["prompt"] for beat in payload["beats"]] == [
        "English lighthouse",
        "English harbour",
    ]
    assert all(beat["languages"] == ["en", "de"] for beat in payload["beats"])
    assert payload["metadata"]["cost_usd"] == 0.08
    assert (tmp_path / DEFAULT_SHARED_VISUALS_FILENAME).exists()


def test_a_second_language_reuses_the_generated_visuals_without_a_new_call() -> None:
    service = FakeImageService()
    store = VisualReuseStore(
        image=service,
        source_language="en",
        strategy="image",
    )

    first = _run(
        store.resolve(
            {"en": _script("en", ["English lighthouse"])},
            language="en",
        )
    )
    second = _run(
        store.resolve(
            {"de": _script("de", ["Deutscher Leuchtturm"])},
            language="de",
        )
    )

    assert len(service.requests) == 1
    assert [beat.asset for beat in first] == [beat.asset for beat in second]
    assert [beat.request for beat in first] == [beat.request for beat in second]
    assert store.shared is not None
    assert store.shared.languages == ["en", "de"]
    assert store.shared.total_cost_usd == 0.04
    assert store.shared.by_language["de"] == store.shared.beats


def test_existing_results_can_be_shared_without_touching_a_provider(
    tmp_path: Path,
) -> None:
    first = FakeImageService()
    scripts = {
        "en": _script("en", ["English lighthouse"]),
        "de": _script("de", ["Deutscher Leuchtturm"]),
    }
    generated = _run(
        resolve_visual_reuse(
            scripts,
            source_language="en",
            strategy="image",
            image=first,
        )
    )

    class ExplodingService:
        async def generate(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("a reused visual must not call the provider")

    stage = VisualReuseStage(
        scripts=scripts,
        source_language="en",
        results=generated,
        image=ExplodingService(),
    )
    result = _run(stage.run(StageContext(run_dir=tmp_path, config={})))

    assert isinstance(result, VisualReuseStageResult)
    assert len(first.requests) == 1
    assert result.total_cost_usd == 0.04
    assert [asset.path for asset in result.assets] == [generated.assets[0].path]


def test_mixed_visuals_are_each_generated_once_and_shared(
    tmp_path: Path,
) -> None:
    image = FakeImageService()
    video = FakeVideoService()
    stock = FakeStockSearch()
    requests = [
        _request("A lighthouse", index=0, visual_type=VisualType.IMAGE),
        _request("A stormy sea", index=1, visual_type=VisualType.VIDEO),
        _request("harbour at night", index=2, visual_type=VisualType.STOCK),
    ]
    stage = VisualReuseStage(
        requests=requests,
        languages=["en", "de", "fr"],
        image=image,
        video=video,
        stock=stock,
    )

    result = cast(
        VisualReuseStageResult,
        _run(stage.run(StageContext(run_dir=tmp_path, config={}))),
    )

    assert len(image.requests) == 1
    assert len(video.requests) == 1
    assert len(stock.requests) == 1
    assert result.languages == ["en", "de", "fr"]
    assert result.total_cost_usd == 0.54
    assert result.metadata["reused_across"] == 3
    assert [beat.index for beat in result.beats] == [0, 1, 2]
    assert all(len(beat.languages) == 3 for beat in result.beats)


def test_the_stage_loads_per_language_scripts_from_the_run_directory(
    tmp_path: Path,
) -> None:
    service = FakeImageService()
    for code, prompt in (
        ("en", "English lighthouse"),
        ("de", "Deutscher Leuchtturm"),
    ):
        (tmp_path / f"script.{code}.json").write_text(
            _script(code, [prompt]).model_dump_json(),
            encoding="utf-8",
        )
    stage = VisualReuseStage(strategy="image", image=service)

    config = {"languages": {"default": "en", "languages": {"en": {}, "de": {}}}}
    result = cast(
        VisualReuseStageResult,
        _run(stage.run(StageContext(run_dir=tmp_path, config=config))),
    )

    assert isinstance(result, VisualReuseStageResult)
    assert result.source_language == "en"
    assert result.languages == ["de", "en"]
    assert [request.prompt for request in service.requests] == ["English lighthouse"]
    assert len(result.assets) == 1
    assert result.by_language["de"] == result.beats


def test_share_visuals_accepts_a_shared_result_as_is() -> None:
    shared = SharedVisuals(
        source_language="en",
        languages=["en", "de"],
        beats=[],
    )
    assert shared.for_language("de") == []
    assert shared.get_language("fr") is None
    assert shared.add_language("fr").languages == ["en", "de", "fr"]


def test_different_beat_counts_share_the_longest_script(
    tmp_path: Path,
) -> None:
    service = FakeImageService()
    stage = VisualReuseStage(
        scripts={
            "en": _script("en", ["English lighthouse", "English harbour"]),
            "de": _script(
                "de",
                ["Deutscher Leuchtturm", "Deutscher Hafen", "Deutscher Markt"],
            ),
        },
        source_language="en",
        strategy="image",
        image=service,
    )

    result = cast(
        VisualReuseStageResult,
        _run(stage.run(StageContext(run_dir=tmp_path, config={}))),
    )

    assert [request.prompt for request in service.requests] == [
        "English lighthouse",
        "English harbour",
        "Deutscher Markt",
    ]
    assert len(result.beats) == 3
    assert result.beat_counts == {"en": 2, "de": 3}
    assert result.by_language["en"] == result.beats[:2]
    assert result.by_language["de"] == result.beats
    assert result.total_cost_usd == 0.12
    assert len(result.resolved) == 3

    manifest = json.loads(
        (tmp_path / DEFAULT_VISUAL_REUSE_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["beat_counts"] == {"en": 2, "de": 3}
    assert manifest["metadata"]["beat_counts"] == {"en": 2, "de": 3}


def test_shared_requests_cover_the_longest_script_and_trim_each_language() -> None:
    scripts = {
        "en": _script("en", ["English lighthouse", "English harbour"]),
        "de": _script(
            "de",
            ["Deutscher Leuchtturm", "Deutscher Hafen", "Deutscher Markt"],
        ),
    }

    shared = shared_visual_requests(scripts, source_language="en", strategy="image")
    by_language = share_visual_requests(
        scripts,
        source_language="en",
        strategy="image",
    )

    assert [request.prompt for request in shared] == [
        "English lighthouse",
        "English harbour",
        "Deutscher Markt",
    ]
    assert by_language["en"] == shared[:2]
    assert by_language["de"] == shared
    assert by_language["en"][0] is by_language["de"][0]


def test_the_store_generates_only_the_extra_beats_for_a_longer_language() -> None:
    service = FakeImageService()
    store = VisualReuseStore(
        image=service,
        source_language="en",
        strategy="image",
    )

    first = _run(
        store.resolve(
            {"en": _script("en", ["English lighthouse", "English harbour"])},
            language="en",
        )
    )
    second = _run(
        store.resolve(
            {
                "de": _script(
                    "de",
                    [
                        "Deutscher Leuchtturm",
                        "Deutscher Hafen",
                        "Deutscher Markt",
                        "Deutscher Bahnhof",
                    ],
                )
            },
            language="de",
        )
    )

    assert len(first) == 2
    assert len(second) == 4
    assert len(service.requests) == 4
    assert [request.prompt for request in service.requests] == [
        "English lighthouse",
        "English harbour",
        "Deutscher Markt",
        "Deutscher Bahnhof",
    ]
    assert [beat.asset for beat in first] == [beat.asset for beat in second[:2]]
    assert store.shared is not None
    assert store.shared.languages == ["en", "de"]
    assert store.shared.beat_counts == {"en": 2, "de": 4}
    assert store.shared.total_cost_usd == 0.16
    assert store.shared.by_language["en"] == store.shared.beats[:2]
    assert store.shared.by_language["de"] == store.shared.beats


def test_the_store_does_not_generate_when_a_later_language_is_shorter() -> None:
    service = FakeImageService()
    store = VisualReuseStore(
        image=service,
        source_language="en",
        strategy="image",
    )

    _run(
        store.resolve(
            {"en": _script("en", ["English lighthouse", "English harbour"])},
            language="en",
        )
    )
    second = _run(
        store.resolve(
            {"de": _script("de", ["Deutscher Leuchtturm"])},
            language="de",
        )
    )

    assert len(service.requests) == 2
    assert len(second) == 1
    assert store.shared is not None
    assert store.shared.beat_counts == {"en": 2, "de": 1}
    assert store.shared.by_language["de"] == store.shared.beats[:1]


def test_existing_results_cover_extra_beats_without_calling_a_provider(
    tmp_path: Path,
) -> None:
    service = FakeImageService()
    scripts = {
        "en": _script("en", ["English lighthouse"]),
        "de": _script("de", ["Deutscher Leuchtturm", "Deutscher Hafen"]),
    }
    generated = _run(
        resolve_visual_reuse(
            {"en": scripts["en"]},
            source_language="en",
            strategy="image",
            image=service,
        )
    )

    class ExplodingService:
        async def generate(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("reused visuals must not call the provider")

    stage = VisualReuseStage(
        scripts=scripts,
        source_language="en",
        results=generated,
        image=ExplodingService(),
    )
    result = cast(
        VisualReuseStageResult,
        _run(stage.run(StageContext(run_dir=tmp_path, config={}))),
    )

    assert len(service.requests) == 1
    assert len(result.beats) == 2
    assert len(result.resolved) == 1
    assert len(result.unresolved) == 1
    assert result.beat_counts == {"en": 1, "de": 2}
    assert result.by_language["de"] == result.beats
    assert result.by_language["en"] == result.beats[:1]


def test_the_stage_applies_the_visuals_config_strategy_and_video_cap(
    tmp_path: Path,
) -> None:
    image = FakeImageService()
    video = FakeVideoService()
    stage = VisualReuseStage(
        scripts={"en": _script("en", ["one", "two", "three"])},
        image=image,
        video=video,
    )
    config = {"visuals": {"strategy": "image", "max_ai_video_beats": 0}}

    result = cast(
        VisualReuseStageResult,
        _run(stage.run(StageContext(run_dir=tmp_path, config=config))),
    )

    assert video.requests == []
    assert len(image.requests) == 3
    assert [request.visual_type for request in result.requests] == [
        VisualType.IMAGE,
        VisualType.IMAGE,
        VisualType.IMAGE,
    ]


def test_the_stage_applies_the_configured_video_cap_to_hybrid_visuals(
    tmp_path: Path,
) -> None:
    image = FakeImageService()
    video = FakeVideoService()
    stage = VisualReuseStage(
        scripts={"en": _script("en", ["one", "two", "three"])},
        image=image,
        video=video,
    )
    config = {"visuals": {"max_ai_video_beats": 1}}

    result = cast(
        VisualReuseStageResult,
        _run(stage.run(StageContext(run_dir=tmp_path, config=config))),
    )

    assert len(video.requests) == 1
    assert len(image.requests) == 2
    assert [request.visual_type for request in result.requests] == [
        VisualType.VIDEO,
        VisualType.IMAGE,
        VisualType.IMAGE,
    ]


def test_explicit_stage_options_win_over_the_visuals_config(
    tmp_path: Path,
) -> None:
    image = FakeImageService()
    video = FakeVideoService()
    stock = FakeStockSearch()
    stage = VisualReuseStage(
        scripts={"en": _script("en", ["one", "two", "three"])},
        strategy="stock",
        max_ai_video_beats=0,
        image=image,
        video=video,
        stock=stock,
    )
    config = {"visuals": {"strategy": "video", "max_ai_video_beats": 3}}

    result = cast(
        VisualReuseStageResult,
        _run(stage.run(StageContext(run_dir=tmp_path, config=config))),
    )

    assert len(stock.requests) == 3
    assert video.requests == []
    assert image.requests == []
    assert all(request.visual_type is VisualType.STOCK for request in result.requests)


def test_a_stage_config_option_is_used_when_the_context_has_none(
    tmp_path: Path,
) -> None:
    image = FakeImageService()
    stage = VisualReuseStage(
        scripts={"en": _script("en", ["one", "two"])},
        config={"visuals": {"strategy": "image", "max_ai_video_beats": 0}},
        image=image,
    )

    result = cast(
        VisualReuseStageResult,
        _run(stage.run(StageContext(run_dir=tmp_path, config={}))),
    )

    assert len(image.requests) == 2
    assert all(request.visual_type is VisualType.IMAGE for request in result.requests)


def test_the_reuse_helpers_apply_the_visuals_config() -> None:
    scripts = {"en": _script("en", ["one", "two"])}
    config = {"visuals": {"strategy": "image", "max_ai_video_beats": 0}}

    requests = shared_visual_requests(scripts, config=config)
    shared = share_visual_requests(scripts, config=config)

    assert [request.visual_type for request in requests] == [
        VisualType.IMAGE,
        VisualType.IMAGE,
    ]
    assert [request.visual_type for request in shared["en"]] == [
        VisualType.IMAGE,
        VisualType.IMAGE,
    ]
