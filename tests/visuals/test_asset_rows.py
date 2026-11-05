"""Every visual asset row keeps its full provenance and rights record."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine

from aivia.core.db import AssetRepository, create_database_engine, upgrade_database
from aivia.core.models.asset import (
    Asset,
    AssetCostLink,
    AssetKind,
    AssetLicense,
    compute_prompt_hash,
)
from aivia.core.models.run import Run
from aivia.core.models.script import MediaKind, VisualType
from aivia.core.orchestrator import Orchestrator
from aivia.core.stages import StageContext
from aivia.media.image_qa import ImageQAAttempt, ImageQAResult, ImageQualityScore
from aivia.media.image_service import ImageOutcome
from aivia.providers.protocols.image import ImageResponse, ImageResult
from aivia.providers.protocols.stock import (
    StockAsset,
    StockMediaKind,
    StockResponse,
)
from aivia.visuals import (
    StockStage,
    VisualQAStage,
    VisualQAStageResult,
    VisualRequest,
    asset_row_from_stock,
    persist_asset_rows,
)


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


def _license() -> AssetLicense:
    return AssetLicense(
        name="CC BY 4.0",
        url="https://creativecommons.org/licenses/by/4.0/",
        attribution="Photo by Ada",
        commercial_use=True,
        attribution_required=True,
    )


def _accepted_qa() -> tuple[Asset, ImageQAResult]:
    asset = Asset(
        id="asset_accepted",
        kind=AssetKind.IMAGE,
        provider="fake-image",
        model="fake-model",
        prompt="A lighthouse",
        path="file:///tmp/lighthouse.png",
        license=_license(),
        cost_links=[AssetCostLink(cost_event_id="cost_3", amount_usd=0.04)],
        pinned=True,
    )
    response = ImageResponse(
        images=[ImageResult(data=b"image", width=1, height=1)],
        provider="fake-image",
        model="fake-model",
    )
    outcome = ImageOutcome(response=response, assets=(asset,))
    score = ImageQualityScore(
        score=0.93,
        description="A lighthouse at sunset",
        threshold=0.5,
    )
    attempt = ImageQAAttempt(
        index=0,
        provider="fake-image",
        outcome=outcome,
        scores=(score,),
        passed=True,
    )
    return asset, ImageQAResult(
        attempts=(attempt,),
        outcome=outcome,
        score=score,
        passed=True,
    )


class _StockSearch:
    def __init__(self, assets: list[StockAsset]) -> None:
        self.assets = assets

    async def search(self, request: Any) -> StockResponse:
        return StockResponse(
            query=request.query,
            assets=list(self.assets),
            provider="fake-stock",
        )


def test_persist_asset_rows_writes_every_provenance_field(engine: Engine) -> None:
    asset = Asset(
        provider="acme",
        model="image-1",
        prompt="A bridge at dusk",
        license=_license(),
        cost_links=[
            AssetCostLink(
                cost_event_id="cost_1",
                amount_usd=0.25,
                provider="acme",
                model="image-1",
                operation="image_gen",
            )
        ],
        quality_score=0.82,
        alt_text="A stone bridge crossing a river at dusk",
        pinned=True,
        run_id="run_1",
    )

    stored = persist_asset_rows(engine, asset)

    assert [row.id for row in stored] == [asset.id]
    row = AssetRepository(engine).get(asset.id)
    assert row is not None
    assert row.provider == "acme"
    assert row.model == "image-1"
    assert row.prompt == "A bridge at dusk"
    assert row.prompt_hash == compute_prompt_hash("A bridge at dusk")
    assert row.license.name == "CC BY 4.0"
    assert row.license.url == "https://creativecommons.org/licenses/by/4.0/"
    assert row.license.attribution == "Photo by Ada"
    assert row.license.commercial_use is True
    assert row.cost_links == asset.cost_links
    assert row.cost_event_ids == ["cost_1"]
    assert row.total_cost_usd == pytest.approx(0.25)
    assert row.quality_score == pytest.approx(0.82)
    assert row.alt_text == "A stone bridge crossing a river at dusk"
    assert row.pinned is True


def test_persist_asset_rows_updates_an_existing_row_without_duplicating_it(
    engine: Engine,
) -> None:
    repository = AssetRepository(engine)
    original = repository.create(
        Asset(
            id="asset_qa",
            provider="acme",
            model="image-1",
            prompt="A lighthouse",
        )
    )
    updated = original.model_copy(
        update={
            "license": _license(),
            "cost_links": [AssetCostLink(cost_event_id="cost_2", amount_usd=0.5)],
            "quality_score": 0.91,
            "alt_text": "A lighthouse at sunset",
            "pinned": True,
        }
    )

    persist_asset_rows(repository, updated)

    rows = repository.list()
    assert [row.id for row in rows] == ["asset_qa"]
    row = rows[0]
    assert row.license.name == "CC BY 4.0"
    assert row.cost_event_ids == ["cost_2"]
    assert row.quality_score == pytest.approx(0.91)
    assert row.alt_text == "A lighthouse at sunset"
    assert row.pinned is True


def test_a_stock_result_becomes_a_licensed_asset_row(engine: Engine) -> None:
    stock = StockAsset(
        id="stock_1",
        provider="pexels",
        kind=StockMediaKind.IMAGE,
        url="https://example.com/boat.jpg",
        width=1920,
        height=1080,
        license="Pexels License",
        license_url="https://www.pexels.com/license/",
        attribution="Photo by Ada",
        metadata={"alt": "A red boat on a lake"},
    )

    row = asset_row_from_stock(stock, prompt="A boat at dawn", run_id="run_1")

    assert row.kind is AssetKind.IMAGE
    assert row.provider == "pexels"
    assert row.path == "https://example.com/boat.jpg"
    assert row.prompt == "A boat at dawn"
    assert row.prompt_hash == compute_prompt_hash("A boat at dawn")
    assert row.license.name == "Pexels License"
    assert row.license.url == "https://www.pexels.com/license/"
    assert row.license.attribution == "Photo by Ada"
    assert row.license.attribution_required is True
    assert row.alt_text == "A red boat on a lake"
    assert row.analysis.width == 1920
    assert row.analysis.height == 1080
    assert row.pinned is True
    assert row.run_id == "run_1"

    persist_asset_rows(engine, row)
    assert AssetRepository(engine).get(row.id) == row


def test_the_qa_stage_persists_the_accepted_asset_with_its_score_and_alt_text(
    engine: Engine,
    tmp_path: Path,
) -> None:
    asset = Asset(
        id="asset_accepted",
        kind=AssetKind.IMAGE,
        provider="fake-image",
        model="fake-model",
        prompt="A lighthouse",
        path="file:///tmp/lighthouse.png",
        license=_license(),
        cost_links=[AssetCostLink(cost_event_id="cost_3", amount_usd=0.04)],
        pinned=True,
    )
    response = ImageResponse(
        images=[ImageResult(data=b"image", width=1, height=1)],
        provider="fake-image",
        model="fake-model",
    )
    outcome = ImageOutcome(response=response, assets=(asset,))
    score = ImageQualityScore(
        score=0.93,
        description="A lighthouse at sunset",
        threshold=0.5,
    )
    attempt = ImageQAAttempt(
        index=0,
        provider="fake-image",
        outcome=outcome,
        scores=(score,),
        passed=True,
    )
    qa_result = ImageQAResult(
        attempts=(attempt,),
        outcome=outcome,
        score=score,
        passed=True,
    )

    async def runner(request: Any, **_: Any) -> ImageQAResult:
        return qa_result

    stage = VisualQAStage(
        requests=[
            VisualRequest(
                index=0,
                visual_type=VisualType.IMAGE,
                kind=MediaKind.IMAGE,
                prompt="A lighthouse",
            )
        ],
        image_qa=runner,
        assets=engine,
    )

    result = asyncio.run(stage.run(StageContext(run_dir=tmp_path, config={})))

    assert isinstance(result, VisualQAStageResult)
    assert result.beats[0].asset is not None
    assert result.beats[0].asset.quality_score == pytest.approx(0.93)
    row = AssetRepository(engine).get("asset_accepted")
    assert row is not None
    assert row.provider == "fake-image"
    assert row.model == "fake-model"
    assert row.prompt_hash == compute_prompt_hash("A lighthouse")
    assert row.license.name == "CC BY 4.0"
    assert row.cost_event_ids == ["cost_3"]
    assert row.quality_score == pytest.approx(0.93)
    assert row.alt_text == "A lighthouse at sunset"
    assert row.pinned is True


def test_the_qa_stage_persists_through_the_context_database(
    engine: Engine,
    tmp_path: Path,
) -> None:
    asset, qa_result = _accepted_qa()

    async def runner(request: Any, **_: Any) -> ImageQAResult:
        return qa_result

    stage = VisualQAStage(
        requests=[
            VisualRequest(
                index=0,
                visual_type=VisualType.IMAGE,
                kind=MediaKind.IMAGE,
                prompt="A lighthouse",
            )
        ],
        image_qa=runner,
    )

    result = asyncio.run(
        stage.run(StageContext(run_dir=tmp_path, config={}, database=engine))
    )

    assert isinstance(result, VisualQAStageResult)
    row = AssetRepository(engine).get(asset.id)
    assert row is not None
    assert row.quality_score == pytest.approx(0.93)
    assert row.alt_text == "A lighthouse at sunset"


def test_the_orchestrator_passes_its_run_engine_to_the_stock_stage(
    engine: Engine,
    tmp_path: Path,
) -> None:
    stock = StockAsset(
        id="stock_1",
        provider="pexels",
        kind=StockMediaKind.IMAGE,
        url="https://example.com/boat.jpg",
        width=1920,
        height=1080,
        license="Pexels License",
        license_url="https://www.pexels.com/license/",
        attribution="Photo by Ada",
        metadata={"alt": "A red boat on a lake"},
    )
    stage = StockStage(
        requests=[
            VisualRequest(
                index=0,
                visual_type=VisualType.STOCK,
                kind=MediaKind.STOCK,
                prompt="A boat at dawn",
                query="boat",
            )
        ],
        search=_StockSearch([stock]),
    )
    orchestrator = Orchestrator(
        stages=[stage],
        run=Run(id="run_stock"),
        repository=engine,
        run_dir=tmp_path,
        config={},
    )

    result = asyncio.run(orchestrator.execute())

    assert result.ok is True
    rows = AssetRepository(engine).list(run_id="run_stock")
    assert [row.label for row in rows] == ["stock_1"]
    row = rows[0]
    assert row.provider == "pexels"
    assert row.prompt == "A boat at dawn"
    assert row.license.name == "Pexels License"
    assert row.alt_text == "A red boat on a lake"
    assert row.pinned is True
