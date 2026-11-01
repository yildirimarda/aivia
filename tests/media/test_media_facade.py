"""The media facade is the one entry point and emits a metrics document."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from aivia.api import create_app
from aivia.cli import main
from aivia.core.config import ApiConfig
from aivia.core.config.capability import CapabilityChainConfig
from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.router import Router
from aivia.core.stages import StageContext
from aivia.media import MEDIA_METRICS_FILENAME, MediaServiceFacade
from aivia.providers.mock import MockProvider
from aivia.providers.storage_local import LocalStorageProvider


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def provider() -> MockProvider:
    return MockProvider(name="mock")


@pytest.fixture
def router(provider: MockProvider) -> Router:
    return Router(
        {provider.name: provider},
        config=CapabilityChainConfig(chain=[provider.name]),
    )


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorageProvider:
    return LocalStorageProvider(tmp_path / "media")


@pytest.fixture
def facade(
    router: Router,
    engine: Engine,
    storage: LocalStorageProvider,
    tmp_path: Path,
) -> MediaServiceFacade:
    return MediaServiceFacade(
        router,
        tracker=CostTracker(engine),
        engine=engine,
        storage=storage,
        metrics_path=tmp_path / MEDIA_METRICS_FILENAME,
    )


def read_metrics(path: Path | None) -> dict[str, Any]:
    assert path is not None
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def test_a_call_writes_a_metrics_document(facade: MediaServiceFacade) -> None:
    result = asyncio.run(facade.image("A lighthouse at sunset"))

    assert result.kind == "image"
    assert result.cost_usd == pytest.approx(0.02)
    assert facade.metrics_path is not None
    document = read_metrics(facade.metrics_path)
    assert document["version"] == 1
    assert document["calls"] == 1
    assert document["succeeded"] == 1
    assert document["failed"] == 0
    assert document["cache_hits"] == 0
    assert document["cost_usd"] == pytest.approx(0.02)
    assert document["kinds"]["image"]["calls"] == 1
    assert document["kinds"]["image"]["cost_usd"] == pytest.approx(0.02)
    assert document["last"]["kind"] == "image"
    assert document["last"]["provider"] == "mock"
    assert document["last"]["cached"] is False
    assert document["last"]["paths"] == list(result.paths)


def test_a_cached_call_counts_a_hit_without_adding_cost(
    facade: MediaServiceFacade,
    provider: MockProvider,
) -> None:
    asyncio.run(facade.image("A lighthouse at sunset"))
    second = asyncio.run(facade.image("A lighthouse at sunset"))

    assert second.cached is True
    assert len(provider.calls) == 1
    document = read_metrics(facade.metrics_path)
    assert document["calls"] == 2
    assert document["succeeded"] == 2
    assert document["cache_hits"] == 1
    assert document["cost_usd"] == pytest.approx(0.02)
    assert document["kinds"]["image"]["cache_hits"] == 1


def test_a_failed_call_is_recorded_and_reraised(
    facade: MediaServiceFacade,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(facade.image_service, "generate", fail)

    with pytest.raises(RuntimeError, match="provider exploded"):
        asyncio.run(facade.image("A storm"))

    document = read_metrics(facade.metrics_path)
    assert document["calls"] == 1
    assert document["succeeded"] == 0
    assert document["failed"] == 1
    assert document["kinds"]["image"]["failed"] == 1
    assert document["last"]["status"] == "failed"
    assert document["last"]["error_code"] == "RuntimeError"
    assert "provider exploded" in document["last"]["error_message"]


def test_generate_dispatches_and_records_each_kind(
    facade: MediaServiceFacade,
) -> None:
    asyncio.run(facade.generate("image", "A lighthouse"))
    asyncio.run(facade.generate("image", "A quiet harbour"))

    document = read_metrics(facade.metrics_path)
    assert document["calls"] == 2
    assert document["kinds"]["image"]["calls"] == 2
    assert document["kinds"]["image"]["succeeded"] == 2


def test_reset_metrics_rewrites_an_empty_document(
    facade: MediaServiceFacade,
) -> None:
    asyncio.run(facade.image("A lighthouse"))
    facade.reset_metrics()

    document = read_metrics(facade.metrics_path)
    assert document["calls"] == 0
    assert document["cost_usd"] == 0.0
    assert document["kinds"] == {}
    assert "last" not in document


def test_a_stage_context_builds_the_facade_and_writes_to_the_run_dir(
    router: Router,
    engine: Engine,
    storage: LocalStorageProvider,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    context = StageContext(
        run_dir=run_dir,
        router=router,
        cost=CostTracker(engine),
        storage=storage,
    )

    result = asyncio.run(context.media.image("A lighthouse"))

    assert result.kind == "image"
    assert context.media is context.media_facade
    document = read_metrics(run_dir / MEDIA_METRICS_FILENAME)
    assert document["calls"] == 1
    assert document["kinds"]["image"]["calls"] == 1


def test_the_cli_writes_metrics_beside_its_database(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    config = tmp_path / "config.yaml"
    config.write_text(
        "providers:\n"
        "  mock:\n"
        "    type: mock\n"
        "storage:\n"
        "  backend: local\n"
        "  local_dir: media\n",
        encoding="utf-8",
    )

    code = main(
        [
            "media",
            "generate",
            "image",
            "--prompt",
            "A lighthouse at sunset",
            "--config",
            str(config),
        ]
    )
    capsys.readouterr()

    assert code == 0
    document = read_metrics(tmp_path / "data" / MEDIA_METRICS_FILENAME)
    assert document["calls"] == 1
    assert document["kinds"]["image"]["calls"] == 1


def test_the_api_runs_jobs_through_the_facade(
    engine: Engine,
    router: Router,
    storage: LocalStorageProvider,
    tmp_path: Path,
) -> None:
    facade = MediaServiceFacade(
        router,
        tracker=CostTracker(engine),
        engine=engine,
        storage=storage,
        metrics_path=tmp_path / MEDIA_METRICS_FILENAME,
    )
    client = TestClient(
        create_app(ApiConfig(require_auth=False), engine, environ={}, toolkit=facade),
    )

    submitted = client.post(
        "/api/v1/media/jobs",
        json={"kind": "image", "prompt": "A lighthouse at sunset"},
    ).json()
    status = client.get(f"/api/v1/media/jobs/{submitted['job_id']}").json()

    assert status["status"] == "succeeded"
    document = read_metrics(facade.metrics_path)
    assert document["calls"] == 1
    assert document["kinds"]["image"]["calls"] == 1
