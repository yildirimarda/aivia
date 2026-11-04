"""The write stage reads the run fact list and writes one script per language."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine, Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine

from aivia.core.cost import CostTracker
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import ConfigError
from aivia.core.models import Run, StageResult
from aivia.core.run_dirs import RunDirectory
from aivia.core.stages import StageContext
from aivia.providers.protocols.llm import LLMRequest, LLMResponse, LLMUsage
from aivia.writing import WriteStage

MODEL = "writer-model"
INPUT_PRICE = 3.0
OUTPUT_PRICE = 15.0

FACTS = {
    "title": "Honey",
    "claims": [
        {
            "text": "Honey never spoils.",
            "source_urls": ["https://example.com/honey"],
        },
        {
            "text": "Bees recognise human faces.",
            "source_urls": ["https://example.com/bees"],
        },
    ],
}


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


class FakeWriterProvider:
    """Return one script per target language, in the order the stage asks."""

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []
        self.payloads = [
            {
                "title": "Honey",
                "hook": "Honey keeps forever.",
                "beats": [{"text": "Honey never spoils."}],
                "outro": "That is the story of honey.",
            },
            {
                "title": "Honig",
                "hook": "Honig bleibt ewig.",
                "beats": [{"text": "Honig verdirbt nie."}],
                "outro": "Das ist die Geschichte von Honig.",
            },
        ]

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        payload = self.payloads.pop(0)
        return LLMResponse(
            content=json.dumps(payload, ensure_ascii=False),
            provider="acme",
            model=MODEL,
            usage=LLMUsage(input_tokens=20, output_tokens=10),
        )


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


def _config() -> dict[str, Any]:
    return {
        "languages": {
            "default": "en",
            "en": {"name": "English"},
            "de": {"name": "German"},
        },
        "llm": {
            "pricing": {
                MODEL: {
                    "input_per_million": INPUT_PRICE,
                    "output_per_million": OUTPUT_PRICE,
                }
            }
        },
    }


def _write_facts(run_dir: Path, value: Any = None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "facts.json"
    path.write_text(json.dumps(FACTS if value is None else value), encoding="utf-8")
    return path


def _context(
    run_dir: Path,
    *,
    engine: Engine,
    config: dict[str, Any] | None = None,
    run_id: str = "run_write",
) -> StageContext:
    return StageContext(
        run_dir=run_dir,
        config=config or _config(),
        cost=CostTracker(engine),
        run=Run(id=run_id),
    )


def test_stage_reads_the_run_fact_list_and_writes_a_script_per_language(
    tmp_path: Path,
    engine: Engine,
) -> None:
    _write_facts(tmp_path)
    provider = FakeWriterProvider()
    stage = WriteStage(options={"languages": ["en", "de"]}, provider=provider)
    context = _context(tmp_path, engine=engine)

    result = _run(stage.run(context))

    assert isinstance(result, StageResult)
    assert result.stage == "write"
    assert result.run_id == "run_write"
    assert result.artifacts == ["script.en.json", "script.de.json"]
    assert result.metadata["languages"] == ["en", "de"]
    assert result.metadata["fact_count"] == 2
    assert result.metadata["source_urls"] == [
        "https://example.com/honey",
        "https://example.com/bees",
    ]
    assert result.metadata["provider"] == "acme"
    assert result.metadata["model"] == MODEL

    english = json.loads((tmp_path / "script.en.json").read_text(encoding="utf-8"))
    german = json.loads((tmp_path / "script.de.json").read_text(encoding="utf-8"))
    assert english["language"] == "en"
    assert english["hook"] == "Honey keeps forever."
    assert english["title"] == "Honey"
    assert german["language"] == "de"
    assert german["hook"] == "Honig bleibt ewig."
    assert len(provider.requests) == 2


def test_stage_reports_tokens_and_cost_from_the_model_calls(
    tmp_path: Path,
    engine: Engine,
) -> None:
    _write_facts(tmp_path)
    stage = WriteStage(
        options={"languages": ["en", "de"]}, provider=FakeWriterProvider()
    )
    context = _context(tmp_path, engine=engine)

    result = _run(stage.run(context))

    assert result.metadata["input_tokens"] == 40
    assert result.metadata["output_tokens"] == 20
    assert result.metadata["total_tokens"] == 60
    assert result.cost_usd == pytest.approx(
        40 * INPUT_PRICE / 1_000_000 + 20 * OUTPUT_PRICE / 1_000_000
    )
    assert context.cost is not None
    assert context.cost.total(run_id="run_write") == pytest.approx(result.cost_usd)


def test_stage_uses_the_configured_languages_when_no_filter_is_given(
    tmp_path: Path,
    engine: Engine,
) -> None:
    _write_facts(tmp_path)
    stage = WriteStage(provider=FakeWriterProvider())
    context = _context(tmp_path, engine=engine)

    result = _run(stage.run(context))

    assert result.metadata["languages"] == ["en", "de"]
    assert result.artifacts == ["script.en.json", "script.de.json"]


def test_stage_accepts_an_explicit_fact_list_without_a_file(
    tmp_path: Path,
    engine: Engine,
) -> None:
    stage = WriteStage(
        fact_list=FACTS,
        options={"languages": ["en"]},
        provider=FakeWriterProvider(),
    )
    context = _context(tmp_path, engine=engine)

    result = _run(stage.run(context))

    assert result.artifacts == ["script.en.json"]
    assert result.metadata["fact_count"] == 2


def test_stage_reads_a_fact_list_registered_as_a_research_artifact(
    tmp_path: Path,
    engine: Engine,
) -> None:
    facts_path = tmp_path / "research" / "claims.json"
    facts_path.parent.mkdir(parents=True)
    facts_path.write_text(json.dumps(FACTS), encoding="utf-8")
    directory = RunDirectory(tmp_path, run_id="run_write").create()
    directory.register("research", "research/claims.json")

    stage = WriteStage(options={"languages": ["en"]}, provider=FakeWriterProvider())
    context = _context(tmp_path, engine=engine)

    result = _run(stage.run(context))

    assert result.artifacts == ["script.en.json"]
    assert result.metadata["fact_count"] == 2


def test_stage_rejects_a_missing_fact_list(tmp_path: Path, engine: Engine) -> None:
    stage = WriteStage(options={"languages": ["en"]}, provider=FakeWriterProvider())
    context = _context(tmp_path, engine=engine)

    with pytest.raises(ConfigError, match="needs a fact list"):
        _run(stage.run(context))


def test_stage_rejects_an_unknown_language_before_spending(
    tmp_path: Path,
    engine: Engine,
) -> None:
    _write_facts(tmp_path)
    stage = WriteStage(options={"languages": ["fr"]}, provider=FakeWriterProvider())
    context = _context(tmp_path, engine=engine)

    with pytest.raises(ConfigError, match="unknown language"):
        _run(stage.run(context))


def test_stage_falls_back_to_cost_events_when_the_writer_hides_usage(
    tmp_path: Path,
    engine: Engine,
) -> None:
    _write_facts(tmp_path)
    tracker = CostTracker(engine)

    class RecordingWriter:
        def write(self, facts: Any, languages: list[str]) -> dict[str, Any]:
            tracker.record(
                provider="acme",
                model=MODEL,
                operation="multilingual_writing",
                units=60.0,
                unit_kind="token",
                amount_usd=0.002,
                run_id="run_write",
                metadata={"input_tokens": 40, "output_tokens": 20},
            )
            return {
                "en": {
                    "language": "en",
                    "hook": "Honey keeps forever.",
                }
            }

    stage = WriteStage(
        options={"languages": ["en"]},
        writer=RecordingWriter(),
    )
    context = StageContext(
        run_dir=tmp_path,
        config=_config(),
        cost=tracker,
        run=Run(id="run_write"),
    )

    result = _run(stage.run(context))

    assert result.cost_usd == pytest.approx(0.002)
    assert result.metadata["input_tokens"] == 40
    assert result.metadata["output_tokens"] == 20
    assert result.metadata["total_tokens"] == 60
