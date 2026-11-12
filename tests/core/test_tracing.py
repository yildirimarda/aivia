"""Tracing emits spans for stages and provider calls without ever failing a run."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy import Engine

from aivia.core.config import CapabilityChainConfig, CapabilityStrategy
from aivia.core.config.sections import TracingConfig
from aivia.core.cost import CostTracker
from aivia.core.db import RunRepository, StageRunRepository, create_database_engine
from aivia.core.db.migrate import upgrade_database
from aivia.core.errors import AiviaError, RetryableError
from aivia.core.models import ProviderMeta
from aivia.core.models.run import Run, StageResult
from aivia.core.orchestrator import Orchestrator
from aivia.core.router import Router
from aivia.core.run_state import RunStateMachine
from aivia.core.stages import Stage, StageContext
from aivia.core.tracing import (
    OTLPExporter,
    Span,
    SpanKind,
    Tracer,
)
from aivia.providers import Provider
from aivia.providers.protocols.storage import StorageProvider


class _RecordingExporter:
    """Collects exported spans and optionally raises to simulate a failure."""

    def __init__(self, *, raise_on: int | None = None) -> None:
        self.spans: list[Span] = []
        self.raise_on = raise_on
        self.exports = 0

    def export(self, span: Span) -> None:
        self.exports += 1
        if self.raise_on is not None and self.exports >= self.raise_on:
            raise AiviaError("exporter is down")
        self.spans.append(span)


class _WriteStage(Stage):
    name = "write"
    artifacts = ("script.md",)

    def __init__(self, cost_usd: float = 0.25) -> None:
        super().__init__()
        self.cost_usd = cost_usd

    async def run(self, context: StageContext) -> StageResult:
        (context.run_dir / "script.md").write_text("script", encoding="utf-8")
        return StageResult(
            stage=self.name,
            artifacts=["script.md"],
            cost_usd=self.cost_usd,
        )


class _FailingStage(Stage):
    name = "boom"

    async def run(self, context: StageContext) -> StageResult:
        raise RetryableError("provider down")


class _FakeProvider(Provider):
    def __init__(self, name: str, capability: str) -> None:
        super().__init__(
            ProviderMeta.model_validate(
                {
                    "name": name,
                    "capabilities": [capability],
                    "pricing": {},
                }
            )
        )


def _router(
    provider: Provider,
    *,
    tracer: Tracer | None = None,
    chain: list[str] | None = None,
) -> Router:
    return Router(
        providers={provider.name: provider},
        config=CapabilityChainConfig(
            strategy=CapabilityStrategy.FIRST_AVAILABLE,
            chain=chain or [],
        ),
        tracer=tracer,
    )


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def machine(engine: Engine) -> RunStateMachine:
    return RunStateMachine(RunRepository(engine))


def _orchestrator(
    engine: Engine,
    run: Run,
    stages: list[Stage],
    tmp_path: Path,
    *,
    tracer: Tracer | None = None,
    **options: Any,
) -> Orchestrator:
    return Orchestrator(
        stages=stages,
        run=run,
        repository=StageRunRepository(engine),
        run_dir=tmp_path,
        config={"format": "sample"},
        cost=cast(CostTracker, object()),
        storage=cast(StorageProvider, object()),
        router=Router(),
        tracer=tracer,
        **options,
    )


def test_disabled_tracer_is_a_noop() -> None:
    tracer = Tracer.from_config(None)
    assert tracer.enabled is False
    assert tracer.exporter is None

    with tracer.span(
        kind=SpanKind.STAGE, name="write", capability="write", run_id="run-1"
    ):
        pass

    assert tracer.spans == []


def test_tracer_from_config_disabled() -> None:
    tracer = Tracer.from_config(TracingConfig(enabled=False))
    assert tracer.enabled is False


def test_tracer_from_config_otlp_without_endpoint_is_disabled() -> None:
    tracer = Tracer.from_config(TracingConfig(enabled=True, exporter="otlp"))
    assert tracer.enabled is False


def test_tracer_from_config_otlp_builds_exporter() -> None:
    tracer = Tracer.from_config(
        TracingConfig(enabled=True, exporter="otlp", endpoint="http://localhost:4318")
    )
    assert tracer.enabled is True
    assert isinstance(tracer.exporter, OTLPExporter)


def test_span_records_run_id_capability_latency_and_cost() -> None:
    exporter = _RecordingExporter()
    tracer = Tracer(enabled=True, exporter=exporter)

    with tracer.span(
        kind=SpanKind.STAGE,
        name="write",
        capability="write",
        run_id="run-123",
    ) as span:
        assert span.kind == SpanKind.STAGE
        assert span.name == "write"
        assert span.capability == "write"
        assert span.run_id == "run-123"
        assert span.status == "ok"
        span.cost_usd = 0.25

    assert len(exporter.spans) == 1
    recorded = exporter.spans[0]
    assert recorded.trace_id
    assert recorded.span_id
    assert recorded.run_id == "run-123"
    assert recorded.capability == "write"
    assert recorded.status == "ok"
    assert recorded.cost_usd == 0.25
    assert recorded.latency_ms is not None
    assert recorded.latency_ms >= 0.0


def test_span_measures_positive_latency() -> None:
    exporter = _RecordingExporter()
    tracer = Tracer(enabled=True, exporter=exporter)

    async def run() -> None:
        with tracer.span(
            kind=SpanKind.PROVIDER_CALL,
            name="gen",
            capability="image_gen",
            provider="fal",
        ):
            await asyncio.sleep(0.01)

    asyncio.run(run())

    assert len(exporter.spans) == 1
    assert exporter.spans[0].latency_ms is not None
    assert exporter.spans[0].latency_ms > 0.0


def test_span_records_error_status_and_propagates_exception() -> None:
    exporter = _RecordingExporter()
    tracer = Tracer(enabled=True, exporter=exporter)

    with (
        pytest.raises(ValueError),
        tracer.span(
            kind=SpanKind.PROVIDER_CALL,
            name="gen",
            capability="image_gen",
            provider="fal",
            run_id="run-9",
        ),
    ):
        raise ValueError("boom")

    assert len(exporter.spans) == 1
    recorded = exporter.spans[0]
    assert recorded.status == "error"
    assert recorded.error == "ValueError"
    assert recorded.run_id == "run-9"


def test_exporter_failure_is_swallowed_and_run_continues() -> None:
    exporter = _RecordingExporter(raise_on=1)
    tracer = Tracer(enabled=True, exporter=exporter)

    # The exporter raises on the first span but the tracer must not propagate.
    with tracer.span(
        kind=SpanKind.STAGE, name="write", capability="write", run_id="run-1"
    ):
        pass
    with tracer.span(
        kind=SpanKind.STAGE, name="render", capability="render", run_id="run-1"
    ):
        pass

    assert exporter.exports == 2
    assert len(tracer.spans) == 2


def test_sample_ratio_zero_emits_no_spans() -> None:
    exporter = _RecordingExporter()
    tracer = Tracer(enabled=True, sample_ratio=0.0, exporter=exporter)

    with tracer.span(
        kind=SpanKind.STAGE, name="write", capability="write", run_id="run-1"
    ):
        pass

    assert tracer.spans == []
    assert exporter.spans == []


def test_sample_ratio_one_emits_every_span() -> None:
    exporter = _RecordingExporter()
    tracer = Tracer(enabled=True, sample_ratio=1.0, exporter=exporter)

    with tracer.span(
        kind=SpanKind.STAGE, name="write", capability="write", run_id="run-1"
    ):
        pass
    with tracer.span(
        kind=SpanKind.PROVIDER_CALL,
        name="gen",
        capability="image_gen",
        provider="fal",
        run_id="run-1",
    ):
        pass

    assert len(tracer.spans) == 2
    assert len(exporter.spans) == 2


def test_exporter_never_receives_a_span_when_tracer_disabled() -> None:
    exporter = _RecordingExporter()
    tracer = Tracer(enabled=False, exporter=exporter)

    with tracer.span(
        kind=SpanKind.STAGE, name="write", capability="write", run_id="run-1"
    ):
        pass

    assert exporter.spans == []
    assert tracer.spans == []


def test_orchestrator_emits_stage_spans(
    engine: Engine, machine: RunStateMachine, tmp_path: Path
) -> None:
    run = machine.create(Run())
    tracer = Tracer(enabled=True)
    exporter = _RecordingExporter()
    tracer.exporter = exporter

    orchestrator = _orchestrator(
        engine, run, [_WriteStage(cost_usd=0.25)], tmp_path, tracer=tracer
    )
    result = asyncio.run(orchestrator.execute())

    assert result.ok is True
    assert len(tracer.spans) == 1
    span = tracer.spans[0]
    assert span.kind == SpanKind.STAGE
    assert span.name == "write"
    assert span.capability == "write"
    assert span.run_id == run.id
    assert span.cost_usd == pytest.approx(0.25)
    assert span.status == "ok"
    assert span.latency_ms is not None
    assert span.latency_ms >= 0.0


def test_orchestrator_emits_stage_span_on_failure(
    engine: Engine, machine: RunStateMachine, tmp_path: Path
) -> None:
    run = machine.create(Run())
    tracer = Tracer(enabled=True)

    orchestrator = _orchestrator(
        engine, run, [_FailingStage()], tmp_path, tracer=tracer, raise_on_failure=False
    )
    result = asyncio.run(orchestrator.execute())

    assert result.ok is False
    assert len(tracer.spans) == 1
    span = tracer.spans[0]
    assert span.kind == SpanKind.STAGE
    assert span.name == "boom"
    assert span.run_id == run.id
    assert span.status == "error"
    assert span.error == "StageRetryExhausted"


def test_orchestrator_does_not_break_when_exporter_fails(
    engine: Engine, machine: RunStateMachine, tmp_path: Path
) -> None:
    run = machine.create(Run())
    exporter = _RecordingExporter(raise_on=1)
    tracer = Tracer(enabled=True, exporter=exporter)

    orchestrator = _orchestrator(
        engine, run, [_WriteStage(cost_usd=0.25)], tmp_path, tracer=tracer
    )
    result = asyncio.run(orchestrator.execute())

    assert result.ok is True
    assert exporter.exports == 1
    assert len(tracer.spans) == 1


def test_router_emits_provider_call_spans() -> None:
    provider = _FakeProvider("fal", "image_gen")
    router = _router(provider, tracer=Tracer(enabled=True))

    async def operation(provider: Provider) -> str:
        return f"{provider.name}:ok"

    result = asyncio.run(router.execute("image_gen", operation))

    assert result == "fal:ok"
    tracer = router.tracer
    assert tracer is not None
    assert len(tracer.spans) == 1
    span = tracer.spans[0]
    assert span.kind == SpanKind.PROVIDER_CALL
    assert span.name == "fal"
    assert span.capability == "image_gen"
    assert span.provider == "fal"
    assert span.latency_ms is not None
    assert span.latency_ms >= 0.0


def test_router_provider_span_records_error_and_continues() -> None:
    provider = _FakeProvider("fal", "image_gen")
    tracer = Tracer(enabled=True)
    router = _router(provider, tracer=tracer)

    async def operation(provider: Provider) -> str:
        raise RetryableError("temporary")

    with pytest.raises(RetryableError):
        asyncio.run(router.execute("image_gen", operation))

    assert len(tracer.spans) == 1
    span = tracer.spans[0]
    assert span.status == "error"
    assert span.error == "RetryableError"


def test_router_does_not_break_when_exporter_fails() -> None:
    provider = _FakeProvider("fal", "image_gen")
    exporter = _RecordingExporter(raise_on=1)
    tracer = Tracer(enabled=True, exporter=exporter)
    router = _router(provider, tracer=tracer)

    async def operation(provider: Provider) -> str:
        return f"{provider.name}:ok"

    result = asyncio.run(router.execute("image_gen", operation))

    assert result == "fal:ok"
    assert exporter.exports == 1
    assert len(tracer.spans) == 1
