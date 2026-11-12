"""Tracing for stages and provider calls.

The :class:`Tracer` emits one span per stage run and one span per provider
call. Every span carries the run id, the capability (stage name or routed
capability), the provider (for provider calls), the latency and the cost.

Spans are handed to an :class:`Exporter`. An exporter that raises is swallowed:
a misconfigured or down telemetry backend must never fail a run.

The tracer is config driven through
:class:`~aivia.core.config.sections.TracingConfig`. When tracing is disabled or
no exporter is configured, the tracer is a no-op that still returns spans so
callers never have to special-case it.
"""

from __future__ import annotations

import random
import secrets
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Protocol

import httpx

from aivia.core.config.sections import TracingConfig
from aivia.core.logging import run_id as _logging_run_id

__all__ = [
    "Exporter",
    "OTLPExporter",
    "Span",
    "SpanKind",
    "Tracer",
]


class SpanKind(str):
    STAGE = "stage"
    PROVIDER_CALL = "provider_call"


@dataclass(slots=True)
class Span:
    """One traced operation."""

    name: str
    kind: str
    run_id: str
    capability: str
    trace_id: str = field(default_factory=lambda: secrets.token_hex(16))
    span_id: str = field(default_factory=lambda: secrets.token_hex(8))
    provider: str = ""
    service_name: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    latency_ms: float | None = None
    cost_usd: float = 0.0
    status: str = "ok"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def finish(
        self,
        *,
        latency_ms: float | None = None,
        cost_usd: float | None = None,
        status: str = "ok",
        error: str | None = None,
        ended_at: datetime | None = None,
    ) -> None:
        """Record the end of the span and compute its latency.

        Fields left as ``None`` keep their current value, so a caller may
        update only the pieces it knows at finish time.
        """
        self.ended_at = ended_at or datetime.now(UTC)
        if latency_ms is not None:
            self.latency_ms = max(0.0, float(latency_ms))
        elif self.latency_ms is None:
            started = self.started_at
            self.latency_ms = max(
                0.0, (self.ended_at - started).total_seconds() * 1000.0
            )
        if cost_usd is not None:
            self.cost_usd = max(0.0, float(cost_usd))
        self.status = status
        self.error = error


class Exporter(Protocol):
    """Send a finished span to a telemetry backend.

    Implementations may raise; the :class:`Tracer` never propagates exporter
    failures into a run.
    """

    def export(self, span: Span) -> None: ...


class OTLPExporter(Exporter):
    """Export spans as OTLP over HTTP to an endpoint.

    A failed request raises so the tracer can swallow it. No network happens
    until :meth:`export` is called with a live endpoint.
    """

    def __init__(
        self,
        endpoint: str,
        service_name: str = "aivia",
        *,
        timeout: float = 5.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.service_name = service_name
        self.timeout = timeout

    def export(self, span: Span) -> None:
        payload = {
            "service_name": self.service_name,
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "name": span.name,
            "kind": span.kind,
            "run_id": span.run_id,
            "capability": span.capability,
            "provider": span.provider,
            "latency_ms": span.latency_ms,
            "cost_usd": span.cost_usd,
            "status": span.status,
            "error": span.error,
            "metadata": span.metadata,
        }
        response = httpx.post(
            f"{self.endpoint}/v1/traces", json=payload, timeout=self.timeout
        )
        response.raise_for_status()


class Tracer:
    """Emit spans for stages and provider calls without ever failing a run."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        exporter: Exporter | None = None,
        sample_ratio: float = 1.0,
        service_name: str = "aivia",
    ) -> None:
        self.enabled = enabled
        self.exporter = exporter
        self.sample_ratio = max(0.0, min(1.0, float(sample_ratio)))
        self.service_name = service_name
        self._rng = random.Random()
        self.spans: list[Span] = []

    @classmethod
    def from_config(
        cls,
        config: TracingConfig | None,
        *,
        exporter: Exporter | None = None,
    ) -> Tracer:
        """Build a tracer from configuration.

        A ``None`` or disabled config produces a no-op tracer. An OTLP
        exporter is built automatically when the config names one and supplies
        an endpoint.
        """
        if config is None or not config.enabled:
            return cls(enabled=False)
        if exporter is None and config.exporter == "otlp":
            if not config.endpoint:
                return cls(enabled=False)
            exporter = OTLPExporter(config.endpoint, config.service_name)
        return cls(
            enabled=True,
            exporter=exporter,
            sample_ratio=config.sample_ratio,
            service_name=config.service_name,
        )

    def _sampled(self) -> bool:
        return self._rng.random() < self.sample_ratio

    def _export(self, span: Span) -> None:
        if self.exporter is None:
            return
        # A down or misconfigured telemetry backend must never fail a run.
        with suppress(Exception):
            self.exporter.export(span)

    def _new_span(
        self,
        *,
        kind: str,
        name: str,
        capability: str,
        run_id: str | None,
        provider: str,
        service_name: str,
    ) -> Span:
        return Span(
            name=name,
            kind=kind,
            run_id=run_id if run_id is not None else "",
            capability=capability,
            provider=provider,
            service_name=service_name,
        )

    @contextmanager
    def span(
        self,
        *,
        kind: str,
        name: str,
        capability: str,
        run_id: str | None = None,
        provider: str = "",
        service_name: str | None = None,
    ) -> Iterator[Span]:
        """Trace one operation.

        The yielded :class:`Span` is started on entry and finished (with its
        latency computed) on exit. Its cost and status may be updated before
        exit. Exporter failures are swallowed and never propagate.
        """
        if not self.enabled or not self._sampled():
            yield Span(
                kind=kind,
                name=name,
                capability=capability,
                run_id=run_id or "",
                provider=provider,
                service_name=service_name or self.service_name,
            )
            return
        resolved_run_id = run_id if run_id is not None else _logging_run_id.get()
        span = self._new_span(
            kind=kind,
            name=name,
            capability=capability,
            run_id=resolved_run_id,
            provider=provider,
            service_name=service_name or self.service_name,
        )
        self.spans.append(span)
        start = perf_counter()
        try:
            yield span
        except BaseException as exc:
            span.finish(
                latency_ms=(perf_counter() - start) * 1000.0,
                cost_usd=span.cost_usd,
                status="error",
                error=type(exc).__name__,
            )
            self._export(span)
            raise
        else:
            span.finish(
                latency_ms=(perf_counter() - start) * 1000.0,
                cost_usd=span.cost_usd,
                status="ok",
            )
            self._export(span)
