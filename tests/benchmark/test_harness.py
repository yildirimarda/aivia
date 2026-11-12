"""Benchmark harness: the same prompt set across providers of one capability."""

from __future__ import annotations

import asyncio
from typing import cast

from aivia.benchmark import (
    BenchmarkHarness,
    BenchmarkProbeResult,
    BenchmarkReport,
)
from aivia.core.errors import PermanentError
from aivia.providers import AudioFormat, MockTTSProvider, Provider
from aivia.providers.protocols.tts import TTSRequest

PROMPTS = ("hello world", "good morning", "benchmark run")


def _providers() -> dict[str, MockTTSProvider]:
    return {
        "cheap": MockTTSProvider(cost_usd=0.00001),
        "expensive": MockTTSProvider(cost_usd=0.00005),
        "failing": MockTTSProvider(error=PermanentError("provider down")),
    }


async def _invoke(provider: Provider, prompt: str) -> BenchmarkProbeResult:
    tts = cast(MockTTSProvider, provider)
    request = TTSRequest(
        text=prompt,
        voice="mock-voice",
        format=AudioFormat.WAV,
        sample_rate=8_000,
    )
    response = await tts.synthesize(request)
    return BenchmarkProbeResult(
        cost_usd=tts.last_cost_usd,
        units=float(len(prompt)),
        unit_kind="character",
        model=response.model,
        artifacts=(f"tts_{response.format.value} ({response.size_bytes} bytes)",),
    )


def _run() -> BenchmarkReport:
    harness = BenchmarkHarness(
        capability="tts",
        providers=_providers(),
        prompts=PROMPTS,
    )
    return asyncio.run(harness.run(_invoke))


def _call_matrix(report: BenchmarkReport) -> dict[tuple[str, str], dict[str, object]]:
    """Comparable fields of every call, ignoring timestamps that differ run to run."""
    matrix: dict[tuple[str, str], dict[str, object]] = {}
    for result in report.results:
        for call in result.calls:
            matrix[(result.provider, call.prompt)] = {
                "ok": call.ok,
                "cost_usd": call.cost_usd,
                "units": call.units,
                "unit_kind": call.unit_kind,
                "model": call.model,
                "artifacts": call.artifacts,
                "error": call.error,
            }
    return matrix


def test_benchmark_runs_every_prompt_across_every_provider() -> None:
    report = _run()

    assert report.capability == "tts"
    assert report.prompts == PROMPTS
    assert tuple(report.providers) == ("cheap", "expensive", "failing")
    assert report.call_count == len(_providers()) * len(PROMPTS)
    for result in report.results:
        assert result.call_count == len(PROMPTS)
        assert {call.prompt for call in result.calls} == set(PROMPTS)


def test_cost_is_recorded_per_provider_side_by_side() -> None:
    report = _run()

    cheap = report.provider_result("cheap")
    failing = report.provider_result("failing")
    assert cheap.total_cost_usd > 0.0
    assert failing.total_cost_usd == 0.0
    assert report.provider_result("expensive").total_cost_usd > cheap.total_cost_usd
    assert report.cheapest().provider == "cheap"


def test_failures_are_recorded_without_stopping_the_benchmark() -> None:
    report = _run()

    failing = report.provider_result("failing")
    assert failing.success_count == 0
    assert failing.failure_count == len(PROMPTS)
    assert failing.errors() == ("PermanentError: provider down",)
    for call in failing.calls:
        assert not call.ok
        assert call.cost_usd == 0.0
        assert call.artifacts == ()

    cheap = report.provider_result("cheap")
    expensive = report.provider_result("expensive")
    assert cheap.failure_count == 0
    assert expensive.failure_count == 0
    assert report.total_failures == len(PROMPTS)
    assert report.total_successes == 2 * len(PROMPTS)


def test_latency_is_recorded_for_every_call() -> None:
    report = _run()

    for result in report.results:
        for call in result.calls:
            assert call.latency_ms >= 0.0
            assert call.started_at is not None
            assert call.finished_at is not None
    fastest = report.fastest()
    assert fastest.provider in {"cheap", "expensive", "failing"}


def test_artifacts_are_captured_for_successful_calls_only() -> None:
    report = _run()

    for result in report.results:
        for call in result.calls:
            if call.ok:
                assert call.artifacts
                assert call.error is None
            else:
                assert call.artifacts == ()
                assert call.error is not None


def test_benchmark_is_deterministic() -> None:
    first = _call_matrix(_run())
    second = _call_matrix(_run())

    assert first == second


def test_cheapest_fastest_most_reliable() -> None:
    report = _run()

    assert report.cheapest().provider == "cheap"
    assert report.most_reliable().provider in {"cheap", "expensive"}
    assert report.total_successes == 2 * len(PROMPTS)
    assert report.total_failures == len(PROMPTS)


def test_to_dict_is_json_serializable_and_complete() -> None:
    report = _run()
    payload = report.to_dict()

    assert payload["capability"] == "tts"
    assert payload["prompts"] == list(PROMPTS)
    assert payload["providers"] == ["cheap", "expensive", "failing"]
    assert payload["call_count"] == 9
    assert payload["total_successes"] == 6
    assert payload["total_failures"] == 3
    provider_results = cast(list[dict[str, object]], payload["provider_results"])
    assert len(provider_results) == 3
    for entry in provider_results:
        assert set(entry) == {
            "provider",
            "calls",
            "total_cost_usd",
            "mean_latency_ms",
            "success_count",
            "failure_count",
        }
        calls = cast(list[object], entry["calls"])
        assert len(calls) == len(PROMPTS)
