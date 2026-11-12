"""Benchmark harness comparing providers of one capability side by side.

The harness runs the same prompt set across every provider of a single
capability, timing each call and recording cost, latency, success or failure
and the artifacts produced. The result is a side by side
:class:`BenchmarkReport` a human or a CI job can read.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypeVar

from aivia.providers.base import Provider

__all__ = [
    "BenchmarkCallResult",
    "BenchmarkHarness",
    "BenchmarkProbeResult",
    "BenchmarkProviderResult",
    "BenchmarkReport",
]

_ProviderT = TypeVar("_ProviderT", bound=Provider)


@dataclass(frozen=True, slots=True)
class BenchmarkProbeResult:
    """The outcome of one provider call, before the harness adds timing.

    The caller supplied to :meth:`BenchmarkHarness.run` fills this from the
    provider response; the harness only adds latency, outcome and error
    handling.
    """

    cost_usd: float = 0.0
    """Simulated or reported cost of the call in USD."""

    units: float = 0.0
    """Quantity consumed, in ``unit_kind`` units."""

    unit_kind: str = "call"
    """Unit of ``units`` (call, character, second, image, …)."""

    model: str = ""
    """Model id the provider used for the call."""

    artifacts: tuple[str, ...] = ()
    """Human readable artifact descriptors produced by the call."""


@dataclass(frozen=True, slots=True)
class BenchmarkCallResult:
    """One timed call: provider, prompt, outcome, cost, latency and artifacts."""

    provider: str
    """Name of the provider that served the call."""

    prompt: str
    """Prompt from the benchmark set the call was made with."""

    ok: bool
    """Whether the call succeeded."""

    latency_ms: float
    """Wall clock time the call took, in milliseconds."""

    cost_usd: float
    """Cost of the call in USD."""

    units: float
    """Quantity consumed, in ``unit_kind`` units."""

    unit_kind: str
    """Unit of ``units``."""

    model: str
    """Model id the provider used."""

    artifacts: tuple[str, ...]
    """Artifact descriptors produced by the call."""

    error: str | None
    """Failure message when the call did not succeed, otherwise ``None``."""

    started_at: datetime
    """Instant the call started."""

    finished_at: datetime
    """Instant the call finished."""

    @property
    def duration_seconds(self) -> float:
        """The call latency expressed in seconds."""
        return self.latency_ms / 1000.0


@dataclass(frozen=True, slots=True)
class BenchmarkProviderResult:
    """Aggregated outcome for one provider across every prompt."""

    provider: str
    calls: tuple[BenchmarkCallResult, ...]

    @property
    def total_cost_usd(self) -> float:
        """Sum of the cost of every call this provider served."""
        return sum(call.cost_usd for call in self.calls)

    @property
    def total_latency_ms(self) -> float:
        """Sum of the latency of every call this provider served."""
        return sum(call.latency_ms for call in self.calls)

    @property
    def success_count(self) -> int:
        """Number of successful calls this provider served."""
        return sum(1 for call in self.calls if call.ok)

    @property
    def failure_count(self) -> int:
        """Number of failed calls this provider served."""
        return sum(1 for call in self.calls if not call.ok)

    @property
    def call_count(self) -> int:
        """Total number of calls this provider served."""
        return len(self.calls)

    @property
    def success_rate(self) -> float:
        """Fraction of calls that succeeded, or ``0.0`` with no calls."""
        if not self.calls:
            return 0.0
        return self.success_count / self.call_count

    @property
    def mean_latency_ms(self) -> float:
        """Mean latency across this provider's calls, or ``0.0`` with none."""
        if not self.calls:
            return 0.0
        return self.total_latency_ms / self.call_count

    @property
    def has_failure(self) -> bool:
        """Whether at least one call this provider served failed."""
        return self.failure_count > 0

    def errors(self) -> tuple[str, ...]:
        """Distinct error messages recorded for this provider, sorted."""
        return tuple(sorted({call.error for call in self.calls if call.error}))


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """A side by side comparison of providers for one capability."""

    capability: str
    """The capability every provider in the report serves."""

    prompts: tuple[str, ...]
    """The prompt set every provider was run against."""

    providers: tuple[str, ...]
    """Provider names, in the order they were benchmarked."""

    results: tuple[BenchmarkProviderResult, ...]
    """Per provider aggregated results, matching ``providers`` order."""

    @property
    def call_count(self) -> int:
        """Total number of calls across every provider."""
        return sum(result.call_count for result in self.results)

    @property
    def total_cost_usd(self) -> float:
        """Total spend across every provider and prompt."""
        return sum(result.total_cost_usd for result in self.results)

    @property
    def total_successes(self) -> int:
        """Total successful calls across every provider."""
        return sum(result.success_count for result in self.results)

    @property
    def total_failures(self) -> int:
        """Total failed calls across every provider."""
        return sum(result.failure_count for result in self.results)

    def provider_result(self, provider: str) -> BenchmarkProviderResult:
        """Return the aggregated result for *provider*."""
        for result in self.results:
            if result.provider == provider:
                return result
        raise KeyError(f"no benchmark result for provider: {provider}")

    def _successful(self) -> tuple[BenchmarkProviderResult, ...]:
        """Providers that served at least one successful call."""
        return tuple(result for result in self.results if result.success_count > 0)

    def cheapest(self) -> BenchmarkProviderResult:
        """The successful provider with the lowest total cost."""
        successful = self._successful()
        if not successful:
            raise ValueError("no successful provider to compare")
        return min(successful, key=lambda result: result.total_cost_usd)

    def fastest(self) -> BenchmarkProviderResult:
        """The successful provider with the lowest mean latency."""
        successful = self._successful()
        if not successful:
            raise ValueError("no successful provider to compare")
        return min(successful, key=lambda result: result.mean_latency_ms)

    def most_reliable(self) -> BenchmarkProviderResult:
        """The successful provider with the highest success rate."""
        successful = self._successful()
        if not successful:
            raise ValueError("no successful provider to compare")
        return max(successful, key=lambda result: result.success_rate)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON serializable representation of the report."""
        return {
            "capability": self.capability,
            "prompts": list(self.prompts),
            "providers": list(self.providers),
            "call_count": self.call_count,
            "total_cost_usd": self.total_cost_usd,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "provider_results": [
                {
                    "provider": result.provider,
                    "calls": [
                        {
                            "prompt": call.prompt,
                            "ok": call.ok,
                            "latency_ms": call.latency_ms,
                            "cost_usd": call.cost_usd,
                            "units": call.units,
                            "unit_kind": call.unit_kind,
                            "model": call.model,
                            "artifacts": list(call.artifacts),
                            "error": call.error,
                        }
                        for call in result.calls
                    ],
                    "total_cost_usd": result.total_cost_usd,
                    "mean_latency_ms": result.mean_latency_ms,
                    "success_count": result.success_count,
                    "failure_count": result.failure_count,
                }
                for result in self.results
            ],
        }


class BenchmarkHarness:
    """Run the same prompt set across providers of one capability.

    For every provider and every prompt the harness times the call, records
    cost, latency, success or failure and the produced artifacts, then assembles
    a side by side :class:`BenchmarkReport`.
    """

    def __init__(
        self,
        capability: str,
        providers: Mapping[str, Provider] | Sequence[tuple[str, Provider]],
        prompts: Sequence[str],
    ) -> None:
        if isinstance(providers, Mapping):
            providers = list(providers.items())
        self.capability = capability
        self._providers = list(providers)
        self.prompts = tuple(prompts)

    @property
    def provider_names(self) -> tuple[str, ...]:
        """Provider names, in benchmark order."""
        return tuple(name for name, _ in self._providers)

    async def run(
        self,
        invoke: Callable[[Provider, str], Awaitable[BenchmarkProbeResult]],
    ) -> BenchmarkReport:
        """Execute *invoke* for every provider against every prompt.

        *invoke* is called with ``(provider, prompt)`` and should perform the
        provider call and return a :class:`BenchmarkProbeResult`. When *invoke*
        raises, the failure is recorded and the next call proceeds.
        """
        results: list[BenchmarkProviderResult] = []
        for name, provider in self._providers:
            calls: list[BenchmarkCallResult] = []
            for prompt in self.prompts:
                started = time.perf_counter()
                started_at = datetime.now(UTC)
                error: str | None = None
                ok = False
                probe = BenchmarkProbeResult()
                try:
                    probe = await invoke(provider, prompt)
                    ok = True
                except Exception as exc:  # record every failure
                    message = str(exc)
                    if message:
                        error = f"{type(exc).__name__}: {message}"
                    else:
                        error = type(exc).__name__
                finished = time.perf_counter()
                finished_at = datetime.now(UTC)
                calls.append(
                    BenchmarkCallResult(
                        provider=name,
                        prompt=prompt,
                        ok=ok,
                        latency_ms=max(0.0, (finished - started) * 1000.0),
                        cost_usd=probe.cost_usd,
                        units=probe.units,
                        unit_kind=probe.unit_kind,
                        model=probe.model,
                        artifacts=probe.artifacts,
                        error=error,
                        started_at=started_at,
                        finished_at=finished_at,
                    )
                )
            results.append(BenchmarkProviderResult(provider=name, calls=tuple(calls)))
        return BenchmarkReport(
            capability=self.capability,
            prompts=self.prompts,
            providers=self.provider_names,
            results=tuple(results),
        )
