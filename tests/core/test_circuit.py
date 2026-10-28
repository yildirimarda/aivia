"""Per-provider circuit breaker behaviour and router integration."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aivia.core.circuit import CircuitBreaker, CircuitState, ProviderCircuitBreaker
from aivia.core.config import CapabilityChainConfig, CapabilityStrategy
from aivia.core.errors import (
    ConfigError,
    PermanentError,
    ProviderExhaustedError,
    RateLimitError,
)
from aivia.core.models import ProviderMeta
from aivia.core.router import Router, RouteSkip
from aivia.providers import Provider


class FakeClock:
    """A manually advanced monotonic clock."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeProvider(Provider):
    """A provider that only needs a name and a capability for these tests."""

    def __init__(self, name: str, capabilities: list[str]) -> None:
        super().__init__(
            ProviderMeta.model_validate({"name": name, "capabilities": capabilities})
        )


def make_router(
    *providers: FakeProvider,
    breakers: dict[str, CircuitBreaker] | None = None,
) -> Router:
    return Router(
        providers={provider.name: provider for provider in providers},
        config=CapabilityChainConfig(chain=[provider.name for provider in providers]),
        circuit_breakers=breakers,
    )


def make_operation(
    errors: dict[str, Exception],
    calls: list[str],
) -> Any:
    async def operation(provider: Provider) -> str:
        calls.append(provider.name)
        error = errors.get(provider.name)
        if error is not None:
            raise error
        return f"{provider.name}:ok"

    return operation


def test_a_new_circuit_is_closed_and_available() -> None:
    breaker = CircuitBreaker()

    assert breaker.state is CircuitState.CLOSED
    assert breaker.is_closed is True
    assert breaker.is_open is False
    assert breaker.failures == 0
    assert breaker.available() is True
    assert breaker.acquire() is True


def test_the_circuit_opens_after_the_failure_threshold() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=30, clock=clock)

    breaker.record_failure()
    breaker.record_failure()

    state = breaker.state
    assert state is CircuitState.CLOSED
    assert breaker.failures == 2

    breaker.record_failure()

    state = breaker.state
    assert state is CircuitState.OPEN
    assert breaker.is_open is True
    assert breaker.failures == 3
    assert breaker.available() is False
    assert breaker.acquire() is False


def test_a_success_resets_consecutive_failures() -> None:
    breaker = CircuitBreaker(failure_threshold=2)

    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()

    assert breaker.state is CircuitState.CLOSED
    assert breaker.failures == 1


def test_an_open_circuit_half_opens_after_the_cooldown() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=10, clock=clock)
    breaker.record_failure()

    clock.advance(9.9)
    state = breaker.state
    assert state is CircuitState.OPEN
    assert breaker.available() is False
    assert breaker.cooldown_remaining() == pytest.approx(0.1)

    clock.advance(0.1)
    state = breaker.state
    assert state is CircuitState.HALF_OPEN
    assert breaker.is_half_open is True
    assert breaker.available() is True


def test_a_successful_half_open_probe_closes_the_circuit() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=5, clock=clock)
    breaker.record_failure()
    clock.advance(5)

    assert breaker.acquire() is True
    breaker.record_success()

    assert breaker.state is CircuitState.CLOSED
    assert breaker.failures == 0
    assert breaker.opened_at is None


def test_a_failed_half_open_probe_reopens_the_circuit() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=5, clock=clock)
    breaker.record_failure()
    clock.advance(5)
    assert breaker.acquire() is True

    breaker.record_failure()

    assert breaker.state is CircuitState.OPEN
    assert breaker.cooldown_remaining() == pytest.approx(5)


def test_a_half_open_circuit_allows_only_one_probe_at_a_time() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=5, clock=clock)
    breaker.record_failure()
    clock.advance(5)

    assert breaker.available() is True
    assert breaker.acquire() is True
    assert breaker.available() is False
    assert breaker.acquire() is False

    breaker.release()

    assert breaker.available() is True
    assert breaker.acquire() is True


def test_a_failure_while_open_does_not_extend_the_cooldown() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=5, clock=clock)
    breaker.record_failure()
    clock.advance(3)

    breaker.record_failure()

    assert breaker.cooldown_remaining() == pytest.approx(2)


def test_reset_closes_an_open_circuit() -> None:
    breaker = CircuitBreaker(failure_threshold=1)
    breaker.record_failure()

    breaker.reset()

    assert breaker.state is CircuitState.CLOSED
    assert breaker.failures == 0


def test_invalid_circuit_settings_are_configuration_errors() -> None:
    with pytest.raises(ConfigError, match="failure_threshold"):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ConfigError, match="cooldown"):
        CircuitBreaker(cooldown_seconds=float("nan"))
    with pytest.raises(ConfigError, match="clock"):
        CircuitBreaker(clock=object())  # type: ignore[arg-type]


def test_provider_circuit_breaker_alias_is_exported() -> None:
    assert ProviderCircuitBreaker is CircuitBreaker


def test_the_router_skips_a_provider_whose_circuit_is_open() -> None:
    clock = FakeClock()
    first = FakeProvider("first", ["image_gen"])
    second = FakeProvider("second", ["image_gen"])
    first_breaker = CircuitBreaker(
        failure_threshold=1, cooldown_seconds=10, clock=clock
    )
    router = make_router(first, second, breakers={"first": first_breaker})
    first_breaker.record_failure()

    selection = router.select_with_record("image_gen")

    assert selection.provider is second
    assert selection.skipped == (RouteSkip(provider="first", reason="circuit_open"),)
    assert [provider.name for provider in router.candidates("image_gen")] == ["second"]


def test_the_router_uses_a_half_open_provider_after_the_cooldown() -> None:
    clock = FakeClock()
    first = FakeProvider("first", ["image_gen"])
    second = FakeProvider("second", ["image_gen"])
    first_breaker = CircuitBreaker(
        failure_threshold=1, cooldown_seconds=10, clock=clock
    )
    router = make_router(first, second, breakers={"first": first_breaker})
    first_breaker.record_failure()
    clock.advance(10)

    selection = router.select_with_record("image_gen")

    assert selection.provider is first
    assert first_breaker.state is CircuitState.HALF_OPEN
    assert selection.skipped == ()


def test_each_provider_has_an_independent_circuit() -> None:
    first = FakeProvider("first", ["image_gen"])
    second = FakeProvider("second", ["image_gen"])
    router = make_router(first, second)

    router.circuit_for("first").record_failure()

    assert router.circuit_for("first").failures == 1
    assert router.circuit_for("second").failures == 0
    assert router.select("image_gen") is first


def test_unknown_circuit_names_are_configuration_errors() -> None:
    image = FakeProvider("image", ["image_gen"])

    with pytest.raises(ConfigError, match="unknown providers: missing"):
        make_router(
            image,
            breakers={"missing": CircuitBreaker()},
        )

    router = make_router(image)
    with pytest.raises(ConfigError, match="unknown provider"):
        router.circuit_for("missing")


def test_execute_records_retryable_failures_and_opens_the_circuit() -> None:
    first = FakeProvider("first", ["image_gen"])
    second = FakeProvider("second", ["image_gen"])
    first_breaker = CircuitBreaker(failure_threshold=2)
    router = make_router(first, second, breakers={"first": first_breaker})
    calls: list[str] = []
    operation = make_operation({"first": RateLimitError("slow down")}, calls)

    assert asyncio.run(router.execute("image_gen", operation)) == "second:ok"
    assert first_breaker.failures == 1

    assert asyncio.run(router.execute("image_gen", operation)) == "second:ok"
    assert first_breaker.state is CircuitState.OPEN

    calls.clear()
    assert asyncio.run(router.execute("image_gen", operation)) == "second:ok"
    assert calls == ["second"]
    assert router.last_selection is not None
    assert router.last_selection.skipped == (
        RouteSkip(provider="first", reason="circuit_open"),
    )


def test_execute_closes_a_half_open_circuit_on_success() -> None:
    clock = FakeClock()
    image = FakeProvider("image", ["image_gen"])
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=5, clock=clock)
    router = make_router(image, breakers={"image": breaker})
    breaker.record_failure()
    clock.advance(5)
    calls: list[str] = []

    result = asyncio.run(router.execute("image_gen", make_operation({}, calls)))

    assert result == "image:ok"
    assert calls == ["image"]
    assert breaker.state is CircuitState.CLOSED


def test_execute_reopens_a_half_open_circuit_on_failure() -> None:
    clock = FakeClock()
    image = FakeProvider("image", ["image_gen"])
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=5, clock=clock)
    router = make_router(image, breakers={"image": breaker})
    breaker.record_failure()
    clock.advance(5)
    calls: list[str] = []

    with pytest.raises(ProviderExhaustedError):
        asyncio.run(
            router.execute(
                "image_gen",
                make_operation({"image": RateLimitError("still down")}, calls),
            )
        )

    assert calls == ["image"]
    assert breaker.state is CircuitState.OPEN


def test_permanent_errors_do_not_trip_the_circuit() -> None:
    image = FakeProvider("image", ["image_gen"])
    breaker = CircuitBreaker(failure_threshold=1)
    router = make_router(image, breakers={"image": breaker})
    calls: list[str] = []

    with pytest.raises(PermanentError):
        asyncio.run(
            router.execute(
                "image_gen",
                make_operation({"image": PermanentError("bad request")}, calls),
            )
        )

    assert calls == ["image"]
    assert breaker.state is CircuitState.CLOSED
    assert breaker.failures == 0


def test_circuit_breakers_can_be_built_by_a_factory() -> None:
    clock = FakeClock()
    first = FakeProvider("first", ["image_gen"])
    second = FakeProvider("second", ["image_gen"])
    router = Router(
        providers=[first, second],
        config=CapabilityChainConfig(
            strategy=CapabilityStrategy.CHEAPEST,
            chain=["first", "second"],
        ),
        circuit_breaker_factory=lambda name: CircuitBreaker(
            failure_threshold=1, cooldown_seconds=10, clock=clock
        ),
    )

    assert set(router.breakers) == {"first", "second"}
    router.circuit_for("first").record_failure()
    clock.advance(10)

    assert router.select("image_gen") is first
