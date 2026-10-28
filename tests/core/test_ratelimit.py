"""Token bucket rate limiting per provider and router integration."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aivia.core.config import CapabilityChainConfig, OnRateLimit, RateLimiterConfig
from aivia.core.errors import ConfigError, RateLimitError
from aivia.core.models import ProviderMeta, RateLimit
from aivia.core.ratelimit import (
    ConcurrencyLimiter,
    ProviderRateLimiter,
    RateLimiter,
    TokenBucket,
)
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


class FakeSleep:
    """Record requested delays and advance the fake clock immediately."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)
        self.clock.advance(seconds)


class FakeProvider(Provider):
    """A provider with an optional declared rate limit."""

    def __init__(
        self,
        name: str,
        capabilities: list[str],
        rate_limit: RateLimit | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "name": name,
            "capabilities": capabilities,
        }
        if rate_limit is not None:
            payload["rate_limit"] = rate_limit
        super().__init__(ProviderMeta.model_validate(payload))


def make_router(
    *providers: FakeProvider,
    limiters: dict[str, TokenBucket] | None = None,
    factory: Any = None,
) -> Router:
    return Router(
        providers={provider.name: provider for provider in providers},
        config=CapabilityChainConfig(chain=[provider.name for provider in providers]),
        rate_limiters=limiters,
        rate_limiter_factory=factory,
    )


def test_a_new_bucket_starts_full_and_consumes_tokens() -> None:
    clock = FakeClock()
    bucket = TokenBucket(requests_per_minute=60, burst=2, clock=clock)

    assert bucket.capacity == 2
    assert bucket.tokens == 2
    assert bucket.try_acquire() is True
    assert bucket.tokens == 1
    assert bucket.try_acquire() is True
    assert bucket.tokens == 0
    assert bucket.try_acquire() is False
    assert bucket.available() is False


def test_tokens_refill_continuously_and_stop_at_capacity() -> None:
    clock = FakeClock()
    bucket = TokenBucket(requests_per_minute=60, burst=2, clock=clock)
    assert bucket.try_acquire(2) is True

    clock.advance(0.5)
    assert bucket.tokens == pytest.approx(0.5)

    clock.advance(10)
    assert bucket.tokens == pytest.approx(2)
    assert bucket.time_until(2) == pytest.approx(0)


def test_a_non_waiting_acquire_raises_with_retry_after() -> None:
    clock = FakeClock()
    bucket = TokenBucket(
        requests_per_minute=60,
        burst=1,
        on_limit=OnRateLimit.RAISE,
        name="image",
        clock=clock,
    )
    assert bucket.try_acquire() is True

    with pytest.raises(RateLimitError) as caught:
        asyncio.run(bucket.acquire())

    error = caught.value
    assert error.retry_after == pytest.approx(1.0)
    assert error.details["provider"] == "image"
    assert error.details["requested"] == pytest.approx(1.0)
    assert bucket.tokens == pytest.approx(0)


def test_a_waiting_acquire_sleeps_for_the_next_token() -> None:
    clock = FakeClock()
    sleep = FakeSleep(clock)
    bucket = TokenBucket(
        requests_per_minute=60,
        burst=1,
        on_limit=OnRateLimit.WAIT,
        clock=clock,
        sleep=sleep,
    )
    assert bucket.try_acquire() is True

    waited = asyncio.run(bucket.acquire())

    assert waited == pytest.approx(1.0)
    assert sleep.delays == [pytest.approx(1.0)]
    assert bucket.tokens == pytest.approx(0)


def test_a_per_call_wait_flag_overrides_the_default_policy() -> None:
    clock = FakeClock()
    sleep = FakeSleep(clock)
    bucket = TokenBucket(
        requests_per_minute=60,
        burst=1,
        on_limit=OnRateLimit.WAIT,
        clock=clock,
        sleep=sleep,
    )
    assert bucket.try_acquire() is True

    with pytest.raises(RateLimitError):
        asyncio.run(bucket.acquire(wait=False))

    assert sleep.delays == []


def test_concurrent_non_waiting_acquires_cannot_overdraw_the_bucket() -> None:
    clock = FakeClock()

    async def scenario() -> list[object]:
        bucket = TokenBucket(
            requests_per_minute=60,
            burst=2,
            on_limit=OnRateLimit.RAISE,
            clock=clock,
        )
        return await asyncio.gather(
            *(bucket.acquire() for _ in range(5)),
            return_exceptions=True,
        )

    results = asyncio.run(scenario())
    successes = [result for result in results if not isinstance(result, Exception)]
    failures = [result for result in results if isinstance(result, RateLimitError)]

    assert len(successes) == 2
    assert len(failures) == 3


def test_concurrent_waiting_acquires_queue_in_token_order() -> None:
    clock = FakeClock()
    sleep = FakeSleep(clock)

    async def scenario() -> list[float]:
        bucket = TokenBucket(
            requests_per_minute=60,
            burst=2,
            on_limit=OnRateLimit.WAIT,
            clock=clock,
            sleep=sleep,
        )
        return await asyncio.gather(*(bucket.acquire() for _ in range(4)))

    waits = asyncio.run(scenario())

    assert sorted(waits) == pytest.approx([0.0, 0.0, 1.0, 1.0])
    assert sum(sleep.delays) == pytest.approx(2.0)


def test_an_unlimited_bucket_never_blocks() -> None:
    bucket = TokenBucket(requests_per_minute=0)

    assert bucket.unlimited is True
    assert all(bucket.try_acquire() for _ in range(100))
    assert asyncio.run(bucket.acquire()) == 0.0


def test_a_bucket_can_be_built_from_provider_metadata() -> None:
    bucket = TokenBucket(
        rate_limit=RateLimit(requests_per_minute=120, burst=5),
        name="image",
    )

    assert bucket.requests_per_minute == 120
    assert bucket.rate_per_second == pytest.approx(2.0)
    assert bucket.capacity == 5
    assert bucket.name == "image"


def test_rate_and_capacity_are_accepted_as_token_bucket_terms() -> None:
    bucket = TokenBucket(rate=2, capacity=3)

    assert bucket.requests_per_minute == pytest.approx(120)
    assert bucket.rate_per_second == pytest.approx(2.0)
    assert bucket.capacity == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"requests_per_minute": -1},
        {"requests_per_minute": float("nan")},
        {"requests_per_minute": 60, "burst": -1},
        {"requests_per_minute": 60, "capacity": 0},
        {"requests_per_minute": 60, "on_limit": "sometimes"},
    ],
)
def test_invalid_rate_limit_values_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        TokenBucket(**kwargs)


def test_the_limiter_aliases_point_at_the_same_class() -> None:
    assert RateLimiter is TokenBucket
    assert ProviderRateLimiter is TokenBucket


def test_the_router_builds_one_bucket_per_provider() -> None:
    first = FakeProvider(
        "first",
        ["image_gen"],
        RateLimit(requests_per_minute=60, burst=2),
    )
    second = FakeProvider(
        "second",
        ["image_gen"],
        RateLimit(requests_per_minute=120, burst=4),
    )
    router = make_router(first, second)

    assert set(router.rate_limiters) == {"first", "second"}
    assert router.limiter_for("first") is not router.limiter_for("second")
    assert router.limiter_for("first").capacity == 2
    assert router.rate_limiter_for("second").requests_per_minute == 120


def test_the_router_accepts_injected_limiters_and_a_factory() -> None:
    first = FakeProvider("first", ["image_gen"])
    second = FakeProvider("second", ["image_gen"])
    injected = TokenBucket(requests_per_minute=10, burst=1, name="first")
    built: list[str] = []

    def factory(name: str, provider: Provider) -> TokenBucket:
        built.append(name)
        return TokenBucket(
            rate_limit=provider.metadata.rate_limit,
            name=name,
        )

    router = make_router(first, second, limiters={"first": injected}, factory=factory)

    assert router.limiter_for("first") is injected
    assert built == ["second"]
    assert router.limiter_for("second").name == "second"


def test_the_router_rejects_limiters_for_unknown_providers() -> None:
    image = FakeProvider("image", ["image_gen"])

    with pytest.raises(ConfigError, match="unknown providers"):
        make_router(
            image,
            limiters={"missing": TokenBucket(requests_per_minute=60)},
        )


def test_the_router_uses_the_configured_on_limit_policy() -> None:
    image = FakeProvider("image", ["image_gen"])
    router = Router(
        providers=[image],
        config=CapabilityChainConfig(chain=["image"]),
        rate_limit_config=RateLimiterConfig(on_limit=OnRateLimit.RAISE),
    )

    assert router.limiter_for("image").on_limit is OnRateLimit.RAISE


def test_execute_waits_for_a_token_on_the_same_provider() -> None:
    clock = FakeClock()
    sleep = FakeSleep(clock)
    image = FakeProvider(
        "image",
        ["image_gen"],
        RateLimit(requests_per_minute=60, burst=1),
    )
    limiter = TokenBucket(
        rate_limit=image.metadata.rate_limit,
        name="image",
        on_limit=OnRateLimit.WAIT,
        clock=clock,
        sleep=sleep,
    )
    router = make_router(image, limiters={"image": limiter})
    calls: list[str] = []

    async def scenario() -> None:
        async def operation(provider: Provider) -> str:
            calls.append(provider.name)
            return f"{provider.name}:ok"

        assert await router.execute("image_gen", operation) == "image:ok"
        assert await router.execute("image_gen", operation) == "image:ok"

    asyncio.run(scenario())

    assert calls == ["image", "image"]
    assert sleep.delays == [pytest.approx(1.0)]


def test_execute_falls_back_when_a_fail_fast_bucket_is_empty() -> None:
    clock = FakeClock()
    first = FakeProvider("first", ["image_gen"])
    second = FakeProvider("second", ["image_gen"])
    limiter = TokenBucket(
        requests_per_minute=60,
        burst=1,
        on_limit=OnRateLimit.RAISE,
        name="first",
        clock=clock,
    )
    assert limiter.try_acquire() is True
    router = make_router(first, second, limiters={"first": limiter})
    calls: list[str] = []

    async def operation(provider: Provider) -> str:
        calls.append(provider.name)
        return f"{provider.name}:ok"

    result = asyncio.run(router.execute("image_gen", operation))

    assert result == "second:ok"
    assert calls == ["second"]
    assert router.last_selection is not None
    assert router.last_selection.skipped == (
        RouteSkip(provider="first", reason="rate_limited"),
    )


def test_candidates_exclude_a_provider_whose_fail_fast_bucket_is_empty() -> None:
    first = FakeProvider("first", ["image_gen"])
    second = FakeProvider("second", ["image_gen"])
    limiter = TokenBucket(
        requests_per_minute=60,
        burst=1,
        on_limit=OnRateLimit.RAISE,
        name="first",
    )
    assert limiter.try_acquire() is True
    router = make_router(first, second, limiters={"first": limiter})

    assert [provider.name for provider in router.candidates("image_gen")] == ["second"]


def test_a_concurrency_limiter_fails_fast_when_full() -> None:
    gate = ConcurrencyLimiter(1, on_limit=OnRateLimit.RAISE, name="image")

    async def scenario() -> None:
        assert await gate.acquire() == 0.0
        assert gate.active == 1
        assert gate.available() is False
        with pytest.raises(RateLimitError) as caught:
            await gate.acquire()
        assert caught.value.details["max_concurrency"] == 1
        assert caught.value.details["provider"] == "image"
        gate.release()
        assert gate.active == 0
        assert gate.available() is True

    asyncio.run(scenario())


def test_a_waiting_concurrency_limiter_queues_until_release() -> None:
    async def scenario() -> list[int]:
        gate = ConcurrencyLimiter(1)
        await gate.acquire()
        order: list[int] = []

        async def waiter(index: int) -> None:
            await gate.acquire()
            order.append(index)
            gate.release()

        tasks = [asyncio.create_task(waiter(index)) for index in range(3)]
        await asyncio.sleep(0)
        assert order == []

        gate.release()
        await asyncio.gather(*tasks)
        return order

    assert asyncio.run(scenario()) == [0, 1, 2]


def test_a_bucket_reads_max_concurrency_from_provider_metadata() -> None:
    bucket = TokenBucket(
        rate_limit=RateLimit(requests_per_minute=120, max_concurrency=3),
        name="image",
    )

    assert bucket.max_concurrency == 3
    assert bucket.concurrency.max_concurrency == 3
    assert bucket.active == 0


def test_execute_queues_calls_above_the_provider_concurrency_cap() -> None:
    image = FakeProvider(
        "image",
        ["image_gen"],
        RateLimit(requests_per_minute=0, max_concurrency=2),
    )
    router = make_router(image)
    active = 0
    peak = 0

    async def scenario() -> None:
        nonlocal active, peak
        release = asyncio.Event()
        reached = asyncio.Event()

        async def operation(provider: Provider) -> str:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                reached.set()
            await release.wait()
            active -= 1
            return f"{provider.name}:ok"

        tasks = [
            asyncio.create_task(router.execute("image_gen", operation))
            for _ in range(5)
        ]
        await asyncio.wait_for(reached.wait(), timeout=5)
        await asyncio.sleep(0)
        assert peak == 2
        assert active == 2

        release.set()
        assert await asyncio.gather(*tasks) == ["image:ok"] * 5
        assert peak == 2
        assert router.limiter_for("image").active == 0

    asyncio.run(scenario())


def test_execute_falls_back_when_the_concurrency_cap_is_full() -> None:
    first = FakeProvider(
        "first",
        ["image_gen"],
        RateLimit(requests_per_minute=0, max_concurrency=1),
    )
    second = FakeProvider("second", ["image_gen"], RateLimit(requests_per_minute=0))
    limiter = TokenBucket(
        rate_limit=first.metadata.rate_limit,
        name="first",
        on_limit=OnRateLimit.RAISE,
    )
    router = make_router(first, second, limiters={"first": limiter})
    calls: list[str] = []

    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def first_operation(provider: Provider) -> str:
            calls.append(provider.name)
            entered.set()
            await release.wait()
            return f"{provider.name}:ok"

        async def second_operation(provider: Provider) -> str:
            calls.append(provider.name)
            return f"{provider.name}:ok"

        task = asyncio.create_task(router.execute("image_gen", first_operation))
        await entered.wait()
        assert [provider.name for provider in router.candidates("image_gen")] == [
            "second"
        ]

        result = await router.execute("image_gen", second_operation)
        assert result == "second:ok"

        release.set()
        assert await task == "first:ok"

    asyncio.run(scenario())

    assert calls == ["first", "second"]
