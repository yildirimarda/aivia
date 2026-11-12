"""Scope runs, orders and posts to an owner with optional per-owner budgets
and rate limits.

Single-user config (no owner) is left unchanged: an empty owner inherits the
global budget config and receives no per-owner rate limit.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine

from aivia.core.budget import BudgetGuard
from aivia.core.config import BudgetConfig
from aivia.core.config.ratelimit import RateLimiterConfig
from aivia.core.cost import CostTracker
from aivia.core.db import (
    OrderRepository,
    PostRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.errors import BudgetError, RateLimitError
from aivia.core.models.io import Post
from aivia.core.models.run import Run, RunKind
from aivia.core.owner_scope import (
    OwnerConfig,
    OwnerScope,
    budget_guard_for_owner,
    rate_limiter_for_owner,
)
from aivia.core.ratelimit import TokenBucket


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def tracker(engine: Engine) -> CostTracker:
    return CostTracker(engine)


# -- owner field on run and post models -------------------------------------


def test_run_carries_an_owner() -> None:
    run = Run(kind=RunKind.ORDER, owner="alice")
    assert run.owner == "alice"
    assert run.run_id == run.id


def test_run_owner_defaults_to_empty() -> None:
    run = Run(kind=RunKind.PIPELINE)
    assert run.owner == ""


def test_post_carries_an_owner() -> None:
    post = Post(platform="youtube", owner="bob")
    assert post.owner == "bob"


def test_post_owner_defaults_to_empty() -> None:
    post = Post(platform="youtube")
    assert post.owner == ""


# -- orders and posts are stored and read back with their owner -------------


def _record_order(repository: OrderRepository, **overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "state": "draft",
        "occasion": "birthday",
        "name": "Ada",
        "message": "Happy birthday",
        "language": "en",
        "profile": "default",
        "level": "standard",
    }
    values.update(overrides)
    return repository.create(**values)


def test_order_is_stored_and_read_back_with_its_owner(engine: Engine) -> None:
    repository = OrderRepository(engine)
    record = _record_order(repository, owner="alice")

    assert record["owner"] == "alice"

    stored = repository.get(record["id"])
    assert stored is not None
    assert stored["owner"] == "alice"

    orders = repository.list(owner="alice")
    assert len(orders) == 1
    assert orders[0]["owner"] == "alice"


def test_order_owner_defaults_to_empty(engine: Engine) -> None:
    repository = OrderRepository(engine)
    record = _record_order(repository)

    assert record["owner"] == ""
    stored = repository.get(record["id"])
    assert stored is not None
    assert stored["owner"] == ""


def test_post_is_stored_and_read_back_with_its_owner(engine: Engine) -> None:
    repository = PostRepository(engine)
    post = repository.create(platform="youtube", owner="bob")

    assert post.owner == "bob"

    stored = repository.get(post.id)
    assert stored is not None
    assert stored.owner == "bob"

    posts = repository.list(owner="bob")
    assert len(posts) == 1
    assert posts[0].owner == "bob"


def test_post_repository_owner_defaults_to_empty(engine: Engine) -> None:
    repository = PostRepository(engine)
    post = repository.create(platform="youtube")

    assert post.owner == ""
    stored = repository.get(post.id)
    assert stored is not None
    assert stored.owner == ""


# -- per-owner budgets -------------------------------------------------------


def test_owner_budget_cap_rejects_calls_above_it(tracker: CostTracker) -> None:
    scope = OwnerScope(
        {"alice": OwnerConfig(budget=BudgetConfig(per_run=0.50, per_call=0.25))}
    )
    guard = scope.budget_guard_for("alice", tracker)

    assert guard.check(0.20, run_id="run_1").allowed is True
    with pytest.raises(BudgetError) as caught:
        guard.check(0.30, run_id="run_1")

    assert caught.value.details["scope"] == "per_call"
    assert caught.value.details["cap"] == pytest.approx(0.25)


def test_owner_budget_cap_allows_calls_within_it(tracker: CostTracker) -> None:
    scope = OwnerScope({"alice": OwnerConfig(budget=BudgetConfig(per_run=0.50))})
    guard = scope.budget_guard_for("alice", tracker)

    assert guard.check(0.30, run_id="run_1").allowed is True
    assert guard.check(0.20, run_id="run_1").allowed is True


def test_owner_budget_is_scoped_per_owner(tracker: CostTracker) -> None:
    scope = OwnerScope(
        {
            "alice": OwnerConfig(budget=BudgetConfig(per_run=0.10)),
            "bob": OwnerConfig(budget=BudgetConfig(per_run=10.0)),
        }
    )
    alice = scope.budget_guard_for("alice", tracker)
    bob = scope.budget_guard_for("bob", tracker)

    tracker.record(
        provider="acme",
        model="image-1",
        operation="image_gen",
        units=1.0,
        unit_kind="image",
        amount_usd=0.05,
        run_id="run_alice",
    )

    assert alice.evaluate(0.06, run_id="run_alice").allowed is False
    with pytest.raises(BudgetError):
        alice.check(0.06, run_id="run_alice")

    assert bob.check(5.0, run_id="run_bob").allowed is True


def test_empty_owner_has_no_per_owner_budget_caps(tracker: CostTracker) -> None:
    scope = OwnerScope()
    guard = scope.budget_guard_for("", tracker)

    result = guard.check(1_000.0, run_id="run_1")
    assert result.allowed is True
    assert result.remaining_usd is None


def test_unconfigured_owner_inherits_global_defaults(tracker: CostTracker) -> None:
    scope = OwnerScope({"alice": OwnerConfig(budget=BudgetConfig(per_run=0.10))})
    guard = scope.budget_guard_for("carol", tracker)

    result = guard.check(1_000.0, run_id="run_1")
    assert result.allowed is True
    assert result.remaining_usd is None


# -- per-owner rate limits ---------------------------------------------------


class FakeClock:
    """A manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.value = 0.0

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


def test_owner_rate_limit_allows_calls_up_to_the_rate() -> None:
    clock = FakeClock()
    sleep = FakeSleep(clock)
    scope = OwnerScope(
        {"alice": OwnerConfig(rate_limit=RateLimiterConfig(requests_per_minute=2))}
    )
    limiter = scope.rate_limiter_for("alice", clock, sleep)
    assert limiter is not None

    import asyncio

    async def run() -> None:
        delays1 = await limiter.acquire()
        delays2 = await limiter.acquire()
        assert delays1 == 0.0
        assert delays2 == 0.0

    asyncio.run(run())


def test_owner_rate_limit_raises_when_exceeded() -> None:
    clock = FakeClock()
    sleep = FakeSleep(clock)
    scope = OwnerScope(
        {
            "alice": OwnerConfig(
                rate_limit=RateLimiterConfig(requests_per_minute=1, burst=0)
            )
        }
    )
    limiter = scope.rate_limiter_for("alice", clock, sleep)
    assert limiter is not None

    import asyncio

    async def run() -> None:
        await limiter.acquire()
        with pytest.raises(RateLimitError):
            await limiter.acquire(wait=False)

    asyncio.run(run())


def test_empty_owner_has_no_rate_limiter() -> None:
    scope = OwnerScope()
    assert scope.rate_limiter_for("", None, None) is None


def test_unconfigured_owner_has_no_rate_limiter() -> None:
    scope = OwnerScope({"alice": OwnerConfig()})
    assert scope.rate_limiter_for("carol", None, None) is None


# -- module-level convenience functions --------------------------------------


def test_module_level_budget_guard_for_owner(tracker: CostTracker) -> None:
    scope = OwnerScope({"alice": OwnerConfig(budget=BudgetConfig(per_call=0.10))})
    guard = budget_guard_for_owner(scope, "alice", tracker)

    assert isinstance(guard, BudgetGuard)
    assert guard.check(0.05, run_id="run_1").allowed is True


def test_module_level_rate_limiter_for_owner() -> None:
    scope = OwnerScope(
        {"alice": OwnerConfig(rate_limit=RateLimiterConfig(requests_per_minute=60))}
    )
    limiter = rate_limiter_for_owner(scope, "alice")

    assert isinstance(limiter, TokenBucket)


# -- config round trip -------------------------------------------------------


def test_owner_config_accepts_budget_and_rate_limit() -> None:
    config = OwnerConfig(
        budget=BudgetConfig(per_run=1.0),
        rate_limit=RateLimiterConfig(requests_per_minute=30, burst=5),
    )
    assert config.budget is not None
    assert config.budget.per_run == 1.0
    assert config.rate_limit is not None
    assert config.rate_limit.requests_per_minute == 30


def test_owner_config_defaults_to_none() -> None:
    config = OwnerConfig()
    assert config.budget is None
    assert config.rate_limit is None


def test_owner_scope_normalises_owner_names() -> None:
    scope = OwnerScope({"  Alice  ": OwnerConfig()})
    assert scope.config_for("Alice") is not None
    assert scope.config_for("alice") is None
