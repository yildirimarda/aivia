"""Schedules and deliveries crossing spring forward and fall back fire exactly
once at the intended local time.

Berlin DST in 2026:
  * spring forward: 2026-03-29, 02:00 CET becomes 03:00 CEST (02:00-02:59 local
    does not exist, offsets UTC+1 -> UTC+2).
  * fall back:      2026-10-25, 03:00 CEST becomes 02:00 CET (02:00-02:59 local
    occurs twice, offsets UTC+2 -> UTC+1).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import Engine

from aivia.core.config.orders import DeliveryConfig
from aivia.core.config.schedule import ScheduleConfig
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.delivery import ScheduledDelivery
from aivia.core.jobs import JobQueue
from aivia.core.locks import RunLockManager
from aivia.core.models.io import Delivery
from aivia.core.models.provider import ProviderMeta
from aivia.core.scheduler import CronScheduler, next_fire_time
from aivia.providers import DeliveryChain
from aivia.providers.base import HealthStatus, ProviderHealth
from aivia.providers.protocols.delivery import (
    CAPABILITY_DELIVERY,
    DeliveryArtifact,
    DeliveryMethod,
    DeliveryReceipt,
    DeliveryRequest,
)

BERLIN = ZoneInfo("Europe/Berlin")
BERLIN_TZ = "Europe/Berlin"


def utc(y: int, m: int, d: int, h: int, mi: int = 0, s: int = 0) -> datetime:
    return datetime(y, m, d, h, mi, s, tzinfo=UTC)


class Clock:
    """A mutable clock so tests never depend on wall time."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def engine(tmp_path: Path) -> Generator[Engine, None, None]:
    database = tmp_path / "aivia.db"
    created = create_database_engine(database)
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def queue(engine: Engine) -> JobQueue:
    return JobQueue(engine)


@pytest.fixture
def locks(tmp_path: Path) -> RunLockManager:
    return RunLockManager(max_parallel=1, lock_dir=tmp_path / "locks")


# ---------------------------------------------------------------------------
# next_fire_time: spring forward


def test_spring_forward_daily_fires_at_intended_local_time_once() -> None:
    # 09:00 local on the spring-forward day exists (09:00 CEST = 07:00 UTC).
    fire = next_fire_time("0 9 * * *", BERLIN_TZ, utc(2026, 3, 28, 23, 0))
    assert fire == utc(2026, 3, 29, 7, 0)
    assert fire.astimezone(BERLIN).hour == 9
    # The following fire is the next day, never a repeat of the same day.
    assert next_fire_time("0 9 * * *", BERLIN_TZ, fire) == utc(2026, 3, 30, 7, 0)


def test_spring_forward_nonexistent_hour_is_skipped_and_fires_once() -> None:
    # 02:00 local does not exist on 2026-03-29 (the clock jumps 02:00 -> 03:00).
    fire = next_fire_time("0 2 * * *", BERLIN_TZ, utc(2026, 3, 28, 23, 0))
    assert fire == utc(2026, 3, 30, 0, 0)
    assert fire.astimezone(BERLIN).hour == 2
    # No fire lands on the nonexistent spring-forward hour; the next fire is
    # the following day at the intended local time, exactly once.
    assert next_fire_time("0 2 * * *", BERLIN_TZ, fire) == utc(2026, 3, 31, 0, 0)


# ---------------------------------------------------------------------------
# next_fire_time: fall back
# ---------------------------------------------------------------------------


def test_fall_back_daily_fires_at_intended_local_time_once() -> None:
    # 09:00 local occurs once on the fall-back day (09:00 CET = 08:00 UTC).
    fire = next_fire_time("0 9 * * *", BERLIN_TZ, utc(2026, 10, 24, 22, 0))
    assert fire == utc(2026, 10, 25, 8, 0)
    assert fire.astimezone(BERLIN).hour == 9
    assert next_fire_time("0 9 * * *", BERLIN_TZ, fire) == utc(2026, 10, 26, 8, 0)


def test_fall_back_repeated_hour_fires_once_first_occurrence() -> None:
    # 02:00 local occurs twice on 2026-10-25. The scheduler fires at the first
    # occurrence (02:00 CEST = 00:00 UTC) and never at the second (01:00 UTC).
    fire = next_fire_time("0 2 * * *", BERLIN_TZ, utc(2026, 10, 24, 22, 0))
    assert fire == utc(2026, 10, 25, 0, 0)
    assert fire.astimezone(BERLIN).hour == 2
    assert fire.astimezone(BERLIN).utcoffset() == timedelta(hours=2)
    # Stepping forward from the first occurrence lands on the next day, never
    # on the repeated second occurrence of the same wall-clock time.
    assert next_fire_time("0 2 * * *", BERLIN_TZ, fire) == utc(2026, 10, 26, 1, 0)


# ---------------------------------------------------------------------------
# CronScheduler.tick: enqueues exactly once across a DST transition
# ---------------------------------------------------------------------------


def test_tick_enqueues_once_around_spring_forward(
    queue: JobQueue, locks: RunLockManager
) -> None:
    schedule = ScheduleConfig.from_mapping(
        {
            "schedules": {
                "daily_nine": {
                    "profile": "default",
                    "cron": "0 9 * * *",
                    "timezone": "Europe/Berlin",
                }
            }
        }
    )
    clock = Clock(utc(2026, 3, 28, 23, 0))
    scheduler = CronScheduler(schedule, queue, locks, clock=clock)
    # First tick schedules the spring-forward fire at 09:00 CEST (07:00 UTC).
    assert scheduler.tick() == []
    clock.now = utc(2026, 3, 29, 7, 0)
    jobs = scheduler.tick()
    assert [j.schedule_name for j in jobs] == ["daily_nine"]
    assert jobs[0].job.not_before == utc(2026, 3, 29, 7, 0)
    assert queue.depth() == 1
    # The pending fire advanced to the next day; ticking later in the same day
    # does not enqueue a second time.
    clock.now = utc(2026, 3, 29, 8, 0)
    assert scheduler.tick() == []
    assert queue.depth() == 1


def test_tick_enqueues_once_around_fall_back_repeated_hour(
    queue: JobQueue, locks: RunLockManager
) -> None:
    schedule = ScheduleConfig.from_mapping(
        {
            "schedules": {
                "daily_two": {
                    "profile": "default",
                    "cron": "0 2 * * *",
                    "timezone": "Europe/Berlin",
                }
            }
        }
    )
    clock = Clock(utc(2026, 10, 24, 22, 0))
    scheduler = CronScheduler(schedule, queue, locks, clock=clock)
    # First tick schedules the fall-back fire at the first 02:00 (00:00 UTC).
    assert scheduler.tick() == []
    clock.now = utc(2026, 10, 25, 0, 0)
    jobs = scheduler.tick()
    assert [j.schedule_name for j in jobs] == ["daily_two"]
    assert jobs[0].job.not_before == utc(2026, 10, 25, 0, 0)
    assert queue.depth() == 1
    # The clock now lies in the repeated second 02:00 (02:00 CET = 01:00 UTC):
    # no second job is enqueued for the same wall-clock time.
    clock.now = utc(2026, 10, 25, 1, 0)
    assert scheduler.tick() == []
    assert queue.depth() == 1


# ---------------------------------------------------------------------------
# ScheduledDelivery: due instant maps the intended local time, exactly once
# ---------------------------------------------------------------------------


def test_delivery_due_at_spring_forward_intended_local_time() -> None:
    naive = datetime(2026, 3, 29, 9, 0, 0)
    delivery = Delivery.model_construct(scheduled_for=naive)
    service = ScheduledDelivery(_chain({"chain": ["a"]}, {"a": _FakeDelivery("a")}))
    due = service.due_at(delivery, timezone=BERLIN_TZ)
    assert due == utc(2026, 3, 29, 7, 0)  # 09:00 CEST = 07:00 UTC
    before = utc(2026, 3, 29, 6, 59)
    assert service.is_due(delivery, timezone=BERLIN_TZ, now=before) is False
    assert service.is_due(delivery, timezone=BERLIN_TZ, now=due) is True


def test_delivery_due_at_fall_back_intended_local_time() -> None:
    naive = datetime(2026, 10, 25, 9, 0, 0)
    delivery = Delivery.model_construct(scheduled_for=naive)
    service = ScheduledDelivery(_chain({"chain": ["a"]}, {"a": _FakeDelivery("a")}))
    due = service.due_at(delivery, timezone=BERLIN_TZ)
    assert due == utc(2026, 10, 25, 8, 0)  # 09:00 CET = 08:00 UTC
    before = utc(2026, 10, 25, 7, 59)
    assert service.is_due(delivery, timezone=BERLIN_TZ, now=before) is False
    assert service.is_due(delivery, timezone=BERLIN_TZ, now=due) is True


def test_delivery_ambiguous_fall_back_time_uses_one_occurrence() -> None:
    # 02:00 local is ambiguous on the fall-back day; fold defaults to 0, i.e.
    # the first occurrence (02:00 CEST = 00:00 UTC), never both.
    naive = datetime(2026, 10, 25, 2, 0, 0)
    delivery = Delivery.model_construct(scheduled_for=naive)
    service = ScheduledDelivery(_chain({"chain": ["a"]}, {"a": _FakeDelivery("a")}))
    due = service.due_at(delivery, timezone=BERLIN_TZ)
    assert due == utc(2026, 10, 25, 0, 0)
    before = utc(2026, 10, 24, 23, 59)
    assert service.is_due(delivery, timezone=BERLIN_TZ, now=before) is False
    assert service.is_due(delivery, timezone=BERLIN_TZ, now=due) is True


def test_deliver_invokes_the_chain_exactly_once_at_due_time() -> None:
    clock_now = utc(2026, 10, 25, 7, 59, 30)
    clock = _AdvancingClock(clock_now)
    provider = _FakeDelivery("a")
    chain = _chain({"chain": ["a"], "max_attempts": 1}, {"a": provider})
    service = ScheduledDelivery(chain, clock=clock, sleep=_advancing_sleep(clock))
    # 09:00 CET on the fall-back day = 08:00 UTC.
    delivery = Delivery.model_construct(scheduled_for=datetime(2026, 10, 25, 9, 0, 0))
    request = _request()

    asyncio.run(service.deliver(delivery, request, timezone=BERLIN_TZ))

    assert provider.calls == 1
    assert delivery.is_delivered is True


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _chain(config_values: dict[str, Any], providers: dict[str, Any]) -> DeliveryChain:
    values = dict(config_values)
    values.setdefault("providers", {})
    for name in providers:
        values["providers"].setdefault(name, {})
    values.setdefault("default", next(iter(providers)))
    config = DeliveryConfig.from_mapping(values)
    return DeliveryChain(config, providers, sleep=lambda _s: None)


class _FakeDelivery:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    @property
    def metadata(self) -> ProviderMeta:
        return ProviderMeta(name=self.name, capabilities=[CAPABILITY_DELIVERY])

    def supports(self, capability: str) -> bool:
        return capability == CAPABILITY_DELIVERY

    async def health(self) -> ProviderHealth:
        return ProviderHealth(status=HealthStatus.HEALTHY, message="ok")

    async def close(self) -> None:
        return None

    async def deliver(self, request: DeliveryRequest) -> DeliveryReceipt:
        self.calls += 1
        return DeliveryReceipt(
            method=request.method,
            target=request.target,
            provider=self.name,
            reference="ref-1",
            delivered_at=datetime.now(UTC),
        )


def _request() -> DeliveryRequest:
    return DeliveryRequest(
        method=DeliveryMethod.FOLDER,
        target="/tmp/deliveries",
        order_id="order-1",
        run_id="run-1",
        subject="Your card",
        message="Enjoy",
        artifacts=[
            DeliveryArtifact(data=b"card", media_type="image/png", filename="card.png")
        ],
    )


class _AdvancingClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _advancing_sleep(clock: _AdvancingClock) -> Callable[[float], object]:
    async def _sleep(delay: float) -> None:
        clock.now += timedelta(seconds=delay)

    return _sleep
