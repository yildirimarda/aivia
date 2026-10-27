"""Pytest configuration for opt-in live tests and their spend cap."""

from __future__ import annotations

import os
from decimal import Decimal, InvalidOperation

import pytest

LIVE_BUDGET_ENV = "AIVA_LIVE_BUDGET_USD"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help=f"run tests marked live; requires {LIVE_BUDGET_ENV}",
    )


def _live_budget() -> Decimal:
    raw = os.environ.get(LIVE_BUDGET_ENV)
    if raw is None or not raw.strip():
        raise pytest.UsageError(
            f"--live requires {LIVE_BUDGET_ENV} to be a positive amount"
        )

    try:
        budget = Decimal(raw)
    except InvalidOperation as exc:
        raise pytest.UsageError(
            f"{LIVE_BUDGET_ENV} must be a positive amount, got {raw!r}"
        ) from exc

    if not budget.is_finite() or budget <= 0:
        raise pytest.UsageError(
            f"{LIVE_BUDGET_ENV} must be a positive amount, got {raw!r}"
        )
    return budget


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--live"):
        _live_budget()


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--live"):
        return

    skip_live = pytest.mark.skip(reason="live tests require --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture
def live_budget_usd() -> Decimal:
    """The spend cap configured for this live run."""
    return _live_budget()
