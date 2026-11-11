"""Aggregator publisher: multi-platform routing and per-account gap reporting."""

from __future__ import annotations

import asyncio

import pytest

from aivia.providers.aggregator_publish import AggregatorPublisher, CapabilityGap
from aivia.providers.mock import MockPublisher
from aivia.providers.protocols.publish import (
    CAPABILITY_PUBLISH,
    Publisher,
    PublishRequest,
    PublishResult,
)


def _publisher(name: str) -> MockPublisher:
    return MockPublisher(name=name)


def _request(
    *,
    platform: str = "x",
    account: str = "acct",
    caption: str = "hello",
    dry_run: bool = False,
) -> PublishRequest:
    return PublishRequest(
        platform=platform, account=account, caption=caption, dry_run=dry_run
    )


def test_metadata_and_capabilities_describe_the_aggregator() -> None:
    aggregator = AggregatorPublisher()

    assert aggregator.name == "aggregator"
    assert aggregator.capabilities == (CAPABILITY_PUBLISH,)
    assert aggregator.supports(CAPABILITY_PUBLISH)
    assert aggregator.platforms == ()


def test_aggregator_is_a_publisher() -> None:
    aggregator = AggregatorPublisher(publishers={"x": _publisher("x-pub")})

    assert isinstance(aggregator, Publisher)


def test_routes_to_the_platform_publisher_and_passes_the_result_through() -> None:
    x_pub = _publisher("x-pub")
    ig_pub = _publisher("instagram-pub")
    aggregator = AggregatorPublisher(
        publishers={"x": x_pub, "instagram": ig_pub},
    )

    assert aggregator.platforms == ("instagram", "x")
    assert aggregator.supports_platform("x")
    assert aggregator.publisher_for("x") is x_pub
    assert aggregator.publisher_for("tiktok") is None

    result = asyncio.run(
        aggregator.publish(_request(platform="x", account="acct-1", caption="hi there"))
    )

    assert isinstance(result, PublishResult)
    assert result.platform == "x"
    assert result.account == "acct-1"
    assert result.provider == "x-pub"
    assert result.remote_id.startswith("mock-post-")
    assert result.is_live
    assert len(x_pub.calls) == 1
    assert x_pub.calls[0].request.caption == "hi there"
    assert ig_pub.calls == ()


def test_instagram_request_routes_to_instagram_publisher() -> None:
    x_pub = _publisher("x-pub")
    ig_pub = _publisher("instagram-pub")
    aggregator = AggregatorPublisher(
        publishers={"x": x_pub, "instagram": ig_pub},
    )

    result = asyncio.run(aggregator.publish(_request(platform="instagram")))

    assert result.provider == "instagram-pub"
    assert len(ig_pub.calls) == 1
    assert len(x_pub.calls) == 0


def test_dry_run_request_is_handled_by_the_platform_publisher() -> None:
    x_pub = _publisher("x-pub")
    aggregator = AggregatorPublisher(publishers={"x": x_pub})

    result = asyncio.run(aggregator.publish(_request(platform="x", dry_run=True)))

    assert result.dry_run is True
    assert result.is_live is False


def test_uncovered_platform_returns_a_capability_gap_result() -> None:
    aggregator = AggregatorPublisher(publishers={"x": _publisher("x-pub")})

    result = asyncio.run(
        aggregator.publish(_request(platform="tiktok", account="acct-2"))
    )

    assert isinstance(result, PublishResult)
    assert result.platform == "tiktok"
    assert result.account == "acct-2"
    assert result.provider == "aggregator"
    assert result.remote_id == ""
    assert result.is_live is False
    assert result.metadata["capability_gap"] is True
    assert result.metadata["reason"] == "no publisher configured"
    assert result.metadata["platform"] == "tiktok"


def test_capability_gaps_report_expected_but_uncovered_platforms_per_account() -> None:
    aggregator = AggregatorPublisher(
        publishers={"x": _publisher("x-pub")},
        accounts={
            "alice": ("x", "instagram", "tiktok"),
            "bob": ("x", "instagram"),
            "carol": ("x",),
        },
    )

    gaps = aggregator.capability_gaps()

    assert gaps == [
        CapabilityGap(account="alice", platform="instagram"),
        CapabilityGap(account="alice", platform="tiktok"),
        CapabilityGap(account="bob", platform="instagram"),
    ]


def test_gaps_by_account_groups_gaps_per_account() -> None:
    aggregator = AggregatorPublisher(
        publishers={"x": _publisher("x-pub")},
        accounts={
            "alice": ("x", "instagram"),
            "bob": ("tiktok",),
        },
    )

    by_account = aggregator.gaps_by_account()

    assert set(by_account) == {"alice", "bob"}
    assert [gap.platform for gap in by_account["alice"]] == ["instagram"]
    assert [gap.platform for gap in by_account["bob"]] == ["tiktok"]


def test_no_gaps_when_every_expected_platform_is_covered() -> None:
    aggregator = AggregatorPublisher(
        publishers={"x": _publisher("x-pub"), "instagram": _publisher("ig-pub")},
        accounts={"alice": ("x", "instagram")},
    )

    assert aggregator.capability_gaps() == []
    assert aggregator.gaps_by_account() == {}


def test_accounts_without_gaps_are_omitted_from_gaps_by_account() -> None:
    aggregator = AggregatorPublisher(
        publishers={"x": _publisher("x-pub"), "tiktok": _publisher("tt-pub")},
        accounts={"alice": ("x", "tiktok"), "bob": ("x",)},
    )

    assert aggregator.gaps_by_account() == {}


def test_duplicate_platform_or_account_is_rejected() -> None:
    x_pub = _publisher("x-pub")

    with pytest.raises(ValueError):
        AggregatorPublisher(publishers={"x": x_pub, "X": x_pub})

    with pytest.raises(ValueError):
        AggregatorPublisher(
            publishers={"x": x_pub},
            accounts={"alice": ("x",), "ALICE": ("x",)},
        )


def test_publisher_must_expose_publish() -> None:
    with pytest.raises(TypeError):
        AggregatorPublisher(publishers={"x": object()})  # type: ignore[dict-item]


def test_non_publish_request_is_rejected() -> None:
    aggregator = AggregatorPublisher(publishers={"x": _publisher("x-pub")})

    with pytest.raises(TypeError):
        asyncio.run(aggregator.publish("not a request"))  # type: ignore[arg-type]
