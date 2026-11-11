"""Aggregator publisher covering several platforms with per-account gap reporting.

The aggregator wraps one publisher per platform and routes each validated
:class:`~aivia.providers.protocols.publish.PublishRequest` to the publisher
matching ``request.platform``. When no publisher covers a platform, the attempt
is not silently dropped: the aggregator returns a result flagged as a capability
gap so callers can surface it per account.

``capability_gaps()`` reports, for every configured account, the platforms it
expects to cover but has no publisher for.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from aivia.core.config.http import HttpClientConfig
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.providers.base import Provider
from aivia.providers.protocols.publish import (
    CAPABILITY_PUBLISH,
    Publisher,
    PublishRequest,
    PublishResult,
)

__all__ = [
    "AggregatorPublisher",
    "CapabilityGap",
]

_DEFAULT_METADATA = ProviderMeta(
    name="aggregator",
    capabilities=[CAPABILITY_PUBLISH],
    pricing={CAPABILITY_PUBLISH: ProviderPrice(amount_usd=0.0)},
)


def _normalize_platform(value: Any) -> str:
    """Normalize a platform id the same way the platform catalog does."""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_account(value: Any) -> str:
    return str(value).strip().lower()


@dataclass(frozen=True, slots=True)
class CapabilityGap:
    """One platform an account expects to cover but has no publisher for."""

    account: str
    platform: str
    reason: str = "no publisher configured"


class AggregatorPublisher(Provider):
    """Publish to several platforms and report capability gaps per account.

    ``publishers`` maps a platform id to the publisher that serves it.
    ``accounts`` maps an account to the platforms it expects to cover; any
    expected platform without a publisher is a capability gap reported by
    :meth:`capability_gaps`.
    """

    default_metadata: ClassVar[ProviderMeta] = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        publishers: Mapping[str, Publisher] | None = None,
        accounts: Mapping[str, Collection[str]] | None = None,
        metadata: ProviderMeta | Mapping[str, Any] | None = None,
        name: str | None = None,
        health_timeout_seconds: float | None = None,
        close_timeout_seconds: float | None = None,
        http_config: HttpClientConfig | None = None,
    ) -> None:
        super().__init__(
            metadata,
            name=name,
            health_timeout_seconds=health_timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            http_config=http_config,
        )
        normalized: dict[str, Publisher] = {}
        for platform, publisher in (publishers or {}).items():
            key = _normalize_platform(platform)
            if not key:
                raise ValueError("publisher platforms must be non-empty")
            if key in normalized:
                raise ValueError(f"duplicate publisher for platform {key!r}")
            if not callable(getattr(publisher, "publish", None)):
                raise TypeError(f"publisher for {key!r} must expose publish(request)")
            normalized[key] = publisher
        self._publishers = normalized

        expected: dict[str, set[str]] = {}
        for account, platforms in (accounts or {}).items():
            account_key = _normalize_account(account)
            if not account_key:
                raise ValueError("account names must be non-empty")
            if account_key in expected:
                raise ValueError(f"duplicate account {account_key!r}")
            platform_set: set[str] = set()
            for platform in platforms:
                platform_key = _normalize_platform(platform)
                if not platform_key:
                    raise ValueError("expected platforms must be non-empty")
                platform_set.add(platform_key)
            expected[account_key] = platform_set
        self._accounts = expected

    @property
    def platforms(self) -> tuple[str, ...]:
        """The platforms this aggregator can publish to."""
        return tuple(sorted(self._publishers))

    def publisher_for(self, platform: str) -> Publisher | None:
        """Return the publisher for *platform*, or ``None`` when uncovered."""
        return self._publishers.get(_normalize_platform(platform))

    def supports_platform(self, platform: str) -> bool:
        """Return whether *platform* has a configured publisher."""
        return _normalize_platform(platform) in self._publishers

    def capability_gaps(self) -> list[CapabilityGap]:
        """Every expected platform without a publisher, grouped by account."""

        gaps: list[CapabilityGap] = []
        for account in sorted(self._accounts):
            for platform in sorted(self._accounts[account]):
                if platform not in self._publishers:
                    gaps.append(
                        CapabilityGap(
                            account=account,
                            platform=platform,
                            reason="no publisher configured",
                        )
                    )
        return gaps

    def gaps_by_account(self) -> dict[str, list[CapabilityGap]]:
        """Capability gaps grouped by account (accounts without gaps omitted)."""

        grouped: dict[str, list[CapabilityGap]] = {}
        for gap in self.capability_gaps():
            grouped.setdefault(gap.account, []).append(gap)
        return grouped

    async def publish(self, request: PublishRequest) -> PublishResult:
        """Publish *request* to its platform's publisher or report a gap."""

        if not isinstance(request, PublishRequest):
            raise TypeError("request must be a PublishRequest")
        platform = _normalize_platform(request.platform)
        publisher = self._publishers.get(platform)
        if publisher is None:
            return PublishResult(
                platform=platform,
                account=request.account,
                provider=self.name,
                remote_id="",
                metadata={
                    "capability_gap": True,
                    "reason": "no publisher configured",
                    "platform": platform,
                },
            )
        return await publisher.publish(request)
