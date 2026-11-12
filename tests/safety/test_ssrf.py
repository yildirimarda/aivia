"""SSRF hardening: numeric IP encodings, IPv6-mapped addresses, DNS rebinding
and per-hop redirect validation.

These tests exercise :func:`aivia.core.ssrf.validate_ssrf_target` and
:func:`aivia.core.ssrf.is_blocked_ip` directly, plus the page fetch provider's
per-hop revalidation of redirect targets through an injectable resolver and
mock transport. No real DNS is performed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from aivia.core.errors import InvalidRequestError
from aivia.core.ssrf import is_blocked_ip, validate_ssrf_target
from aivia.providers.pagefetch import HttpPageFetchProvider
from aivia.providers.protocols import PageFetchRequest

URL = "https://example.com/article"
PUBLIC_IP = "93.184.216.34"
PRIVATE_IP = "10.0.0.7"


def _provider(handler: Any, **kwargs: Any) -> HttpPageFetchProvider:
    return HttpPageFetchProvider(
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _resolver(addresses: list[str]) -> Callable[[str, int], list[str]]:
    """Return an injectable resolver yielding one address list per call."""

    def resolve(host: str, port: int) -> list[str]:
        return list(addresses)

    return resolve


def _rebinding_resolver() -> Callable[[str, int], list[str]]:
    """A resolver that first answers public, then private (DNS rebinding)."""

    answers = iter([[PUBLIC_IP], [PRIVATE_IP]])

    def resolve(host: str, port: int) -> list[str]:
        return next(answers)

    return resolve


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "0.0.0.0",
        "::1",
        "fe80::1",
        "ff02::1",
        "224.0.0.1",
        "240.0.0.1",
        "::ffff:127.0.0.1",
        "::ffff:10.0.0.1",
        "::ffff:169.254.169.254",
        "2130706433",
        "0x7f000001",
    ],
)
def test_is_blocked_ip_rejects_private_and_internal(address: str) -> None:
    assert is_blocked_ip(address) is True


@pytest.mark.parametrize(
    "address",
    [
        PUBLIC_IP,
        "2606:2800:220:1:248:1893:25c8:1946",
        "8.8.8.8",
    ],
)
def test_is_blocked_ip_allows_public_addresses(address: str) -> None:
    assert is_blocked_ip(address) is False


def test_is_blocked_ip_rejects_ipaddress_objects() -> None:
    import ipaddress

    assert is_blocked_ip(ipaddress.IPv4Address("127.0.0.1")) is True
    assert is_blocked_ip(ipaddress.IPv6Address("::ffff:127.0.0.1")) is True
    assert is_blocked_ip(ipaddress.IPv4Address(PUBLIC_IP)) is False


def test_is_blocked_ip_ignores_unparseable_strings() -> None:
    assert is_blocked_ip("not an address") is False
    assert is_blocked_ip("") is False


@pytest.mark.parametrize(
    "url",
    [
        "http://2130706433/",
        "http://0x7f000001/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://[::ffff:10.0.0.1]/",
        "http://169.254.169.254/latest/meta-data/",
        "http://localhost/",
        "http://metadata.google.internal/",
        "http://printer.local/",
    ],
)
def test_validate_rejects_numeric_and_internal_literal_hosts(url: str) -> None:
    with pytest.raises(InvalidRequestError):
        asyncio.run(validate_ssrf_target(url))


def test_validate_rejects_a_public_host_that_resolves_to_private() -> None:
    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(validate_ssrf_target(URL, resolver=_resolver([PRIVATE_IP])))

    assert caught.value.details["host"] == "example.com"
    assert caught.value.details["address"] == PRIVATE_IP


def test_validate_rejects_when_any_resolved_address_is_private() -> None:
    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(
            validate_ssrf_target(
                URL, resolver=_resolver([PUBLIC_IP, PRIVATE_IP, PUBLIC_IP])
            )
        )

    assert caught.value.details["address"] == PRIVATE_IP


def test_validate_accepts_a_public_host_resolving_only_to_public() -> None:
    asyncio.run(validate_ssrf_target(URL, resolver=_resolver([PUBLIC_IP])))


def test_validate_rebinding_resolver_is_blocked_on_second_resolution() -> None:
    resolver = _rebinding_resolver()
    asyncio.run(validate_ssrf_target(URL, resolver=resolver))
    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(validate_ssrf_target(URL, resolver=resolver))

    assert caught.value.details["host"] == "example.com"
    assert caught.value.details["address"] == PRIVATE_IP


def test_validate_applies_allowlist_before_dns() -> None:
    calls: list[tuple[str, int]] = []

    def resolver(host: str, port: int) -> list[str]:
        calls.append((host, port))
        return [PUBLIC_IP]

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(
            validate_ssrf_target(URL, resolver=resolver, allowed_hosts=["allowed.test"])
        )

    assert caught.value.details["host"] == "example.com"
    assert calls == []


def test_validate_permits_listed_host_resolving_to_public() -> None:
    asyncio.run(
        validate_ssrf_target(
            URL, resolver=_resolver([PUBLIC_IP]), allowed_hosts=["example.com"]
        )
    )


def test_redirect_to_private_address_is_rejected_on_second_hop() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            302, headers={"location": "http://127.0.0.1/admin"}
        ),
        resolver=_resolver([PUBLIC_IP]),
    )

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(provider.fetch(PageFetchRequest(url=URL)))

    assert caught.value.details["host"] == "127.0.0.1"


def test_redirect_chain_ending_at_private_is_rejected_on_final_hop() -> None:
    responses = iter(
        [
            httpx.Response(302, headers={"location": "https://example.net/"}),
            httpx.Response(302, headers={"location": "http://10.0.0.7/"}),
            httpx.Response(200, text="should not happen"),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return next(responses)

    provider = _provider(handler, resolver=_resolver([PUBLIC_IP]))

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(provider.fetch(PageFetchRequest(url=URL)))

    assert caught.value.details["host"] == "10.0.0.7"


def test_redirect_to_numeric_encoded_private_is_rejected() -> None:
    provider = _provider(
        lambda request: httpx.Response(302, headers={"location": "http://0x7f000001/"}),
        resolver=_resolver([PUBLIC_IP]),
    )

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(provider.fetch(PageFetchRequest(url=URL)))

    assert caught.value.details["host"] == "0x7f000001"
