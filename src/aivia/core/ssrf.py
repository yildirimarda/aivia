"""Shared SSRF policy for outbound fetches.

Every outbound fetch resolves the target host and rejects any address that is
not globally routable, so a caller cannot be pointed at loopback, private,
link-local, reserved or metadata endpoints. An allowlist is applied before DNS,
so an unlisted host never triggers a lookup. The resolver is injectable so the
offline suite can exercise address validation without touching DNS.

The policy is shared by page fetch, remote reference downloads, webhooks,
search results and provider inputs such as a remote video first frame.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable, Iterable, Sequence
from urllib.parse import urlsplit

from aivia.core.errors import ConfigError, InvalidRequestError, RetryableError

__all__ = [
    "BLOCKED_HOSTNAMES",
    "BLOCKED_HOST_SUFFIXES",
    "ResolveHost",
    "host_is_allowed",
    "is_blocked_ip",
    "normalise_allowed_host",
    "normalise_allowed_hosts",
    "resolve_host",
    "resolve_port",
    "validate_ssrf_target",
]

ResolveHost = Callable[[str, int], Sequence[str] | Awaitable[Sequence[str]]]
"""A callable returning the addresses for ``(host, port)``."""

BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata",
        "metadata.google.internal",
        "instance-data",
        "host.docker.internal",
        "gateway.docker.internal",
    }
)
"""Well-known host names that must never be fetched."""

BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".home.arpa")
"""Host suffixes reserved for local or internal names."""


def _literal_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    candidate = host.strip().strip("[]")
    try:
        return ipaddress.ip_address(candidate)
    except ValueError:
        pass
    lowered = candidate.lower()
    try:
        if lowered.startswith("0x"):
            return ipaddress.ip_address(int(lowered, 16))
        if lowered.isdigit():
            return ipaddress.ip_address(int(lowered, 10))
    except (ValueError, OverflowError):
        return None
    return None


def is_blocked_ip(value: str | ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return whether *value* is an address an outbound fetch must never contact."""
    address = _literal_ip(value) if isinstance(value, str) else value
    if address is None:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if (
        address.is_unspecified
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_private
        or address.is_reserved
    ):
        return True
    return not address.is_global


def normalise_allowed_host(value: str) -> str:
    """Normalise one allowlist entry to a bare lower-case host name."""
    if not isinstance(value, str):
        raise TypeError("allowed host must be a string")
    text = value.strip().lower().rstrip(".")
    if not text:
        raise ConfigError("allowed host must be non-empty")
    if "://" in text:
        host = urlsplit(text).hostname
        if not host:
            raise ConfigError(f"allowed host must name a host: {value!r}")
        text = host.lower().rstrip(".")
    text = text.strip("[]")
    if text.startswith("*."):
        text = text[2:]
    if not text or any(character.isspace() for character in text):
        raise ConfigError(f"allowed host must be a bare host name: {value!r}")
    return text


def normalise_allowed_hosts(
    allowed_hosts: Iterable[str] | None,
) -> tuple[str, ...]:
    """Normalise an allowlist, rejecting duplicate entries."""
    hosts: list[str] = []
    seen: set[str] = set()
    for entry in allowed_hosts or ():
        host = normalise_allowed_host(entry)
        if host in seen:
            raise ConfigError(f"duplicate allowed host: {host}")
        seen.add(host)
        hosts.append(host)
    return tuple(hosts)


def host_is_allowed(host: str, allowed_hosts: Sequence[str]) -> bool:
    """Return whether *host* matches the allowlist, including subdomains.

    An empty allowlist allows every host; the SSRF guard still applies.
    """
    if not allowed_hosts:
        return True
    candidate = host.strip().lower().rstrip(".")
    for entry in allowed_hosts:
        normalised = entry.strip().lower().rstrip(".").strip("[]")
        if normalised.startswith("*."):
            normalised = normalised[2:]
        if candidate == normalised:
            return True
        if not _literal_ip(normalised) and candidate.endswith(f".{normalised}"):
            return True
    return False


def resolve_port(url: str, *, label: str = "url") -> int:
    """Return the explicit or default port for *url*."""
    parsed = urlsplit(url)
    try:
        return parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise InvalidRequestError(
            f"{label} URL has an invalid port: {url!r}",
            details={"url": url},
        ) from exc


async def resolve_host(
    host: str,
    port: int,
    *,
    resolver: ResolveHost | None = None,
    skip_dns: bool = False,
) -> Sequence[str]:
    """Resolve *host* through *resolver* or the system resolver."""
    if resolver is not None:
        resolved = resolver(host, port)
        if isinstance(resolved, Awaitable):
            return await resolved
        return resolved
    if skip_dns:
        return ()
    try:
        infos = await asyncio.to_thread(
            socket.getaddrinfo,
            host,
            port,
            socket.AF_UNSPEC,
            socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise RetryableError(
            "host could not be resolved",
            details={"host": host, "error": type(exc).__name__},
        ) from exc
    addresses: list[str] = []
    for info in infos:
        address = str(info[4][0])
        if address not in addresses:
            addresses.append(address)
    return addresses


async def validate_ssrf_target(
    url: str,
    *,
    resolver: ResolveHost | None = None,
    allowed_hosts: Sequence[str] = (),
    skip_dns: bool = False,
    label: str = "url",
) -> None:
    """Reject *url* when its host or any resolved address is not public.

    *label* names the kind of URL in error messages, for example ``"page"`` or
    ``"first frame"``. An allowlist is checked before DNS, so an unlisted host
    never triggers a lookup. ``skip_dns`` is for offline transports that cannot
    resolve names; the literal and host-name checks still apply.
    """
    parsed = urlsplit(url)
    host = parsed.hostname
    if host is None:
        raise InvalidRequestError(
            f"{label} URL must name a host: {url!r}",
            details={"url": url},
        )
    if not host_is_allowed(host, allowed_hosts):
        raise InvalidRequestError(
            f"host is not in the {label} allowlist: {host}",
            details={"url": url, "host": host},
        )
    literal = _literal_ip(host)
    if literal is not None:
        if is_blocked_ip(literal):
            raise InvalidRequestError(
                f"{label} URL points at a blocked address: {host}",
                details={"url": url, "host": host},
            )
        return
    lowered = host.lower().rstrip(".")
    if lowered in BLOCKED_HOSTNAMES or lowered.endswith(BLOCKED_HOST_SUFFIXES):
        raise InvalidRequestError(
            f"{label} URL points at a blocked host: {host}",
            details={"url": url, "host": host},
        )
    addresses = await resolve_host(
        host,
        resolve_port(url, label=label),
        resolver=resolver,
        skip_dns=skip_dns,
    )
    for address in addresses:
        if is_blocked_ip(address):
            raise InvalidRequestError(
                f"{label} host resolves to a blocked address: {host}",
                details={"url": url, "host": host, "address": address},
            )
