"""Local API security primitives: loopback binding and API key checks.

The local API must never be reachable from another machine by accident, so the
configured bind address is validated against the loopback ranges before the
server starts. Requests are authenticated with a constant-time comparison of a
key supplied through the ``X-API-Key`` header or a ``Bearer`` token.
"""

from __future__ import annotations

import hmac
import ipaddress
from collections.abc import Mapping

from aivia.core.errors import ConfigError

__all__ = [
    "API_KEY_HEADER",
    "AUTHORIZATION_HEADER",
    "api_key_matches",
    "extract_api_key",
    "is_loopback_host",
    "validate_bind_host",
]

API_KEY_HEADER = "x-api-key"
AUTHORIZATION_HEADER = "authorization"


def is_loopback_host(host: str) -> bool:
    """Return whether *host* names the local machine only.

    ``localhost`` is accepted as a convenience. IPv6 literals may be written
    with or without brackets. Hostnames are rejected because resolving them
    could point at a non-loopback interface.
    """
    candidate = host.strip()
    if candidate.lower() == "localhost":
        return True
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def validate_bind_host(host: str) -> str:
    """Return *host* when it binds to loopback, otherwise raise.

    Wildcard addresses such as ``0.0.0.0`` and ``::`` are rejected explicitly
    because they would expose the API to the local network.
    """
    if not isinstance(host, str) or not host.strip():
        raise ConfigError("api.host must be a non-empty loopback address")
    if not is_loopback_host(host):
        raise ConfigError(
            "api.host must bind to localhost only",
            details={"host": host},
        )
    return host.strip()


def extract_api_key(headers: Mapping[str, str]) -> str | None:
    """Return the API key carried by *headers*, if any.

    Both ``X-API-Key`` and ``Authorization: Bearer <key>`` are accepted. The
    latter is the OpenAPI-native form, while the former is easier for scripts.
    """
    direct = headers.get(API_KEY_HEADER)
    if direct:
        return direct.strip()
    authorization = headers.get(AUTHORIZATION_HEADER)
    if not authorization:
        return None
    scheme, separator, value = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer":
        return None
    token = value.strip()
    return token or None


def api_key_matches(expected: str, supplied: str | None) -> bool:
    """Compare *supplied* with *expected* in constant time."""
    if supplied is None:
        return False
    return hmac.compare_digest(expected.encode("utf-8"), supplied.encode("utf-8"))
