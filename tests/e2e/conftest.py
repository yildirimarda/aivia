"""Keep end-to-end tests offline and away from the developer's real config."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from pathlib import Path

import pytest


def _blocked(*args: object, **kwargs: object) -> None:
    raise AssertionError("outbound socket call during an offline test")


@pytest.fixture(autouse=True)
def isolate_user_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Point the default XDG lookup at a temporary home for every e2e test."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


@pytest.fixture(autouse=True)
def forbid_outbound_sockets(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Fail any test in this package that tries to open an outbound connection."""
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    yield
