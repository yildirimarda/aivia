"""The ``aivia api serve`` command validates the bind and starts the server."""

from __future__ import annotations

from pathlib import Path

import pytest
import uvicorn

from aivia.cli import main


class _RunRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[object, str, int]] = []

    def __call__(self, app: object, *, host: str, port: int) -> None:
        self.calls.append((app, host, port))


@pytest.fixture
def uvicorn_run(monkeypatch: pytest.MonkeyPatch) -> _RunRecorder:
    recorder = _RunRecorder()
    monkeypatch.setattr(uvicorn, "run", recorder)
    return recorder


def test_serve_starts_the_configured_loopback_server(
    uvicorn_run: _RunRecorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["api", "serve", "--set", "api.require_auth=false"]) == 0

    assert len(uvicorn_run.calls) == 1
    _, host, port = uvicorn_run.calls[0]
    assert host == "127.0.0.1"
    assert port == 8765
    assert "Serving AIVIA API on http://127.0.0.1:8765" in capsys.readouterr().out


def test_serve_applies_host_and_port_overrides(uvicorn_run: _RunRecorder) -> None:
    code = main(
        [
            "api",
            "serve",
            "--host",
            "localhost",
            "--port",
            "9123",
            "--set",
            "api.require_auth=false",
        ]
    )

    assert code == 0
    _, host, port = uvicorn_run.calls[0]
    assert (host, port) == ("localhost", 9123)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "example.test"])
def test_serve_rejects_a_non_loopback_host_before_starting(
    host: str,
    uvicorn_run: _RunRecorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["api", "serve", "--host", host]) == 1

    assert uvicorn_run.calls == []
    assert "must bind to localhost only" in capsys.readouterr().err


def test_serve_validates_the_host_from_the_config_file(
    tmp_path: Path,
    uvicorn_run: _RunRecorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("api:\n  host: 0.0.0.0\n", encoding="utf-8")

    assert main(["api", "serve", "--config", str(config)]) == 1

    assert uvicorn_run.calls == []
    assert "must bind to localhost only" in capsys.readouterr().err


def test_serve_rejects_an_out_of_range_port(
    uvicorn_run: _RunRecorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["api", "serve", "--port", "0"]) == 1

    assert uvicorn_run.calls == []
    assert "port" in capsys.readouterr().err


def test_api_without_a_subcommand_prints_usage(
    uvicorn_run: _RunRecorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["api"]) == 2

    assert uvicorn_run.calls == []
    assert "usage: aivia api serve" in capsys.readouterr().err
