"""Credential rotation: reload secrets without restart."""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import pytest

from aivia.cli import main
from aivia.core.secrets import SecretsLoader


def test_reload_picks_up_changed_secrets_file(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets.env"
    secrets.write_text("API_KEY=old\n", encoding="utf-8")
    loader = SecretsLoader(secrets_file=secrets, environ={})

    assert loader.get("API_KEY") == "old"

    secrets.write_text("API_KEY=new\n", encoding="utf-8")
    loader.reload()

    assert loader.get("API_KEY") == "new"


def test_reload_picks_up_changed_environment(tmp_path: Path) -> None:
    environ = {"AIVA_API_KEY": "old"}
    loader = SecretsLoader(environ=environ)

    assert loader.get("API_KEY") == "old"

    environ["AIVA_API_KEY"] = "new"
    loader.reload()

    assert loader.get("API_KEY") == "new"


def test_in_flight_call_finishes_with_old_value(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets.env"
    secrets.write_text("API_KEY=old\n", encoding="utf-8")
    loader = SecretsLoader(secrets_file=secrets, environ={})

    captured = loader.get("API_KEY")

    secrets.write_text("API_KEY=new\n", encoding="utf-8")
    loader.reload()

    assert loader.get("API_KEY") == "new"
    assert captured == "old"


def test_get_optional_keeps_cached_value_after_reload(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets.env"
    secrets.write_text("API_KEY=old\n", encoding="utf-8")
    loader = SecretsLoader(secrets_file=secrets, environ={})

    assert loader.get_optional("API_KEY") == "old"

    secrets.write_text("API_KEY=new\n", encoding="utf-8")
    loader.reload()

    assert loader.get_optional("API_KEY") == "new"


def test_reload_clears_missing_secret_cache(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets.env"
    secrets.write_text("", encoding="utf-8")
    loader = SecretsLoader(secrets_file=secrets, environ={})

    assert loader.get_optional("MISSING") is None

    secrets.write_text("MISSING=now-present\n", encoding="utf-8")
    loader.reload()

    assert loader.get("MISSING") == "now-present"


def test_reload_logs_without_secret_values(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    secrets = tmp_path / "secrets.env"
    secrets.write_text("API_KEY=test-value\n", encoding="utf-8")
    loader = SecretsLoader(secrets_file=secrets, environ={})

    with caplog.at_level(logging.INFO, logger="aivia.core.secrets"):
        loader.reload()

    messages = [record.getMessage() for record in caplog.records]
    assert any("credentials reloaded" in message for message in messages)
    assert "test-value" not in caplog.text


def test_cli_credentials_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    secrets = xdg / "aivia" / "secrets.env"
    secrets.parent.mkdir(parents=True)
    secrets.write_text("API_KEY=old\n", encoding="utf-8")

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    root.propagate = False

    try:
        assert main(["credentials", "reload"]) == 0
    finally:
        root.handlers[:] = previous_handlers
        root.setLevel(previous_level)
        root.propagate = True

    output = stream.getvalue()
    assert "credentials reloaded" in output
    assert "old" not in output
    assert "API_KEY" not in output
