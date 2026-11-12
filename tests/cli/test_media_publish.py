"""``aivia media publish``: publish a file and record a publish cost event.

The command publishes an already-generated media file to a configured platform
account and writes one publish cost event to the ledger.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aivia.cli import main
from aivia.core.db import CostEventRepository, create_database_engine


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text.lstrip(), encoding="utf-8")
    return path


def run_cli(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, str, str]:
    code = main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


BASE_CONFIG = """
platforms:
  accounts:
    x/en:
      credential_refs: [X_TOKEN]
providers:
  mock:
    type: mock
publish:
  accounts:
    x/en:
      credential_refs: [X_TOKEN]
"""


def _config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    text: str | None = None,
) -> Path:
    monkeypatch.setenv("AIVA_X_TOKEN", "test-token")
    return write_config(tmp_path, text if text is not None else BASE_CONFIG)


def _db_path(config: Path) -> Path:
    return config.parent / "data" / "aivia.db"


def _published_events(config: Path) -> list[Any]:
    engine = create_database_engine(_db_path(config))
    try:
        return CostEventRepository(engine).list()
    finally:
        engine.dispose()


def test_media_publish_publishes_a_file_and_records_a_cost_event(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    media = tmp_path / "poster.png"
    media.write_bytes(b"\x89PNG\r\n\x1a\nfake-image")

    code, out, err = run_cli(
        capsys,
        "media",
        "publish",
        str(media),
        "--config",
        str(config),
        "--platform",
        "x",
        "--caption",
        "hello world",
    )
    assert code == 0, err
    assert err == ""
    assert "Platform: x" in out
    assert "Provider: mock" in out
    assert "Remote id: mock-post-" in out

    events = _published_events(config)
    assert len(events) == 1
    event = events[0]
    assert event.operation == "publish"
    assert event.provider == "mock"
    assert event.estimated is False
    assert event.unit_kind == "call"
    assert event.units == 1.0


def test_media_publish_estimate_prints_without_a_paid_call(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    media = tmp_path / "poster.png"
    media.write_bytes(b"\x89PNG\r\n\x1a\nfake-image")

    code, _out, err = run_cli(
        capsys,
        "media",
        "publish",
        str(media),
        "--config",
        str(config),
        "--platform",
        "x",
        "--estimate",
    )
    assert code == 0, err
    assert err == ""
    assert not (config.parent / "data").exists()


def test_media_publish_dry_run_records_a_publish_event(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    media = tmp_path / "poster.png"
    media.write_bytes(b"\x89PNG\r\n\x1a\nfake-image")

    code, out, err = run_cli(
        capsys,
        "media",
        "publish",
        str(media),
        "--config",
        str(config),
        "--platform",
        "x",
        "--dry-run",
    )
    assert code == 0, err
    assert err == ""
    assert "Dry run: yes" in out

    events = _published_events(config)
    assert len(events) == 1
    assert events[0].operation == "publish"
    assert events[0].estimated is False


def test_media_publish_uses_the_first_account_when_platform_is_omitted(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    media = tmp_path / "poster.png"
    media.write_bytes(b"\x89PNG\r\n\x1a\nfake-image")

    code, out, err = run_cli(
        capsys,
        "media",
        "publish",
        str(media),
        "--config",
        str(config),
    )
    assert code == 0, err
    assert err == ""
    assert "Platform: x" in out


def test_media_publish_fails_when_the_media_file_is_missing(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)

    code, _out, err = run_cli(
        capsys,
        "media",
        "publish",
        str(tmp_path / "does-not-exist.png"),
        "--config",
        str(config),
        "--platform",
        "x",
    )
    assert code == 1
    assert "media file not found" in err


def test_media_publish_fails_when_no_publish_accounts_are_configured(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = """
platforms:
  accounts:
    x/en:
      credential_refs: [X_TOKEN]
providers:
  mock:
    type: mock
"""
    config = _config(tmp_path, monkeypatch, text=text)
    media = tmp_path / "poster.png"
    media.write_bytes(b"\x89PNG\r\n\x1a\nfake-image")

    code, _out, err = run_cli(
        capsys,
        "media",
        "publish",
        str(media),
        "--config",
        str(config),
        "--platform",
        "x",
    )
    assert code == 1
    assert "no publish accounts are configured" in err


def test_media_publish_fails_when_no_provider_serves_publish(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = """
platforms:
  accounts:
    x/en:
      credential_refs: [X_TOKEN]
providers:
  mock:
    type: mock-image
publish:
  accounts:
    x/en:
      credential_refs: [X_TOKEN]
"""
    config = _config(tmp_path, monkeypatch, text=text)
    media = tmp_path / "poster.png"
    media.write_bytes(b"\x89PNG\r\n\x1a\nfake-image")

    code, _out, err = run_cli(
        capsys,
        "media",
        "publish",
        str(media),
        "--config",
        str(config),
        "--platform",
        "x",
    )
    assert code == 1
    assert "publish capability" in err
