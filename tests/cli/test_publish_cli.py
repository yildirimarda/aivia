"""``aivia publish check`` and ``aivia publish dry-run``: pre-flight publish validation.

Both commands validate tokens, media, formatting, compatibility, scopes and
remaining quota without calling a platform API or spending money.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.cli import main


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
  x:
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


def _dimension_line(out: str, dimension: str) -> str:
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] == dimension:
            return line
    raise AssertionError(f"no report line for dimension {dimension!r}:\n{out}")


def _assert_dimension(out: str, dimension: str, status: str) -> str:
    line = _dimension_line(out, dimension)
    assert f"[{status}]" in line, f"expected [{status}] for {dimension}: {line}"
    return line


def _assert_all_pass(out: str) -> None:
    for dimension in (
        "tokens",
        "media",
        "formatting",
        "compatibility",
        "scopes",
        "quota",
    ):
        _assert_dimension(out, dimension, "ok")


def test_publish_check_passes_when_every_dimension_is_valid(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    media = tmp_path / "img.png"
    media.write_bytes(b"fake-image")
    code, out, err = run_cli(
        capsys,
        "publish",
        "check",
        "--config",
        str(config),
        "--platform",
        "x",
        "--media",
        str(media),
        "--caption",
        "hello",
        "--title",
        "hi",
    )
    assert code == 0, err
    assert err == ""
    _assert_all_pass(out)


def test_publish_dry_run_passes_with_mock_provider(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    media = tmp_path / "img.png"
    media.write_bytes(b"fake-image")
    code, out, err = run_cli(
        capsys,
        "publish",
        "dry-run",
        "--config",
        str(config),
        "--platform",
        "x",
        "--media",
        str(media),
        "--caption",
        "hello",
    )
    assert code == 0, err
    assert err == ""
    assert _assert_dimension(out, "dry-run", "ok")
    assert "dry-run publish accepted by the provider" in out


def test_publish_check_fails_when_token_is_missing(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AIVA_X_TOKEN", raising=False)
    config = write_config(tmp_path, BASE_CONFIG)
    media = tmp_path / "img.png"
    media.write_bytes(b"fake-image")
    code, out, err = run_cli(
        capsys,
        "publish",
        "check",
        "--config",
        str(config),
        "--platform",
        "x",
        "--media",
        str(media),
    )
    assert code == 1, err
    assert _assert_dimension(out, "tokens", "FAIL")
    assert "unresolvable credential reference" in (out + err)


def test_publish_check_fails_when_media_file_is_missing(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    code, out, _err = run_cli(
        capsys,
        "publish",
        "check",
        "--config",
        str(config),
        "--platform",
        "x",
        "--media",
        str(tmp_path / "does-not-exist.png"),
    )
    assert code == 1
    assert _assert_dimension(out, "media", "FAIL")
    assert "media file not found" in out


def test_publish_check_fails_when_media_exceeds_platform_limit(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    # X allows 5 MiB images; write 6 MiB.
    media = tmp_path / "img.png"
    media.write_bytes(b"\x00" * (6 * 1024 * 1024))
    code, out, err = run_cli(
        capsys,
        "publish",
        "check",
        "--config",
        str(config),
        "--platform",
        "x",
        "--media",
        str(media),
    )
    assert code == 1, err
    assert _assert_dimension(out, "media", "FAIL")
    assert "exceeds" in out


def test_publish_check_fails_when_caption_exceeds_platform_limit(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    # X allows 280 characters.
    caption = "x" * 300
    code, out, err = run_cli(
        capsys,
        "publish",
        "check",
        "--config",
        str(config),
        "--platform",
        "x",
        "--caption",
        caption,
    )
    assert code == 1, err
    assert _assert_dimension(out, "formatting", "FAIL")
    assert "caption is 300 characters" in out


def test_publish_check_fails_when_format_is_incompatible(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    # audio_only outputs audio, which X does not support.
    code, out, err = run_cli(
        capsys,
        "publish",
        "check",
        "--config",
        str(config),
        "--platform",
        "x",
        "--format",
        "audio_only",
    )
    assert code == 1, err
    assert _assert_dimension(out, "compatibility", "FAIL")
    assert "does not support" in out


def test_publish_check_fails_when_no_provider_serves_publish(
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
  x:
    type: mock-image
publish:
  accounts:
    x/en:
      credential_refs: [X_TOKEN]
"""
    config = _config(tmp_path, monkeypatch, text=text)
    code, out, err = run_cli(
        capsys,
        "publish",
        "check",
        "--config",
        str(config),
        "--platform",
        "x",
    )
    assert code == 1, err
    assert _assert_dimension(out, "scopes", "FAIL")
    assert "publish capability" in out


def test_publish_check_skips_media_when_no_media_supplied(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    code, out, err = run_cli(
        capsys,
        "publish",
        "check",
        "--config",
        str(config),
        "--platform",
        "x",
    )
    assert code == 0, err
    assert _assert_dimension(out, "media", "ok")
    assert "no media supplied" in out


def test_publish_check_skips_formatting_when_no_text_supplied(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    code, out, err = run_cli(
        capsys,
        "publish",
        "check",
        "--config",
        str(config),
        "--platform",
        "x",
    )
    assert code == 0, err
    assert _assert_dimension(out, "formatting", "ok")
    assert "no caption or title supplied" in out


def test_publish_check_skips_compatibility_when_no_format_supplied(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    code, out, err = run_cli(
        capsys,
        "publish",
        "check",
        "--config",
        str(config),
        "--platform",
        "x",
    )
    assert code == 0, err
    assert _assert_dimension(out, "compatibility", "ok")
    assert "no format supplied" in out


def test_publish_check_fails_when_account_is_unknown(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    code, _out, err = run_cli(
        capsys,
        "publish",
        "check",
        "--config",
        str(config),
        "--account",
        "youtube/en",
    )
    assert code == 1
    assert "unknown publish account" in err


def test_publish_check_fails_when_platform_is_unknown(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    code, _out, err = run_cli(
        capsys,
        "publish",
        "check",
        "--config",
        str(config),
        "--platform",
        "no-such-platform",
    )
    assert code == 1
    assert "no publish account configured for platform" in err


def test_publish_report_lists_every_dimension(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, monkeypatch)
    code, out, err = run_cli(
        capsys,
        "publish",
        "check",
        "--config",
        str(config),
        "--platform",
        "x",
    )
    assert code == 0, err
    for dimension in (
        "tokens",
        "media",
        "formatting",
        "compatibility",
        "scopes",
        "quota",
    ):
        _assert_dimension(out, dimension, "ok")
