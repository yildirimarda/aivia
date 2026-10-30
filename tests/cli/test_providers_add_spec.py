"""``aivia providers add-spec``: inspect, confirm and install a provider spec."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from aivia.cli import main
from aivia.providers.http_spec import load_http_specs

_SPEC = """
name: example-image
capabilities: [image_gen]
base_url: https://api.example.test
headers:
  authorization: Bearer {{ secrets.api_key }}
request:
  method: POST
  path: /v1/images
response:
  results: data
  fields:
    url: image.url
pricing:
  image_gen:
    amount_usd: 0.02
    unit: image
secrets:
  api_key: EXAMPLE_API_KEY
""".lstrip()


class _Stream(io.StringIO):
    """A stdin replacement whose terminal status the test controls."""

    def __init__(self, text: str = "", *, tty: bool) -> None:
        super().__init__(text)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def write_spec(tmp_path: Path, text: str = _SPEC) -> Path:
    path = tmp_path / "example.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def providers_dir(tmp_path: Path) -> Path:
    return tmp_path / "home" / ".config" / "aivia" / "providers.d"


def run_add_spec(
    path: Path,
    *extra: str,
    stdin: io.StringIO | None = None,
    monkeypatch: pytest.MonkeyPatch,
) -> int:
    monkeypatch.setattr(sys, "stdin", stdin or _Stream(tty=False))
    return main(["providers", "add-spec", str(path), *extra])


def test_add_spec_prints_hosts_and_secrets_and_installs_after_yes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_spec(tmp_path)

    code = run_add_spec(
        source,
        "--allow-host",
        "api.example.test",
        "--yes",
        monkeypatch=monkeypatch,
    )
    captured = capsys.readouterr()

    assert code == 0
    assert "Target hosts: api.example.test" in captured.out
    assert "Secret names: api_key" in captured.out
    installed = providers_dir(tmp_path) / "example.yaml"
    assert installed.exists()
    assert [spec.name for spec in load_http_specs(installed.parent)] == [
        "example-image"
    ]
    assert installed.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_add_spec_refuses_a_non_interactive_run_without_yes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_spec(tmp_path)

    code = run_add_spec(
        source,
        "--allow-host",
        "api.example.test",
        monkeypatch=monkeypatch,
    )
    captured = capsys.readouterr()

    assert code == 1
    assert "Target hosts: api.example.test" in captured.out
    assert "non-interactive runs require --yes" in captured.err
    assert not providers_dir(tmp_path).exists()


def test_add_spec_accepts_an_interactive_yes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_spec(tmp_path)

    code = run_add_spec(
        source,
        "--allow-host",
        "api.example.test",
        stdin=_Stream("y\n", tty=True),
        monkeypatch=monkeypatch,
    )

    assert code == 0
    assert "Add provider spec example-image? [y/N]" in capsys.readouterr().out
    assert (providers_dir(tmp_path) / "example.yaml").exists()


def test_add_spec_declines_an_interactive_no(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_spec(tmp_path)

    code = run_add_spec(
        source,
        "--allow-host",
        "api.example.test",
        stdin=_Stream("n\n", tty=True),
        monkeypatch=monkeypatch,
    )

    assert code == 1
    assert "confirmation declined" in capsys.readouterr().err
    assert not providers_dir(tmp_path).exists()


def test_add_spec_rejects_cleartext_http_for_a_remote_host(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_spec(
        tmp_path,
        _SPEC.replace("https://api.example.test", "http://api.example.test"),
    )

    code = run_add_spec(
        source,
        "--allow-host",
        "api.example.test",
        "--yes",
        monkeypatch=monkeypatch,
    )
    captured = capsys.readouterr()

    assert code == 1
    assert "must use TLS" in captured.err
    assert "http://api.example.test" in captured.err
    assert not providers_dir(tmp_path).exists()


@pytest.mark.parametrize(
    "base_url",
    ["http://localhost:8080", "http://127.0.0.1:8080", "http://[::1]:8080"],
)
def test_add_spec_allows_cleartext_http_only_on_loopback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    base_url: str,
) -> None:
    source = write_spec(
        tmp_path,
        _SPEC.replace("https://api.example.test", base_url),
    )
    host = base_url.split("//", 1)[1].rsplit(":", 1)[0]

    code = run_add_spec(
        source,
        "--allow-host",
        host,
        "--yes",
        monkeypatch=monkeypatch,
    )

    assert code == 0
    assert "Added provider spec" in capsys.readouterr().out


def test_add_spec_rejects_a_host_outside_the_allowlist(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_spec(tmp_path)

    code = run_add_spec(
        source,
        "--allowlist",
        "other.test",
        "--yes",
        monkeypatch=monkeypatch,
    )
    captured = capsys.readouterr()

    assert code == 1
    assert "outside the allowlist" in captured.err
    assert "api.example.test" in captured.err
    assert not providers_dir(tmp_path).exists()


def test_add_spec_accepts_a_parent_domain_allowlist_entry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_spec(tmp_path)

    code = run_add_spec(
        source,
        "--allow-host",
        "*.example.test",
        "--yes",
        monkeypatch=monkeypatch,
    )

    assert code == 0
    assert "Added provider spec" in capsys.readouterr().out


def test_add_spec_does_not_overwrite_an_installed_spec_without_force(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_spec(tmp_path)
    installed = providers_dir(tmp_path) / "example.yaml"
    installed.parent.mkdir(parents=True)
    installed.write_text("previous", encoding="utf-8")

    code = run_add_spec(
        source,
        "--allow-host",
        "api.example.test",
        "--yes",
        monkeypatch=monkeypatch,
    )
    captured = capsys.readouterr()

    assert code == 1
    assert "already exists" in captured.err
    assert "--force" in captured.err
    assert installed.read_text(encoding="utf-8") == "previous"


def test_add_spec_force_replaces_an_installed_spec(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_spec(tmp_path)
    installed = providers_dir(tmp_path) / "example.yaml"
    installed.parent.mkdir(parents=True)
    installed.write_text("previous", encoding="utf-8")

    code = run_add_spec(
        source,
        "--allow-host",
        "api.example.test",
        "--yes",
        "--force",
        monkeypatch=monkeypatch,
    )

    assert code == 0
    assert "Added provider spec" in capsys.readouterr().out
    assert installed.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_add_spec_dry_run_does_not_write(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_spec(tmp_path)

    code = run_add_spec(
        source,
        "--allow-host",
        "api.example.test",
        "--yes",
        "--dry-run",
        monkeypatch=monkeypatch,
    )

    assert code == 0
    assert "Would add provider spec" in capsys.readouterr().out
    assert not providers_dir(tmp_path).exists()


def test_add_spec_reports_an_invalid_spec(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_spec(tmp_path, "name: broken\ncapabilities: []\n")

    code = run_add_spec(
        source,
        "--yes",
        monkeypatch=monkeypatch,
    )

    assert code == 1
    assert "invalid" in capsys.readouterr().err.lower()
