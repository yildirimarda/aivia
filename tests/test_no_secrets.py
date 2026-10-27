"""Secret hygiene contract: env example and repository scanning."""

from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from aivia.core.secret_scan import (
    format_findings,
    main,
    scan_file,
    scan_repository,
    scan_text,
)

ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = ROOT / ".env.example"

_ENV_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def _token() -> str:
    return base64.b64encode(hashlib.sha256(b"aivia-secret-scan-test").digest()).decode()


def _aws_text() -> str:
    return "AKIA" + "Q" * 16


def _github_text() -> str:
    return "ghp_" + "x" * 30


def _slack_text() -> str:
    return "xoxb-" + "1234567890-" + "x" * 12


def _google_text() -> str:
    return "AIza" + "x" * 35


def _openai_text() -> str:
    return "sk-" + "x" * 24


def _jwt_text() -> str:
    return "eyJ" + "x" * 20 + "." + "y" * 20 + "." + "z" * 20


def _credential_text() -> str:
    return "api_key=" + _token()


def _bearer_text() -> str:
    return "Authorization: Bearer " + _token()


def _url_credentials_text() -> str:
    return "https://user:" + "Aa1" * 8 + "@example.com/path"


def _private_key_text() -> str:
    begin = "-----BEGIN " + "RSA PRIVATE KEY-----"
    end = "-----END " + "RSA PRIVATE KEY-----"
    return "\n".join([begin, "MIIEowIBAAKCAQEA" + _token(), end])


@pytest.mark.parametrize(
    ("kind", "factory"),
    [
        ("private key", _private_key_text),
        ("aws access key", _aws_text),
        ("github token", _github_text),
        ("slack token", _slack_text),
        ("google api key", _google_text),
        ("openai key", _openai_text),
        ("jwt", _jwt_text),
        ("credential assignment", _credential_text),
        ("bearer token", _bearer_text),
        ("url credentials", _url_credentials_text),
    ],
)
def test_key_like_strings_are_found(kind: str, factory: Callable[[], str]) -> None:
    findings = scan_text(factory())

    assert kind in {finding.kind for finding in findings}


def test_prefixed_environment_credentials_are_found() -> None:
    findings = scan_text("OPENAI_API_KEY=" + _token())

    assert "credential assignment" in {finding.kind for finding in findings}


def test_high_entropy_strings_are_found() -> None:
    findings = scan_text("value = " + _token())

    assert [finding.kind for finding in findings] == ["high entropy"]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "AIVA_TOKEN_STORE_KEY=<your-token-store-encryption-key>",
        "api_key=",
        'api_key="${AIVA_OPENAI_API_KEY}"',
        'apiKey="{env:NVIDIA_API_KEY}"',
        "token: ${{ secrets.GITHUB_TOKEN }}",
        "The token budget is 42 and the secret sauce recipe is good.",
        'CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"',
        "value = " + "a" * 64,
    ],
)
def test_placeholders_and_ordinary_text_are_clean(text: str) -> None:
    assert scan_text(text) == []


def test_findings_never_include_the_matched_value() -> None:
    token = _token()

    findings = scan_text("api_key=" + token)

    assert findings
    rendered = format_findings(findings)
    assert token not in rendered
    assert "credential assignment" in rendered


def test_repository_is_clean() -> None:
    findings = scan_repository(ROOT)

    assert findings == [], format_findings(findings)


def test_repository_scan_flags_a_leaking_file(tmp_path: Path) -> None:
    (tmp_path / "clean.txt").write_text("ordinary text\n", encoding="utf-8")
    (tmp_path / "leak.txt").write_text("api_key=" + _token() + "\n", encoding="utf-8")

    findings = scan_repository(tmp_path)

    assert {finding.path for finding in findings} == {"leak.txt"}


def test_allow_file_marker_skips_intentional_fixtures(tmp_path: Path) -> None:
    (tmp_path / "fixture.txt").write_text(
        "# secret-scan: allow-file\napi_key=" + _token() + "\n",
        encoding="utf-8",
    )

    assert scan_repository(tmp_path) == []


def test_binary_files_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "blob.bin").write_bytes(
        b"\x00\x01" + ("api_key=" + _token()).encode("utf-8")
    )

    assert scan_repository(tmp_path) == []


def test_cli_fails_and_names_the_file_without_leaking_the_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    token = _token()
    (tmp_path / "leak.txt").write_text("api_key=" + token + "\n", encoding="utf-8")

    assert main([str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert "leak.txt" in captured.out
    assert token not in captured.out + captured.err


def test_env_example_exists_and_only_uses_aiva_variables() -> None:
    assert ENV_EXAMPLE.is_file()

    names = []
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        match = _ENV_ASSIGNMENT.match(line)
        if match:
            names.append(match.group(1))

    assert names
    assert all(name.startswith("AIVA_") for name in names)


def test_env_example_documents_the_supported_variables() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")

    assert "AIVA_SECRETS_FILE=" in text
    assert "AIVA_TOKEN_STORE_KEY=" in text
    assert "AIVA_LIVE_BUDGET_USD=" in text


def test_env_example_contains_no_secrets() -> None:
    assert scan_file(ENV_EXAMPLE, root=ROOT) == []


def test_gitignore_ignores_real_env_files_but_keeps_the_example() -> None:
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".env" in lines
    assert ".env.*" in lines
    assert "!.env.example" in lines
