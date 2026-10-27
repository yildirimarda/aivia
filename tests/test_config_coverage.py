"""Config coverage contract: advisory reporting outside adapters and data."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.core.config_coverage import (
    ALLOW_FILE_MARKER,
    format_findings,
    main,
    scan_repository,
    scan_text,
)

ROOT = Path(__file__).resolve().parents[1]

UNCOVERED_TEXT = (
    "vendor = OpenAI\nmodel = claude-3-opus\ncost = $0.04\nmax_retries = 3\n"
)


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize(
    ("text", "kind", "value"),
    [
        ("client = OpenAI(api_key=key)", "vendor name", "OpenAI"),
        ("model = claude-3-opus", "model id", "claude-3-opus"),
        ("cost = $0.04", "price", "$0.04"),
        ("max_retries = 3", "threshold", "max_retries = 3"),
        ("if score > 0.8:", "threshold", "score > 0.8"),
    ],
)
def test_scan_text_reports_each_kind(text: str, kind: str, value: str) -> None:
    findings = scan_text(text)

    assert [(finding.kind, finding.value) for finding in findings] == [(kind, value)]


def test_findings_carry_the_line_number() -> None:
    findings = scan_text("clean line\nclient = OpenAI\n")

    assert [(finding.line, finding.kind) for finding in findings] == [
        (2, "vendor name")
    ]


def test_allowed_areas_are_exempt(tmp_path: Path) -> None:
    for relative in (
        "src/aivia/providers/example.py",
        "src/aivia/config/example.py",
        "src/aivia/data/defaults/example.yaml",
    ):
        _write(tmp_path, relative, UNCOVERED_TEXT)

    assert scan_repository(tmp_path) == []


def test_values_outside_allowed_areas_are_reported(tmp_path: Path) -> None:
    _write(tmp_path, "src/aivia/core/example.py", UNCOVERED_TEXT)

    findings = scan_repository(tmp_path)

    assert {finding.kind for finding in findings} == {
        "vendor name",
        "model id",
        "price",
        "threshold",
    }
    assert {finding.path for finding in findings} == {"src/aivia/core/example.py"}


def test_scans_source_and_scripts_but_not_docs_or_tests(tmp_path: Path) -> None:
    _write(tmp_path, "tests/test_example.py", UNCOVERED_TEXT)
    _write(tmp_path, "docs/example.md", UNCOVERED_TEXT)
    _write(tmp_path, "run.sh", UNCOVERED_TEXT)
    _write(tmp_path, "scripts/tool.sh", UNCOVERED_TEXT)

    findings = scan_repository(tmp_path)

    assert {finding.path for finding in findings} == {"scripts/tool.sh"}


def test_allow_file_marker_skips_intentional_mentions(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/aivia/core/example.py",
        f"# {ALLOW_FILE_MARKER}\nvendor = OpenAI\n",
    )

    assert scan_repository(tmp_path) == []


def test_cli_is_advisory_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tmp_path, "src/aivia/core/example.py", "vendor = OpenAI\n")

    assert main([str(tmp_path)]) == 0

    captured = capsys.readouterr()
    assert "vendor name: OpenAI" in captured.out
    assert "example.py" in captured.out
    assert "advisory" in captured.err


def test_strict_cli_fails_when_findings_exist(tmp_path: Path) -> None:
    _write(tmp_path, "src/aivia/core/example.py", "vendor = OpenAI\n")

    assert main([str(tmp_path), "--strict"]) == 1


def test_cli_succeeds_when_there_is_nothing_to_report(tmp_path: Path) -> None:
    _write(tmp_path, "src/aivia/core/example.py", "value = 1\n")

    assert main([str(tmp_path)]) == 0


def test_repository_is_clean() -> None:
    findings = scan_repository(ROOT)

    assert findings == [], format_findings(findings)
