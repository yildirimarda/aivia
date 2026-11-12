# secret-scan: allow-file -- this module intentionally contains fake tokens.
"""Secret audit of a mock run's logs, artifacts, errors and sidecars.

A configured secret must never leak into a run's outputs. The audit scans the
text of every log, artifact, error and sidecar produced by a mock run and
reports any configured secret it finds so the run can be failed.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aivia.core.models import CostEvent, Run, ToolchainFingerprint
from aivia.core.redact import clear_secrets, register_secret
from aivia.core.run_dirs import RunDirectory
from aivia.core.secret_scan import format_findings, scan_for_secrets
from aivia.core.sidecar import write_cost_sidecar

GENERATED_AT = datetime(2026, 1, 5, 10, 30, tzinfo=UTC)
SECRET = "sk-audit-secret-1234567890"


@pytest.fixture(autouse=True)
def _empty_secret_registry() -> Iterator[None]:
    clear_secrets()
    yield
    clear_secrets()


def _event(**overrides: object) -> CostEvent:
    values: dict[str, object] = {
        "provider": "mock",
        "model": "mock-model",
        "operation": "llm",
        "units": 1.0,
        "unit_kind": "request",
        "amount_usd": 0.0005,
        "run_id": "run-1",
        "pricing_version": "2026-01",
        "estimated": False,
    }
    values.update(overrides)
    return CostEvent.model_validate(values)


def _fingerprint() -> ToolchainFingerprint:
    return ToolchainFingerprint(
        renderer_version="renderer-1",
        ffmpeg_version="ffmpeg-7",
        font_set="fonts-abc",
    )


def _build_run(tmp_path: Path) -> tuple[RunDirectory, Run]:
    """Build a mock run directory carrying a log, an artifact and a sidecar."""
    run_dir = RunDirectory(tmp_path / "run", run_id="run-1")
    run_dir.create()

    (run_dir.path / "run.log").write_text(
        "stage=write provider=mock\n", encoding="utf-8"
    )

    artifact = run_dir.path / "script.md"
    artifact.write_text("# script\n", encoding="utf-8")
    run_dir.register("write", artifacts=["script.md"])

    output = run_dir.path / "clip.mp4"
    output.write_bytes(b"clip")
    write_cost_sidecar(
        output,
        [_event()],
        toolchain_fingerprint=_fingerprint(),
        generated_at=GENERATED_AT,
    )

    run = Run(id="run-1")
    return run_dir, run


def _outputs(run_dir: RunDirectory, run: Run) -> list[tuple[str, str]]:
    """Collect the text of the run's log, artifacts, errors and sidecars."""
    texts: list[tuple[str, str]] = []

    log = run_dir.path / "run.log"
    if log.is_file():
        texts.append(("log", log.read_text(encoding="utf-8")))

    for record in run_dir.artifacts():
        path = run_dir.resolve(record.path)
        if path.is_file():
            texts.append(
                (
                    f"artifact:{record.stage}:{record.path}",
                    path.read_text(encoding="utf-8"),
                )
            )

    for sidecar in sorted(run_dir.path.glob("*.cost.json")):
        texts.append((f"sidecar:{sidecar.name}", sidecar.read_text(encoding="utf-8")))

    if run.error_message:
        texts.append(("error", run.error_message))

    return texts


def test_audit_reports_a_secret_in_an_artifact(tmp_path: Path) -> None:
    register_secret(SECRET)
    run_dir, run = _build_run(tmp_path)
    (run_dir.path / "script.md").write_text(f"# script\n{SECRET}\n", encoding="utf-8")

    findings = scan_for_secrets(_outputs(run_dir, run))

    assert findings
    assert any(
        path.startswith("artifact:") for path in {finding.path for finding in findings}
    )


def test_audit_reports_a_secret_in_the_run_error(tmp_path: Path) -> None:
    register_secret(SECRET)
    run_dir, run = _build_run(tmp_path)
    run = Run(id="run-1", error_message=f"failed with {SECRET}")

    findings = scan_for_secrets(_outputs(run_dir, run))

    assert findings
    assert any(path == "error" for path in {finding.path for finding in findings})


def test_audit_reports_a_secret_in_a_sidecar(tmp_path: Path) -> None:
    register_secret(SECRET)
    run_dir, run = _build_run(tmp_path)
    write_cost_sidecar(
        run_dir.path / "clip.mp4",
        [_event(run_id=f"run-1 {SECRET}")],
        toolchain_fingerprint=_fingerprint(),
        generated_at=GENERATED_AT,
    )

    findings = scan_for_secrets(_outputs(run_dir, run))

    assert findings
    assert any(
        path.startswith("sidecar:") for path in {finding.path for finding in findings}
    )


def test_audit_reports_a_secret_in_a_log(tmp_path: Path) -> None:
    register_secret(SECRET)
    run_dir, run = _build_run(tmp_path)
    (run_dir.path / "run.log").write_text(
        f"stage=write provider=mock token={SECRET}\n", encoding="utf-8"
    )

    findings = scan_for_secrets(_outputs(run_dir, run))

    assert findings
    assert any(path == "log" for path in {finding.path for finding in findings})


def test_audit_passes_when_no_output_contains_a_secret(tmp_path: Path) -> None:
    register_secret(SECRET)
    run_dir, run = _build_run(tmp_path)

    findings = scan_for_secrets(_outputs(run_dir, run))

    assert findings == []


def test_audit_findings_never_reveal_the_secret_value(tmp_path: Path) -> None:
    register_secret(SECRET)
    run_dir, run = _build_run(tmp_path)
    (run_dir.path / "script.md").write_text(f"# script\n{SECRET}\n", encoding="utf-8")

    findings = scan_for_secrets(_outputs(run_dir, run))

    assert findings
    rendered = format_findings(findings)
    assert SECRET not in rendered
    assert "configured secret" in rendered
