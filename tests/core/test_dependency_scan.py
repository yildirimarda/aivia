"""CI dependency vulnerability scanning wiring.

Proves, without touching the network, that the CI proposal wires up
``pip-audit`` so it fails the build on confirmed vulnerabilities (including
every high severity issue) and that the documented allowlist exists.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[2]
CI_PROPOSAL = ROOT / "ci-proposals" / "ci.yml"
ALLOWLIST_DOC = ROOT / "docs" / "dependency-allowlist.md"
PYPROJECT = ROOT / "pyproject.toml"


def _ci_steps() -> list[dict[str, Any]]:
    data = yaml.safe_load(CI_PROPOSAL.read_text(encoding="utf-8"))
    assert data is not None
    return cast("list[dict[str, Any]]", data["jobs"]["ci"]["steps"])


def test_ci_proposal_exists_and_is_valid_yaml() -> None:
    assert CI_PROPOSAL.is_file()
    data = yaml.safe_load(CI_PROPOSAL.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data["jobs"]["ci"]["runs-on"] == "ubuntu-24.04"


def test_ci_proposal_runs_pip_audit_without_dry_run() -> None:
    scan_steps = [
        step
        for step in _ci_steps()
        if isinstance(step.get("run"), str) and "pip-audit" in step["run"]
    ]
    assert scan_steps, "ci-proposals/ci.yml has no pip-audit scan step"
    run = scan_steps[0]["run"]
    assert "--dry-run" not in run, "scan must actually fail on vulnerabilities"
    assert "uv run --no-sync pip-audit" in run


def test_dependency_allowlist_documents_policy() -> None:
    assert ALLOWLIST_DOC.is_file()
    text = ALLOWLIST_DOC.read_text(encoding="utf-8")
    for token in (
        "high",
        "critical",
        "fails CI",
        "allowlist",
        "--ignore-vuln",
        "| CVE ID |",
    ):
        assert token.lower() in text.lower(), f"allowlist doc missing: {token}"


def test_pip_audit_is_a_dev_dependency() -> None:
    with PYPROJECT.open("rb") as fh:
        project = tomllib.load(fh)
    dev = project["dependency-groups"]["dev"]
    assert any("pip-audit" in entry for entry in dev)
