"""Golden CI proposal contract: pinned ffmpeg and enabled comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / "ci-proposals" / "golden.yml"
APPLIED = ROOT / ".github" / "workflows" / "golden.yml"
MANIFEST = ROOT / "tests" / "golden" / "render" / "manifest.json"


def _workflow_path() -> Path:
    candidates = [path for path in (PROPOSAL, APPLIED) if path.is_file()]
    assert len(candidates) == 1, (
        "the golden workflow must be proposed or applied exactly once"
    )
    return candidates[0]


def _proposal() -> str:
    return _workflow_path().read_text(encoding="utf-8")


def _workflow() -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(_proposal()))


def _manifest() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(MANIFEST.read_text(encoding="utf-8")))


def _golden_job() -> dict[str, Any]:
    workflow = _workflow()
    jobs = cast(dict[str, Any], workflow["jobs"])
    return cast(dict[str, Any], jobs["golden"])


def _steps() -> dict[str, dict[str, Any]]:
    steps = cast(list[dict[str, Any]], _golden_job()["steps"])
    return {
        cast(str, step["name"]): step
        for step in steps
        if isinstance(step, dict) and "name" in step
    }


def test_golden_workflow_is_proposed_or_applied_exactly_once() -> None:
    assert _workflow_path() in (PROPOSAL, APPLIED)


def test_golden_proposal_runs_on_pull_requests_and_main() -> None:
    text = _proposal()

    assert "pull_request:" in text
    assert "branches: [main]" in text
    assert "contents: read" in text


def test_golden_proposal_pins_the_linux_runner_and_job_name() -> None:
    assert _golden_job()["runs-on"] == "ubuntu-24.04"


def test_golden_proposal_reads_the_pinned_toolchain_from_the_manifest() -> None:
    pin = _steps()["Read the pinned toolchain"]

    assert "tests/golden/render/manifest.json" in cast(str, pin["run"])
    assert "ffmpeg['version']" in cast(str, pin["run"])
    assert "ffmpeg['source']" in cast(str, pin["run"])


def test_golden_proposal_installs_the_manifest_source_not_a_moving_build() -> None:
    manifest = _manifest()
    ffmpeg = cast(dict[str, str], manifest["ffmpeg"])
    source = ffmpeg["source"]
    text = _proposal()

    assert source.startswith("https://")
    assert "latest" not in source
    assert "latest" not in text
    assert "steps.pin.outputs.source" in cast(
        str, _steps()["Install the pinned ffmpeg"]["run"]
    )
    assert "steps.pin.outputs.version" in cast(
        str, cast(dict[str, Any], _steps()["Cache the pinned ffmpeg"]["with"])["key"]
    )


def test_golden_proposal_fails_when_the_installed_ffmpeg_is_not_the_pin() -> None:
    verify = cast(str, _steps()["Verify the pinned ffmpeg"]["run"])

    assert 'expected="${{ steps.pin.outputs.version }}"' in verify
    assert 'test "$found" = "$expected"' in verify
    assert "exit 1" in verify


def test_golden_proposal_enables_comparisons_and_requires_the_pin() -> None:
    golden = _steps()["Golden render comparisons"]
    env = cast(dict[str, str], golden["env"])
    run = cast(str, golden["run"])

    assert env["AIVA_COMPARE_GOLDENS"] == "1"
    assert env["AIVA_GOLDEN_REQUIRE_FFMPEG"] == "1"
    assert "AIVA_GOLDEN_FFMPEG" in _proposal()
    assert "AIVA_GOLDEN_FFPROBE" in _proposal()
    assert "pytest -q tests/golden" in run
    assert "pytest -q -m golden tests/compose/test_render_still.py" in run
