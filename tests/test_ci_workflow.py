"""CI proposal contract: ffmpeg, lint, type and the offline suite."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / "ci-proposals" / "ci.yml"


def _proposal() -> str:
    return PROPOSAL.read_text(encoding="utf-8")


def test_ci_proposal_exists_and_is_inert() -> None:
    assert PROPOSAL.is_file()
    assert PROPOSAL.parent.name == "ci-proposals"
    assert not (ROOT / ".github" / "workflows" / "ci-proposals").exists()


def test_ci_proposal_runs_on_pull_requests_and_main() -> None:
    text = _proposal()

    assert "pull_request:" in text
    assert "branches: [main]" in text
    assert "contents: read" in text


def test_ci_proposal_keeps_the_required_ci_job_name() -> None:
    text = _proposal()

    assert "\n  ci:\n" in text
    assert "runs-on: ubuntu-latest" in text


def test_ci_proposal_installs_ffmpeg_and_ffprobe() -> None:
    text = _proposal()

    assert "apt-get install" in text
    assert "ffmpeg" in text
    assert "ffprobe -version" in text


def test_ci_proposal_runs_lint_and_type_checks() -> None:
    text = _proposal()

    assert "uv run --no-sync ruff check ." in text
    assert "uv run --no-sync ruff format --check ." in text
    assert "uv run --no-sync mypy" in text


def test_ci_proposal_runs_the_offline_suite_without_live_tests() -> None:
    text = _proposal()

    assert 'uv run --no-sync pytest -q -m "not live"' in text
    assert "--live" not in text


def test_ci_proposal_runs_the_installed_tool_smoke_after_the_offline_suite() -> None:
    text = _proposal()

    offline = text.index("Offline tests")
    smoke = text.index("Installed tool smoke")
    assert smoke > offline
    assert "scripts/install_smoke.sh" in text[smoke:]


def test_ci_proposal_delegates_the_smoke_to_the_script() -> None:
    text = _proposal()

    smoke = text.index("Installed tool smoke")
    step = text[smoke:]

    assert "run: scripts/install_smoke.sh" in step
    assert "uv tool install" not in step
    assert "uv build" not in step
