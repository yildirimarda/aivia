"""Documentation contract: architecture note, decisions log and progress log."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

DESIGN = DOCS / "DESIGN.md"
DECISIONS = DOCS / "decisions.md"
PROGRESS = DOCS / "PROGRESS.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_required_documents_exist_and_are_non_empty() -> None:
    for path in (DESIGN, DECISIONS, PROGRESS):
        assert path.is_file(), f"missing {path.relative_to(ROOT)}"
        assert _read(path).strip(), f"{path.relative_to(ROOT)} is empty"


def test_architecture_note_records_the_deterministic_core() -> None:
    text = _read(DESIGN).lower()

    assert "deterministic" in text
    assert "control flow" in text
    assert "llms run *inside* stages" in text
    assert "validated structured output" in text


def test_architecture_note_records_core_independence() -> None:
    text = _read(DESIGN).lower()

    assert "core independence" in text
    assert "the core never imports a module" in text


def test_fixtures_policy_records_responses_per_provider() -> None:
    text = _read(DESIGN).lower()

    assert "recorded responses" in text
    assert "per provider" in text
    assert "tests/fixtures/providers/" in text


def test_fixtures_policy_records_one_example_per_schema() -> None:
    text = _read(DESIGN).lower()

    assert "one example per schema" in text
    assert "examples/" in text


def test_fixtures_policy_records_the_goldens_folder() -> None:
    text = _read(DESIGN).lower()

    assert "goldens folder" in text
    assert "tests/golden/" in text


def test_fixtures_policy_forbids_real_personal_data() -> None:
    text = _read(DESIGN).lower()

    assert "no real personal data" in text
    assert "synthetic" in text
    assert "redacted" in text


def test_decisions_log_records_deterministic_core_and_core_independence() -> None:
    text = _read(DECISIONS).lower()

    assert "d-003 · deterministic core, llm at the edges" in text
    assert "d-006 · core never imports a module" in text


def test_decisions_log_records_the_fixtures_policy() -> None:
    text = _read(DECISIONS).lower()

    assert "d-023 · fixtures are recorded, examples synthetic" in text
    assert "synthetic data only" in text


def test_progress_log_records_deterministic_core_and_core_independence() -> None:
    text = _read(PROGRESS).lower()

    assert "deterministic core" in text
    assert "no module imports in core" in text
    assert "validated structured output" in text
    assert "research" in text
    assert "orders" in text
    assert "publish" in text
    assert "analytics" in text


def test_documents_cross_reference_each_other() -> None:
    design = _read(DESIGN)
    decisions = _read(DECISIONS)
    progress = _read(PROGRESS)

    assert "docs/decisions.md" in design
    assert "docs/PROGRESS.md" in design
    assert "docs/DESIGN.md" in decisions
    assert "docs/DESIGN.md" in progress
    assert "docs/decisions.md" in progress
