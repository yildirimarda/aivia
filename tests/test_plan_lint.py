"""Plan contract: canonical items, complete fields and backward dependencies."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.core.plan_lint import (
    format_findings,
    lint_design_dependencies,
    lint_plan,
    lint_repository,
    main,
    parse_plan,
)

ROOT = Path(__file__).resolve().parents[1]

VALID_PLAN = """# Plan

## Milestone 1: Start

- [ ] Add the first thing (tests/test_one.py)
- [x] Add the second thing (tests/test_two.py)

## Discovered

- [ ] Add a discovered thing (tests/test_three.py)
"""


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _plan(*items: str, heading: str = "## Milestone 1: Start") -> str:
    return "\n".join(["# Plan", "", heading, "", *items, ""])


def test_valid_plan_parses_items_and_milestones() -> None:
    plan = parse_plan(VALID_PLAN)

    assert plan.findings == ()
    assert [(milestone.number, milestone.name) for milestone in plan.milestones] == [
        (1, "Start")
    ]
    assert [(item.ordinal, item.done, item.milestone) for item in plan.items] == [
        (1, False, 1),
        (2, True, 1),
        (3, False, None),
    ]
    assert [item.verification for item in plan.items] == [
        "tests/test_one.py",
        "tests/test_two.py",
        "tests/test_three.py",
    ]


@pytest.mark.parametrize(
    "line",
    [
        "- [~] Add a thing (tests/test_one.py)",
        "* [ ] Add a thing (tests/test_one.py)",
        "- Add a thing (tests/test_one.py)",
        "- [] Add a thing (tests/test_one.py)",
    ],
)
def test_malformed_item_lines_are_reported(line: str) -> None:
    findings = lint_plan(_plan(line))

    assert {finding.kind for finding in findings} == {"malformed item line"}


def test_item_without_a_verification_field_is_reported() -> None:
    findings = lint_plan(_plan("- [ ] Add a thing"))

    assert [finding.kind for finding in findings] == ["missing field"]


def test_item_with_an_empty_verification_field_is_reported() -> None:
    findings = lint_plan(_plan("- [ ] Add a thing ()"))

    assert [finding.kind for finding in findings] == ["missing field"]


def test_item_with_duplicate_verification_fields_is_reported() -> None:
    findings = lint_plan(_plan("- [ ] Add a thing (details) (tests/test_one.py)"))

    assert [finding.kind for finding in findings] == ["duplicate field"]


def test_item_outside_a_milestone_is_reported() -> None:
    findings = lint_plan("# Plan\n\n- [ ] Add a thing (tests/test_one.py)\n")

    assert [finding.kind for finding in findings] == ["missing field"]


def test_duplicate_item_text_is_reported() -> None:
    findings = lint_plan(
        _plan(
            "- [ ] Add a thing (tests/test_one.py)",
            "- [x] Add a thing (tests/test_two.py)",
        )
    )

    assert [finding.kind for finding in findings] == ["duplicate item"]


def test_duplicate_milestone_number_is_reported() -> None:
    findings = lint_plan(
        "\n".join(
            [
                "# Plan",
                "",
                "## Milestone 1: Start",
                "- [ ] Add one (tests/test_one.py)",
                "## Milestone 1: Again",
                "- [ ] Add two (tests/test_two.py)",
            ]
        )
    )

    assert "duplicate milestone" in {finding.kind for finding in findings}


def test_malformed_heading_is_reported() -> None:
    findings = lint_plan("# Plan\n\n## Milestones\n")

    assert [finding.kind for finding in findings] == ["malformed heading"]


def test_dependency_on_a_later_item_is_reported() -> None:
    findings = lint_plan(
        _plan(
            "- [ ] First depends on item 2 (tests/test_one.py)",
            "- [ ] Second (tests/test_two.py)",
        )
    )

    assert [finding.kind for finding in findings] == ["dependency"]
    assert "later item 2" in findings[0].message


def test_dependency_on_an_earlier_item_is_allowed() -> None:
    findings = lint_plan(
        _plan(
            "- [ ] First (tests/test_one.py)",
            "- [ ] Second depends on item 1 (tests/test_two.py)",
        )
    )

    assert findings == []


def test_dependency_on_a_later_milestone_is_reported() -> None:
    text = "\n".join(
        [
            "# Plan",
            "",
            "## Milestone 1: Start",
            "- [ ] First depends on milestone 2 (tests/test_one.py)",
            "",
            "## Milestone 2: Later",
            "- [ ] Second (tests/test_two.py)",
        ]
    )

    findings = lint_plan(text)

    assert [finding.kind for finding in findings] == ["dependency"]
    assert "later milestone 2" in findings[0].message


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("item 9", "missing item 9"),
        ("milestone 9", "missing milestone 9"),
    ],
)
def test_dependency_on_a_missing_target_is_reported(
    reference: str, message: str
) -> None:
    findings = lint_plan(
        _plan(f"- [ ] First depends on {reference} (tests/test_one.py)")
    )

    assert [finding.kind for finding in findings] == ["dependency"]
    assert message in findings[0].message


def test_design_table_rejects_a_later_milestone_dependency() -> None:
    text = "\n".join(
        [
            "| # | Milestone | Layer | Depends on | Items |",
            "|---|---|---|---|---|",
            "| 1 | First | Core | — | 1 |",
            "| 2 | Second | Core | 1 | 1 |",
            "| 3 | Third | Core | 2, 4 | 1 |",
        ]
    )

    findings = lint_design_dependencies(text)

    assert [finding.kind for finding in findings] == ["dependency"]
    assert findings[0].line == 5
    assert "milestone 3 depends on later milestone 4" in findings[0].message


def test_design_table_allows_backward_dependencies() -> None:
    text = "\n".join(
        [
            "| # | Milestone | Layer | Depends on | Items |",
            "|---|---|---|---|---|",
            "| 1 | First | Core | — | 1 |",
            "| 2 | Second | Core | 1 | 1 |",
            "| 3 | Third | Core | 1, 2 | 1 |",
        ]
    )

    assert lint_design_dependencies(text) == []


def test_missing_design_table_is_not_an_error() -> None:
    assert lint_design_dependencies("# Design\n\nNo table here.\n") == []


def test_cli_fails_on_an_invalid_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tmp_path, "PLAN.md", _plan("- [ ] Add a thing"))

    assert main([str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert "missing field" in captured.out
    assert "plan problem(s) found" in captured.err


def test_cli_succeeds_on_a_valid_plan(tmp_path: Path) -> None:
    _write(tmp_path, "PLAN.md", VALID_PLAN)

    assert main([str(tmp_path)]) == 0


def test_repository_plan_is_clean() -> None:
    findings = lint_repository(ROOT)

    assert findings == [], format_findings(findings)
