"""Lint PLAN.md for canonical items, complete fields and backward dependencies.

The plan is deliberately plain: milestones are headings and work is a list of
checkbox items whose last parenthesised field names the test or command that
proves the item.  This lint keeps that contract honest.  It rejects item lines
that do not use the canonical marker, items with a missing or duplicated
verification field, duplicate items or milestones and dependencies that point
at an item or milestone which comes later in the plan.

The only dependency notation understood is the explicit phrase ``depends on
item N`` or ``depends on milestone N``.  The current plan uses ordering alone,
so the check stays dormant until an item chooses to spell a dependency out.
Milestone dependencies declared in the ``Depends on`` table in DESIGN.md are
checked as well: a milestone may only depend on an earlier milestone.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Finding",
    "Item",
    "Milestone",
    "Plan",
    "format_findings",
    "lint_design_dependencies",
    "lint_plan",
    "lint_repository",
    "main",
    "parse_plan",
]

_MILESTONE = re.compile(r"^## Milestone (?P<number>\d+): (?P<name>\S.*)$")
_DISCOVERED = re.compile(r"^## Discovered$")
_ITEM = re.compile(r"^\s*- \[(?P<status>[ xX])\] (?P<body>.*)$")
_ITEM_LIKE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")
_HEADING = re.compile(r"^#{1,6}\s")
_FIELD = re.compile(r"^(?P<text>.*?)\s+\((?P<field>[^()]*)\)$")
_PARENTHESES = re.compile(r"\([^()]*\)")
_DEPENDENCY = re.compile(
    r"\bdepends? on\s+(?P<kind>item|milestone)\s+(?P<number>\d+)\b",
    re.IGNORECASE,
)
_TABLE_HEADING = "| # | Milestone |"


@dataclass(frozen=True, slots=True)
class Finding:
    """One reason the plan is not valid, with the line that caused it."""

    path: str
    line: int
    kind: str
    message: str


@dataclass(frozen=True, slots=True)
class Item:
    """A parsed checkbox item with its position and verification field."""

    line: int
    ordinal: int
    milestone: int | None
    done: bool
    text: str
    verification: str


@dataclass(frozen=True, slots=True)
class Milestone:
    """A parsed milestone heading and its items."""

    line: int
    number: int
    name: str
    items: tuple[Item, ...]


@dataclass(frozen=True, slots=True)
class Plan:
    """The parsed plan plus every structural finding collected while parsing."""

    milestones: tuple[Milestone, ...]
    items: tuple[Item, ...]
    findings: tuple[Finding, ...]


def parse_plan(text: str, *, path: str = "PLAN.md") -> Plan:
    """Parse *text* and report malformed lines, fields and duplicates."""
    findings: list[Finding] = []
    milestones: list[tuple[int, int, str, list[Item]]] = []
    items: list[Item] = []
    seen_milestones: dict[int, int] = {}
    seen_names: dict[str, int] = {}
    seen_items: dict[str, int] = {}
    current: tuple[int, int, str, list[Item]] | None = None
    section: int | None = None
    in_discovered = False
    previous_milestone = 0
    in_comment = False

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()

        if in_comment:
            if "-->" in line:
                in_comment = False
            continue
        if line.startswith("<!--"):
            if "-->" not in line:
                in_comment = True
            continue
        if not line:
            continue

        milestone_match = _MILESTONE.match(line)
        if milestone_match is not None:
            number = int(milestone_match.group("number"))
            name = milestone_match.group("name").strip()
            if number in seen_milestones:
                findings.append(
                    _finding(
                        path,
                        line_number,
                        "duplicate milestone",
                        f"milestone {number} is declared more than once",
                    )
                )
            if name.lower() in seen_names:
                findings.append(
                    _finding(
                        path,
                        line_number,
                        "duplicate milestone",
                        f"milestone name {name!r} is used more than once",
                    )
                )
            if number <= previous_milestone:
                findings.append(
                    _finding(
                        path,
                        line_number,
                        "milestone order",
                        f"milestone {number} does not follow milestone "
                        f"{previous_milestone}",
                    )
                )
            seen_milestones[number] = line_number
            seen_names[name.lower()] = line_number
            previous_milestone = number
            current = (line_number, number, name, [])
            milestones.append(current)
            section = number
            in_discovered = False
            continue

        if _DISCOVERED.match(line):
            current = None
            section = None
            in_discovered = True
            continue

        if line.startswith("# "):
            continue

        if _HEADING.match(line):
            findings.append(
                _finding(
                    path,
                    line_number,
                    "malformed heading",
                    (
                        "only '## Milestone N: name' and '## Discovered' "
                        "headings are allowed"
                    ),
                )
            )
            continue

        item_match = _ITEM.match(line)
        if item_match is not None:
            ordinal = len(items) + 1
            if current is None and not in_discovered:
                findings.append(
                    _finding(
                        path,
                        line_number,
                        "missing field",
                        "item is not inside a milestone",
                    )
                )
            body = item_match.group("body").strip()
            text_value, verification = _item_fields(path, line_number, body, findings)
            if "`" in body or "|" in body:
                findings.append(
                    _finding(
                        path,
                        line_number,
                        "malformed item line",
                        "items may not contain backticks or pipes",
                    )
                )
            key = " ".join(text_value.lower().split())
            if key and key in seen_items:
                findings.append(
                    _finding(
                        path,
                        line_number,
                        "duplicate item",
                        f"same item text as line {seen_items[key]}",
                    )
                )
            elif key:
                seen_items[key] = line_number
            item = Item(
                line=line_number,
                ordinal=ordinal,
                milestone=section,
                done=item_match.group("status").lower() == "x",
                text=text_value,
                verification=verification,
            )
            items.append(item)
            if current is not None:
                current[3].append(item)
            continue

        if _ITEM_LIKE.match(line):
            findings.append(
                _finding(
                    path,
                    line_number,
                    "malformed item line",
                    "expected '- [ ] text (tests/path.py)' or '- [x] ...'",
                )
            )
            continue

    findings.extend(_dependency_findings(path, items, seen_milestones))
    parsed_milestones = tuple(
        Milestone(line=line, number=number, name=name, items=tuple(group))
        for line, number, name, group in milestones
    )
    return Plan(
        milestones=parsed_milestones,
        items=tuple(items),
        findings=tuple(findings),
    )


def lint_plan(text: str, *, path: str = "PLAN.md") -> list[Finding]:
    """Return every structural and dependency finding for *text*."""
    return list(parse_plan(text, path=path).findings)


def lint_design_dependencies(
    text: str, *, path: str = "docs/DESIGN.md"
) -> list[Finding]:
    """Check the milestone dependency table, rejecting later dependencies."""
    findings: list[Finding] = []
    lines = text.splitlines()
    header_index: int | None = None
    dependency_column: int | None = None

    for index, line in enumerate(lines):
        if line.strip().startswith(_TABLE_HEADING):
            header_index = index
            cells = _table_cells(line)
            if "Depends on" in cells:
                dependency_column = cells.index("Depends on")
            break
    if header_index is None or dependency_column is None:
        return findings

    for line_number, line in enumerate(
        lines[header_index + 1 :], start=header_index + 2
    ):
        cells = _table_cells(line)
        if not cells or not cells[0].isdigit():
            continue
        if len(cells) <= dependency_column:
            continue
        milestone = int(cells[0])
        dependencies = [
            int(number) for number in re.findall(r"\d+", cells[dependency_column])
        ]
        for dependency in dependencies:
            if dependency >= milestone:
                findings.append(
                    _finding(
                        path,
                        line_number,
                        "dependency",
                        f"milestone {milestone} depends on later milestone "
                        f"{dependency}",
                    )
                )
    return findings


def lint_repository(root: Path | str = Path(".")) -> list[Finding]:
    """Lint PLAN.md and the milestone dependency table under *root*."""
    root_path = Path(root)
    findings: list[Finding] = []
    plan_path = root_path / "PLAN.md"
    if plan_path.is_file():
        findings.extend(
            lint_plan(
                plan_path.read_text(encoding="utf-8"),
                path=plan_path.relative_to(root_path).as_posix(),
            )
        )
    design_path = root_path / "docs" / "DESIGN.md"
    if design_path.is_file():
        findings.extend(
            lint_design_dependencies(
                design_path.read_text(encoding="utf-8"),
                path=design_path.relative_to(root_path).as_posix(),
            )
        )
    return findings


def format_findings(findings: Iterable[Finding]) -> str:
    """Render findings as one line per problem."""
    return "\n".join(
        f"{finding.path}:{finding.line}: {finding.kind}: {finding.message}"
        for finding in findings
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the lint against a repository root and fail when it is invalid."""
    parser = argparse.ArgumentParser(
        prog="aivia-plan-lint",
        description=(
            "Fail on malformed plan items, duplicate or missing fields and "
            "dependencies that point forward."
        ),
    )
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)

    findings = lint_repository(Path(args.root))
    if findings:
        print(format_findings(findings))
        print(f"{len(findings)} plan problem(s) found", file=sys.stderr)
        return 1
    return 0


def _item_fields(
    path: str, line: int, body: str, findings: list[Finding]
) -> tuple[str, str]:
    if not body:
        findings.append(_finding(path, line, "missing field", "item has no text"))
        return "", ""

    groups = _PARENTHESES.findall(body)
    if len(groups) > 1:
        findings.append(
            _finding(
                path,
                line,
                "duplicate field",
                "item has more than one verification field",
            )
        )

    match = _FIELD.match(body)
    if match is None:
        findings.append(
            _finding(
                path,
                line,
                "missing field",
                "item is missing a trailing test or verification field",
            )
        )
        return body, ""

    text = match.group("text").strip()
    verification = match.group("field").strip()
    if not text:
        findings.append(
            _finding(path, line, "missing field", "item has no description")
        )
    if not verification:
        findings.append(
            _finding(
                path,
                line,
                "missing field",
                "item has an empty verification field",
            )
        )
    return text, verification


def _dependency_findings(
    path: str, items: Sequence[Item], milestones: dict[int, int]
) -> list[Finding]:
    findings: list[Finding] = []
    total = len(items)
    for item in items:
        for match in _DEPENDENCY.finditer(item.text):
            kind = match.group("kind").lower()
            number = int(match.group("number"))
            if kind == "item":
                if number < 1 or number > total:
                    findings.append(
                        _finding(
                            path,
                            item.line,
                            "dependency",
                            f"depends on missing item {number}",
                        )
                    )
                elif number >= item.ordinal:
                    findings.append(
                        _finding(
                            path,
                            item.line,
                            "dependency",
                            f"depends on later item {number}",
                        )
                    )
            else:
                if number not in milestones:
                    findings.append(
                        _finding(
                            path,
                            item.line,
                            "dependency",
                            f"depends on missing milestone {number}",
                        )
                    )
                elif item.milestone is not None and number >= item.milestone:
                    findings.append(
                        _finding(
                            path,
                            item.line,
                            "dependency",
                            f"depends on later milestone {number}",
                        )
                    )
    return findings


def _table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _finding(path: str, line: int, kind: str, message: str) -> Finding:
    return Finding(path=path, line=line, kind=kind, message=message)


if __name__ == "__main__":
    raise SystemExit(main())
