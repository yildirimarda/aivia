"""Advisory lint for vendor and tuning values that escaped configuration.

The lint scans production source and reports vendor names, model ids, prices
and threshold literals found outside provider adapters, config modules and data
files. It is advisory: by default findings are printed but the command exits
successfully, so the report can guide cleanup without blocking a branch.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import cast

from aivia.core.paths import package_resource

__all__ = [
    "ALLOWED_PATH_PARTS",
    "ALLOW_FILE_MARKER",
    "Finding",
    "Rules",
    "format_findings",
    "load_rules",
    "main",
    "scan_file",
    "scan_repository",
    "scan_text",
]

ALLOW_FILE_MARKER = "config-coverage: allow-file"

ALLOWED_PATH_PARTS = frozenset(
    {
        "adapter",
        "adapters",
        "config",
        "data",
        "defaults",
        "llm",
        "provider",
        "providers",
    }
)

SOURCE_DIRS = ("src", "scripts")

_TEXT_SUFFIXES = frozenset(
    {
        ".bash",
        ".cfg",
        ".ini",
        ".json",
        ".py",
        ".pyi",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)

_THRESHOLD_ASSIGNMENT = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?P<op>==|!=|<=|>=|[:=<>])\s*"
    r"(?P<value>-?\d+(?:\.\d+)?)\b"
)


@dataclass(frozen=True, slots=True)
class Finding:
    """One value that belongs in configuration, with where it was found."""

    path: str
    line: int
    kind: str
    value: str


@dataclass(frozen=True, slots=True)
class Rules:
    """The data-driven rules the lint applies to a line of text."""

    vendor_pattern: re.Pattern[str]
    model_patterns: tuple[re.Pattern[str], ...]
    price_patterns: tuple[re.Pattern[str], ...]
    threshold_names: tuple[str, ...]


def load_rules(path: Path | None = None) -> Rules:
    """Load rules from *path*, or from the packaged data file by default."""
    rules_path = (
        package_resource("data", "config_coverage.json") if path is None else path
    )
    payload = cast(
        dict[str, list[str]],
        json.loads(rules_path.read_text(encoding="utf-8")),
    )
    vendors = tuple(payload["vendors"])
    vendor_alternation = "|".join(
        re.escape(vendor) for vendor in sorted(vendors, key=len, reverse=True)
    )
    return Rules(
        vendor_pattern=re.compile(
            rf"(?<![A-Za-z0-9])(?:{vendor_alternation})(?![A-Za-z0-9])"
        ),
        model_patterns=tuple(
            re.compile(pattern, re.IGNORECASE) for pattern in payload["model_patterns"]
        ),
        price_patterns=tuple(
            re.compile(pattern, re.IGNORECASE) for pattern in payload["price_patterns"]
        ),
        threshold_names=tuple(name.lower() for name in payload["threshold_names"]),
    )


@lru_cache(maxsize=1)
def _default_rules() -> Rules:
    return load_rules()


def scan_text(
    text: str, *, path: str = "<text>", rules: Rules | None = None
) -> list[Finding]:
    """Return advisory findings for *text*, using *path* only for reporting."""
    resolved = rules or _default_rules()
    findings: list[Finding] = []

    for match in resolved.vendor_pattern.finditer(text):
        findings.append(
            Finding(
                path=path,
                line=_line_number(text, match.start()),
                kind="vendor name",
                value=match.group(0),
            )
        )

    for kind, patterns in (
        ("model id", resolved.model_patterns),
        ("price", resolved.price_patterns),
    ):
        for pattern in patterns:
            for match in pattern.finditer(text):
                findings.append(
                    Finding(
                        path=path,
                        line=_line_number(text, match.start()),
                        kind=kind,
                        value=match.group(0),
                    )
                )

    findings.extend(_threshold_findings(text, path=path, rules=resolved))
    return _dedupe(findings)


def scan_file(
    path: Path, *, root: Path | None = None, rules: Rules | None = None
) -> list[Finding]:
    """Scan one file, skipping binary files and explicit allow-file markers."""
    try:
        data = path.read_bytes()
    except OSError:
        return []
    if b"\x00" in data[:8192]:
        return []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return []
    if ALLOW_FILE_MARKER in text:
        return []

    display = path.relative_to(root).as_posix() if root is not None else path.as_posix()
    return scan_text(text, path=display, rules=rules)


def scan_repository(
    root: Path | str = Path("."), *, rules: Rules | None = None
) -> list[Finding]:
    """Scan production source directories under *root* for uncovered values."""
    root_path = Path(root)
    findings: list[Finding] = []
    for path in _source_files(root_path):
        findings.extend(scan_file(path, root=root_path, rules=rules))
    return _dedupe(findings)


def format_findings(findings: Iterable[Finding]) -> str:
    """Render findings as one line per hit."""
    return "\n".join(
        f"{item.path}:{item.line}: {item.kind}: {item.value}" for item in findings
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the advisory lint, optionally failing when ``--strict`` is set."""
    parser = argparse.ArgumentParser(
        prog="aivia-config-coverage",
        description=(
            "Report vendor names, model ids, prices and thresholds found outside "
            "adapters, config and data."
        ),
    )
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when findings exist instead of only advising",
    )
    args = parser.parse_args(argv)

    findings = scan_repository(Path(args.root))
    if findings:
        print(format_findings(findings))
        print(
            f"{len(findings)} advisory config coverage finding(s)",
            file=sys.stderr,
        )
        return 1 if args.strict else 0
    return 0


def _threshold_findings(text: str, *, path: str, rules: Rules) -> list[Finding]:
    findings: list[Finding] = []
    for match in _THRESHOLD_ASSIGNMENT.finditer(text):
        name = match.group("name").lower()
        if any(term in name for term in rules.threshold_names):
            findings.append(
                Finding(
                    path=path,
                    line=_line_number(text, match.start()),
                    kind="threshold",
                    value=match.group(0).strip(),
                )
            )
    return findings


def _source_files(root: Path) -> Iterator[Path]:
    for directory in SOURCE_DIRS:
        base = root / directory
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            if _is_allowed(path.relative_to(root)):
                continue
            yield path


def _is_allowed(relative: Path) -> bool:
    return any(part in ALLOWED_PATH_PARTS for part in relative.parts)


def _line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _dedupe(findings: Iterable[Finding]) -> list[Finding]:
    unique = {(item.path, item.line, item.kind, item.value): item for item in findings}
    return [unique[key] for key in sorted(unique)]


if __name__ == "__main__":
    raise SystemExit(main())
