# secret-scan: allow-file
"""Scan repository text for credentials and high-entropy strings.

The scan is deliberately conservative about output: a finding records where a
suspicious string is and what kind of rule matched, never the matched value.
That keeps a failing test or CI log from becoming a second leak.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from aivia.core.redact import registered_secrets

__all__ = [
    "ALLOW_FILE_MARKER",
    "DEFAULT_SKIP_DIRS",
    "Finding",
    "format_findings",
    "main",
    "scan_file",
    "scan_for_secrets",
    "scan_repository",
    "scan_text",
]

ALLOW_FILE_MARKER = "secret-scan: allow-file"

DEFAULT_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "graphify-out",
        "logs",
        "node_modules",
        "venv",
    }
)

_MIN_ENTROPY = 4.5
_MIN_ENTROPY_LENGTH = 24

_KEY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private key",
        re.compile(
            r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"
            r".*?"
            r"-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    (
        "aws access key",
        re.compile(
            r"\b(?:AKIA|ASIA|ABIA|ACCA|AGPA|AIDA|AIPA|ANPA|ANVA|APKA|AROA|ASCA)"
            r"[A-Z0-9]{16}\b"
        ),
    ),
    (
        "github token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ),
    ("slack token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b")),
    ("google api key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("openai key", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{16,}\b")),
    (
        "stripe key",
        re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b"),
    ),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    ),
    (
        "bearer token",
        re.compile(r"\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE),
    ),
    ("url credentials", re.compile(r"\bhttps?://[^/\s:@]+:[^/\s@]+@")),
    (
        "credential assignment",
        re.compile(
            r"(?:api[_-]?key|secret|token|password|passwd|pwd|"
            r"access[_-]?token|refresh[_-]?token|client[_-]?secret|"
            r"auth[_-]?token|bearer[_-]?token)"
            r"(?![A-Za-z0-9_])"
            r"\s*[:=]\s*"
            r"(?P<quote>[\"']?)"
            r"(?P<value>[A-Za-z0-9+/_=-]{12,})"
            r"(?P=quote)",
            re.IGNORECASE,
        ),
    ),
)

_ENTROPY_CANDIDATES: tuple[re.Pattern[str], ...] = (
    re.compile(r"[A-Za-z0-9+]{24,}={0,2}"),
    re.compile(r"[A-Za-z0-9_-]{24,}"),
)

_PLACEHOLDER = re.compile(
    r"""
    (?:
        <[^>\n]+>
      | \$\{[^}\n]+\}
      | \$\([^)\n]+\)
      | \{\{[^}\n]+\}\}
      | \{env:[^}\n]+\}
      | %[A-Za-z_][A-Za-z0-9_]*%
      | change[-_ ]?me
      | replace[-_ ]?me
      | your[-_ ].*
      | example
      | placeholder
      | dummy
      | fake
      | redacted
      | none
      | null
      | true
      | false
      | x{3,}
      | \*+
      | \.{3,}
      | -+
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True, slots=True)
class Finding:
    """One suspicious location, with no copy of the matched value."""

    path: str
    line: int
    kind: str


def scan_text(text: str, *, path: str = "<text>") -> list[Finding]:
    """Return findings for *text*, using *path* only for reporting."""
    findings: list[Finding] = []
    for kind, pattern in _KEY_PATTERNS:
        for match in pattern.finditer(text):
            if kind == "credential assignment" and not _credential_value_is_secret(
                match.group("value")
            ):
                continue
            findings.append(
                Finding(path=path, line=_line_number(text, match.start()), kind=kind)
            )

    for pattern in _ENTROPY_CANDIDATES:
        for match in pattern.finditer(text):
            if _looks_high_entropy(match.group(0)):
                findings.append(
                    Finding(
                        path=path,
                        line=_line_number(text, match.start()),
                        kind="high entropy",
                    )
                )
    return _dedupe(findings)


def scan_file(path: Path, *, root: Path | None = None) -> list[Finding]:
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
    return scan_text(text, path=display)


def scan_repository(root: Path | str = Path(".")) -> list[Finding]:
    """Scan every file that git would include, or walk the tree as a fallback."""
    root_path = Path(root)
    findings: list[Finding] = []
    for path in _repository_files(root_path):
        if path.is_file():
            findings.extend(scan_file(path, root=root_path))
    return _dedupe(findings)


def format_findings(findings: Iterable[Finding]) -> str:
    """Render findings as one line per hit without any matched value."""
    return "\n".join(f"{item.path}:{item.line}: {item.kind}" for item in findings)


def scan_for_secrets(
    texts: Iterable[tuple[str, str]],
    *,
    secrets: Iterable[str] | None = None,
) -> list[Finding]:
    """Scan each ``(source, text)`` pair for configured secret values.

    *texts* pairs a label naming the run output (a log, an artifact, an error
    or a sidecar) with its text. Returns one finding per matching line, never
    copying the secret value itself so a failing audit cannot leak it.
    """
    if secrets is None:
        secrets = registered_secrets()
    needles = tuple(secret for secret in secrets if secret)
    findings: list[Finding] = []
    for source, text in texts:
        for line_number, line in enumerate(text.splitlines(), 1):
            for secret in needles:
                if secret in line:
                    findings.append(
                        Finding(path=source, line=line_number, kind="configured secret")
                    )
                    break
    return _dedupe(findings)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scanner and return a non-zero exit code when anything is found."""
    parser = argparse.ArgumentParser(
        prog="aivia-secret-scan",
        description=(
            "Fail when the repository contains credentials or high entropy strings."
        ),
    )
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args(argv)

    findings = scan_repository(Path(args.root))
    if findings:
        print(format_findings(findings))
        print(
            f"{len(findings)} potential secret(s) found",
            file=sys.stderr,
        )
        return 1
    return 0


def _repository_files(root: Path) -> Iterator[Path]:
    tracked = _git_files(root)
    if tracked is not None:
        yield from tracked
        return
    yield from _walk_files(root)


def _git_files(root: Path) -> list[Path] | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    names = result.stdout.decode("utf-8", "surrogateescape").split("\0")
    return [root / name for name in names if name]


def _walk_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in DEFAULT_SKIP_DIRS for part in relative.parts):
            continue
        if path.is_file():
            yield path


def _line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _credential_value_is_secret(value: str) -> bool:
    if _is_placeholder(value):
        return False
    if value.startswith("_"):
        return False
    if re.fullmatch(r"[A-Z][A-Z0-9_]*", value):
        return False
    return not (value.islower() and "_" in value)


def _looks_high_entropy(value: str) -> bool:
    if len(value) < _MIN_ENTROPY_LENGTH:
        return False
    if not any(character.islower() for character in value):
        return False
    if not any(character.isupper() for character in value) and not any(
        character.isdigit() for character in value
    ):
        return False
    if _is_placeholder(value):
        return False
    if len(set(value)) <= 4:
        return False
    return _entropy(value) >= _MIN_ENTROPY


def _entropy(value: str) -> float:
    length = len(value)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in Counter(value).values()
    )


def _is_placeholder(value: str) -> bool:
    stripped = value.strip().strip("\"'")
    if not stripped:
        return True
    return _PLACEHOLDER.fullmatch(stripped) is not None


def _dedupe(findings: Iterable[Finding]) -> list[Finding]:
    unique = {(item.path, item.line, item.kind): item for item in findings}
    return [unique[key] for key in sorted(unique)]


if __name__ == "__main__":
    raise SystemExit(main())
