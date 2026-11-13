"""Troubleshooting guide contract: every quoted message exists in code."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
GUIDE = DOCS / "troubleshooting.md"
SRC = ROOT / "src" / "aivia"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _source_text() -> str:
    """The concatenated text of every Python file under src/aivia/."""
    return "\n".join(path.read_text(encoding="utf-8") for path in SRC.rglob("*.py"))


def test_guide_exists_and_is_non_empty() -> None:
    assert GUIDE.is_file(), f"missing {GUIDE.relative_to(ROOT)}"
    assert _read(GUIDE).strip(), f"{GUIDE.relative_to(ROOT)} is empty"


def _messages() -> list[str]:
    """Every user-facing message quoted in the guide, matched against code.

    Each entry is a substring that must appear verbatim in the source under
    src/aivia/. The variable parts of a message (paths, ids, amounts) are
    omitted so the stable prefix is what is asserted.
    """
    return [
        # Common failures -- credentials
        "missing secret: ",
        "secret reference must not be empty",
        "secrets file not found: ",
        "cannot read secrets file: ",
        "invalid secrets file line ",
        "credentials reloaded",
        "error: cannot reload credentials: ",
        # Common failures -- configuration
        "error: config file not found: ",
        "error: cannot read ",
        "configuration is valid",
        # Common failures -- budget and provider errors
        "budget",
        "provider_exhausted",
        "rate_limit",
        "auth",
        "timeout",
        "invalid_request",
        "content_policy",
        "validation",
        "config",
        "quota",
        # Common failures -- bootstrap
        "error: cannot populate the config directory: ",
        "packaged defaults directory is missing: ",
        "error: config already exists at ",
        "error: cannot overwrite ",
        # Cost reports -- costs run
        "error: a run id is required",
        "error: database not found: ",
        "error: run not found: ",
        "error: cannot read the cost ledger: ",
        "Run: ",
        "Total: ",
        "BY STAGE",
        "BY PROVIDER",
        "BY MODEL",
        "BY ASSET",
        "(unassigned)",
        # Cost reports -- summary, export, dashboard
        "error: invalid date: ",
        "unknown group ",
        "wrote ",
        "error: cannot write ",
        # Resuming
        "error: --by is required to record who resumed the run",
        "error: --reason is required to record why the run was resumed",
        "error: run ",
        "is not paused on budget (state: ",
        "error: cannot resume run ",
        "error: cannot resume the run: ",
        "Resumed run ",
        # Config locations -- init output
        "Config directory: ",
        "Config file: ",
        "Data directory: ",
        "Defaults directory: ",
    ]


def test_every_quoted_message_exists_in_code() -> None:
    source = _source_text()
    for message in _messages():
        assert message in source, f"guide message not found in code: {message!r}"
