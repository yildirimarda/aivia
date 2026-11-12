"""Path traversal fuzz tests for every file input and output.

File inputs and outputs are attacker-controlled strings. Every entry point that
resolves or derives a filesystem path must keep the result inside its intended
root, so a hostile name can never read, write, list or escape outside it.

This module fuzzes:

* the assistant sandbox file access (``access``/``read``/``write``/``list_dir``/
  ``verified``), which confines every session file access to the session root,
* ``sanitize_filename``, which reduces an attacker-controlled upload filename to
  a single safe path component.

A path is considered safe when it cannot escape its root under any of the
traversal tricks an attacker can inject: ``..`` sequences, backslashes, absolute
paths, null bytes, URL/Unicode encodings, symlinks and Windows device names.
The sandbox treats its input as an opaque path (it does not URL-decode or
expand ``~``), so encoded segments that do not collapse to ``..`` simply name a
literal directory inside the root -- that is safe. The invariant tested here is
stronger than "every hostile string is blocked": *no resolved path may ever
leave the root*, whether the input is blocked outright or confined to a literal
component.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from aivia.api.uploads import sanitize_filename
from aivia.assistant import SandboxedEnvironment, SandboxPathError

SEED = 20260112


# --- curated hostile inputs -------------------------------------------------

_HOSTILE_PATHS = [
    "../../etc/passwd",
    "../..",
    "../../tmp/evil",
    "..\\..\\etc\\passwd",
    "..\\..\\..\\windows\\system32",
    "../etc/passwd",
    "..%2f..%2fetc%2fpasswd",
    "..%5c..%5cetc%5cpasswd",
    "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
    "%2e%2e\\%2e%2e\\etc\\passwd",
    "../../etc/passwd%00.jpg",
    "/etc/passwd",
    "/absolute/escape/root",
    "file:///etc/passwd",
    "//server/share/evil",
    "\\\\?\\C:\\etc\\passwd",
    "\\\\server\\share\\evil",
    "....//....//etc/passwd",
    "..../..../etc/passwd",
    "sub/../../etc/passwd",
    "a/../../../b",
    "./../../etc/passwd",
    "a/./b/../../../c",
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "LPT9",
    "con",
    "nul.txt",
    "a/../../b/../../c",
    "\\x00../../etc/passwd",
    "../../etc/passwd ",
    "....",
    "...",
    ".",
    "..",
    "",
]


# Paths that genuinely escape the root when resolved (joined with root and
# normalized). These must be blocked outright.
_ESCAPING_PATHS = [
    "../../etc/passwd",
    "../..",
    "../../tmp/evil",
    "../etc/passwd",
    "sub/../../etc/passwd",
    "a/../../../b",
    "./../../etc/passwd",
    "a/./b/../../../c",
    "/etc/passwd",
    "/absolute/escape/root",
    "a/../../b/../../c",
]


_LEGITIMATE_PATHS = [
    "hello.txt",
    "sub/hello.txt",
    "a/b/c.txt",
    "./sub/./file.txt",
    "sub//file.txt",
    "deep/nested/dir/artifact.png",
    "report.v2.json",
    "résumé.txt",
]


def _fuzz_paths() -> list[str]:
    """Deterministically generated hostile path variations."""
    rng = random.Random(SEED)
    prefixes = [
        "",
        "../",
        "../../",
        "../../../",
        "..\\..\\..\\",
        "./../",
        "sub/../../",
        "a/../b/../../",
        "%2e%2e%2f",
        "..%2f",
        "..%5c",
        "..\\u2215",
        "..\\uff0f",
        "/",
        "~/",
        "file:///",
        "\\\\server\\share\\",
    ]
    middles = ["etc", "tmp", "windows", "system32", "secret", "root", "home"]
    targets = [
        "passwd",
        "shadow",
        "hosts",
        "evil.txt",
        "secret.key",
        "credential.json",
        "id_rsa",
        "config.yaml",
    ]
    suffixes = [
        "",
        "%00.jpg",
        "\\x00.png",
        " ",
        " ../../extra",
        "/../../etc/passwd",
        "\\..\\..\\etc\\passwd",
    ]
    out: list[str] = []
    for _ in range(400):
        path = (
            rng.choice(prefixes)
            + rng.choice(middles)
            + "/"
            + rng.choice(targets)
            + rng.choice(suffixes)
        )
        out.append(path)
    return out


_HOSTILE_FILENAMES = [
    "../../etc/passwd",
    "../..",
    "..\\..\\etc\\passwd",
    "/etc/passwd",
    "~",
    "~/escape",
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "LPT9",
    "con",
    "nul.txt",
    ".env",
    ".hidden",
    "..",
    ".",
    "",
    "a/b/c",
    "a\\b\\c",
    "a//b",
    "a\\\\b",
    "evil<script>.png",
    "evil\x00<script>.png",
    "résumé.png",
    "a" * 300 + ".png",
    "CON.txt",
    "PRN.log",
    "NUL.dat",
    "....",
    "...",
    " . . ",
    "  ..  ",
    "file:///etc/passwd",
    "%2e%2e%2fpasswd",
    "normal.txt",
    "report.v2.json",
    "image.png",
]


def _fuzz_filenames() -> list[str]:
    """Deterministically generated hostile filename variations."""
    rng = random.Random(SEED + 1)
    chunks = [
        "..",
        "../",
        "..\\",
        "%2e%2e",
        "%2f",
        "%5c",
        "\\x00",
        "~",
        "etc",
        "passwd",
        "secret",
        "windows",
        "CON",
        "PRN",
        "NUL",
        "<script>",
        "\x00",
        " ",
        ".",
        "..",
        "résumé",
        "evil",
        "a" * 50,
    ]
    separators = ["/", "\\", "//", "\\\\", "/..", "\\..", ""]
    out: list[str] = []
    for _ in range(400):
        name = "".join(rng.choice(chunks) for _ in range(rng.randint(1, 6)))
        if rng.random() < 0.5:
            name = rng.choice(separators) + name + rng.choice(separators)
        out.append(name)
    return out


# --- sandbox file access ----------------------------------------------------


@pytest.fixture
def sandbox(tmp_path: Path) -> SandboxedEnvironment:
    return SandboxedEnvironment(tmp_path / "session")


@pytest.mark.parametrize("path", _HOSTILE_PATHS + _fuzz_paths())
def test_sandbox_never_resolves_outside_root(
    sandbox: SandboxedEnvironment, path: str
) -> None:
    """Any input either is blocked or resolves inside the session root."""
    try:
        resolved = sandbox.access(path)
    except SandboxPathError:
        return
    assert resolved.is_relative_to(sandbox.root)


@pytest.mark.parametrize("escaping", _ESCAPING_PATHS)
def test_sandbox_blocks_genuine_traversal(
    sandbox: SandboxedEnvironment, escaping: str
) -> None:
    """Genuine traversal inputs are rejected by every file access method."""
    assert sandbox.verified(escaping) is False
    with pytest.raises(SandboxPathError):
        sandbox.access(escaping)
    with pytest.raises(SandboxPathError):
        sandbox.read(escaping)
    with pytest.raises(SandboxPathError):
        sandbox.write(escaping, b"x")
    with pytest.raises(SandboxPathError):
        sandbox.list_dir(escaping)


def test_sandbox_symlink_escape_is_blocked(tmp_path: Path) -> None:
    """A symlink pointing outside the root must not be traversable."""
    sandbox = SandboxedEnvironment(tmp_path / "session")
    target = tmp_path / "outside.txt"
    target.write_bytes(b"secret")
    (sandbox.root / "link").symlink_to(target)

    assert sandbox.verified("link") is False
    with pytest.raises(SandboxPathError):
        sandbox.access("link")
    with pytest.raises(SandboxPathError):
        sandbox.read("link")
    with pytest.raises(SandboxPathError):
        sandbox.write("link", b"leaked")


@pytest.mark.parametrize("legit", _LEGITIMATE_PATHS)
def test_sandbox_allows_legitimate_paths_inside_root(
    sandbox: SandboxedEnvironment, legit: str
) -> None:
    """Well-formed relative paths still resolve inside the session root."""
    assert sandbox.verified(legit) is True
    resolved = sandbox.access(legit)
    assert resolved.is_relative_to(sandbox.root)

    sandbox.write(legit, b"contents")
    assert sandbox.read(legit) == b"contents"

    parent = resolved.parent
    children = sandbox.list_dir(str(parent.relative_to(sandbox.root)))
    assert any(child.name == resolved.name for child in children)


def test_sandbox_write_creates_files_inside_root(tmp_path: Path) -> None:
    """Writing through a nested relative path never leaves the root."""
    sandbox = SandboxedEnvironment(tmp_path / "session")
    sandbox.write("a/b/c.txt", b"data")
    expected = sandbox.root / "a" / "b" / "c.txt"
    assert expected.read_bytes() == b"data"
    assert expected.is_relative_to(sandbox.root)


# --- upload filename sanitization ------------------------------------------


@pytest.mark.parametrize("hostile", _HOSTILE_FILENAMES + _fuzz_filenames())
def test_sanitized_filename_is_a_single_safe_component(hostile: str) -> None:
    """Every hostile filename collapses to one safe path component."""
    safe = sanitize_filename(hostile)

    assert safe != ""
    assert safe not in {".", ".."}
    assert "/" not in safe and "\\" not in safe
    assert not safe.startswith(".")
    assert Path(safe).name == safe
    assert len(safe) <= 128

    # The sanitized name cannot escape any base directory it is joined to.
    base = Path("/tmp/safe_base_root")
    joined = (base / safe).resolve()
    assert joined.is_relative_to(base)

    # Sanitization is idempotent.
    assert sanitize_filename(safe) == safe


@pytest.mark.parametrize("max_length", [1, 2, 5, 16, 64, 128])
def test_sanitized_filename_respects_max_length(max_length: int) -> None:
    """The truncated result never exceeds max_length and stays safe."""
    hostile = "../../../" + "a" * 500 + "/etc/passwd"
    safe = sanitize_filename(hostile, max_length=max_length)

    assert 0 < len(safe) <= max_length
    assert "/" not in safe and "\\" not in safe
    assert safe not in {".", ".."}
    joined = Path("/tmp/base") / safe
    assert joined.resolve().is_relative_to(Path("/tmp/base"))


def test_sanitize_filename_never_escapes_under_fuzzing(tmp_path: Path) -> None:
    """Cross-check: joining every sanitized fuzz output stays inside any root."""
    base = tmp_path / "root"
    base.mkdir()
    for hostile in _HOSTILE_FILENAMES + _fuzz_filenames():
        safe = sanitize_filename(hostile)
        joined = (base / safe).resolve()
        assert joined.is_relative_to(base), safe
