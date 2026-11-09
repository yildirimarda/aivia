"""``aivia session repl``: an interactive session REPL.

The REPL reads one message per line from stdin. Each message is matched against
a fixed set of sample edits; the matched edit becomes a composition patch that
refines or replans the session's current composition and is stored as a new
session version. ``undo`` steps back one version, ``redo`` steps forward, and
applying an edit after an undo starts a new branch (the abandoned forward
history is dropped).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from aivia.cli.registry import register_command
from aivia.compose import Canvas, Composition
from aivia.compose.patch import (
    CompositionPatch,
    PatchOperation,
    apply_patch_reversible,
)
from aivia.compose.schema import TextLayer

__all__ = ["configure", "run_repl", "run_session"]

_PROMPT = "session> "


@dataclass(frozen=True, slots=True)
class SampleEdit:
    """One canned message and the patch it maps to."""

    keywords: tuple[str, ...]
    patch: CompositionPatch
    note: str


def _initial_composition() -> Composition:
    """Return the composition a fresh session starts from."""
    return Composition(
        id="session",
        canvas=Canvas(width=1280, height=720, background="#000000"),
        duration_seconds=10.0,
        fps=30,
        layers=[
            TextLayer(id="title", text="Hello"),
        ],
        audio_tracks=[],
        effects=[],
    )


SAMPLE_EDITS: tuple[SampleEdit, ...] = (
    SampleEdit(
        keywords=("background", "bg", "color"),
        patch=CompositionPatch.from_operations(
            [
                PatchOperation(op="set", path="/canvas/background", value="#ffffff"),
            ],
            description="Set the canvas background to white.",
        ),
        note="Set canvas background to white.",
    ),
    SampleEdit(
        keywords=("duration", "length", "longer"),
        patch=CompositionPatch.from_operations(
            [
                PatchOperation(op="set", path="/duration_seconds", value=15.0),
            ],
            description="Extend the composition to 15 seconds.",
        ),
        note="Extend duration to 15 seconds.",
    ),
    SampleEdit(
        keywords=("fps", "frame rate", "frames"),
        patch=CompositionPatch.from_operations(
            [
                PatchOperation(op="set", path="/fps", value=60),
            ],
            description="Set the frame rate to 60 fps.",
        ),
        note="Set frame rate to 60 fps.",
    ),
    SampleEdit(
        keywords=("title", "text", "caption"),
        patch=CompositionPatch.from_operations(
            [
                PatchOperation(op="set", path="/layers/0/text", value="Refined title"),
            ],
            description="Refine the title text.",
        ),
        note="Refine the title text.",
    ),
    SampleEdit(
        keywords=("opacity", "transparent", "fade"),
        patch=CompositionPatch.from_operations(
            [
                PatchOperation(op="set", path="/layers/0/opacity", value=0.5),
            ],
            description="Fade the title to 50% opacity.",
        ),
        note="Fade the title to 50% opacity.",
    ),
)


def _match_edit(message: str) -> SampleEdit | None:
    """Return the sample edit whose keywords appear in *message*, or ``None``."""
    lowered = message.casefold()
    for edit in SAMPLE_EDITS:
        if any(keyword in lowered for keyword in edit.keywords):
            return edit
    return None


@dataclass(frozen=True, slots=True)
class _Version:
    version: int
    note: str
    composition: Composition


class SessionEditor:
    """Track a session's version history with undo, redo and branching.

    Applying an edit appends a new version and becomes the tip. ``undo`` moves
    the tip back one version and ``redo`` moves it forward. Applying an edit
    while the tip is not at the end of the history truncates the abandoned
    forward history, so editing after an undo starts a new branch.
    """

    def __init__(self, composition: Composition) -> None:
        self._versions: list[_Version] = [
            _Version(version=1, note="initial", composition=composition)
        ]
        self._index = 0
        self._next_version = (
            max((version.version for version in self._versions), default=0) + 1
        )

    @property
    def current(self) -> Composition:
        """The composition at the current tip."""
        return self._versions[self._index].composition

    @property
    def current_version(self) -> int:
        """The version number at the current tip."""
        return self._versions[self._index].version

    @property
    def version_count(self) -> int:
        """The number of recorded versions."""
        return len(self._versions)

    @property
    def can_undo(self) -> bool:
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        return self._index < len(self._versions) - 1

    def apply(self, patch: CompositionPatch, note: str) -> Composition:
        """Apply *patch* to the current composition and record a new version."""
        result = apply_patch_reversible(
            self.current,
            patch,
            validate=True,
            asset_exists=lambda source: True,
        )
        # Drop anything ahead of the tip: editing after an undo branches.
        self._versions = self._versions[: self._index + 1]
        new_version = self._next_version
        self._next_version += 1
        self._versions.append(
            _Version(version=new_version, note=note, composition=result.composition)
        )
        self._index = len(self._versions) - 1
        return result.composition

    def undo(self) -> Composition | None:
        """Step the tip back one version, or ``None`` when already at the start."""
        if not self.can_undo:
            return None
        self._index -= 1
        return self.current

    def redo(self) -> Composition | None:
        """Step the tip forward one version, or ``None`` when already at the tip."""
        if not self.can_redo:
            return None
        self._index += 1
        return self.current

    def history(self) -> tuple[_Version, ...]:
        return tuple(self._versions)


def _summarize(composition: Composition) -> str:
    layer_bits: list[str] = []
    for layer in composition.layers:
        bit = layer.id
        text = getattr(layer, "text", None)
        if text is not None:
            bit += f" text={text!r}"
        opacity = getattr(layer, "opacity", None)
        if opacity is not None:
            bit += f" opacity={opacity}"
        layer_bits.append(bit)
    return (
        f"canvas={composition.canvas.width}x{composition.canvas.height}"
        f" bg={composition.canvas.background}"
        f" duration={composition.duration_seconds}"
        f" fps={composition.fps}"
        f" layers=[{', '.join(layer_bits)}]"
    )


def _read_line() -> str | None:
    """Read one line from stdin, returning ``None`` at end of input."""
    try:
        line = sys.stdin.readline()
    except (EOFError, KeyboardInterrupt):
        return None
    if line == "":
        return None
    return line


def _print(text: str) -> None:
    print(text, flush=True)


def _handle(
    line: str,
    editor: SessionEditor,
) -> bool:
    """Handle one REPL line. Return ``False`` when the REPL should stop."""
    lowered = line.casefold()
    if lowered in {"quit", "exit", "q"}:
        _print("Goodbye.")
        return False
    if lowered == "undo":
        composition = editor.undo()
        if composition is None:
            _print("(nothing to undo)")
        else:
            _print(f"undone to version {editor.current_version}")
            _print(_summarize(composition))
        return True
    if lowered == "redo":
        composition = editor.redo()
        if composition is None:
            _print("(nothing to redo)")
        else:
            _print(f"redone to version {editor.current_version}")
            _print(_summarize(composition))
        return True
    if lowered in {"show", "current"}:
        _print(_summarize(editor.current))
        return True
    if lowered in {"list", "history", "versions"}:
        for version in editor.history():
            marker = " * " if version.version == editor.current_version else "   "
            _print(f"{marker}version {version.version}: {version.note}")
        return True
    if lowered.startswith("branch "):
        _print("branching is implicit: undo then edit to start a new branch")
        return True
    edit = _match_edit(line)
    if edit is None:
        _print(f"(no sample edit matches {line!r})")
        return True
    composition = editor.apply(edit.patch, edit.note)
    _print(f"applied {edit.note} -> version {editor.current_version}")
    _print(_summarize(composition))
    return True


def run_repl(args: argparse.Namespace) -> int:
    """Run the interactive session REPL and return its exit code."""
    editor = SessionEditor(_initial_composition())
    _print("AIVIA session REPL")
    _print("Send a message to refine the composition; try undo/redo/show/quit.")
    while True:
        line = _read_line()
        if line is None:
            return 0
        line = line.strip()
        if not line:
            continue
        if not _handle(line, editor):
            return 0


def run_session(args: argparse.Namespace) -> int:
    """Dispatch to the selected session subcommand."""
    handler = getattr(args, "_session_handler", None)
    if handler is None:
        _print("error: missing session command")
        _print("usage: aivia session repl")
        return 2
    return int(handler(args))


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the session group and its ``repl`` subcommand."""
    subparsers = parser.add_subparsers(dest="session_command", metavar="COMMAND")
    repl = subparsers.add_parser(
        "repl",
        help="Start the interactive session REPL",
        description="Start the interactive session REPL",
    )
    repl.set_defaults(_session_handler=run_repl)


register_command(
    "session",
    run_session,
    help="Manage an interactive editing session",
    configure=configure,
)
