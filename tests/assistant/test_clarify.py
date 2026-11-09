"""Clarifier: one short question interactively, configured defaults in batch."""

from __future__ import annotations

import io
import sys

import pytest

from aivia.assistant import Clarification, Clarifier
from aivia.core.config.planner import PlannerDefaultsConfig


class _Stream(io.StringIO):
    """A stdin/stdout replacement whose terminal status the test controls."""

    def __init__(self, text: str = "", *, tty: bool) -> None:
        super().__init__(text)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_interactive_asks_one_question_and_incorporates_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream("the hero image\n", tty=True))
    stdout = _Stream(tty=True)
    monkeypatch.setattr(sys, "stdout", stdout)

    result = Clarifier().clarify("caption it", question="Which asset?")

    assert isinstance(result, Clarification)
    assert result.asked is True
    assert result.notes == []
    assert result.instruction == "caption it the hero image"
    assert "Which asset? " in stdout.getvalue()


def test_interactive_empty_answer_leaves_instruction_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream("\n", tty=True))
    stdout = _Stream(tty=True)
    monkeypatch.setattr(sys, "stdout", stdout)

    result = Clarifier().clarify("caption it", question="Which asset?")

    assert result.asked is True
    assert result.notes == []
    assert result.instruction == "caption it"


def test_interactive_without_a_question_asks_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream("ignored\n", tty=True))
    stdout = _Stream(tty=True)
    monkeypatch.setattr(sys, "stdout", stdout)

    result = Clarifier().clarify("caption it")

    assert result.asked is False
    assert result.notes == []
    assert result.instruction == "caption it"
    assert stdout.getvalue() == ""


def test_batch_mode_applies_configured_defaults_and_records_a_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Non-interactive stream: nothing may be read or written.
    monkeypatch.setattr(sys, "stdin", _Stream("would-be-read\n", tty=False))
    stdout = _Stream(tty=False)
    monkeypatch.setattr(sys, "stdout", stdout)

    defaults = PlannerDefaultsConfig(format="video", language="en", voice="")
    result = Clarifier(defaults=defaults).clarify("caption it")

    assert result.asked is False
    assert result.instruction == "caption it format=video language=en"
    assert len(result.notes) == 1
    note = result.notes[0]
    assert "batch mode" in note
    assert "clarifying question skipped" in note
    assert "{'format': 'video', 'language': 'en'}" in note
    assert stdout.getvalue() == ""


def test_batch_mode_without_defaults_records_an_empty_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream("would-be-read\n", tty=False))
    stdout = _Stream(tty=False)
    monkeypatch.setattr(sys, "stdout", stdout)

    result = Clarifier().clarify("caption it")

    assert result.asked is False
    assert result.instruction == "caption it"
    assert result.notes == [
        "clarify: batch mode; clarifying question skipped; defaults applied: {}"
    ]
    assert stdout.getvalue() == ""


def test_explicit_interactive_flag_overrides_a_non_tty_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream("overridden\n", tty=False))
    stdout = _Stream(tty=False)
    monkeypatch.setattr(sys, "stdout", stdout)

    result = Clarifier().clarify(
        "caption it", question="Which asset?", interactive=True
    )

    assert result.asked is True
    assert result.instruction == "caption it overridden"
    assert "Which asset? " in stdout.getvalue()


def test_explicit_batch_flag_overrides_a_tty_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream("would-be-read\n", tty=True))
    stdout = _Stream(tty=True)
    monkeypatch.setattr(sys, "stdout", stdout)

    defaults = PlannerDefaultsConfig(output="mp4")
    result = Clarifier(defaults=defaults).clarify("caption it", interactive=False)

    assert result.asked is False
    assert result.instruction == "caption it output=mp4"
    assert len(result.notes) == 1
    assert "batch mode" in result.notes[0]
    assert stdout.getvalue() == ""
