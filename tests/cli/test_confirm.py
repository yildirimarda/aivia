"""``aivia run``: confirmation above the configured spend threshold."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from aivia.cli import main
from aivia.cli.confirm import confirm_spend, confirmation_required
from aivia.core.config import BudgetConfig
from aivia.core.errors import ConfigError


class _Stream(io.StringIO):
    """A stdin replacement whose terminal status the test controls."""

    def __init__(self, text: str = "", *, tty: bool) -> None:
        super().__init__(text)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def write_config(tmp_path: Path, confirm_above: float) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
providers:
  text:
    type: mock-llm
  speech:
    type: mock-tts
  pictures:
    type: mock-image
formats:
  confirm_sample:
    description: A priced sample recipe.
    stages:
      - stage: input_topic
      - stage: write
      - stage: voice
      - stage: visuals
        options:
          strategy: image
          count: 2
      - stage: render
    capabilities: [llm, tts, image_gen]
    output_type: video
    handler: save
budget:
  confirm_above: {confirm_above}
""".lstrip(),
        encoding="utf-8",
    )
    return path


def run_sample(
    config: Path,
    *extra: str,
) -> int:
    return main(
        [
            "run",
            "--format",
            "confirm_sample",
            "--config",
            str(config),
            *extra,
        ]
    )


def test_non_interactive_run_above_the_threshold_refuses_without_yes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream("yes\n", tty=False))

    code = run_sample(write_config(tmp_path, 0.01))
    captured = capsys.readouterr()

    assert code == 1
    assert "non-interactive runs require --yes" in captured.err
    assert "execution is not available" not in captured.err
    assert "Continue?" not in captured.out


def test_yes_flag_allows_an_above_threshold_run_without_prompting(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream(tty=False))

    code = run_sample(write_config(tmp_path, 0.01), "--yes")
    captured = capsys.readouterr()

    assert code == 0
    assert "State: queued" in captured.out
    assert "execution is not available" not in captured.err
    assert "Continue?" not in captured.out


def test_interactive_run_above_the_threshold_accepts_yes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream("y\n", tty=True))

    code = run_sample(write_config(tmp_path, 0.01))
    captured = capsys.readouterr()

    assert code == 0
    assert "Continue? [y/N]" in captured.out
    assert "Run: run_" in captured.out


def test_interactive_run_above_the_threshold_refuses_no(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream("no\n", tty=True))

    code = run_sample(write_config(tmp_path, 0.01))
    captured = capsys.readouterr()

    assert code == 1
    assert "Continue? [y/N]" in captured.out
    assert "confirmation declined" in captured.err
    assert "execution is not available" not in captured.err


def test_run_below_the_threshold_does_not_prompt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream(tty=False))

    code = run_sample(write_config(tmp_path, 1.0))
    captured = capsys.readouterr()

    assert code == 0
    assert "Continue?" not in captured.out
    assert "Run: run_" in captured.out


def test_zero_threshold_disables_confirmation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream(tty=False))

    code = run_sample(write_config(tmp_path, 0.0))
    captured = capsys.readouterr()

    assert code == 0
    assert "non-interactive runs require --yes" not in captured.err
    assert "Run: run_" in captured.out


def test_estimate_above_the_threshold_never_prompts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _Stream(tty=False))

    code = run_sample(write_config(tmp_path, 0.01), "--estimate")
    captured = capsys.readouterr()

    assert code == 0
    assert "TOTAL" in captured.out
    assert "Continue?" not in captured.out
    assert captured.err == ""


@pytest.mark.parametrize(
    ("amount", "threshold", "required"),
    [
        (0.02, 0.01, True),
        (0.01, 0.01, False),
        (0.0, 0.01, False),
        (1.0, 0.0, False),
        (0.0, 0.0, False),
    ],
)
def test_confirmation_is_required_only_strictly_above_a_positive_threshold(
    amount: float,
    threshold: float,
    required: bool,
) -> None:
    assert confirmation_required(amount, threshold) is required


def test_confirm_spend_with_yes_does_not_touch_the_streams() -> None:
    stdin = _Stream("no\n", tty=True)
    stdout = io.StringIO()

    assert confirm_spend(1.0, 0.5, assume_yes=True, stdin=stdin, stdout=stdout)
    assert stdin.tell() == 0
    assert stdout.getvalue() == ""


def test_confirm_threshold_aliases_and_property() -> None:
    budget = BudgetConfig.from_mapping({"confirm_over": 0.5})

    assert budget.confirm_above == 0.5
    assert budget.confirmation_threshold == 0.5


def test_confirm_threshold_must_not_be_negative() -> None:
    with pytest.raises(ConfigError) as caught:
        BudgetConfig.from_mapping({"confirm_above": -0.01})

    assert "confirm_above" in str(caught.value)
