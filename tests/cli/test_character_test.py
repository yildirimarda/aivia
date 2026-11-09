"""``aivia character test``: render scenes across strategies, print scores and cost."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from aivia.cli import main
from aivia.core.config.character import ConsistencyStrategy


def write_png(path: Path, size: tuple[int, int] = (16, 16), color: str = "red") -> Path:
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text.lstrip(), encoding="utf-8")
    return path


def data_dir(tmp_path: Path, config: Path) -> Path:
    """The data directory resolved from the config file passed on the command line."""
    return config.parent / "data"


def _keyword_strategy(
    keyword: str, asset: bytes | None
) -> Callable[[str], bytes | None]:
    """Return a strategy that produces *asset* when *keyword* is in the scene."""

    def strategy(scene: str) -> bytes | None:
        return asset if keyword in scene else None

    return strategy


def _inject(
    monkeypatch: pytest.MonkeyPatch,
    strategies: Mapping[ConsistencyStrategy, Any],
    scorer: Any,
) -> None:
    monkeypatch.setattr(
        "aivia.cli.character._build_strategies",
        lambda config, character: strategies,
    )
    monkeypatch.setattr(
        "aivia.cli.character._build_scorer",
        lambda config, character, data_dir: scorer,
    )


def test_character_test_prints_scores_and_cost_across_strategies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path, "schema_version: 1\n")
    characters = data_dir(tmp_path, config) / "characters"
    characters.mkdir(parents=True)
    write_png(characters / "ref.png", color="blue")
    (characters / "mascot.yaml").write_text(
        "name: mascot\nversion: 1\nreference_images:\n  - ref.png\n",
        encoding="utf-8",
    )

    strategies = {
        ConsistencyStrategy.COMPOSITE: _keyword_strategy("composite", b"png-composite"),
        ConsistencyStrategy.REFERENCE: _keyword_strategy("reference", b"png-reference"),
        ConsistencyStrategy.ADAPTER: _keyword_strategy("adapter", b"png-adapter"),
        ConsistencyStrategy.PROMPT_ONLY: _keyword_strategy(
            "prompt_only", b"png-prompt"
        ),
    }
    scores = iter([0.91, 0.82, 0.73])

    def scorer(asset: object) -> float | None:
        try:
            return next(scores)
        except StopIteration:
            return 0.5

    _inject(monkeypatch, strategies, scorer)

    db = tmp_path / "cost.db"
    code = main(
        [
            "character",
            "test",
            "mascot",
            "--config",
            str(config),
            "--database",
            str(db),
            "--scene",
            "a composite scene",
            "--scene",
            "a reference scene",
            "--scene",
            "an adapter scene",
        ]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "composite" in output
    assert "reference" in output
    assert "adapter" in output
    assert "0.910" in output
    assert "0.820" in output
    assert "0.730" in output
    assert "Total cost: $0.060000" in output


def test_character_test_limits_to_a_single_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path, "schema_version: 1\n")
    characters = data_dir(tmp_path, config) / "characters"
    characters.mkdir(parents=True)
    write_png(characters / "ref.png", color="blue")
    (characters / "mascot.yaml").write_text(
        "name: mascot\nversion: 1\nreference_images:\n  - ref.png\n",
        encoding="utf-8",
    )

    strategies = {
        ConsistencyStrategy.COMPOSITE: _keyword_strategy("composite", b"png-composite"),
        ConsistencyStrategy.REFERENCE: _keyword_strategy("reference", b"png-reference"),
    }

    def scorer(asset: object) -> float | None:
        return 0.88

    _inject(monkeypatch, strategies, scorer)

    code = main(
        [
            "character",
            "test",
            "mascot",
            "--config",
            str(config),
            "--strategy",
            "reference",
            "--scene",
            "a composite scene",
            "--scene",
            "a reference scene",
        ]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "reference" in output
    assert "0.880" in output


def test_character_test_all_strategies_decline_record_none_and_zero_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path, "schema_version: 1\n")
    characters = data_dir(tmp_path, config) / "characters"
    characters.mkdir(parents=True)
    write_png(characters / "ref.png", color="blue")
    (characters / "mascot.yaml").write_text(
        "name: mascot\nversion: 1\nreference_images:\n  - ref.png\n",
        encoding="utf-8",
    )

    strategies = {
        ConsistencyStrategy.COMPOSITE: _keyword_strategy("nomatch", None),
        ConsistencyStrategy.REFERENCE: _keyword_strategy("nomatch", None),
        ConsistencyStrategy.ADAPTER: _keyword_strategy("nomatch", None),
        ConsistencyStrategy.PROMPT_ONLY: _keyword_strategy("nomatch", None),
    }

    def scorer(asset: object) -> float | None:
        return 0.0

    _inject(monkeypatch, strategies, scorer)

    code = main(
        ["character", "test", "mascot", "--config", str(config), "--scene", "a scene"]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "-" in output
    assert "Total cost: $0.000000" in output


def test_character_test_estimate_prints_cost_without_a_database(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path, "schema_version: 1\n")

    code = main(["character", "test", "mascot", "--config", str(config), "--estimate"])

    assert code == 0
    output = capsys.readouterr().out
    assert "composite" in output
    assert "reference" in output
    assert "adapter" in output
    assert "prompt_only" in output
    assert "$0.020000" in output
    assert "Total estimated cost: $0.080000" in output


def test_character_test_without_a_character_still_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path, "schema_version: 1\n")

    strategies = {
        ConsistencyStrategy.COMPOSITE: _keyword_strategy("composite", b"png-composite"),
    }

    def scorer(asset: object) -> float | None:
        return None

    _inject(monkeypatch, strategies, scorer)

    code = main(["character", "test", "--config", str(config), "--scene", "a scene"])

    assert code == 0
    output = capsys.readouterr().out
    assert "-" in output
    assert "Total cost:" in output


def test_character_test_unknown_character_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path, "schema_version: 1\n")

    strategies = {
        ConsistencyStrategy.COMPOSITE: _keyword_strategy("composite", b"png-composite"),
    }

    def scorer(asset: object) -> float | None:
        return 0.5

    _inject(monkeypatch, strategies, scorer)

    code = main(["character", "test", "nope", "--config", str(config)])

    assert code == 1
    assert "nope" in capsys.readouterr().err
