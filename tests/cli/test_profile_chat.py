"""``aivia profile create``: a short chat, sample topics and approval."""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from aivia.cli import main
from aivia.cli.profile import (
    SAMPLE_TOPIC_COUNT,
    ProfileAnswers,
    build_profile,
    generate_sample_topics,
    save_profile,
)
from aivia.core.config import ContentProfileConfig
from aivia.core.errors import ConfigError


def _load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(path.read_text(encoding="utf-8")))


def _topics(output: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(r"^\s*\d+\.\s+(.+)$", output, re.MULTILINE)
    ]


def _run(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    *arguments: str,
    stdin: str = "",
) -> tuple[int, str, str]:
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    code = main(list(arguments))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_create_asks_three_questions_shows_ten_topics_and_saves_after_approval(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "schema_version: 1\npipeline:\n  timezone: UTC\n",
        encoding="utf-8",
    )

    code, output, err = _run(
        capsys,
        monkeypatch,
        "profile",
        "create",
        "space",
        "--config",
        str(config),
        stdin=(
            "Space exploration\n"
            "Explain how space missions work.\n"
            "rockets, nasa | excited and clear\n"
            "yes\n"
        ),
    )

    assert code == 0, err
    assert output.count("What niche or topic area") == 1
    assert output.count("Describe the profile's brief") == 1
    assert output.count("Keywords and tone") == 1
    assert "Save profile 'space'? [y/N]" in output

    topics = _topics(output)
    assert len(topics) == SAMPLE_TOPIC_COUNT == 10
    assert len(set(topics)) == 10

    saved = _load(config)
    assert saved["schema_version"] == 1
    assert saved["pipeline"] == {"timezone": "UTC"}
    profile = saved["profiles"]["space"]
    assert profile["niche"] == "Space exploration"
    assert profile["brief"] == "Explain how space missions work."
    assert profile["keywords"] == ["rockets", "nasa"]
    assert profile["tone"] == "excited and clear"
    assert "format" not in profile
    assert "Saved profile: space" in output
    assert str(config) in output


def test_declining_approval_leaves_the_config_untouched(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"

    code, output, err = _run(
        capsys,
        monkeypatch,
        "profile",
        "create",
        "space",
        "--config",
        str(config),
        stdin="Space\nExplain space.\nrockets\nno\n",
    )

    assert code == 1
    assert len(_topics(output)) == SAMPLE_TOPIC_COUNT
    assert "approval declined" in err
    assert not config.exists()


def test_flags_skip_the_questions_and_yes_skips_the_approval(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"

    code, output, err = _run(
        capsys,
        monkeypatch,
        "profile",
        "create",
        "history",
        "--config",
        str(config),
        "--niche",
        "History",
        "--brief",
        "Explain turning points clearly.",
        "--keywords",
        "rome, republic",
        "--tone",
        "calm",
        "--format",
        "narrated_still",
        "--yes",
    )

    assert code == 0, err
    assert "What niche" not in output
    assert "Describe the profile's brief" not in output
    assert "Keywords and tone" not in output
    assert "Save profile" not in output
    assert len(_topics(output)) == SAMPLE_TOPIC_COUNT

    profile = _load(config)["profiles"]["history"]
    assert profile["keywords"] == ["rome", "republic"]
    assert profile["tone"] == "calm"
    assert profile["format"] == "narrated_still"


def test_an_unknown_format_is_rejected_before_saving(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"

    code, output, err = _run(
        capsys,
        monkeypatch,
        "profile",
        "create",
        "space",
        "--config",
        str(config),
        "--niche",
        "Space",
        "--brief",
        "Explain space.",
        "--format",
        "quote_card_typo",
        "--yes",
    )

    assert code == 1
    assert "unknown format" in err
    assert "quote_card_typo" in err
    assert "narrated_still" in err
    assert _topics(output) == []
    assert not config.exists()


def test_a_format_from_the_resolved_config_is_accepted(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "schema_version: 1\nformats:\n  custom_card: {}\n",
        encoding="utf-8",
    )

    code, _, err = _run(
        capsys,
        monkeypatch,
        "profile",
        "create",
        "cards",
        "--config",
        str(config),
        "--niche",
        "Cards",
        "--brief",
        "Explain cards.",
        "--format",
        "custom_card",
        "--yes",
    )

    assert code == 0, err
    assert _load(config)["profiles"]["cards"]["format"] == "custom_card"


def test_a_missing_name_is_derived_from_the_niche(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"

    code, _, err = _run(
        capsys,
        monkeypatch,
        "profile",
        "create",
        "--config",
        str(config),
        "--niche",
        "Deep sea biology",
        "--brief",
        "Explain ocean life.",
        "--keywords",
        "fish",
        "--yes",
    )

    assert code == 0, err
    assert "deep-sea-biology" in _load(config)["profiles"]


def test_an_existing_profile_is_only_replaced_with_force(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "schema_version: 1\nprofiles:\n  space:\n    niche: old\n",
        encoding="utf-8",
    )

    code, _, err = _run(
        capsys,
        monkeypatch,
        "profile",
        "create",
        "space",
        "--config",
        str(config),
        "--niche",
        "new",
        "--yes",
    )

    assert code == 1
    assert "already exists" in err
    assert _load(config)["profiles"]["space"]["niche"] == "old"

    code, _, err = _run(
        capsys,
        monkeypatch,
        "profile",
        "create",
        "space",
        "--config",
        str(config),
        "--niche",
        "new",
        "--force",
        "--yes",
    )

    assert code == 0, err
    assert _load(config)["profiles"]["space"]["niche"] == "new"


def test_dry_run_shows_samples_but_writes_nothing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"

    code, output, err = _run(
        capsys,
        monkeypatch,
        "profile",
        "create",
        "space",
        "--config",
        str(config),
        "--niche",
        "Space",
        "--brief",
        "Explain space.",
        "--keywords",
        "rockets",
        "--yes",
        "--dry-run",
    )

    assert code == 0, err
    assert len(_topics(output)) == SAMPLE_TOPIC_COUNT
    assert "Dry run: profile not saved." in output
    assert not config.exists()


def test_sample_topics_are_deterministic_and_distinct_offline() -> None:
    profile = ContentProfileConfig.from_mapping(
        {
            "niche": "Bridge engineering",
            "keywords": ["bridges", "history"],
        }
    )

    topics = generate_sample_topics(profile)

    assert len(topics) == SAMPLE_TOPIC_COUNT
    assert len(set(topics)) == SAMPLE_TOPIC_COUNT
    assert topics == generate_sample_topics(profile)
    assert all(
        any(term in topic for term in ("Bridge engineering", "bridges", "history"))
        for topic in topics
    )


def test_build_profile_rejects_an_empty_chat() -> None:
    with pytest.raises(ConfigError, match="niche or a brief"):
        build_profile(ProfileAnswers())


def test_save_profile_merges_into_an_existing_config(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        (
            "schema_version: 1\npipeline:\n  timezone: UTC\n"
            "profiles:\n  old:\n    niche: old\n"
        ),
        encoding="utf-8",
    )
    profile = ContentProfileConfig.from_mapping({"niche": "new"})

    save_profile(config, "new", profile)

    saved = _load(config)
    assert saved["pipeline"] == {"timezone": "UTC"}
    assert set(saved["profiles"]) == {"old", "new"}
