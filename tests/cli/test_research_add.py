"""``aivia research add``: manual topics with dedupe and verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml

from aivia.cli import main
from aivia.cli.research import topic_store_path

_FIRST = "https://example.com/story"
_SECOND = "https://other.test/background"


def _run(
    capsys: Any,
    *arguments: str,
) -> tuple[int, dict[str, Any], str]:
    code = main(list(arguments))
    captured = capsys.readouterr()
    payload = (
        cast(dict[str, Any], json.loads(captured.out)) if captured.out.strip() else {}
    )
    return code, payload, captured.err


def _add(
    capsys: Any,
    *arguments: str,
) -> tuple[int, dict[str, Any], str]:
    return _run(capsys, "research", "add", *arguments, "--json")


def _stored(path: str | Path) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(Path(path).read_text(encoding="utf-8")),
    )


def test_research_add_verifies_and_persists_a_topic(
    capsys: Any,
) -> None:
    code, data, err = _add(
        capsys,
        "Bridge history",
        "--source",
        _FIRST,
        "--source",
        _SECOND,
    )

    assert code == 0, err
    assert data["counts"] == {
        "added": 1,
        "rejected": 0,
        "verified": 1,
        "unverified": 0,
    }
    added = data["added"][0]
    assert added["title"] == "Bridge history"
    assert added["verified"] is True
    assert added["status"] == "verified"
    assert added["domains"] == ["example.com", "other.test"]
    assert added["source_urls"] == [_FIRST, _SECOND]
    assert data["store"] == str(topic_store_path())

    stored = _stored(data["store"])
    assert stored["version"] == 1
    assert [topic["title"] for topic in stored["topics"]] == ["Bridge history"]
    assert stored["topics"][0]["verified"] is True


def test_research_add_marks_a_single_source_topic_unverified(
    capsys: Any,
) -> None:
    code, data, err = _add(capsys, "Single source story", "--source", _FIRST)

    assert code == 0, err
    added = data["added"][0]
    assert added["verified"] is False
    assert added["status"] == "unverified"
    assert added["domains"] == ["example.com"]
    assert "single source domain" in added["verification_reason"]
    assert data["counts"]["unverified"] == 1

    stored = _stored(data["store"])
    assert stored["topics"][0]["verified"] is False


def test_research_add_rejects_a_duplicate_from_an_earlier_run(
    capsys: Any,
) -> None:
    first_code, _, first_err = _add(
        capsys,
        "Moon landing",
        "--source",
        _FIRST,
        "--source",
        _SECOND,
    )
    assert first_code == 0, first_err

    code, data, err = _add(
        capsys,
        "  moon   LANDING ",
        "--source",
        _FIRST,
        "--source",
        _SECOND,
    )

    assert code == 1
    assert err == ""
    assert data["added"] == []
    assert data["counts"] == {
        "added": 0,
        "rejected": 1,
        "verified": 0,
        "unverified": 0,
    }
    duplicate = data["rejected"][0]
    assert duplicate["title"] == "moon   LANDING"
    assert duplicate["reason"] == "title"
    assert duplicate["matched_title"] == "Moon landing"
    assert "title matches" in duplicate["explanation"]

    stored = _stored(data["store"])
    assert [topic["title"] for topic in stored["topics"]] == ["Moon landing"]


def test_research_add_rejects_duplicates_inside_one_batch(
    capsys: Any,
    tmp_path: Path,
) -> None:
    topics = tmp_path / "topics.yaml"
    topics.write_text(
        yaml.safe_dump(
            {
                "topics": [
                    {
                        "title": "Bridge history",
                        "source_urls": [_FIRST, _SECOND],
                    },
                    {
                        "title": "bridge HISTORY",
                        "source_urls": [_FIRST, _SECOND],
                    },
                    {
                        "title": "A different topic",
                        "source_urls": [_FIRST, _SECOND],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    code, data, err = _add(capsys, "--file", str(topics))

    assert code == 0, err
    assert [topic["title"] for topic in data["added"]] == [
        "Bridge history",
        "A different topic",
    ]
    assert data["counts"]["added"] == 2
    assert data["counts"]["rejected"] == 1
    assert data["rejected"][0]["reason"] == "title"
    assert data["rejected"][0]["matched_title"] == "Bridge history"


def test_research_add_accepts_titles_and_bare_source_urls(
    capsys: Any,
) -> None:
    code, data, err = _run(
        capsys,
        "research",
        "add",
        "--title",
        "Bridge history",
        _FIRST,
        _SECOND,
        "--json",
    )

    assert code == 0, err
    added = data["added"][0]
    assert added["title"] == "Bridge history"
    assert added["source_urls"] == [_FIRST, _SECOND]
    assert added["verified"] is True


def test_research_add_never_runs_collection(
    capsys: Any,
    monkeypatch: Any,
) -> None:
    import importlib

    collect_module = importlib.import_module("aivia.research.collect")

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("research add must skip collection")

    monkeypatch.setattr(collect_module, "collect", unexpected)
    monkeypatch.setattr(collect_module, "collect_sources", unexpected)

    code, data, err = _add(
        capsys,
        "Bridge history",
        "--source",
        _FIRST,
        "--source",
        _SECOND,
    )

    assert code == 0, err
    assert data["counts"]["added"] == 1


def test_research_add_dry_run_does_not_write_the_store(
    capsys: Any,
) -> None:
    code, data, err = _run(
        capsys,
        "--dry-run",
        "research",
        "add",
        "Bridge history",
        "--source",
        _FIRST,
        "--source",
        _SECOND,
        "--json",
    )

    assert code == 0, err
    assert data["recorded"] is False
    assert data["counts"]["added"] == 1
    assert not Path(data["store"]).exists()


def test_research_add_reports_input_errors(
    capsys: Any,
) -> None:
    code = main(["research", "add"])
    captured = capsys.readouterr()
    assert code == 1
    assert "no topics to add" in captured.err

    code = main(
        [
            "research",
            "add",
            "Bridge history",
            "--source",
            "not-a-url",
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "source URL" in captured.err
