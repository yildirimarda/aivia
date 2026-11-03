"""``aivia research preview``: scores, reasons, brief matches and cost."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from aivia.cli import main
from aivia.cli.research import preview_topics, topic_store_path
from aivia.core.config.profile import ContentProfileConfig
from aivia.core.config.research import ResearchConfig
from aivia.research.brief import BriefDerivation
from aivia.research.scoring import ScoringRubric

_FIRST = "https://example.com/story"
_SECOND = "https://other.test/background"


def _write_config(path: Path) -> Path:
    path.write_text(
        """
schema_version: 1
profiles:
  briefed:
    brief: Explain bridge engineering and history.
    niche: civil engineering
    keywords: [bridge, engineering]
    sources: [wikipedia]
""".lstrip(),
        encoding="utf-8",
    )
    return path


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


def test_preview_prints_scores_reasons_matched_brief_parts_and_cost(
    tmp_path: Path,
    capsys: Any,
) -> None:
    config = _write_config(tmp_path / "config.yaml")

    code, data, err = _run(
        capsys,
        "research",
        "preview",
        "Bridge history",
        "--source",
        _FIRST,
        "--source",
        _SECOND,
        "--profile",
        "briefed",
        "--config",
        str(config),
        "--json",
    )

    assert code == 0, err
    assert err == ""
    assert data["profile"] == "briefed"
    assert data["brief"] == "Explain bridge engineering and history."
    assert data["counts"]["topics"] == 1

    topic = data["topics"][0]
    assert topic["title"] == "Bridge history"
    assert 0.0 <= topic["score"] <= 1.0
    assert set(topic["dimensions"]) == {
        "relevance",
        "originality",
        "accuracy",
        "engagement",
    }
    assert topic["scores"] == topic["dimensions"]
    assert topic["matched_brief_parts"]["keywords"] == ["bridge"]
    assert topic["matched_brief_parts"]["include_terms"] == ["bridge"]
    assert "bridge" in topic["matched"]
    assert "bridge" in topic["reason"]
    assert topic["cost_usd"] > 0.0
    assert data["cost_usd"] == topic["cost_usd"]
    assert data["cost"]["paid_usd"] == 0.0
    assert data["cost"]["estimated"] is True
    assert data["rubric"]["version"] == "1"
    assert data["rubric"]["min_score"] == 0.6

    assert not topic_store_path().exists()


def test_preview_text_output_names_every_requested_section(
    tmp_path: Path,
    capsys: Any,
) -> None:
    config = _write_config(tmp_path / "config.yaml")

    code = main(
        [
            "research",
            "preview",
            "Bridge engineering history",
            "--source",
            _FIRST,
            "--source",
            _SECOND,
            "--profile",
            "briefed",
            "--config",
            str(config),
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    assert captured.err == ""
    assert "Topic: Bridge engineering history" in captured.out
    assert "Score:" in captured.out
    assert "Reason:" in captured.out
    assert "Matched brief parts:" in captured.out
    assert "Cost: $" in captured.out
    assert "Total cost: $" in captured.out


def test_preview_reads_topics_from_the_store_when_none_are_given(
    tmp_path: Path,
    capsys: Any,
) -> None:
    store = tmp_path / "topics.json"
    add_code = main(
        [
            "research",
            "add",
            "Bridge history",
            "--source",
            _FIRST,
            "--source",
            _SECOND,
            "--store",
            str(store),
        ]
    )
    assert add_code == 0
    capsys.readouterr()

    code, data, err = _run(
        capsys,
        "research",
        "preview",
        "--store",
        str(store),
        "--json",
    )

    assert code == 0, err
    assert data["store"] == str(store)
    assert [topic["title"] for topic in data["topics"]] == ["Bridge history"]
    assert data["topics"][0]["cost_usd"] > 0.0


def test_preview_rejects_an_empty_store(
    tmp_path: Path,
    capsys: Any,
) -> None:
    code, data, err = _run(
        capsys,
        "research",
        "preview",
        "--store",
        str(tmp_path / "missing.json"),
        "--json",
    )

    assert code == 1
    assert data == {}
    assert "no topics to preview" in err


def test_preview_honours_an_explicit_minimum_score(
    tmp_path: Path,
    capsys: Any,
) -> None:
    config = _write_config(tmp_path / "config.yaml")

    code, data, err = _run(
        capsys,
        "research",
        "preview",
        "Bridge history",
        "--source",
        _FIRST,
        "--source",
        _SECOND,
        "--profile",
        "briefed",
        "--config",
        str(config),
        "--min-score",
        "1.0",
        "--json",
    )

    assert code == 0, err
    assert data["rubric"]["min_score"] == 1.0
    assert data["topics"][0]["passed"] is False
    assert data["counts"]["failed"] == 1


def test_preview_library_reports_excluded_brief_terms() -> None:
    derivation = BriefDerivation(
        include_terms=["bridge"],
        exclude_terms=["celebrity"],
        brief="Bridge engineering.",
        brief_hash="hash",
    )
    profile = ContentProfileConfig(
        brief="Bridge engineering.",
        niche="civil engineering",
        keywords=["bridge"],
    )

    result = preview_topics(
        [
            {
                "title": "Celebrity bridge scandal",
                "source_urls": [_FIRST, _SECOND],
            }
        ],
        profile=profile,
        derivation=derivation,
        cost_usd=0.0,
    )

    topic = result.topics[0]
    assert topic.score.dimensions["relevance"] == 0.0
    assert topic.excluded_terms == ("celebrity",)
    assert "excluded brief terms" in topic.score.reason
    assert result.to_dict()["topics"][0]["excluded_terms"] == ["celebrity"]


def test_preview_reads_the_research_config_dedupe_settings() -> None:
    config = ResearchConfig.from_mapping(
        {"dedupe_window": 7, "embedding_threshold": 0.9}
    )

    result = preview_topics(
        [
            {"title": "Bridge history"},
            {"title": "Bridge past"},
        ],
        config=config,
        embeddings={
            "Bridge history": (1.0, 0.0),
            "Bridge past": (0.99, 0.1),
        },
        cost_usd=0.0,
    )

    assert result.dedupe is not None
    assert result.dedupe.window == timedelta(days=7)
    assert result.dedupe.embedding_threshold == 0.9
    assert result.dedupe.embedding_duplicates == 1
    assert [topic.title for topic in result.topics] == ["Bridge history"]
    payload = result.to_dict()
    assert payload["counts"]["duplicates"] == 1
    assert payload["dedupe"]["metadata"]["window_seconds"] == 7 * 86400
    assert payload["dedupe"]["metadata"]["embedding_threshold"] == 0.9


def test_preview_reads_the_research_config_dedupe_settings_from_a_file(
    tmp_path: Path,
    capsys: Any,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
schema_version: 1
research:
  dedupe_window: 7
  embedding_threshold: 0.9
profiles:
  briefed:
    brief: Explain bridge engineering and history.
    niche: civil engineering
    keywords: [bridge, engineering]
    sources: [wikipedia]
""".lstrip(),
        encoding="utf-8",
    )

    code, data, err = _run(
        capsys,
        "research",
        "preview",
        "Bridge history",
        "bridge HISTORY",
        "--source",
        _FIRST,
        "--source",
        _SECOND,
        "--profile",
        "briefed",
        "--config",
        str(config),
        "--json",
    )

    assert code == 0, err
    assert data["counts"]["topics"] == 1
    assert data["counts"]["duplicates"] == 1
    assert data["dedupe"]["metadata"]["window_seconds"] == 7 * 86400
    assert data["dedupe"]["metadata"]["embedding_threshold"] == 0.9


def test_preview_reads_the_research_config_selection_limits() -> None:
    config = ResearchConfig.from_mapping({"max_topics": 1, "max_per_category": 2})

    result = preview_topics(
        [
            {"title": "Bridge history", "category": "history"},
            {"title": "Bridge engineering", "category": "engineering"},
            {"title": "Bridge design", "category": "design"},
        ],
        config=config,
        rubric=ScoringRubric(min_score=0.0),
        cost_usd=0.0,
    )

    assert result.selection is not None
    assert result.selection.limit == 1
    assert result.selection.max_per_category == 2
    assert result.selection.selected_count == 1
    assert sum(1 for topic in result.topics if topic.selected) == 1
    payload = result.to_dict()
    assert payload["selection"]["limit"] == 1
    assert payload["selection"]["max_per_category"] == 2
    assert payload["counts"]["selected"] == 1


def test_preview_reads_the_per_category_limit_from_research_config() -> None:
    config = ResearchConfig.from_mapping({"max_per_category": 1})

    result = preview_topics(
        [
            {"title": "Bridge history", "category": "history"},
            {"title": "Bridge engineering", "category": "history"},
            {"title": "Bridge design", "category": "history"},
        ],
        config=config,
        rubric=ScoringRubric(min_score=0.0),
        cost_usd=0.0,
    )

    assert result.selection is not None
    assert result.selection.max_per_category == 1
    assert result.selection.selected_count == 1
    assert [topic.title for topic in result.topics if topic.selected] == [
        result.selection.selected_titles[0]
    ]


def test_explicit_selection_options_override_the_preview_config() -> None:
    config = ResearchConfig.from_mapping({"max_topics": 1, "max_per_category": 1})

    result = preview_topics(
        [
            {"title": "Bridge history", "category": "history"},
            {"title": "Bridge engineering", "category": "engineering"},
            {"title": "Bridge design", "category": "design"},
        ],
        config=config,
        limit=None,
        max_per_category=None,
        rubric=ScoringRubric(min_score=0.0),
        cost_usd=0.0,
    )

    assert result.selection is not None
    assert result.selection.limit is None
    assert result.selection.max_per_category is None
    assert result.selection.selected_count == 3
    assert all(topic.selected for topic in result.topics)


def test_preview_dedupes_against_topics_added_in_an_earlier_run(
    tmp_path: Path,
    capsys: Any,
) -> None:
    store = tmp_path / "topics.json"
    config = _write_config(tmp_path / "config.yaml")
    add_code = main(
        [
            "research",
            "add",
            "Bridge history",
            "--source",
            _FIRST,
            "--source",
            _SECOND,
            "--store",
            str(store),
        ]
    )
    assert add_code == 0
    capsys.readouterr()

    code, data, err = _run(
        capsys,
        "research",
        "preview",
        "bridge HISTORY",
        "--source",
        _FIRST,
        "--source",
        _SECOND,
        "--profile",
        "briefed",
        "--config",
        str(config),
        "--store",
        str(store),
        "--json",
    )

    assert code == 0, err
    assert data["counts"]["topics"] == 0
    assert data["counts"]["duplicates"] == 1
    assert data["topics"] == []
    decision = data["dedupe"]["decisions"][0]
    assert decision["reason"] == "title"
    assert decision["matched_title"] == "Bridge history"


def test_preview_dedupes_against_stored_history_only_inside_the_window(
    tmp_path: Path,
    capsys: Any,
) -> None:
    now = datetime.now(UTC)
    store = tmp_path / "topics.json"
    store.write_text(
        json.dumps(
            {
                "version": 1,
                "topics": [
                    {
                        "title": "Bridge history",
                        "source_urls": [_FIRST],
                        "added_at": (now - timedelta(days=30)).isoformat(),
                    },
                    {
                        "title": "Bridge engineering",
                        "source_urls": [_FIRST],
                        "added_at": (now - timedelta(days=1)).isoformat(),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        """
schema_version: 1
research:
  dedupe_window: 7
profiles:
  briefed:
    brief: Explain bridge engineering and history.
    niche: civil engineering
    keywords: [bridge, engineering]
    sources: [wikipedia]
""".lstrip(),
        encoding="utf-8",
    )

    code, data, err = _run(
        capsys,
        "research",
        "preview",
        "Bridge history",
        "Bridge engineering",
        "--source",
        _FIRST,
        "--source",
        _SECOND,
        "--profile",
        "briefed",
        "--config",
        str(config),
        "--store",
        str(store),
        "--json",
    )

    assert code == 0, err
    assert [topic["title"] for topic in data["topics"]] == ["Bridge history"]
    assert data["counts"]["topics"] == 1
    assert data["counts"]["duplicates"] == 1
    assert data["dedupe"]["decisions"][1]["matched_title"] == "Bridge engineering"
