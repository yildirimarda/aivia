"""``aivia topics feedback``: store likes, dislikes and comments as guidance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from aivia.cli import main
from aivia.core.config import ContentProfileConfig
from aivia.core.db import FeedbackRepository, create_database_engine
from aivia.research import (
    TopicFeedback,
    brief_prompt_context,
    build_scoring_prompt,
    load_feedback_guidance,
    summarize_feedback,
)


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


def _feedback(capsys: Any, *arguments: str) -> tuple[int, dict[str, Any], str]:
    return _run(capsys, "topics", "feedback", *arguments, "--json")


def _context_from_prompt(text: str) -> dict[str, Any]:
    context = text.split("\n\n", 1)[1]
    _, _, payload = context.partition("\n")
    return cast(dict[str, Any], json.loads(payload.strip()))


def test_feedback_stores_likes_and_dislikes_with_comments(capsys: Any) -> None:
    code, data, err = _feedback(
        capsys,
        "Bridge history",
        "--like",
        "--comment",
        "Great engineering angle",
    )
    assert code == 0, err
    assert data["recorded"][0]["sentiment"] == "like"
    assert data["guidance"]["likes"] == ["Bridge history"]
    assert data["guidance"]["dislikes"] == []
    assert data["guidance"]["comments"] == ["Bridge history: Great engineering angle"]
    assert data["counts"] == {"like": 1, "dislike": 0, "neutral": 0, "total": 1}

    code, data, err = _feedback(
        capsys,
        "Celebrity gossip",
        "--dislike",
        "--comment",
        "Off theme",
    )
    assert code == 0, err
    assert data["guidance"]["likes"] == ["Bridge history"]
    assert data["guidance"]["dislikes"] == ["Celebrity gossip"]
    assert data["guidance"]["comments"] == [
        "Bridge history: Great engineering angle",
        "Celebrity gossip: Off theme",
    ]
    assert "Prefer topics like: Bridge history." in data["guidance"]["summary"]
    assert "Avoid topics like: Celebrity gossip." in data["guidance"]["summary"]

    engine = create_database_engine(path=Path(data["database"]))
    try:
        rows = FeedbackRepository(engine).list(subject_kind="topic")
    finally:
        engine.dispose()
    assert [row["sentiment"] for row in rows] == ["like", "dislike"]
    assert [row["metadata"]["title"] for row in rows] == [
        "Bridge history",
        "Celebrity gossip",
    ]


def test_feedback_refreshes_guidance_used_by_later_prompts(capsys: Any) -> None:
    code, data, err = _feedback(
        capsys,
        "Bridge history",
        "--like",
        "--comment",
        "Great engineering angle",
    )
    assert code == 0, err
    code, data, err = _feedback(capsys, "Celebrity gossip", "--dislike")
    assert code == 0, err

    guidance = load_feedback_guidance(path=data["store"])
    assert guidance is not None
    assert guidance.likes == ["Bridge history"]
    assert guidance.dislikes == ["Celebrity gossip"]

    profile = ContentProfileConfig(
        brief="Explain bridge engineering clearly.",
        niche="civil engineering",
    )
    prompt = build_scoring_prompt(profile, feedback=guidance)
    context = _context_from_prompt(prompt)
    assert context["topic_feedback"]["likes"] == ["Bridge history"]
    assert context["topic_feedback"]["dislikes"] == ["Celebrity gossip"]
    assert context["topic_feedback"]["summary"] == guidance.summary
    assert "Prefer topics like: Bridge history." in prompt

    direct = brief_prompt_context(profile, feedback=guidance)
    assert direct["topic_feedback"] == guidance.to_prompt_dict()


def test_feedback_lists_existing_entries_without_recording_more(capsys: Any) -> None:
    code, data, err = _feedback(capsys, "Bridge history", "--like")
    assert code == 0, err

    code, data, err = _run(capsys, "topics", "feedback", "--list", "--json")

    assert code == 0, err
    assert data["recorded"] == []
    assert [entry["topic"] for entry in data["feedback"]] == ["Bridge history"]
    assert data["guidance"]["likes"] == ["Bridge history"]
    assert data["counts"]["total"] == 1


def test_feedback_records_multiple_topics_and_a_comment_only_note(
    capsys: Any,
) -> None:
    code, data, err = _feedback(
        capsys,
        "Bridge history",
        "Roman engineering",
        "--comment",
        "More of this",
    )

    assert code == 0, err
    assert [entry["sentiment"] for entry in data["recorded"]] == [
        "neutral",
        "neutral",
    ]
    assert data["guidance"]["likes"] == []
    assert data["guidance"]["dislikes"] == []
    assert data["guidance"]["comments"] == [
        "Bridge history: More of this",
        "Roman engineering: More of this",
    ]
    assert data["counts"]["neutral"] == 2


def test_feedback_uses_an_explicit_database_and_guidance_store(
    tmp_path: Path,
    capsys: Any,
) -> None:
    database = tmp_path / "feedback.db"
    store = tmp_path / "guidance.json"

    code, data, err = _feedback(
        capsys,
        "Bridge history",
        "--like",
        "--database",
        str(database),
        "--store",
        str(store),
    )

    assert code == 0, err
    assert data["database"] == str(database)
    assert data["store"] == str(store)
    assert database.exists()
    assert store.exists()
    payload = json.loads(store.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["guidance"]["likes"] == ["Bridge history"]


def test_feedback_requires_a_topic_unless_listing(capsys: Any) -> None:
    code, _, err = _feedback(capsys)

    assert code == 1
    assert "no topics to record" in err


def test_feedback_rejects_like_and_dislike_together(capsys: Any) -> None:
    code, _, err = _feedback(capsys, "Bridge history", "--like", "--dislike")

    assert code == 2
    assert "not allowed with argument" in err


def test_summary_is_deterministic_and_deduplicates_titles() -> None:
    entries = [
        TopicFeedback(topic="Bridge history", sentiment="like"),
        TopicFeedback(topic="  bridge   HISTORY ", sentiment="liked"),
        TopicFeedback(topic="Celebrity gossip", sentiment="dislike"),
        TopicFeedback(topic="Bridge history", sentiment="neutral", comment="Useful"),
    ]

    first = summarize_feedback(entries)
    second = summarize_feedback(entries)

    assert first == second
    assert first.likes == ["Bridge history"]
    assert first.dislikes == ["Celebrity gossip"]
    assert first.comments == ["Bridge history: Useful"]
    assert first.counts == {"like": 2, "dislike": 1, "neutral": 1, "total": 4}
