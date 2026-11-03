"""``aivia topics``: record and summarize feedback about research topics.

Likes, dislikes and comments are stored in the ``feedback`` table so a topic
can collect many notes over time. Every write also refreshes a small guidance
sidecar under ``data/research``. Later relevance, scoring, tone and style
prompts can load that sidecar and carry the summary as data, which lets a
reviewer steer topic selection without editing the profile brief.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sqlalchemy import Engine

from aivia.cli.registry import register_command
from aivia.core import paths
from aivia.core.config.database import DatabaseConfig
from aivia.core.config.resolve import resolve_config
from aivia.core.db import (
    FeedbackRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.errors import ConfigError
from aivia.research.feedback import (
    FeedbackGuidance,
    FeedbackGuidanceStore,
    TopicFeedback,
    normalize_sentiment,
    summarize_feedback,
)
from aivia.research.modes.otd import normalize_title

__all__ = [
    "configure",
    "run_feedback",
    "run_topics",
]


def _configure_feedback(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "topics",
        nargs="*",
        metavar="TOPIC",
        help="topic title to record feedback for",
    )
    parser.add_argument(
        "--topic",
        dest="topic_options",
        action="append",
        default=None,
        metavar="TOPIC",
        help="topic title; repeatable",
    )
    sentiment = parser.add_mutually_exclusive_group()
    sentiment.add_argument(
        "--like",
        dest="sentiment",
        action="store_const",
        const="like",
        help="record a like",
    )
    sentiment.add_argument(
        "--dislike",
        dest="sentiment",
        action="store_const",
        const="dislike",
        help="record a dislike",
    )
    sentiment.add_argument(
        "--neutral",
        dest="sentiment",
        action="store_const",
        const="neutral",
        help="record a neutral comment",
    )
    parser.add_argument(
        "--comment",
        default="",
        metavar="TEXT",
        help="comment about the topic(s)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        metavar="NAME",
        help="profile the feedback belongs to",
    )
    parser.add_argument(
        "--database",
        default=None,
        metavar="PATH",
        help="database file instead of the configured data directory database",
    )
    parser.add_argument(
        "--store",
        default=None,
        metavar="PATH",
        help="guidance file instead of data/research/topic_feedback.json",
    )
    parser.add_argument(
        "--list",
        dest="list_feedback",
        action="store_true",
        help="print every stored entry and the current guidance",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="print a machine-readable summary",
    )
    parser.set_defaults(_topics_handler=run_feedback)


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the topics group and its subcommands."""
    subparsers = parser.add_subparsers(dest="topics_command", metavar="COMMAND")
    help_text = "Record topic likes, dislikes and comments and refresh the guidance"
    feedback = subparsers.add_parser(
        "feedback",
        help=help_text,
        description=help_text,
    )
    _configure_feedback(feedback)


def run_topics(args: argparse.Namespace) -> int:
    """Dispatch to the selected topics subcommand."""
    handler = getattr(args, "_topics_handler", None)
    if handler is None:
        print("error: missing topics command", file=sys.stderr)
        print("usage: aivia topics feedback", file=sys.stderr)
        return 2
    return int(handler(args))


def _titles(args: argparse.Namespace) -> list[str]:
    values: list[str] = []
    for value in list(getattr(args, "topics", None) or []):
        if not isinstance(value, str):
            raise ConfigError("topic titles must be strings")
        values.append(value)
    for value in list(getattr(args, "topic_options", None) or []):
        if not isinstance(value, str):
            raise ConfigError("topic titles must be strings")
        values.append(value)
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        title = " ".join(value.split())
        key = normalize_title(title)
        if title and key not in seen:
            seen.add(key)
            unique.append(title)
    return unique


def _profile_name(args: argparse.Namespace) -> str:
    value = getattr(args, "profile", None)
    return value.strip() if isinstance(value, str) else ""


def _resolve(args: argparse.Namespace) -> tuple[Path, Path]:
    resolved = resolve_config(
        config_file=getattr(args, "config", None),
        environment=getattr(args, "environment", None),
        profile=getattr(args, "profile", None),
        preset=getattr(args, "preset", None),
        set_values=getattr(args, "set_values", None),
    )
    config = resolved.to_app_config()
    data_dir = paths.resolve_paths(getattr(args, "config", None)).data_dir
    database = getattr(config, "database", None)
    filename = (
        database.filename
        if isinstance(database, DatabaseConfig)
        else DatabaseConfig().filename
    )
    explicit_database = getattr(args, "database", None)
    database_path = (
        Path(explicit_database).expanduser()
        if isinstance(explicit_database, str) and explicit_database.strip()
        else data_dir / filename
    )
    explicit_store = getattr(args, "store", None)
    store = (
        FeedbackGuidanceStore(explicit_store)
        if isinstance(explicit_store, str) and explicit_store.strip()
        else FeedbackGuidanceStore(data_dir=data_dir)
    )
    return database_path, store.path


def _open_database(path: Path) -> Engine:
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_database_engine(path=path)
    try:
        upgrade_database(engine=engine)
    except BaseException:
        engine.dispose()
        raise
    return engine


def _row_feedback(row: Mapping[str, Any]) -> TopicFeedback:
    metadata = row.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    topic = metadata.get("title") or row.get("subject_id") or ""
    values: dict[str, Any] = {
        "topic": str(topic),
        "sentiment": str(row.get("sentiment") or "neutral"),
        "comment": str(row.get("comment") or ""),
        "profile": str(metadata.get("profile") or ""),
        "source": str(row.get("source") or ""),
        "run_id": str(row.get("run_id") or ""),
    }
    created_at = row.get("created_at")
    if isinstance(created_at, datetime):
        values["created_at"] = created_at
    return TopicFeedback(**values)


def _stored_feedback(repository: FeedbackRepository) -> list[TopicFeedback]:
    rows = repository.list(subject_kind="topic")
    return [_row_feedback(row) for row in rows]


def _record(
    repository: FeedbackRepository,
    titles: Sequence[str],
    *,
    sentiment: str,
    comment: str,
    profile: str,
) -> list[TopicFeedback]:
    recorded: list[TopicFeedback] = []
    for title in titles:
        row = repository.create(
            subject_kind="topic",
            subject_id=normalize_title(title),
            sentiment=sentiment,
            comment=comment,
            source="cli",
            run_id="",
            metadata={"title": title, "profile": profile},
        )
        recorded.append(_row_feedback(row))
    return recorded


def _print_feedback(
    recorded: Sequence[TopicFeedback],
    feedback: Sequence[TopicFeedback],
    guidance: FeedbackGuidance,
    *,
    store: Path,
) -> None:
    for entry in recorded:
        suffix = f" ({entry.comment})" if entry.comment else ""
        print(f"recorded: {entry.topic} ({entry.sentiment}){suffix}")
    if recorded:
        print(f"Guidance: {guidance.summary}")
        print(f"Store: {store}")
        return
    for entry in feedback:
        suffix = f": {entry.comment}" if entry.comment else ""
        print(f"{entry.sentiment}: {entry.topic}{suffix}")
    print(f"Guidance: {guidance.summary}")


def run_feedback(args: argparse.Namespace) -> int:
    """Record feedback, refresh the guidance sidecar and print the summary."""
    try:
        database_path, store_path = _resolve(args)
        titles = _titles(args)
        list_feedback = bool(getattr(args, "list_feedback", False))
        if not titles and not list_feedback:
            raise ConfigError("no topics to record (pass a title or use --list)")
        sentiment = normalize_sentiment(getattr(args, "sentiment", "neutral"))
        comment = " ".join(str(getattr(args, "comment", "") or "").split())
        profile = _profile_name(args)
        engine = _open_database(database_path)
        try:
            repository = FeedbackRepository(engine)
            recorded = (
                _record(
                    repository,
                    titles,
                    sentiment=sentiment,
                    comment=comment,
                    profile=profile,
                )
                if titles
                else []
            )
            feedback = _stored_feedback(repository)
            guidance = summarize_feedback(feedback)
            if recorded:
                FeedbackGuidanceStore(store_path).save(guidance)
        finally:
            engine.dispose()
    except (ConfigError, OSError, ValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if bool(getattr(args, "json_output", False)):
        print(
            json.dumps(
                {
                    "database": str(database_path),
                    "store": str(store_path),
                    "recorded": [entry.to_dict() for entry in recorded],
                    "feedback": [entry.to_dict() for entry in feedback],
                    "guidance": guidance.to_dict(),
                    "counts": dict(guidance.counts),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        _print_feedback(recorded, feedback, guidance, store=store_path)
    return 0


register_command(
    "topics",
    run_topics,
    help="Record topic feedback and summarize it into prompt guidance",
    configure=configure,
)
