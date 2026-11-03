"""Persistent topic history: the dedupe window survives a process restart."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from aivia.core.errors import ConfigError
from aivia.providers.protocols.source import SourceRequest, SourceResponse
from aivia.research import (
    DEFAULT_DEDUPE_WINDOW,
    History,
    HistoryStore,
    OnThisDayMode,
    PersistentHistory,
    PersistentHistoryStore,
    PersistentTopicHistory,
    PersistentTopicHistoryStore,
    SeenTopic,
    StoredTopicHistory,
    TopicHistory,
    TopicHistoryFile,
    TopicHistoryPersistence,
    TopicHistoryRepository,
    TopicHistoryStore,
    history_store_path,
    load_history,
    load_topic_history,
    save_history,
    save_topic_history,
    topic_history_path,
    topic_history_store_path,
)

NOW = datetime(2026, 5, 5, 12, tzinfo=UTC)


class Clock:
    def __init__(self, value: datetime = NOW) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, delta: timedelta) -> None:
        self.value += delta


class FakeSource:
    def __init__(self, candidates: list[Any]) -> None:
        self.candidates = list(candidates)

    async def discover(self, request: SourceRequest) -> SourceResponse:
        return SourceResponse(
            provider="fake-otd",
            query=request.query,
            candidates=[candidate.model_copy() for candidate in self.candidates],
        )


def _candidate(title: str, category: str = "selected") -> Any:
    from aivia.core.models.topic import TopicCandidate

    return TopicCandidate(
        title=title,
        category=category,
        source_urls=[f"https://example.test/{title.replace(' ', '_')}"],
    )


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def test_the_default_path_lives_under_the_research_data_directory(
    tmp_path: Path,
) -> None:
    assert topic_history_path(tmp_path) == tmp_path / "research" / "topic_history.json"
    assert topic_history_path(environ={"XDG_CONFIG_HOME": str(tmp_path)}) == (
        tmp_path / "aivia" / "data" / "research" / "topic_history.json"
    )


def test_a_restart_still_suppresses_titles_inside_the_dedupe_window(
    tmp_path: Path,
) -> None:
    path = tmp_path / "topic_history.json"
    clock = Clock()
    source = FakeSource([_candidate("Apollo 11"), _candidate("Moon landing", "events")])

    first_mode = OnThisDayMode(
        source,
        history=TopicHistoryStore(path, now=clock),
        now=clock,
    )
    first = _run(first_mode.discover(SourceRequest()))
    assert first.titles == ["Apollo 11", "Moon landing"]

    second_mode = OnThisDayMode(
        source,
        history=PersistentTopicHistory(path, now=clock),
        now=clock,
    )
    second = _run(second_mode.discover(SourceRequest()))

    assert second.titles == []
    assert second.metadata["skipped_duplicates"] == 2
    assert second_mode.history.titles == ("Apollo 11", "Moon landing")

    clock.advance(DEFAULT_DEDUPE_WINDOW)
    third = _run(second_mode.discover(SourceRequest()))
    assert third.titles == ["Apollo 11", "Moon landing"]


def test_entries_round_trip_with_profile_mode_and_embedding(tmp_path: Path) -> None:
    path = tmp_path / "topic_history.json"
    store = TopicHistoryStore(path, now=NOW)
    store.record(
        [_candidate("Apollo 11")],
        at=NOW,
        mode="on_this_day",
        profile="space",
        embeddings={"Apollo 11": [1.0, 0.0]},
    )

    reloaded = TopicHistoryStore(path, now=NOW)
    entry = reloaded.entries[0]

    assert entry.title == "Apollo 11"
    assert entry.seen_at == NOW
    assert entry.mode == "on_this_day"
    assert entry.profile == "space"
    assert entry.embedding == (1.0, 0.0)
    assert reloaded.seen_within("apollo_11", DEFAULT_DEDUPE_WINDOW, now=NOW)


def test_a_plain_history_can_be_saved_and_loaded(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    history = TopicHistory([SeenTopic("Apollo 11", NOW)], now=NOW)

    save_topic_history(history, path)
    loaded = load_topic_history(path, now=NOW)

    assert loaded.titles == ("Apollo 11",)
    assert loaded.seen_within("apollo 11", DEFAULT_DEDUPE_WINDOW, now=NOW)


def test_module_helpers_load_and_save_the_store(tmp_path: Path) -> None:
    path = tmp_path / "history.json"

    save_topic_history(["Apollo 11"], path, now=NOW)
    loaded = load_topic_history(path, now=NOW)

    assert isinstance(loaded, TopicHistory)
    assert loaded.titles == ("Apollo 11",)
    assert path.exists()


def test_prune_and_clear_are_persisted(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    store = TopicHistoryStore([SeenTopic("Old", NOW)], path=path, now=NOW)

    assert store.prune(timedelta(days=30), now=NOW + timedelta(days=30)) == 1
    assert TopicHistoryStore(path, now=NOW).titles == ()

    store.add(SeenTopic("New", NOW))
    store.clear()

    assert TopicHistoryStore(path, now=NOW).titles == ()


def test_load_refreshes_from_the_file_written_by_another_store(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    writer = TopicHistoryStore(path, now=NOW)
    reader = TopicHistoryStore(path, now=NOW)

    writer.add(SeenTopic("Apollo 11", NOW))
    assert reader.titles == ()

    assert reader.load()[0].title == "Apollo 11"
    assert list(reader.titles) == ["Apollo 11"]


def test_autosave_off_keeps_changes_in_memory_until_saved(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    store = TopicHistoryStore(path, now=NOW, autosave=False)

    store.add(SeenTopic("Apollo 11", NOW))
    assert not path.exists()

    store.save()
    assert TopicHistoryStore(path, now=NOW).titles == ("Apollo 11",)


def test_a_missing_or_empty_file_is_an_empty_history(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    assert TopicHistoryStore(path, now=NOW).titles == ()

    path.write_text("", encoding="utf-8")
    assert TopicHistoryStore(path, now=NOW).titles == ()


def test_a_malformed_store_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid topic history JSON"):
        TopicHistoryStore(path, now=NOW)

    payload = {"entries": [{"mode": "on_this_day"}]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="needs a title"):
        TopicHistoryStore(path, now=NOW)


def test_aliases_share_the_store_class() -> None:
    for alias in (
        History,
        HistoryStore,
        PersistentHistory,
        PersistentHistoryStore,
        PersistentTopicHistory,
        PersistentTopicHistoryStore,
        StoredTopicHistory,
        TopicHistoryFile,
        TopicHistoryPersistence,
        TopicHistoryRepository,
    ):
        assert alias is TopicHistoryStore
    assert topic_history_store_path is topic_history_path
    assert history_store_path is topic_history_path
    assert load_history is load_topic_history
    assert save_history is save_topic_history


def test_the_file_is_deterministic_json(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    store = TopicHistoryStore(path, now=NOW)
    store.record(["B", "A"], at=NOW)

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["version"] == 1
    assert [entry["title"] for entry in payload["entries"]] == ["B", "A"]
    assert payload["entries"][0]["seen_at"] == NOW.isoformat()
