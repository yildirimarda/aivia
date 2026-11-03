"""Persist the topic history used to suppress repeated topics across runs.

The On this day and evergreen modes suppress titles they returned inside a
dedupe window. Their in-memory :class:`~aivia.research.modes.otd.TopicHistory`
forgets everything when the process exits, so a restart would repeat an entry
that the window was supposed to block. :class:`TopicHistoryStore` keeps the
same history surface while writing every change to a small JSON file under the
user data directory.

The file is written atomically and deterministically, and a missing or empty
file is simply an empty history. A caller can pass the store straight to a mode
because it is a :class:`TopicHistory`; it can also preload a plain history from
:func:`load_topic_history`.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from aivia.core import paths
from aivia.core.errors import ConfigError
from aivia.core.models.base import canonical_json
from aivia.core.models.topic import TopicCandidate
from aivia.research.modes.otd import (
    SeenTopic,
    TopicHistory,
    _ensure_utc,
    _resolve_now,
)

__all__ = [
    "DEFAULT_HISTORY_FILENAME",
    "DEFAULT_HISTORY_STORE_FILENAME",
    "DEFAULT_TOPIC_HISTORY_FILENAME",
    "DEFAULT_TOPIC_HISTORY_STORE_FILENAME",
    "History",
    "HistoryStore",
    "PersistentHistory",
    "PersistentHistoryStore",
    "PersistentTopicHistory",
    "PersistentTopicHistoryStore",
    "StoredTopicHistory",
    "TopicHistoryFile",
    "TopicHistoryPersistence",
    "TopicHistoryRepository",
    "TopicHistoryStore",
    "history_store_path",
    "load_history",
    "load_topic_history",
    "save_history",
    "save_topic_history",
    "topic_history_path",
    "topic_history_store_path",
]

DEFAULT_TOPIC_HISTORY_FILENAME = "topic_history.json"
"""The file under ``data/research`` that holds the persisted topic history."""

DEFAULT_HISTORY_FILENAME = DEFAULT_TOPIC_HISTORY_FILENAME
"""Alias for :data:`DEFAULT_TOPIC_HISTORY_FILENAME`."""

DEFAULT_TOPIC_HISTORY_STORE_FILENAME = DEFAULT_TOPIC_HISTORY_FILENAME
"""Alias for :data:`DEFAULT_TOPIC_HISTORY_FILENAME`."""

DEFAULT_HISTORY_STORE_FILENAME = DEFAULT_TOPIC_HISTORY_FILENAME
"""Alias for :data:`DEFAULT_TOPIC_HISTORY_FILENAME`."""

_STORE_VERSION = 1
_ENTRY_KEYS = ("entries", "topics", "history", "records")


def topic_history_path(
    data_dir: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the default persistent topic history path."""
    root = paths.data_dir(environ) if data_dir is None else Path(data_dir)
    return Path(root).expanduser() / "research" / DEFAULT_TOPIC_HISTORY_FILENAME


topic_history_store_path = topic_history_path
"""Alias for :func:`topic_history_path`."""

history_store_path = topic_history_path
"""Alias for :func:`topic_history_path`."""


def _parse_time(value: Any, default: datetime) -> datetime:
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, str) and value.strip():
        try:
            return _ensure_utc(datetime.fromisoformat(value.strip()))
        except ValueError:
            return default
    return default


def _record_to_dict(record: SeenTopic) -> dict[str, Any]:
    return {
        "title": record.title,
        "seen_at": record.seen_at.isoformat(),
        "mode": record.mode,
        "key": record.key,
        "profile": record.profile,
        "embedding": list(record.embedding),
    }


def _coerce_record(value: Any, default: datetime) -> SeenTopic:
    if isinstance(value, SeenTopic):
        return value
    if isinstance(value, str):
        if not value.strip():
            raise ConfigError("history entry needs a title")
        return SeenTopic(title=value, seen_at=default)
    if not isinstance(value, Mapping):
        raise ConfigError("history entries must be titles, records or mappings")
    title = value.get("title") or value.get("name") or value.get("topic")
    if not isinstance(title, str) or not title.strip():
        raise ConfigError("history entry needs a title")
    seen_at = _parse_time(
        value.get("seen_at", value.get("at", value.get("added_at"))),
        default,
    )
    raw_embedding = value.get("embedding", value.get("vector", ()))
    raw_profile = value.get("profile", value.get("profile_name", ""))
    raw_mode = value.get("mode", "")
    key = value.get("key", "")
    return SeenTopic(
        title=title,
        seen_at=seen_at,
        mode=str(raw_mode or ""),
        key=str(key or ""),
        profile=str(raw_profile or ""),
        embedding=raw_embedding,
    )


def _entry_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(key in value for key in ("title", "name", "topic")):
            return [value]
        return [
            dict(entry, title=entry.get("title", name))
            if isinstance(entry, Mapping)
            else entry
            for name, entry in value.items()
        ]
    return value


def _read_store(path: Path, now: datetime) -> list[SeenTopic]:
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read topic history: {path}") from exc
    if not text.strip():
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid topic history JSON: {path}: {exc}") from exc

    if isinstance(payload, Mapping):
        raw_entries: Any = None
        for key in _ENTRY_KEYS:
            if key in payload:
                raw_entries = payload[key]
                break
        if raw_entries is None:
            if any(key in payload for key in ("title", "name", "topic")):
                raw_entries = [payload]
            elif payload:
                raw_entries = payload
            else:
                raw_entries = []
    elif isinstance(payload, list):
        raw_entries = payload
    else:
        raise ConfigError(f"topic history root must be a mapping or list: {path}")

    raw_entries = _entry_values(raw_entries)
    if not isinstance(raw_entries, Iterable) or isinstance(
        raw_entries, (str, bytes, bytearray, Mapping)
    ):
        raise ConfigError(f"topic history entries must be a list: {path}")

    records: list[SeenTopic] = []
    for item in raw_entries:
        records.append(_coerce_record(item, now))
    return records


def _write_store(path: Path, records: Iterable[SeenTopic]) -> None:
    entries = [_record_to_dict(record) for record in records]
    payload = {
        "version": _STORE_VERSION,
        "entries": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(payload, indent=2))
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


class TopicHistoryStore(TopicHistory):
    """A :class:`TopicHistory` that writes every change to a JSON file.

    ``path`` may be supplied directly or resolved from ``data_dir`` /
    ``environ``. The file is read once on construction and again on
    :meth:`load`, so a store created after a restart sees the titles recorded by
    an earlier process. Passing the store to a mode is enough for the mode's
    dedupe window to survive that restart.
    """

    def __init__(
        self,
        *args: Any,
        path: str | Path | None = None,
        data_dir: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
        entries: Iterable[SeenTopic | str | Mapping[str, Any]] | None = None,
        now: datetime | Callable[[], datetime] | None = None,
        autosave: bool = True,
    ) -> None:
        if len(args) > 1:
            raise TypeError("TopicHistoryStore takes at most one positional argument")
        if args:
            value = args[0]
            if isinstance(value, (str, os.PathLike)):
                if path is not None:
                    raise ConfigError("path was given twice")
                path = Path(value)
            else:
                if entries is not None:
                    raise ConfigError("use either entries or a path, not both")
                entries = value
        if not isinstance(autosave, bool):
            raise ConfigError("autosave must be a boolean")
        self._path = (
            topic_history_path(data_dir, environ=environ)
            if path is None
            else Path(path).expanduser()
        )
        self._now = now
        self._autosave = autosave
        self._suspend = 0
        super().__init__(None, now=now)
        self._replace(_read_store(self._path, _resolve_now(now)))
        if entries is not None:
            self._suspend += 1
            try:
                for entry in entries:
                    self.add(entry, seen_at=_resolve_now(now))
            finally:
                self._suspend -= 1
            self._persist()

    @property
    def path(self) -> Path:
        """The JSON file backing this history."""
        return self._path

    def _persist(self, *, force: bool = False) -> None:
        if (force or self._autosave) and self._suspend == 0:
            _write_store(self._path, self.entries)

    def _replace(self, records: Iterable[SeenTopic]) -> None:
        self._seen.clear()
        for record in records:
            TopicHistory.add(self, record)

    def load(self, path: str | Path | None = None) -> list[SeenTopic]:
        """Reload the file into memory and return the stored entries."""
        target = self._path if path is None else Path(path).expanduser()
        records = _read_store(target, _resolve_now(self._now))
        self._replace(records)
        return records

    refresh = load

    def save(
        self,
        entries: Iterable[SeenTopic | str | Mapping[str, Any]] | None = None,
    ) -> None:
        """Write the current history, or replace it with *entries*, atomically."""
        if entries is not None:
            now = _resolve_now(self._now)
            self._replace(_coerce_record(entry, now) for entry in entries)
        self._persist(force=True)

    def add(
        self,
        entry: SeenTopic | TopicCandidate | str | Mapping[str, Any],
        *,
        seen_at: datetime | None = None,
        mode: str = "",
        profile: str = "",
        embedding: Any = None,
    ) -> SeenTopic:
        """Remember *entry* and persist the updated history."""
        record = super().add(
            entry,
            seen_at=seen_at,
            mode=mode,
            profile=profile,
            embedding=embedding,
        )
        self._persist()
        return record

    def record(
        self,
        candidates: Iterable[Any],
        *,
        at: datetime | None = None,
        mode: str = "",
        profile: str = "",
        embeddings: Mapping[str, Any] | None = None,
    ) -> list[SeenTopic]:
        """Remember every candidate and persist once for the batch."""
        self._suspend += 1
        try:
            records = super().record(
                candidates,
                at=at,
                mode=mode,
                profile=profile,
                embeddings=embeddings,
            )
        finally:
            self._suspend -= 1
        self._persist()
        return records

    def prune(
        self,
        window: timedelta | int | float | None,
        *,
        now: datetime | Callable[[], datetime] | None = None,
    ) -> int:
        """Forget entries outside *window*, persist, and return the count."""
        removed = super().prune(window, now=now)
        self._persist()
        return removed

    def clear(self) -> None:
        """Forget every remembered title and persist the empty history."""
        super().clear()
        self._persist()

    def history(
        self,
        *,
        now: datetime | Callable[[], datetime] | None = None,
    ) -> TopicHistoryStore:
        """Return this store as the history a mode should read and write."""
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the persisted history."""
        return {
            "version": _STORE_VERSION,
            "entries": [_record_to_dict(record) for record in self.entries],
        }


PersistentTopicHistory = TopicHistoryStore
"""Alias for :class:`TopicHistoryStore`."""

PersistentTopicHistoryStore = TopicHistoryStore
"""Alias for :class:`TopicHistoryStore`."""

PersistentHistory = TopicHistoryStore
"""Alias for :class:`TopicHistoryStore`."""

PersistentHistoryStore = TopicHistoryStore
"""Alias for :class:`TopicHistoryStore`."""

TopicHistoryFile = TopicHistoryStore
"""Alias for :class:`TopicHistoryStore`."""

TopicHistoryPersistence = TopicHistoryStore
"""Alias for :class:`TopicHistoryStore`."""

HistoryStore = TopicHistoryStore
"""Alias for :class:`TopicHistoryStore`."""

History = TopicHistoryStore
"""Alias for :class:`TopicHistoryStore`."""

TopicHistoryRepository = TopicHistoryStore
"""Alias for :class:`TopicHistoryStore`."""

StoredTopicHistory = TopicHistoryStore
"""Alias for :class:`TopicHistoryStore`."""


def load_topic_history(
    path: str | Path | None = None,
    *,
    data_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    now: datetime | Callable[[], datetime] | None = None,
) -> TopicHistoryStore:
    """Return a persistent history loaded from *path*."""
    return TopicHistoryStore(path, data_dir=data_dir, environ=environ, now=now)


def save_topic_history(
    history: TopicHistory | Iterable[SeenTopic | str | Mapping[str, Any]],
    path: str | Path | None = None,
    *,
    data_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    now: datetime | Callable[[], datetime] | None = None,
) -> Path:
    """Write *history* to *path* atomically and return the file path."""
    store = TopicHistoryStore(path, data_dir=data_dir, environ=environ, now=now)
    if isinstance(history, TopicHistory):
        store.save(history.entries)
    else:
        store.save(history)
    return store.path


load_history = load_topic_history
"""Alias for :func:`load_topic_history`."""

save_history = save_topic_history
"""Alias for :func:`save_topic_history`."""
