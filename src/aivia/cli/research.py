"""``aivia research``: curate topics without running source collection.

``research add`` is the manual entry point into the research engine. The editor
supplies one or more topics (and their source URLs) directly, so the collection
stage is skipped. The command still runs the two safety stages that matter for
a hand-picked topic: dedupe against every topic added before, and verification
against independent source domains.

Accepted topics are persisted in a small JSON store under the user data
directory. A topic that only one domain reports is still stored, but it is
marked unverified and the reason is recorded, matching the verification stage
used by the rest of research.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from pydantic import ValidationError

from aivia.cli.registry import register_command
from aivia.core import paths
from aivia.core.errors import ConfigError
from aivia.core.models.base import canonical_json, utcnow
from aivia.core.models.topic import TopicCandidate
from aivia.research.dedupe import (
    DEFAULT_DEDUPE_WINDOW,
    DedupeDecision,
    DedupeHistory,
    dedupe_candidates,
)
from aivia.research.modes.otd import SeenTopic
from aivia.research.verify import (
    DEFAULT_MIN_DOMAINS,
    VerificationDecision,
    VerificationResult,
    verify_candidates,
)

__all__ = [
    "DEFAULT_STORE_FILENAME",
    "AddResult",
    "StoredTopic",
    "TopicStore",
    "add_topics",
    "configure",
    "run_add",
    "run_research",
    "topic_store_path",
]

DEFAULT_STORE_FILENAME = "topics.json"
"""The file under ``data/research`` that holds manually added topics."""

_STORE_VERSION = 1


def topic_store_path(
    data_dir: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the default persistent topic store path."""
    root = paths.data_dir(environ) if data_dir is None else Path(data_dir)
    return Path(root).expanduser() / "research" / DEFAULT_STORE_FILENAME


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_time(value: Any, default: datetime) -> datetime:
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, str) and value.strip():
        try:
            return _ensure_utc(datetime.fromisoformat(value.strip()))
        except ValueError:
            return default
    return default


@dataclass(frozen=True, slots=True)
class StoredTopic:
    """One accepted topic, when it was added and by which profile."""

    candidate: TopicCandidate
    added_at: datetime
    profile: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "added_at", _ensure_utc(self.added_at))
        if not isinstance(self.profile, str):
            raise ConfigError("stored topic profile must be a string")
        object.__setattr__(self, "profile", self.profile.strip())
        if not isinstance(self.reason, str):
            raise ConfigError("stored topic reason must be a string")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable store record."""
        payload = self.candidate.model_dump(mode="json")
        payload["added_at"] = self.added_at.isoformat()
        payload["profile"] = self.profile
        payload["verification_reason"] = self.reason
        return payload


class TopicStore:
    """A deterministic JSON store of manually added research topics."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        data_dir: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._path = (
            topic_store_path(data_dir, environ=environ)
            if path is None
            else Path(path).expanduser()
        )

    @property
    def path(self) -> Path:
        """The JSON file backing this store."""
        return self._path

    def load(self) -> list[StoredTopic]:
        """Read every stored topic in insertion order."""
        if not self._path.exists():
            return []
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read topic store: {self._path}") from exc
        if not text.strip():
            return []
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid topic store JSON: {self._path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ConfigError(f"topic store root must be a mapping: {self._path}")
        raw_topics = payload.get("topics", [])
        if not isinstance(raw_topics, Sequence) or isinstance(
            raw_topics, (str, bytes, bytearray)
        ):
            raise ConfigError(f"topic store topics must be a list: {self._path}")
        now = utcnow()
        stored: list[StoredTopic] = []
        for item in raw_topics:
            if isinstance(item, str):
                item = {"title": item}
            if not isinstance(item, Mapping):
                raise ConfigError(f"invalid topic entry in {self._path}")
            values = dict(item)
            added_at = _parse_time(values.pop("added_at", None), now)
            profile = values.pop("profile", "")
            reason = values.pop("verification_reason", "")
            try:
                candidate = TopicCandidate.model_validate(values)
            except ValidationError as exc:
                raise ConfigError(
                    f"invalid topic entry in {self._path}: {exc}"
                ) from exc
            stored.append(
                StoredTopic(
                    candidate=candidate,
                    added_at=added_at,
                    profile=str(profile or ""),
                    reason=str(reason or ""),
                )
            )
        return stored

    def save(self, entries: Iterable[StoredTopic]) -> None:
        """Write *entries* atomically and deterministically."""
        payload = {
            "version": _STORE_VERSION,
            "topics": [entry.to_dict() for entry in entries],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self._path.name}.",
            suffix=".tmp",
            dir=self._path.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(canonical_json(payload, indent=2))
                handle.write("\n")
            os.replace(temporary, self._path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def append(self, entries: Iterable[StoredTopic]) -> list[StoredTopic]:
        """Append *entries* to the store and return the full list."""
        stored = self.load()
        stored.extend(entries)
        self.save(stored)
        return stored

    def history(
        self,
        *,
        now: datetime | None = None,
    ) -> DedupeHistory:
        """Return the store as dedupe history for the next add."""
        current = utcnow() if now is None else now
        return DedupeHistory(
            [
                SeenTopic(
                    title=entry.candidate.title,
                    seen_at=entry.added_at,
                    mode=entry.candidate.mode,
                    profile=entry.profile,
                    embedding=getattr(entry.candidate, "embedding", ()) or (),
                )
                for entry in self.load()
            ],
            now=current,
        )


@dataclass(frozen=True, slots=True)
class AddResult:
    """The accepted topics, rejected duplicates and verification decisions."""

    stored: tuple[StoredTopic, ...] = ()
    verification: VerificationResult | None = None
    duplicates: tuple[DedupeDecision, ...] = ()
    store_path: Path | None = None
    recorded: bool = True

    @property
    def added(self) -> list[TopicCandidate]:
        """The accepted candidates in input order."""
        return [entry.candidate for entry in self.stored]

    @property
    def added_count(self) -> int:
        """The number of accepted topics."""
        return len(self.stored)

    @property
    def rejected_count(self) -> int:
        """The number of rejected duplicate topics."""
        return len(self.duplicates)

    @property
    def verified_count(self) -> int:
        """The number of accepted topics that cleared verification."""
        if self.verification is None:
            return 0
        return self.verification.verified_count

    @property
    def unverified_count(self) -> int:
        """The number of accepted topics that did not clear verification."""
        if self.verification is None:
            return 0
        return self.verification.unverified_count

    def _decisions(self) -> list[VerificationDecision]:
        if self.verification is None:
            return []
        return list(self.verification.decisions)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the add operation."""
        added: list[dict[str, Any]] = []
        for entry, decision in zip(
            self.stored,
            self._decisions(),
            strict=False,
        ):
            payload = entry.candidate.model_dump(mode="json")
            payload.update(
                {
                    "added_at": entry.added_at.isoformat(),
                    "profile": entry.profile,
                    "verified": decision.verified,
                    "status": decision.status.value,
                    "domains": list(decision.domains),
                    "domain_count": decision.domain_count,
                    "verification_reason": decision.reason,
                }
            )
            added.append(payload)
        return {
            "added": added,
            "rejected": [decision.to_dict() for decision in self.duplicates],
            "store": None if self.store_path is None else str(self.store_path),
            "recorded": self.recorded,
            "counts": {
                "added": self.added_count,
                "rejected": self.rejected_count,
                "verified": self.verified_count,
                "unverified": self.unverified_count,
            },
        }


def _as_topic_candidate(value: Any) -> TopicCandidate:
    if isinstance(value, TopicCandidate):
        return value
    return TopicCandidate.model_validate(
        value.model_dump(exclude={"claims", "candidate"})
    )


def add_topics(
    candidates: Iterable[TopicCandidate],
    *,
    store: TopicStore,
    profile: str = "",
    min_domains: int = DEFAULT_MIN_DOMAINS,
    window: timedelta | int | float | None = DEFAULT_DEDUPE_WINDOW,
    now: datetime | None = None,
    record: bool = True,
) -> AddResult:
    """Dedupe, verify and store manually supplied *candidates*.

    Collection is deliberately not part of this function: the caller has
    already chosen the topics. Dedupe runs first so a repeat is rejected before
    any verification work, then the surviving topics are verified against
    independent domains and persisted in input order.
    """
    current = utcnow() if now is None else now
    values = [
        candidate.model_copy(update={"mode": candidate.mode or "manual"})
        for candidate in candidates
    ]
    deduped = dedupe_candidates(
        values,
        history=store.history(now=current),
        window=window,
        profile=profile,
        now=current,
        record=False,
    )
    verification = verify_candidates(deduped.candidates, min_domains=min_domains)
    stored = tuple(
        StoredTopic(
            candidate=_as_topic_candidate(decision.candidate),
            added_at=current,
            profile=profile,
            reason=decision.reason,
        )
        for decision in verification.decisions
    )
    if record and stored:
        store.append(stored)
    return AddResult(
        stored=stored,
        verification=verification,
        duplicates=tuple(deduped.duplicates),
        store_path=store.path,
        recorded=record,
    )


def _configure_add(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "topics",
        nargs="*",
        metavar="TITLE",
        help="topic title; a bare http(s) argument is treated as a source URL",
    )
    parser.add_argument(
        "--title",
        dest="title_options",
        action="append",
        default=None,
        metavar="TITLE",
        help="topic title; repeatable",
    )
    parser.add_argument(
        "--source",
        "--url",
        "--source-url",
        dest="sources",
        action="append",
        default=None,
        metavar="URL",
        help="source URL for the topic; repeatable",
    )
    parser.add_argument(
        "--sources",
        "--urls",
        "--source-urls",
        dest="source_lists",
        action="append",
        default=None,
        metavar="URLS",
        help="comma-separated source URLs",
    )
    parser.add_argument(
        "--file",
        "--input",
        "--topics-file",
        dest="file",
        metavar="PATH",
        help="read topics from a JSON or YAML file (use - for standard input)",
    )
    parser.add_argument(
        "--store",
        metavar="PATH",
        help="topic store file instead of data/research/topics.json",
    )
    parser.add_argument(
        "--min-domains",
        type=int,
        default=DEFAULT_MIN_DOMAINS,
        metavar="N",
        help="independent source domains required to verify a topic",
    )
    parser.add_argument(
        "--dedupe-window",
        "--window",
        dest="dedupe_window",
        type=float,
        default=DEFAULT_DEDUPE_WINDOW.total_seconds() / 86400,
        metavar="DAYS",
        help="days of topic history to check for duplicates",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="print a machine-readable summary",
    )
    parser.set_defaults(_research_handler=run_add)


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the research group and its ``add`` subcommand."""
    subparsers = parser.add_subparsers(dest="research_command", metavar="COMMAND")
    add = subparsers.add_parser(
        "add",
        help="Add a hand-picked topic without collecting sources",
        description="Add a hand-picked topic without collecting sources",
    )
    _configure_add(add)


def run_research(args: argparse.Namespace) -> int:
    """Dispatch to the selected research subcommand."""
    handler = getattr(args, "_research_handler", None)
    if handler is None:
        print("error: missing research command", file=sys.stderr)
        print("usage: aivia research {add}", file=sys.stderr)
        return 2
    return int(handler(args))


def _looks_like_url(value: str) -> bool:
    parsed = urlsplit(value.strip())
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _shared_sources(args: argparse.Namespace) -> list[str]:
    values: list[str] = []
    for source in list(getattr(args, "sources", None) or []):
        if isinstance(source, str) and source.strip():
            values.append(source.strip())
    for group in list(getattr(args, "source_lists", None) or []):
        if isinstance(group, str):
            values.extend(
                part.strip()
                for part in group.replace(";", ",").split(",")
                if part.strip()
            )
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _read_file(path: str) -> Any:
    if path == "-":
        text = sys.stdin.read()
    else:
        text = Path(path).expanduser().read_text(encoding="utf-8")
    if not text.strip():
        return []
    return yaml.safe_load(text)


def _file_entries(path: str) -> list[Any]:
    payload = _read_file(path)
    if isinstance(payload, Mapping):
        nested = payload.get("topics")
        if nested is not None and not any(
            key in payload for key in ("title", "name", "topic", "subject")
        ):
            payload = nested
        else:
            payload = [payload]
    if isinstance(payload, str):
        payload = [payload]
    if not isinstance(payload, Sequence) or isinstance(
        payload, (str, bytes, bytearray)
    ):
        raise ConfigError("topic file must contain a list or a single topic")
    return list(payload)


def _candidate_from_file(item: Any, shared: Sequence[str]) -> TopicCandidate:
    if isinstance(item, str):
        return TopicCandidate(
            title=item,
            source_urls=list(shared),
            mode="manual",
        )
    if not isinstance(item, Mapping):
        raise ConfigError("topics in a file must be titles or mappings")
    candidate = TopicCandidate.model_validate(dict(item))
    if not candidate.source_urls and shared:
        candidate = candidate.model_copy(update={"source_urls": list(shared)})
    if not candidate.mode:
        candidate = candidate.model_copy(update={"mode": "manual"})
    return candidate


def _collect_candidates(args: argparse.Namespace) -> list[TopicCandidate]:
    shared = _shared_sources(args)
    titles: list[str] = []
    for value in list(getattr(args, "topics", None) or []):
        if not isinstance(value, str):
            raise ConfigError("topic titles must be strings")
        if _looks_like_url(value):
            if value.strip() not in shared:
                shared.append(value.strip())
        else:
            titles.append(value)
    for value in list(getattr(args, "title_options", None) or []):
        if not isinstance(value, str):
            raise ConfigError("topic titles must be strings")
        titles.append(value)

    candidates = [
        TopicCandidate(title=title, source_urls=list(shared), mode="manual")
        for title in titles
    ]
    path = getattr(args, "file", None)
    if path:
        candidates.extend(
            _candidate_from_file(item, shared) for item in _file_entries(path)
        )
    return candidates


def _profile_name(args: argparse.Namespace) -> str:
    value = getattr(args, "profile", None)
    return value.strip() if isinstance(value, str) else ""


def _store_for(args: argparse.Namespace) -> TopicStore:
    path = getattr(args, "store", None)
    if isinstance(path, str) and path.strip():
        return TopicStore(path.strip())
    config = getattr(args, "config", None)
    return TopicStore(data_dir=paths.resolve_paths(config).data_dir)


def _print_result(result: AddResult) -> None:
    decisions = list(result.verification.decisions) if result.verification else []
    for entry, decision in zip(result.stored, decisions, strict=False):
        status = "verified" if decision.verified else "unverified"
        print(f"added: {entry.candidate.title} ({status})")
    for duplicate in result.duplicates:
        print(f"rejected: {duplicate.title} ({duplicate.explanation})")
    print(
        f"added {result.added_count}, rejected {result.rejected_count}, "
        f"verified {result.verified_count}, unverified {result.unverified_count}"
    )


def run_add(args: argparse.Namespace) -> int:
    """Add, dedupe and verify hand-picked topics without collecting sources."""
    try:
        candidates = _collect_candidates(args)
        if not candidates:
            raise ConfigError("no topics to add")
        window = timedelta(days=float(getattr(args, "dedupe_window", 365)))
        result = add_topics(
            candidates,
            store=_store_for(args),
            profile=_profile_name(args),
            min_domains=int(getattr(args, "min_domains", DEFAULT_MIN_DOMAINS)),
            window=window,
            record=not bool(getattr(args, "dry_run", False)),
        )
    except (ConfigError, OSError, ValidationError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if bool(getattr(args, "json_output", False)):
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_result(result)
    return 0 if result.added_count else 1


register_command(
    "research",
    run_research,
    help="Add hand-picked research topics with verification and dedupe",
    configure=configure,
)
