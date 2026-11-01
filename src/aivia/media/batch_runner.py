"""Run a batch of items with a concurrency cap, a budget stop and a manifest.

The runner is deliberately independent of any CLI or file format. Callers pass
an iterable of items and a handler; the runner decides when the next item may
start, stops the whole batch when the shared budget is exhausted and records
one entry per item in a JSON manifest so a later invocation can resume the
items that already succeeded.

Concurrency is capped by a pool of worker tasks, not by starting every item and
then throttling it. A budget stop is batch wide: once it is raised no new item
starts, while items already running are allowed to finish so their result and
cost are not lost. The remaining items are recorded as ``budget_stopped`` and
are retried on a later run.

No provider is called directly and no CLI argument is parsed here. The handler
owns the paid work; the runner only schedules it and persists the outcome.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from aivia.core.budget import BudgetGuard
from aivia.core.config.budget import BudgetConfig
from aivia.core.cost import CostTracker
from aivia.core.errors import AiviaError, BudgetError
from aivia.core.models.base import canonical_json, utcnow
from aivia.core.redact import redact
from aivia.core.resume import stable_value

__all__ = [
    "BatchItemResult",
    "BatchItemStatus",
    "BatchManifest",
    "BatchOutcome",
    "BatchResult",
    "BatchRunResult",
    "BatchRunner",
    "BatchStatus",
    "ItemStatus",
    "run_batch",
    "run_batch_sync",
]

MANIFEST_VERSION = 1
"""The manifest schema version written by :class:`BatchManifest`."""

MANIFEST_NAME = "batch-manifest.json"
"""The default file name used when a manifest directory is passed."""

Handler = Callable[[Any], Any]
KeyFunction = Callable[..., str]
FingerprintFunction = Callable[..., str]
CostFunction = Callable[[Any], float]
EstimateFunction = Callable[[Any, int], float]
Clock = Callable[[], datetime]

_MAX_COST_TOLERANCE = 1e-9


class BatchItemStatus(StrEnum):
    """The lifecycle of one item in a batch."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BUDGET_STOPPED = "budget_stopped"
    CANCELLED = "cancelled"


BatchStatus = BatchItemStatus
ItemStatus = BatchItemStatus


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("batch timestamp must be an ISO 8601 string")
    return datetime.fromisoformat(value)


def _positive_int(value: object, *, name: str = "concurrency") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_cost(value: object, *, name: str = "cost_usd") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite, non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite, non-negative number")
    return number


def _optional_cost(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite_cost(value, name=name)


def _key_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return str(value).strip()


def _default_key(item: Any, index: int) -> str:
    if isinstance(item, Mapping):
        for name in ("id", "key"):
            if name in item:
                text = _key_text(item[name])
                if text:
                    return text
    for name in ("id", "key"):
        if hasattr(item, name):
            text = _key_text(getattr(item, name))
            if text:
                return text
    return str(index)


def _accepts_index(function: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return False
    positional = 0
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional += 1
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return True
    return positional >= 2


def _extract_cost(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return _finite_cost(value)
    if isinstance(value, Mapping):
        for name in ("cost_usd", "amount_usd", "cost", "amount"):
            if name in value and value[name] is not None:
                return _finite_cost(value[name], name=name)
    for name in ("cost_usd", "amount_usd", "cost", "amount"):
        candidate = getattr(value, name, None)
        if candidate is not None and not callable(candidate):
            return _finite_cost(candidate, name=name)
    event = getattr(value, "event", None)
    if event is not None:
        amount = getattr(event, "amount_usd", None)
        if amount is not None:
            return _finite_cost(amount)
    events = getattr(value, "events", None)
    if (
        events is not None
        and isinstance(events, Iterable)
        and not isinstance(events, (str, bytes, Mapping))
    ):
        total = 0.0
        for event in events:
            amount = getattr(event, "amount_usd", None)
            if amount is not None:
                total += _finite_cost(amount)
        return total
    return 0.0


@dataclass(frozen=True, slots=True)
class BatchItemResult:
    """The recorded outcome of one batch item."""

    key: str
    index: int
    status: BatchItemStatus
    value: Any = None
    error: str = ""
    error_type: str = ""
    error_code: str = ""
    cost_usd: float = 0.0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    resumed: bool = False
    fingerprint: str = ""

    @property
    def succeeded(self) -> bool:
        """Whether the item completed successfully."""
        return self.status is BatchItemStatus.SUCCEEDED

    @property
    def ok(self) -> bool:
        """Alias for :attr:`succeeded`."""
        return self.succeeded

    @property
    def failed(self) -> bool:
        """Whether the item ran and raised an error."""
        return self.status is BatchItemStatus.FAILED

    @property
    def budget_stopped(self) -> bool:
        """Whether the item never started because the batch budget ran out."""
        return self.status is BatchItemStatus.BUDGET_STOPPED

    @property
    def cancelled(self) -> bool:
        """Whether the item never started because an earlier item failed."""
        return self.status is BatchItemStatus.CANCELLED

    @property
    def completed(self) -> bool:
        """Whether the item reached a terminal outcome."""
        return self.status is not BatchItemStatus.PENDING and (
            self.status is not BatchItemStatus.RUNNING
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible manifest entry."""
        return {
            "key": self.key,
            "index": self.index,
            "status": self.status.value,
            "value": stable_value(self.value),
            "error": self.error,
            "error_type": self.error_type,
            "error_code": self.error_code,
            "cost_usd": self.cost_usd,
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "resumed": self.resumed,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BatchItemResult:
        """Rebuild an entry written by :meth:`to_dict`."""
        try:
            status = BatchItemStatus(str(data["status"]))
        except (KeyError, ValueError) as exc:
            raise ValueError("invalid batch item status") from exc
        try:
            index = int(data.get("index", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid batch item index") from exc
        return cls(
            key=str(data.get("key", "")),
            index=index,
            status=status,
            value=data.get("value"),
            error=str(data.get("error", "") or ""),
            error_type=str(data.get("error_type", "") or ""),
            error_code=str(data.get("error_code", "") or ""),
            cost_usd=_finite_cost(data.get("cost_usd", 0.0)),
            started_at=_parse_time(data.get("started_at")),
            finished_at=_parse_time(data.get("finished_at")),
            resumed=bool(data.get("resumed", False)),
            fingerprint=str(data.get("fingerprint", "") or ""),
        )


@dataclass(frozen=True, slots=True)
class BatchResult:
    """The ordered outcomes of one batch run."""

    items: tuple[BatchItemResult, ...]
    manifest_path: Path | None = None
    stopped_for_budget: bool = False
    stopped_for_error: bool = False
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def __iter__(self) -> Iterator[BatchItemResult]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    @property
    def succeeded(self) -> tuple[BatchItemResult, ...]:
        """Every item that completed successfully, including resumed items."""
        return tuple(item for item in self.items if item.succeeded)

    @property
    def failed(self) -> tuple[BatchItemResult, ...]:
        """Every item that ran and raised an error."""
        return tuple(item for item in self.items if item.failed)

    @property
    def budget_stopped(self) -> tuple[BatchItemResult, ...]:
        """Every item that never started because the budget ran out."""
        return tuple(item for item in self.items if item.budget_stopped)

    @property
    def cancelled(self) -> tuple[BatchItemResult, ...]:
        """Every item that never started because the batch was stopped."""
        return tuple(item for item in self.items if item.cancelled)

    @property
    def ok(self) -> bool:
        """Whether every item succeeded and nothing was skipped for budget."""
        return not self.failed and not self.budget_stopped and not self.cancelled

    @property
    def total_cost_usd(self) -> float:
        """The sum of every recorded item cost, including resumed items."""
        return math.fsum(item.cost_usd for item in self.items)

    @property
    def keys(self) -> tuple[str, ...]:
        """Every item key in batch order."""
        return tuple(item.key for item in self.items)

    def by_key(self, key: str) -> BatchItemResult | None:
        """Return the outcome for *key*, or ``None``."""
        for item in self.items:
            if item.key == key:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible summary of the run."""
        return {
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "stopped_for_budget": self.stopped_for_budget,
            "stopped_for_error": self.stopped_for_error,
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "total_cost_usd": self.total_cost_usd,
            "items": [item.to_dict() for item in self.items],
        }


BatchRunResult = BatchResult
BatchOutcome = BatchResult


class BatchManifest:
    """The durable per-item manifest for one batch.

    Entries are keyed by the item key and written atomically after every state
    change. ``load`` reads an existing file so a resumed run can reuse entries
    that already succeeded; a non-resumed run starts from an empty manifest.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        if path is not None:
            target = Path(path).expanduser()
            if target.is_dir():
                target = target / MANIFEST_NAME
            self._path: Path | None = target
        else:
            self._path = None
        self._clock = clock or utcnow
        self._entries: dict[str, BatchItemResult] = {}

    @property
    def path(self) -> Path | None:
        """The manifest file, or ``None`` for an in-memory manifest."""
        return self._path

    @property
    def exists(self) -> bool:
        """Whether the manifest file currently exists."""
        return self._path is not None and self._path.is_file()

    @property
    def entries(self) -> tuple[BatchItemResult, ...]:
        """Every entry ordered by item index and then key."""
        return tuple(
            sorted(self._entries.values(), key=lambda item: (item.index, item.key))
        )

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> BatchItemResult | None:
        """Return the recorded entry for *key*, or ``None``."""
        return self._entries.get(key)

    def completed(self) -> dict[str, BatchItemResult]:
        """Return the entries that completed successfully, keyed by item key."""
        return {
            key: item
            for key, item in self._entries.items()
            if item.status is BatchItemStatus.SUCCEEDED
        }

    def clear(self) -> None:
        """Drop every in-memory entry without touching the file."""
        self._entries.clear()

    def reset(self) -> None:
        """Drop every entry and remove the manifest file when it exists."""
        self._entries.clear()
        if self._path is not None:
            with contextlib.suppress(FileNotFoundError):
                self._path.unlink()

    def load(self) -> dict[str, BatchItemResult]:
        """Read the manifest file and return its entries keyed by item key."""
        self._entries.clear()
        if self._path is None or not self._path.is_file():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid batch manifest: {self._path}") from exc
        if not isinstance(data, Mapping):
            raise ValueError(f"invalid batch manifest: {self._path}")
        raw_items = data.get("items", {})
        if not isinstance(raw_items, Mapping):
            raise ValueError(f"invalid batch manifest: {self._path}")
        for key, raw in raw_items.items():
            if not isinstance(raw, Mapping):
                raise ValueError(f"invalid batch manifest entry: {key}")
            entry = BatchItemResult.from_dict(raw)
            self._entries[str(key)] = entry
        return dict(self._entries)

    def upsert(self, result: BatchItemResult) -> None:
        """Insert or replace one item outcome."""
        self._entries[result.key] = result

    def save(self) -> Path | None:
        """Atomically write every entry and return the manifest path."""
        if self._path is None:
            return None
        payload = {
            "version": MANIFEST_VERSION,
            "updated_at": _iso(self._clock()),
            "items": {
                item.key: item.to_dict()
                for item in sorted(
                    self._entries.values(), key=lambda entry: (entry.index, entry.key)
                )
            },
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(self._path, canonical_json(payload, indent=2) + "\n")
        return self._path


def _write_atomic(path: Path, text: str) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
        raise


class BatchRunner:
    """Schedule items with a concurrency cap and a shared budget stop.

    ``handler`` receives one item and may be synchronous or asynchronous. A
    synchronous handler runs in a worker thread so the concurrency cap applies
    to blocking work too. The return value is stored in the manifest and is
    used to extract an item cost unless ``cost_of`` overrides it.

    ``concurrency`` is the maximum number of handlers in flight. ``max_cost_usd``
    (alias ``budget_usd``) stops the batch once the sum of completed item costs
    reaches the cap; a ``BudgetGuard`` can also be passed as ``budget`` and is
    consulted before every item, optionally with a per-item ``estimate``. A
    :class:`~aivia.core.errors.BudgetError` raised by the handler stops the
    batch as well, so a provider rejecting an over-budget call cannot let the
    rest of the batch keep spending.
    """

    def __init__(
        self,
        handler: Handler | None = None,
        *,
        concurrency: int = 1,
        max_cost_usd: float | None = None,
        budget_usd: float | None = None,
        budget: BudgetGuard | BudgetConfig | Mapping[str, Any] | None = None,
        tracker: CostTracker | None = None,
        run_id: str = "",
        estimate: float | EstimateFunction | None = None,
        operation: str = "batch",
        provider: str = "",
        manifest: str | Path | BatchManifest | None = None,
        key: KeyFunction | str | None = None,
        fingerprint: FingerprintFunction | str | None = None,
        cost_of: CostFunction | None = None,
        resume: bool = False,
        stop_on_error: bool = False,
        clock: Clock | None = None,
    ) -> None:
        if handler is not None and not callable(handler):
            raise TypeError("handler must be callable")
        if key is not None and not callable(key) and not isinstance(key, str):
            raise TypeError("key must be callable, a field name or None")
        if (
            fingerprint is not None
            and not callable(fingerprint)
            and not isinstance(fingerprint, str)
        ):
            raise TypeError("fingerprint must be callable, a field name or None")
        if cost_of is not None and not callable(cost_of):
            raise TypeError("cost_of must be callable")
        if (
            estimate is not None
            and not isinstance(estimate, (int, float))
            and not callable(estimate)
        ):
            raise TypeError("estimate must be a number, callable or None")
        if tracker is not None and not isinstance(tracker, CostTracker):
            raise TypeError("tracker must be a CostTracker")
        if (
            max_cost_usd is not None
            and budget_usd is not None
            and not math.isclose(
                float(max_cost_usd), float(budget_usd), rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise ValueError("max_cost_usd and budget_usd disagree")
        limit = max_cost_usd if max_cost_usd is not None else budget_usd
        self._max_cost_usd = (
            _optional_cost(limit, name="max_cost_usd") if limit is not None else None
        )
        active_budget = budget
        if isinstance(active_budget, (BudgetConfig, Mapping)):
            active_budget = BudgetGuard(active_budget, tracker)
        if active_budget is not None and not isinstance(active_budget, BudgetGuard):
            raise TypeError("budget must be a BudgetGuard, BudgetConfig or mapping")
        if not isinstance(operation, str):
            raise TypeError("operation must be a string")
        if not isinstance(provider, str):
            raise TypeError("provider must be a string")
        if not isinstance(run_id, str):
            raise TypeError("run_id must be a string")
        if manifest is None:
            self._manifest = BatchManifest()
        elif isinstance(manifest, BatchManifest):
            self._manifest = manifest
        elif isinstance(manifest, (str, Path)):
            self._manifest = BatchManifest(manifest)
        else:
            raise TypeError("manifest must be a path, BatchManifest or None")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")

        self._handler = handler
        self._concurrency = _positive_int(concurrency)
        self._budget = active_budget
        self._run_id = run_id
        self._estimate = estimate
        self._operation = operation.strip()
        self._provider = provider.strip()
        self._key = key
        self._fingerprint = fingerprint
        self._cost_of = cost_of or _extract_cost
        self._resume = bool(resume)
        self._stop_on_error = bool(stop_on_error)
        self._clock = clock or utcnow

    @property
    def concurrency(self) -> int:
        """The maximum number of handlers that may run at once."""
        return self._concurrency

    @property
    def max_cost_usd(self) -> float | None:
        """The batch-wide cost cap, or ``None`` when unlimited."""
        return self._max_cost_usd

    @property
    def budget(self) -> BudgetGuard | None:
        """The budget guard consulted before each item, when configured."""
        return self._budget

    @property
    def manifest(self) -> BatchManifest:
        """The manifest this runner writes."""
        return self._manifest

    @property
    def fingerprint(self) -> FingerprintFunction | str | None:
        """The content fingerprint source used to validate resumed entries."""
        return self._fingerprint

    def _key_for(self, item: Any, index: int) -> str:
        if self._key is None:
            return _default_key(item, index)
        if isinstance(self._key, str):
            if isinstance(item, Mapping):
                value = item.get(self._key)
            else:
                value = getattr(item, self._key, None)
            text = _key_text(value)
            return text or str(index)
        function = self._key
        if _accepts_index(function):
            text = _key_text(function(item, index))
        else:
            text = _key_text(function(item))
        return text or str(index)

    def _fingerprint_for(self, item: Any, index: int) -> str:
        if self._fingerprint is None:
            return ""
        if isinstance(self._fingerprint, str):
            if isinstance(item, Mapping):
                value = item.get(self._fingerprint)
            else:
                value = getattr(item, self._fingerprint, None)
            return _key_text(value)
        function = self._fingerprint
        if _accepts_index(function):
            return _key_text(function(item, index))
        return _key_text(function(item))

    def _estimate_for(self, item: Any, index: int) -> float:
        if self._estimate is None:
            return 0.0
        if callable(self._estimate):
            return _finite_cost(self._estimate(item, index), name="estimate")
        return _finite_cost(self._estimate, name="estimate")

    def _budget_exceeded(self, item: Any, index: int, spent: float) -> bool:
        if self._max_cost_usd is not None and spent >= (
            self._max_cost_usd - _MAX_COST_TOLERANCE
        ):
            return True
        guard = self._budget
        if guard is None:
            return False
        estimate = self._estimate_for(item, index)
        remaining = guard.remaining(
            self._run_id or None,
            provider=self._provider or None,
            operation=self._operation or None,
        )
        if remaining is not None and (
            remaining <= _MAX_COST_TOLERANCE
            or estimate > remaining + _MAX_COST_TOLERANCE
        ):
            return True
        try:
            guard.check(
                estimate,
                run_id=self._run_id or None,
                provider=self._provider or None,
                operation=self._operation or None,
            )
        except BudgetError:
            return True
        return False

    async def _invoke(self, handler: Handler, item: Any) -> Any:
        if inspect.iscoroutinefunction(handler):
            return await handler(item)
        result = await asyncio.to_thread(handler, item)
        if inspect.isawaitable(result):
            return await result
        return result

    def _error_fields(self, error: BaseException) -> tuple[str, str, str]:
        code = error.code if isinstance(error, AiviaError) else ""
        return (
            redact(str(error)) or type(error).__name__,
            type(error).__name__,
            code,
        )

    async def run(
        self,
        items: Iterable[Any],
        *,
        handler: Handler | None = None,
        resume: bool | None = None,
    ) -> BatchResult:
        """Run *items* and return their ordered outcomes.

        ``resume=True`` loads the manifest first and reuses every item that
        already succeeded without calling the handler. A ``False`` value starts
        a fresh manifest. Passing ``handler`` here overrides the constructor's
        handler.
        """
        active_handler = handler if handler is not None else self._handler
        if active_handler is None:
            raise TypeError("a batch runner needs a handler")
        if not callable(active_handler):
            raise TypeError("handler must be callable")
        values = tuple(items)
        keys = tuple(self._key_for(item, index) for index, item in enumerate(values))
        fingerprints = tuple(
            self._fingerprint_for(item, index) for index, item in enumerate(values)
        )
        seen: set[str] = set()
        duplicates: set[str] = set()
        for key in keys:
            if key in seen:
                duplicates.add(key)
            seen.add(key)
        if duplicates:
            raise ValueError(f"duplicate batch item keys: {sorted(duplicates)}")

        do_resume = self._resume if resume is None else bool(resume)
        if do_resume:
            self._manifest.load()
        else:
            self._manifest.reset()

        started_at = self._clock()
        results = [
            BatchItemResult(
                key=key,
                index=index,
                status=BatchItemStatus.PENDING,
                fingerprint=fingerprints[index],
            )
            for index, key in enumerate(keys)
        ]
        spent = 0.0
        if do_resume:
            for index, key in enumerate(keys):
                previous = self._manifest.get(key)
                if (
                    previous is not None
                    and previous.succeeded
                    and (
                        self._fingerprint is None
                        or previous.fingerprint == fingerprints[index]
                    )
                ):
                    results[index] = replace(
                        previous,
                        index=index,
                        key=key,
                        resumed=True,
                        fingerprint=fingerprints[index],
                    )
                    spent += previous.cost_usd

        lock = asyncio.Lock()
        manifest_lock = asyncio.Lock()
        next_index = 0
        stop = False
        stop_error = False
        budget_stop = False

        async def save(result: BatchItemResult) -> None:
            async with manifest_lock:
                self._manifest.upsert(result)
                self._manifest.save()

        async def worker() -> None:
            nonlocal next_index, stop, stop_error, budget_stop, spent
            while True:
                stopped_result: BatchItemResult | None = None
                running: BatchItemResult | None = None
                async with lock:
                    if stop or next_index >= len(values):
                        return
                    index = next_index
                    next_index += 1
                    if results[index].status is not BatchItemStatus.PENDING:
                        continue
                    key = keys[index]
                    item = values[index]
                    if self._budget_exceeded(item, index, spent):
                        stop = True
                        budget_stop = True
                        stopped_result = BatchItemResult(
                            key=key,
                            index=index,
                            status=BatchItemStatus.BUDGET_STOPPED,
                            finished_at=self._clock(),
                            fingerprint=fingerprints[index],
                        )
                        results[index] = stopped_result
                    else:
                        running = BatchItemResult(
                            key=key,
                            index=index,
                            status=BatchItemStatus.RUNNING,
                            started_at=self._clock(),
                            fingerprint=fingerprints[index],
                        )
                        results[index] = running
                if stopped_result is not None:
                    await save(stopped_result)
                    return
                assert running is not None
                await save(running)

                try:
                    value = await self._invoke(active_handler, item)
                except BudgetError as exc:
                    error, error_type, error_code = self._error_fields(exc)
                    result = BatchItemResult(
                        key=key,
                        index=index,
                        status=BatchItemStatus.FAILED,
                        started_at=running.started_at,
                        finished_at=self._clock(),
                        error=error,
                        error_type=error_type,
                        error_code=error_code,
                        fingerprint=fingerprints[index],
                    )
                    async with lock:
                        stop = True
                        budget_stop = True
                        results[index] = result
                    await save(result)
                    continue
                except Exception as exc:
                    error, error_type, error_code = self._error_fields(exc)
                    result = BatchItemResult(
                        key=key,
                        index=index,
                        status=BatchItemStatus.FAILED,
                        started_at=running.started_at,
                        finished_at=self._clock(),
                        error=error,
                        error_type=error_type,
                        error_code=error_code,
                        fingerprint=fingerprints[index],
                    )
                    async with lock:
                        if self._stop_on_error:
                            stop = True
                            stop_error = True
                        results[index] = result
                    await save(result)
                    continue

                try:
                    cost = self._cost_of(value)
                except Exception as exc:
                    error, error_type, error_code = self._error_fields(exc)
                    result = BatchItemResult(
                        key=key,
                        index=index,
                        status=BatchItemStatus.FAILED,
                        started_at=running.started_at,
                        finished_at=self._clock(),
                        error=error,
                        error_type=error_type,
                        error_code=error_code,
                        fingerprint=fingerprints[index],
                    )
                    async with lock:
                        results[index] = result
                    await save(result)
                    continue
                if cost < 0 or not math.isfinite(cost):
                    cost = _finite_cost(0)
                result = BatchItemResult(
                    key=key,
                    index=index,
                    status=BatchItemStatus.SUCCEEDED,
                    value=value,
                    cost_usd=cost,
                    started_at=running.started_at,
                    finished_at=self._clock(),
                    fingerprint=fingerprints[index],
                )
                async with lock:
                    spent += cost
                    results[index] = result
                    if self._max_cost_usd is not None and spent >= (
                        self._max_cost_usd - _MAX_COST_TOLERANCE
                    ):
                        stop = True
                        budget_stop = True
                await save(result)

        workers = [
            asyncio.create_task(worker())
            for _ in range(min(self._concurrency, max(len(values), 1)))
        ]
        if workers:
            await asyncio.gather(*workers)

        stopped_for_error = stop_error
        for index, result in enumerate(results):
            if result.status is not BatchItemStatus.PENDING:
                continue
            status = (
                BatchItemStatus.CANCELLED
                if stopped_for_error
                else BatchItemStatus.BUDGET_STOPPED
            )
            replacement = BatchItemResult(
                key=result.key,
                index=index,
                status=status,
                finished_at=self._clock(),
                fingerprint=fingerprints[index],
            )
            results[index] = replacement
            await save(replacement)

        finished_at = self._clock()
        final = tuple(results)
        return BatchResult(
            items=final,
            manifest_path=self._manifest.path,
            stopped_for_budget=budget_stop
            or any(item.status is BatchItemStatus.BUDGET_STOPPED for item in final),
            stopped_for_error=stopped_for_error,
            started_at=started_at,
            finished_at=finished_at,
        )

    async def run_async(
        self,
        items: Iterable[Any],
        *,
        handler: Handler | None = None,
        resume: bool | None = None,
    ) -> BatchResult:
        """Alias for :meth:`run`."""
        return await self.run(items, handler=handler, resume=resume)

    def run_sync(
        self,
        items: Iterable[Any],
        *,
        handler: Handler | None = None,
        resume: bool | None = None,
    ) -> BatchResult:
        """Run the batch to completion from synchronous code."""
        return asyncio.run(self.run(items, handler=handler, resume=resume))


async def run_batch(
    items: Iterable[Any],
    handler: Handler | None = None,
    **options: Any,
) -> BatchResult:
    """Build a :class:`BatchRunner` from *options* and run *items* once."""
    runner = BatchRunner(handler, **options)
    return await runner.run(items)


def run_batch_sync(
    items: Iterable[Any],
    handler: Handler | None = None,
    **options: Any,
) -> BatchResult:
    """Synchronous convenience wrapper around :func:`run_batch`."""
    return asyncio.run(run_batch(items, handler, **options))
