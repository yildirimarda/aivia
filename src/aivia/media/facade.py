"""One media service facade for the CLI, the HTTP API and pipeline stages.

The facade sits on top of :class:`~aivia.media.toolkit.MediaToolkit` and keeps
the same call surface: every capability still runs through the screened,
cached, budgeted and cost tracked service path. On top of that it records one
small metrics document after every call, so an operator can see how many media
calls an invocation made, how many were served from cache, which provider
answered and what the calls cost without querying the ledger.

The metrics file is opt-in per facade but the CLI, the API and stages all point
it at a sensible location (the data directory or the run directory). Writes are
atomic and a metrics write failure never turns a successful paid call into a
failed one.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from aivia.core.models.base import canonical_json, utcnow
from aivia.core.paths import resolve_paths
from aivia.media.toolkit import MediaResult, MediaToolkit

__all__ = [
    "MEDIA_METRICS_FILENAME",
    "MediaMetrics",
    "MediaServiceFacade",
]

MEDIA_METRICS_FILENAME = "media-metrics.json"
"""The default metrics file name written beside a database or run directory."""

_LOGGER = logging.getLogger("aivia.media.facade")


def _empty_kind() -> dict[str, Any]:
    return {
        "calls": 0,
        "succeeded": 0,
        "failed": 0,
        "cache_hits": 0,
        "cost_usd": 0.0,
    }


@dataclass(slots=True)
class MediaMetrics:
    """Mutable counters for every media call made through one facade.

    ``kinds`` holds the same counters per capability so a mixed invocation can
    report image, speech and video calls separately. ``last`` is the most recent
    call summary and is useful when a metrics file is inspected by a human.
    """

    started_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    calls: int = 0
    succeeded: int = 0
    failed: int = 0
    cache_hits: int = 0
    cost_usd: float = 0.0
    kinds: dict[str, dict[str, Any]] = field(default_factory=dict)
    last: dict[str, Any] | None = None

    def kind(self, name: str) -> dict[str, Any]:
        """Return the counters for *name*, creating them on first use."""
        counters = self.kinds.get(name)
        if counters is None:
            counters = _empty_kind()
            self.kinds[name] = counters
        return counters

    def record_success(
        self,
        kind: str,
        result: MediaResult,
        *,
        duration_seconds: float,
    ) -> None:
        """Count one successful call and remember its summary."""
        amount = float(result.cost_usd)
        cached = bool(result.cached)
        self.calls += 1
        self.succeeded += 1
        self.cost_usd += amount
        if cached:
            self.cache_hits += 1
        counters = self.kind(kind)
        counters["calls"] += 1
        counters["succeeded"] += 1
        counters["cost_usd"] += amount
        if cached:
            counters["cache_hits"] += 1
        self.last = {
            "kind": kind,
            "status": "succeeded",
            "provider": result.provider,
            "model": result.model,
            "cost_usd": amount,
            "cached": cached,
            "paths": list(result.paths),
            "duration_seconds": round(duration_seconds, 6),
            "at": utcnow().isoformat(),
        }
        self.updated_at = utcnow()

    def record_failure(
        self,
        kind: str,
        error: BaseException,
        *,
        duration_seconds: float,
    ) -> None:
        """Count one failed call and remember its redacted summary."""
        self.calls += 1
        self.failed += 1
        counters = self.kind(kind)
        counters["calls"] += 1
        counters["failed"] += 1
        self.last = {
            "kind": kind,
            "status": "failed",
            "error_code": str(getattr(error, "code", "") or type(error).__name__),
            "error_message": str(error),
            "duration_seconds": round(duration_seconds, 6),
            "at": utcnow().isoformat(),
        }
        self.updated_at = utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot."""
        payload: dict[str, Any] = {
            "version": 1,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "calls": self.calls,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "cache_hits": self.cache_hits,
            "cost_usd": self.cost_usd,
            "kinds": {
                name: dict(counters) for name, counters in sorted(self.kinds.items())
            },
        }
        if self.last is not None:
            payload["last"] = dict(self.last)
        return payload


class MediaServiceFacade(MediaToolkit):
    """The one media entry point used by the CLI, API and stages.

    It is a :class:`MediaToolkit` with an optional ``metrics_path``. Every
    capability method (and :meth:`generate`, which dispatches to them) records
    success, failure, cache hits and cost in :class:`MediaMetrics` and rewrites
    the metrics file atomically. The file is a convenience for operators; the
    cost ledger remains the source of truth for money.
    """

    def __init__(
        self,
        *args: Any,
        metrics_path: str | Path | None = None,
        metrics: MediaMetrics | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._metrics = metrics if metrics is not None else MediaMetrics()
        self._metrics_path = (
            Path(metrics_path).expanduser() if metrics_path is not None else None
        )
        self._metrics_lock = threading.Lock()

    @classmethod
    def from_config(
        cls,
        config: Any = None,
        *,
        config_file: str | Path | None = None,
        data_dir: str | Path | None = None,
        metrics_path: str | Path | None = None,
        **values: Any,
    ) -> MediaServiceFacade:
        """Build a facade, defaulting the metrics file to the data directory."""
        if metrics_path is None:
            base = (
                Path(data_dir).expanduser()
                if data_dir is not None
                else resolve_paths(
                    Path(config_file) if config_file is not None else None
                ).data_dir
            )
            metrics_path = base / MEDIA_METRICS_FILENAME
        return super().from_config(
            config,
            config_file=config_file,
            data_dir=data_dir,
            metrics_path=metrics_path,
            **values,
        )

    @classmethod
    def from_stage_context(
        cls,
        context: Any,
        *,
        metrics_path: str | Path | None = None,
        **values: Any,
    ) -> MediaServiceFacade:
        """Build a facade for a stage, writing metrics into its run directory.

        The stage context already carries the router, cost tracker and storage
        the pipeline shares, so a stage never rebuilds providers or a ledger.
        """
        router = getattr(context, "router", None)
        if router is None:
            raise ValueError("a stage context needs a router to build the media facade")
        if metrics_path is None:
            metrics_path = Path(context.run_dir) / MEDIA_METRICS_FILENAME
        return cls(
            router,
            tracker=getattr(context, "cost", None),
            storage=getattr(context, "storage", None),
            metrics_path=metrics_path,
            **values,
        )

    @property
    def metrics_path(self) -> Path | None:
        """The metrics file this facade writes, or ``None`` when disabled."""
        return self._metrics_path

    @property
    def metrics(self) -> MediaMetrics:
        """The live metrics counters."""
        return self._metrics

    def metrics_snapshot(self) -> dict[str, Any]:
        """Return a thread-safe copy of the metrics document."""
        with self._metrics_lock:
            return self._metrics.to_dict()

    def reset_metrics(self) -> None:
        """Clear the counters and rewrite an empty metrics file."""
        with self._metrics_lock:
            self._metrics = MediaMetrics()
            self._write_metrics()

    def _write_metrics(self) -> None:
        path = self._metrics_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.write_text(
                canonical_json(self._metrics.to_dict(), indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        except OSError as exc:  # pragma: no cover - exercised by an unwritable path
            _LOGGER.warning("could not write media metrics to %s: %s", path, exc)

    async def _observe(
        self,
        kind: str,
        call: Callable[..., Awaitable[MediaResult]],
        *args: Any,
        **kwargs: Any,
    ) -> MediaResult:
        started = utcnow()
        try:
            result = await call(*args, **kwargs)
        except BaseException as exc:
            elapsed = (utcnow() - started).total_seconds()
            with self._metrics_lock:
                self._metrics.record_failure(kind, exc, duration_seconds=elapsed)
                self._write_metrics()
            raise
        elapsed = (utcnow() - started).total_seconds()
        with self._metrics_lock:
            self._metrics.record_success(kind, result, duration_seconds=elapsed)
            self._write_metrics()
        return result

    async def image(self, *args: Any, **kwargs: Any) -> MediaResult:
        """Generate images and record the call in the metrics file."""
        return await self._observe("image", super().image, *args, **kwargs)

    async def video(self, *args: Any, **kwargs: Any) -> MediaResult:
        """Generate video and record the call in the metrics file."""
        return await self._observe("video", super().video, *args, **kwargs)

    async def speech(self, *args: Any, **kwargs: Any) -> MediaResult:
        """Synthesize speech and record the call in the metrics file."""
        return await self._observe("speech", super().speech, *args, **kwargs)

    async def transcribe(self, *args: Any, **kwargs: Any) -> MediaResult:
        """Transcribe audio and record the call in the metrics file."""
        return await self._observe("transcription", super().transcribe, *args, **kwargs)

    async def music(self, *args: Any, **kwargs: Any) -> MediaResult:
        """Generate music and record the call in the metrics file."""
        return await self._observe("music", super().music, *args, **kwargs)

    async def sfx(self, *args: Any, **kwargs: Any) -> MediaResult:
        """Generate a sound effect and record the call in the metrics file."""
        return await self._observe("sfx", super().sfx, *args, **kwargs)

    async def edit(self, *args: Any, **kwargs: Any) -> MediaResult:
        """Edit an image and record the call in the metrics file."""
        return await self._observe("edit", super().edit, *args, **kwargs)
