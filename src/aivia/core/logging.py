"""Structured JSON logging with run, stage and provider context."""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, TextIO

run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
stage: ContextVar[str | None] = ContextVar("stage", default=None)
provider: ContextVar[str | None] = ContextVar("provider", default=None)

CONTEXT_VARS: tuple[tuple[str, ContextVar[str | None]], ...] = (
    ("run_id", run_id),
    ("stage", stage),
    ("provider", provider),
)


class JsonFormatter(logging.Formatter):
    """Render each log record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": _timestamp(record.created),
            "level": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
        }
        for name, variable in CONTEXT_VARS:
            value = variable.get()
            if value is not None:
                payload[name] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _timestamp(created: float) -> str:
    return (
        datetime.fromtimestamp(created, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def get_logger(name: str) -> logging.Logger:
    """Return the logger registered under *name*."""
    return logging.getLogger(name)


def configure_logging(
    level: int | str = logging.INFO,
    *,
    stream: TextIO | None = None,
    logger: logging.Logger | None = None,
) -> logging.Handler:
    """Install a single JSON handler and return it.

    The handler writes to *stream* (stderr by default) and is attached to *logger*
    (the root logger by default). Existing handlers on that logger are replaced so
    output is never duplicated.
    """
    target = logger if logger is not None else logging.getLogger()
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonFormatter())
    target.handlers.clear()
    target.addHandler(handler)
    target.setLevel(level)
    target.propagate = False
    return handler
