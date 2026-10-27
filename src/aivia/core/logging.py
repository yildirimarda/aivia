"""Structured JSON logging with run, stage and provider context."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, TextIO

from aivia.core.redact import redact

run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
stage: ContextVar[str | None] = ContextVar("stage", default=None)
provider: ContextVar[str | None] = ContextVar("provider", default=None)

CONTEXT_VARS: tuple[tuple[str, ContextVar[str | None]], ...] = (
    ("run_id", run_id),
    ("stage", stage),
    ("provider", provider),
)

_CONTEXT_VARIABLES: dict[str, ContextVar[str | None]] = dict(CONTEXT_VARS)


class _Unset:
    """Sentinel type distinguishing "leave unchanged" from an explicit ``None``."""

    __slots__ = ()


_UNSET = _Unset()


class LogContext(
    AbstractContextManager["LogContext"],
    AbstractAsyncContextManager["LogContext"],
):
    """Temporarily set logging context values and restore them on exit.

    The values live in :class:`~contextvars.ContextVar` instances, so they are
    isolated between asyncio tasks and survive ``await`` points inside the task
    that entered the context. Nested contexts only change the values they name
    and restore the previous values afterwards.
    """

    def __init__(self, values: Mapping[str, str | None]) -> None:
        self._values = dict(values)
        self._tokens: list[tuple[str, Token[str | None]]] = []
        self._entered = False

    def __enter__(self) -> LogContext:
        if self._entered:
            raise RuntimeError("log context is already active")
        self._entered = True
        try:
            for name, value in self._values.items():
                token = _CONTEXT_VARIABLES[name].set(value)
                self._tokens.append((name, token))
        except BaseException:
            self._reset()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._reset()

    async def __aenter__(self) -> LogContext:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc_value, traceback)

    def _reset(self) -> None:
        tokens, self._tokens = self._tokens, []
        self._entered = False
        for name, token in reversed(tokens):
            _CONTEXT_VARIABLES[name].reset(token)


def log_context(
    *,
    run_id: str | _Unset | None = _UNSET,
    stage: str | _Unset | None = _UNSET,
    provider: str | _Unset | None = _UNSET,
) -> LogContext:
    """Return a context manager that sets the named logging context values.

    Values that are omitted are left untouched. Passing ``None`` explicitly
    clears a value. The returned object works with both ``with`` and
    ``async with``.
    """
    values: dict[str, str | None] = {}
    if not isinstance(run_id, _Unset):
        values["run_id"] = run_id
    if not isinstance(stage, _Unset):
        values["stage"] = stage
    if not isinstance(provider, _Unset):
        values["provider"] = provider
    return LogContext(values)


def current_context() -> dict[str, str]:
    """Return the logging context values that are currently set."""
    context: dict[str, str] = {}
    for name, variable in CONTEXT_VARS:
        value = variable.get()
        if value is not None:
            context[name] = value
    return context


class JsonFormatter(logging.Formatter):
    """Render each log record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": _timestamp(record.created),
            "level": record.levelname,
            "msg": redact(record.getMessage()),
            "logger": record.name,
        }
        for name, variable in CONTEXT_VARS:
            value = variable.get()
            if value is not None:
                payload[name] = redact(value)
        if record.exc_info:
            payload["exc_info"] = redact(self.formatException(record.exc_info))
        if record.stack_info:
            payload["stack_info"] = redact(self.formatStack(record.stack_info))
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
