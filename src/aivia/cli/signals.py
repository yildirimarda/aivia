"""Turn process stop signals into a worker's graceful stop request.

The worker loop already knows how to drain: :meth:`Worker.request_stop` lets the
in-flight handler finish and claims nothing new. The CLI installs these
handlers so ``SIGINT`` (Ctrl-C) and ``SIGTERM`` (service managers) take that
path instead of killing the process in the middle of a job.
"""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Protocol

__all__ = ["STOP_SIGNALS", "Stoppable", "stop_on_signals"]

STOP_SIGNALS: tuple[signal.Signals, ...] = (signal.SIGINT, signal.SIGTERM)


class Stoppable(Protocol):
    """Anything that exposes the worker's graceful stop hook."""

    def request_stop(self) -> None: ...


@contextmanager
def stop_on_signals(
    worker: Stoppable,
    *,
    loop: asyncio.AbstractEventLoop | None = None,
    signals: Sequence[signal.Signals] = STOP_SIGNALS,
) -> Iterator[None]:
    """Request a graceful stop when one of *signals* arrives.

    The handler that is already running finishes and no new job is claimed.
    The previous handlers are restored when the block exits.
    """
    active = loop if loop is not None else asyncio.get_running_loop()
    previous: dict[signal.Signals, Any] = {}
    try:
        for sig in signals:
            previous[sig] = signal.getsignal(sig)
            active.add_signal_handler(sig, worker.request_stop)
        yield
    finally:
        for sig, old in previous.items():
            active.remove_signal_handler(sig)
            if callable(old) or old in (signal.SIG_DFL, signal.SIG_IGN):
                signal.signal(sig, old)
