"""Sandboxed assistant environment.

A session-scoped sandbox that confines file access to the session root,
renders user text as delimited data blocks (data only, never instructions),
enforces the configured tool safety limits and logs every tool call.

File access is resolved against the session root and rejected when it escapes,
so a hostile instruction cannot read or write outside the session. User text
supplied through :meth:`user` or :meth:`data` is rendered inside escaped,
delimited blocks so it can never be interpreted as instructions. Every tool
call made through :meth:`execute` is recorded and logged, and the wrapped
:class:`~aivia.assistant.tool_framework.ToolFramework` enforces the configured
safety limits.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from aivia.assistant.tool_framework import (
    SafetyLimits,
    ToolEstimate,
    ToolFramework,
    ToolRegistry,
    ToolResult,
)
from aivia.core.logging import get_logger, log_context
from aivia.core.models.base import canonical_json, utcnow
from aivia.core.redact import redact
from aivia.llm.untrusted import (
    UntrustedKind,
    render_data_block,
    wrap_user_text,
)

__all__ = [
    "SandboxError",
    "SandboxPathError",
    "SandboxedEnvironment",
]


class SandboxError(Exception):
    """Base class for every error raised by the sandbox."""

    code: str = "sandbox_error"

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = dict(details) if details else {}


class SandboxPathError(SandboxError):
    """Raised when a file access escapes the session root."""

    code = "sandbox_path_error"


class SandboxedEnvironment:
    """A session-scoped sandbox for assistant tool execution.

    ``root`` is the session directory. Every file access is resolved against
    it and rejected when it escapes, so a hostile instruction cannot read or
    write outside the session. User text supplied through :meth:`user` or
    :meth:`data` is rendered inside escaped, delimited blocks so it can never
    be interpreted as instructions. Every tool call made through
    :meth:`execute` is recorded and logged, and the wrapped
    :class:`~aivia.assistant.tool_framework.ToolFramework` enforces the
    configured safety limits.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        framework: ToolFramework | None = None,
        safety_limits: SafetyLimits | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        if framework is None:
            framework = ToolFramework(ToolRegistry(), safety_limits=safety_limits)
        self._framework = framework
        self._logger = logger or get_logger("aivia.assistant.sandbox")
        self._call_log: list[dict[str, Any]] = []

    # --- properties -------------------------------------------------------

    @property
    def root(self) -> Path:
        """The session directory every file access is confined to."""
        return self._root

    @property
    def framework(self) -> ToolFramework:
        """The tool framework this sandbox executes calls through."""
        return self._framework

    @property
    def safety_limits(self) -> SafetyLimits:
        """The runtime safety limits in force."""
        return self._framework.safety_limits

    @property
    def total_cost_usd(self) -> float:
        """The cumulative cost of executed tool calls so far."""
        return self._framework.total_cost_usd

    @property
    def tool_calls(self) -> int:
        """The number of tool calls executed so far."""
        return self._framework.tool_calls

    @property
    def steps(self) -> int:
        """The number of tool steps executed so far."""
        return self._framework.steps

    @property
    def call_log(self) -> list[dict[str, Any]]:
        """The recorded tool calls, in order, as an audit trail."""
        return list(self._call_log)

    # --- user text as data -----------------------------------------------

    def user(self, text: str, *, source: str | Path | None = None) -> str:
        """Wrap *text* as untrusted user data (data only, never instructions).

        The result is prefixed with a data-only instruction and the text is
        rendered inside an escaped, delimited block, so the content cannot
        close the block or masquerade as a new instruction.
        """
        return wrap_user_text(text, source=source)

    def data(
        self,
        content: str,
        *,
        kind: str | UntrustedKind = UntrustedKind.USER_TEXT,
        source: str | Path | None = None,
    ) -> str:
        """Wrap *content* as a single escaped, delimited data block."""
        return render_data_block(content, kind=kind, source=source)

    # --- session scoped file access --------------------------------------

    def access(self, path: str | Path) -> Path:
        """Resolve *path* inside the session root.

        Returns the absolute, resolved path when it stays inside the session
        root. Raises :class:`SandboxPathError` when the resolved path escapes
        the root (for example ``../../etc/passwd`` or an absolute path
        elsewhere on the filesystem), including escapes through symlinks.
        """
        raw = Path(path)
        candidate = raw if raw.is_absolute() else (self._root / raw)
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self._root):
            raise SandboxPathError(
                f"path escapes the session root: {path!r}",
                details={
                    "path": str(path),
                    "root": str(self._root),
                    "resolved": str(resolved),
                },
            )
        return resolved

    def verified(self, path: str | Path) -> bool:
        """Return whether *path* resolves inside the session root."""
        try:
            self.access(path)
            return True
        except SandboxPathError:
            return False

    def read(self, path: str | Path) -> bytes:
        """Read and return the bytes of *path*, confined to the session root."""
        target = self.access(path)
        if not target.is_file():
            raise FileNotFoundError(f"not a file: {path!r}")
        return target.read_bytes()

    def write(self, path: str | Path, data: bytes | str) -> None:
        """Write *data* to *path*, confined to the session root."""
        target = self.access(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            data = data.encode("utf-8")
        target.write_bytes(data)

    def list_dir(self, path: str | Path = "") -> list[Path]:
        """Return the sorted children of *path* inside the session root."""
        target = self.access(path)
        if not target.is_dir():
            raise NotADirectoryError(f"not a directory: {path!r}")
        return sorted(target.iterdir())

    # --- tool execution ---------------------------------------------------

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Execute *name* with *arguments* within the safety limits.

        The call is enforced against the allowlist and safety limits by the
        wrapped framework, recorded in :attr:`call_log` and logged. Blocked
        calls are recorded as failed before the error is re-raised.
        """
        arguments = dict(arguments or {})
        try:
            result = self._framework.execute(name, arguments)
        except BaseException as error:
            self._record_call(name, arguments, ok=False, error=error)
            self._emit_log(name, arguments, ok=False, error=error)
            raise
        self._record_call(name, arguments, ok=True, result=result)
        self._emit_log(name, arguments, ok=True, result=result)
        return result

    def estimate(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> ToolEstimate:
        """Return the estimate for *name* after enforcing the allowlist."""
        return self._framework.estimate(name, arguments)

    def permits(self, name: str) -> bool:
        """Return whether *name* is registered and allowed by the allowlist."""
        return self._framework.permits(name)

    def reset(self) -> None:
        """Clear the accumulated counters and the call log."""
        self._framework.reset()
        self._call_log.clear()

    # --- logging ----------------------------------------------------------

    def _record_call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        ok: bool,
        result: ToolResult | None = None,
        error: BaseException | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "name": name,
            "arguments": dict(arguments),
            "ok": ok,
            "at": utcnow().isoformat(),
        }
        if ok:
            assert result is not None
            record["cost_usd"] = result.cost_usd
            record["estimated"] = result.estimated
        else:
            assert error is not None
            record["error_code"] = getattr(error, "code", "") or type(error).__name__
            record["error_message"] = str(error)
        self._call_log.append(record)

    def _emit_log(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        ok: bool,
        result: ToolResult | None = None,
        error: BaseException | None = None,
    ) -> None:
        if ok:
            assert result is not None
            message = (
                f"tool {name!r} executed "
                f"args={redact(canonical_json(arguments))} "
                f"cost_usd={result.cost_usd} "
                f"estimated={result.estimated}"
            )
        else:
            assert error is not None
            message = (
                f"tool {name!r} failed "
                f"args={redact(canonical_json(arguments))} "
                f"error={type(error).__name__}: {error!r}"
            )
        with log_context(stage="sandbox.tool"):
            self._logger.info(redact(message))
