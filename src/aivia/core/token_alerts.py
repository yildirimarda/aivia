"""Token expiration alerts delivered once per day with a re-auth hint.

Publish tokens are stored encrypted in :class:`~aivia.core.token_store.TokenStore`
and carry an optional expiry. When a stored token is about to expire the user
should be warned before the credential stops working, but only once per day so a
long-lived token does not produce a repeating alarm.

This module mirrors :mod:`aivia.core.budget_alerts`: a
:class:`TokenAlertTracker` decides which expiring tokens are due for a warning
and remembers what has already been emitted so the same token is not nagged twice
in one day, while :class:`TokenAlertNotifier` turns those alerts into
:class:`~aivia.providers.protocols.notify.NotificationMessage` objects and sends
them through any notifier, falling back to
:class:`~aivia.providers.console.ConsoleNotifier`.

State is persisted as JSON so a restart does not repeat an alert inside the same
day, and writes are serialised across processes with a sibling lock file.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, BinaryIO, cast

from aivia.core import paths
from aivia.core.errors import ConfigError
from aivia.core.models.base import utcnow
from aivia.core.token_store import StoredToken
from aivia.providers.console import (
    ConsoleNotifier,
    default_notifier,
)
from aivia.providers.protocols.notify import (
    NotificationButton,
    NotificationLevel,
    NotificationMessage,
    NotificationReceipt,
    Notifier,
)

__all__ = [
    "DEFAULT_TOKEN_ALERT_STATE_FILENAME",
    "DEFAULT_WARN_WINDOW",
    "TokenAlert",
    "TokenAlertNotifier",
    "TokenAlertTracker",
    "default_notifier",
    "emit_token_alerts",
    "token_alert_state_path",
]

DEFAULT_TOKEN_ALERT_STATE_FILENAME = "token_alerts.json"
"""The persisted token alert state file inside the user data directory."""

DEFAULT_WARN_WINDOW = 7 * 24 * 60 * 60
"""Warn when a token expires within this many seconds (one week by default)."""

_STATE_VERSION = 1
"""The on-disk state format version written by this module."""


def _coerce_warn_within(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            "token alert warn_within_seconds must be a finite, non-negative number",
            details={"warn_within_seconds": value},
        )
    amount = float(value)
    if not math.isfinite(amount) or amount < 0:
        raise ConfigError(
            "token alert warn_within_seconds must be a finite, non-negative number",
            details={"warn_within_seconds": value},
        )
    return amount


def _coerce_now(value: object) -> datetime:
    if value is None:
        return utcnow()
    if not isinstance(value, datetime):
        raise ConfigError(
            "token alert timestamp must be a datetime",
            details={"now": value},
        )
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _period_window(now: datetime) -> tuple[datetime, datetime, str]:
    """Return the half-open UTC day window and stable key for *now*."""
    start = datetime.combine(now.date(), time.min, tzinfo=UTC)
    return start, start + timedelta(days=1), now.date().isoformat()


def token_alert_state_path(
    data_dir: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    filename: str = DEFAULT_TOKEN_ALERT_STATE_FILENAME,
) -> Path:
    """Return the default persisted token alert state file."""
    root = paths.data_dir(environ) if data_dir is None else Path(data_dir)
    return root.expanduser() / filename


def _coerce_state_file(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise ConfigError(
            "token alert state_file must be a path",
            details={"state_file": value},
        )
    return Path(value).expanduser()


def _load_state(path: Path) -> tuple[set[tuple[str, str, str]], str | None]:
    """Read the emitted tokens and latest period from *path*."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return set(), None
    except OSError as exc:
        raise ConfigError(
            f"cannot read token alert state: {exc}",
            details={"state_file": str(path)},
        ) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"token alert state is not valid JSON: {exc}",
            details={"state_file": str(path)},
        ) from exc
    if not isinstance(payload, Mapping):
        raise ConfigError(
            "token alert state must be a JSON object",
            details={"state_file": str(path)},
        )
    version = payload.get("version", _STATE_VERSION)
    if version != _STATE_VERSION:
        raise ConfigError(
            f"unsupported token alert state version: {version!r}",
            details={"state_file": str(path), "version": version},
        )
    entries = payload.get("emitted", [])
    if not isinstance(entries, list):
        raise ConfigError(
            "token alert state emitted must be a list",
            details={"state_file": str(path)},
        )
    emitted: set[tuple[str, str, str]] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ConfigError(
                "token alert state entries must be objects",
                details={"state_file": str(path), "entry": entry},
            )
        period_key = entry.get("period_key")
        if not isinstance(period_key, str) or not period_key:
            raise ConfigError(
                "token alert state entries need a period_key",
                details={"state_file": str(path), "entry": entry},
            )
        platform = entry.get("platform")
        if not isinstance(platform, str) or not platform:
            raise ConfigError(
                "token alert state entries need a platform",
                details={"state_file": str(path), "entry": entry},
            )
        account = entry.get("account")
        if not isinstance(account, str) or not account:
            raise ConfigError(
                "token alert state entries need an account",
                details={"state_file": str(path), "entry": entry},
            )
        emitted.add((period_key, platform, account))
    latest = payload.get("latest_period_key")
    if latest is not None and (not isinstance(latest, str) or not latest):
        raise ConfigError(
            "token alert state latest_period_key must be a non-empty string",
            details={"state_file": str(path), "latest_period_key": latest},
        )
    return emitted, latest


def _write_emitted(
    path: Path,
    emitted: Iterable[tuple[str, str, str]],
    *,
    latest_period_key: str | None = None,
) -> None:
    """Atomically replace *path* with the emitted tokens as canonical JSON."""
    payload = {
        "version": _STATE_VERSION,
        "latest_period_key": latest_period_key,
        "emitted": [
            {
                "period_key": period_key,
                "platform": platform,
                "account": account,
            }
            for period_key, platform, account in sorted(emitted)
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
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
            temporary.unlink(missing_ok=True)
        raise


def _lock_handle(handle: BinaryIO, *, unlock: bool) -> None:
    """Take or release an exclusive advisory lock on *handle*."""
    if os.name == "nt":
        import importlib

        msvcrt: Any = importlib.import_module("msvcrt")
        handle.seek(0)
        msvcrt.locking(
            handle.fileno(),
            msvcrt.LK_UNLCK if unlock else msvcrt.LK_LOCK,
            1,
        )
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN if unlock else fcntl.LOCK_EX)


@contextmanager
def _state_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive inter-process lock for a state file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+b") as handle:
        _lock_handle(handle, unlock=False)
        try:
            yield
        finally:
            _lock_handle(handle, unlock=True)


@dataclass(frozen=True, slots=True)
class TokenAlert:
    """One publish token due for an expiration warning.

    ``seconds_until_expiry`` is the time left before the token expires when the
    alert was raised. ``key`` is the identity used to emit this alert once per
    day and can be persisted with :attr:`TokenAlertTracker.emitted`.
    """

    platform: str
    account: str
    expires_at: float
    seconds_until_expiry: float
    period_key: str
    period_start: datetime
    period_end: datetime

    @property
    def key(self) -> tuple[str, str, str]:
        """The identity used to emit this alert once per day."""
        return (self.period_key, self.platform, self.account)

    @property
    def expires_at_iso(self) -> str:
        """The expiry instant as an ISO-8601 UTC string."""
        return datetime.fromtimestamp(self.expires_at, tz=UTC).isoformat()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for logs and messages."""
        return {
            "platform": self.platform,
            "account": self.account,
            "expires_at": self.expires_at,
            "expires_at_iso": self.expires_at_iso,
            "seconds_until_expiry": self.seconds_until_expiry,
            "period_key": self.period_key,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
        }


class TokenAlertTracker:
    """Return each expiring token once per calendar day.

    Pass a mapping of ``(platform, account) -> StoredToken`` (as returned by
    :meth:`TokenStore.entries`) together with a ``warn_within_seconds`` window.
    :meth:`check` returns the tokens whose expiry falls inside that window and
    marks them as emitted for the day, so a subsequent call does not repeat the
    warning. :meth:`peek` only reports what is due without marking anything.

    Pass ``state_file`` (or its ``path`` alias) to persist the emitted tokens as
    JSON and load them again on the next process. Without a state file the
    tracker is purely in-memory. Writes are serialised across processes with a
    sibling lock file, and every check merges what other workers have written.
    A check also drops tokens from earlier days, so the state file only ever
    holds the current day's keys and stays bounded.
    """

    def __init__(
        self,
        tokens: Mapping[tuple[str, str], StoredToken] | None = None,
        *,
        warn_within_seconds: float | None = None,
        state_file: str | Path | None = None,
        path: str | Path | None = None,
        clock: Any = None,
    ) -> None:
        if state_file is not None and path is not None:
            raise ConfigError("pass either state_file or path, not both")
        import time as _time

        self._tokens = dict(tokens or {})
        self._warn_within = (
            DEFAULT_WARN_WINDOW
            if warn_within_seconds is None
            else _coerce_warn_within(warn_within_seconds)
        )
        self._state_file = _coerce_state_file(
            state_file if state_file is not None else path
        )
        self._lock = RLock()
        self._clock = clock if clock is not None else _time.time
        self._latest_period_key: str | None = None
        self._emitted: set[tuple[str, str, str]] = set()
        if self._state_file is not None:
            emitted, latest = _load_state(self._state_file)
            self._emitted.update(emitted)
            self._latest_period_key = latest

    @property
    def warn_within_seconds(self) -> float:
        """How far ahead of expiry a token counts as about to expire."""
        return self._warn_within

    @property
    def state_file(self) -> Path | None:
        """The file emitted tokens are loaded from and saved to, if any."""
        return self._state_file

    @property
    def emitted(self) -> frozenset[tuple[str, str, str]]:
        """Every ``(period_key, platform, account)`` pair already emitted."""
        with self._lock:
            return frozenset(self._emitted)

    def reset(self) -> None:
        """Forget every emitted token, so the next check may fire again."""
        with self._lock:
            if self._state_file is None:
                self._emitted.clear()
                self._latest_period_key = None
                return
            with _state_lock(self._state_file):
                self._emitted.clear()
                self._latest_period_key = None
                self._persist()

    def _persist(self) -> None:
        if self._state_file is not None:
            _write_emitted(
                self._state_file,
                self._emitted,
                latest_period_key=self._latest_period_key,
            )

    def _due_locked(
        self,
        now_ts: float,
        start: datetime,
        end: datetime,
        key: str,
        *,
        mark: bool,
    ) -> tuple[TokenAlert, ...]:
        """Compute due alerts while the caller holds every required lock."""
        if self._latest_period_key is not None and key < self._latest_period_key:
            return ()
        due: list[TokenAlert] = []
        changed = False
        if mark:
            stale = {token for token in self._emitted if token[0] != key}
            if stale:
                self._emitted.difference_update(stale)
                changed = True
            if self._latest_period_key != key:
                self._latest_period_key = key
                changed = True
        for (platform, account), stored in self._tokens.items():
            if stored.expires_at is None:
                continue
            expires_at = float(stored.expires_at)
            if not (now_ts < expires_at):
                continue
            seconds_until_expiry = expires_at - now_ts
            if seconds_until_expiry > self._warn_within:
                continue
            token = (key, platform, account)
            if token in self._emitted:
                continue
            due.append(
                TokenAlert(
                    platform=platform,
                    account=account,
                    expires_at=expires_at,
                    seconds_until_expiry=seconds_until_expiry,
                    period_key=key,
                    period_start=start,
                    period_end=end,
                )
            )
            if mark:
                self._emitted.add(token)
                changed = True
        if changed:
            self._persist()
        return tuple(due)

    def _due(
        self,
        now: datetime | None,
        *,
        mark: bool,
    ) -> tuple[TokenAlert, ...]:
        now_dt = _coerce_now(now)
        now_ts = now_dt.timestamp()
        start, end, key = _period_window(now_dt)
        with self._lock:
            if self._state_file is None:
                return self._due_locked(now_ts, start, end, key, mark=mark)
            with _state_lock(self._state_file):
                emitted, latest = _load_state(self._state_file)
                self._emitted.update(emitted)
                if latest is not None and (
                    self._latest_period_key is None or latest > self._latest_period_key
                ):
                    self._latest_period_key = latest
                return self._due_locked(now_ts, start, end, key, mark=mark)

    def peek(
        self,
        tokens: Mapping[tuple[str, str], StoredToken] | None = None,
        *,
        now: datetime | None = None,
    ) -> tuple[TokenAlert, ...]:
        """Return the tokens due now without marking them emitted."""
        if tokens is not None:
            self._tokens = dict(tokens)
        return self._due(now, mark=False)

    def check(
        self,
        tokens: Mapping[tuple[str, str], StoredToken] | None = None,
        *,
        now: datetime | None = None,
    ) -> tuple[TokenAlert, ...]:
        """Return the tokens due now and mark them emitted for the day."""
        if tokens is not None:
            self._tokens = dict(tokens)
        return self._due(now, mark=True)

    def due(
        self,
        tokens: Mapping[tuple[str, str], StoredToken] | None = None,
        *,
        now: datetime | None = None,
    ) -> tuple[TokenAlert, ...]:
        """Alias for :meth:`check`."""
        return self.check(tokens, now=now)


TokenAlertState = TokenAlertTracker
"""Alias for :class:`TokenAlertTracker` for callers that prefer state wording."""


class TokenAlertNotifier:
    """Turn due token alerts into notifications and send them.

    ``notifier`` may be any object exposing the notifier ``send`` coroutine. When
    it is omitted, the console notifier is used. ``reauth_hint`` may be a string
    or a callable ``(platform, account) -> str | None`` that produces a
    re-authentication hint for one token; when present it is included in the
    message as text and, when it is a URL, as an action button.

    The same :class:`TokenAlertTracker` instance is reused across calls, which is
    what makes each token fire once per day.
    """

    def __init__(
        self,
        notifier: Notifier | None = None,
        *,
        tracker: TokenAlertTracker | None = None,
        state_file: str | Path | None = None,
        path: str | Path | None = None,
        warn_within_seconds: float | None = None,
        reauth_hint: Any = None,
    ) -> None:
        if tracker is not None and (
            state_file is not None
            or path is not None
            or warn_within_seconds is not None
        ):
            raise ConfigError("pass either a tracker or alert options, not both")
        if tracker is not None:
            self._tracker = tracker
        else:
            self._tracker = TokenAlertTracker(
                state_file=state_file,
                path=path,
                warn_within_seconds=warn_within_seconds,
            )
        self._notifier = notifier if notifier is not None else ConsoleNotifier()
        send = getattr(self._notifier, "send", None)
        if not callable(send):
            raise ConfigError(
                "token alert notifier must expose a send coroutine",
                details={"notifier": type(self._notifier).__name__},
            )
        self._reauth_hint = reauth_hint

    @property
    def tracker(self) -> TokenAlertTracker:
        """The tracker that enforces once-per-token-per-day."""
        return self._tracker

    @property
    def notifier(self) -> Notifier:
        """The notifier alerts are delivered through."""
        return self._notifier

    def _resolve_hint(self, platform: str, account: str) -> str | None:
        hint = self._reauth_hint
        if hint is None:
            return None
        if callable(hint):
            return cast(str | None, hint(platform, account))
        return str(hint)

    @staticmethod
    def message_for(alert: TokenAlert, hint: str | None = None) -> NotificationMessage:
        """Build the warning message for one alert."""
        if not isinstance(alert, TokenAlert):
            raise ConfigError(
                "token alert notifier needs a TokenAlert",
                details={"alert": type(alert).__name__},
            )
        expires = alert.expires_at_iso
        seconds = alert.seconds_until_expiry
        days, rem = divmod(int(seconds), 86400)
        hours, rem = divmod(rem, 3600)
        minutes, secs = divmod(rem, 60)
        parts: list[str] = []
        if days:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if not parts or secs:
            parts.append(f"{secs} second{'s' if secs != 1 else ''}")
        remaining = ", ".join(parts)
        text = (
            f"The {alert.platform} publish token for account {alert.account} "
            f"expires in {remaining} (at {expires}). "
            "Re-authenticate the account so publishing keeps working."
        )
        buttons: list[NotificationButton] = []
        if hint is not None:
            text += f" Re-auth hint: {hint}"
            if hint.startswith(("http://", "https://")):
                buttons.append(NotificationButton(label="Re-authenticate", url=hint))
        return NotificationMessage(
            title=f"Publish token about to expire: {alert.platform}",
            text=text,
            level=NotificationLevel.WARNING,
            buttons=buttons,
            metadata=alert.to_dict(),
        )

    async def notify(
        self,
        alerts: Iterable[TokenAlert],
        *,
        hint: Any = None,
    ) -> tuple[NotificationReceipt, ...]:
        """Send one message per alert and return every receipt."""
        receipts: list[NotificationReceipt] = []
        for alert in alerts:
            resolved = hint
            if resolved is None:
                resolved = self._resolve_hint(alert.platform, alert.account)
            receipts.append(
                await self._notifier.send(self.message_for(alert, resolved))
            )
        return tuple(receipts)

    async def check_and_notify(
        self,
        tokens: Mapping[tuple[str, str], StoredToken] | None = None,
        *,
        now: datetime | None = None,
        hint: Any = None,
    ) -> tuple[NotificationReceipt, ...]:
        """Check *tokens* for expiring credentials and notify once per day."""
        alerts = self._tracker.check(tokens, now=now)
        return await self.notify(alerts, hint=hint)

    async def emit(
        self,
        tokens: Mapping[tuple[str, str], StoredToken] | None = None,
        *,
        now: datetime | None = None,
        hint: Any = None,
    ) -> tuple[NotificationReceipt, ...]:
        """Alias for :meth:`check_and_notify`."""
        return await self.check_and_notify(tokens, now=now, hint=hint)

    async def send(
        self,
        tokens: Mapping[tuple[str, str], StoredToken] | None = None,
        *,
        now: datetime | None = None,
        hint: Any = None,
    ) -> tuple[NotificationReceipt, ...]:
        """Alias for :meth:`check_and_notify`."""
        return await self.check_and_notify(tokens, now=now, hint=hint)


async def emit_token_alerts(
    tokens: Mapping[tuple[str, str], StoredToken],
    *,
    notifier: Notifier | None = None,
    tracker: TokenAlertTracker | None = None,
    state_file: str | Path | None = None,
    path: str | Path | None = None,
    warn_within_seconds: float | None = None,
    reauth_hint: Any = None,
    now: datetime | None = None,
    hint: Any = None,
) -> tuple[NotificationReceipt, ...]:
    """Emit the token alerts due for *tokens* without a long-lived object."""
    return await TokenAlertNotifier(
        notifier,
        tracker=tracker,
        state_file=state_file,
        path=path,
        warn_within_seconds=warn_within_seconds,
        reauth_hint=reauth_hint,
    ).check_and_notify(tokens, now=now, hint=hint)
