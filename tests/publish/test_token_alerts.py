"""Token expiration alerts delivered once per day with a re-auth hint.

Covers the :class:`~aivia.core.token_alerts.TokenAlertTracker` (which token is
due for a warning and that each token fires at most once per day) and the
:class:`~aivia.core.token_alerts.TokenAlertNotifier` (which delivers the warning
with a re-authentication hint through any notifier).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from aivia.core.errors import ConfigError
from aivia.core.token_alerts import (
    DEFAULT_WARN_WINDOW,
    TokenAlert,
    TokenAlertNotifier,
    TokenAlertTracker,
    emit_token_alerts,
)
from aivia.core.token_store import StoredToken
from aivia.providers.console import ConsoleNotifier
from aivia.providers.mock import MockNotifier
from aivia.providers.protocols.notify import (
    NotificationLevel,
    NotificationMessage,
    NotificationReceipt,
    Notifier,
)

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
TOMORROW = NOW + timedelta(days=1)
EPOCH = NOW.timestamp()


def _token(
    platform: str = "youtube",
    account: str = "acct-1",
    *,
    expires_in_seconds: float | None = None,
) -> StoredToken:
    expires_at: float | None = None
    if expires_in_seconds is not None:
        expires_at = EPOCH + expires_in_seconds
    return StoredToken(
        platform=platform,
        account=account,
        encrypted="ciphertext",
        expires_at=expires_at,
        refreshed_at=EPOCH,
    )


def _entries(*tokens: StoredToken) -> dict[tuple[str, str], StoredToken]:
    return {(t.platform, t.account): t for t in tokens}


def test_a_token_expiring_soon_is_flagged_once_per_day() -> None:
    tracker = TokenAlertTracker(warn_within_seconds=3600)
    tokens = _entries(_token(expires_in_seconds=600))

    first = tracker.check(tokens, now=NOW)

    assert [alert.platform for alert in first] == ["youtube"]
    assert first[0].account == "acct-1"
    assert 0 < first[0].seconds_until_expiry <= 3600
    assert first[0].period_key == "2026-06-01"
    assert first[0].period_start == datetime(2026, 6, 1, tzinfo=UTC)
    assert first[0].period_end == datetime(2026, 6, 2, tzinfo=UTC)
    assert tracker.check(tokens, now=NOW) == ()
    assert tracker.check(tokens, now=NOW + timedelta(hours=6)) == ()
    assert tracker.emitted == {("2026-06-01", "youtube", "acct-1")}


def test_a_far_future_token_is_not_flagged() -> None:
    tracker = TokenAlertTracker(warn_within_seconds=3600)
    tokens = _entries(_token(expires_in_seconds=7200))

    assert tracker.check(tokens, now=NOW) == ()
    assert tracker.emitted == frozenset()


def test_a_token_at_exactly_the_window_boundary_is_flagged() -> None:
    tracker = TokenAlertTracker(warn_within_seconds=3600)
    tokens = _entries(_token(expires_in_seconds=3600))

    alerts = tracker.check(tokens, now=NOW)

    assert [alert.platform for alert in alerts] == ["youtube"]
    assert alerts[0].seconds_until_expiry == pytest.approx(3600.0)


def test_an_expired_token_is_not_flagged() -> None:
    tracker = TokenAlertTracker(warn_within_seconds=3600)
    tokens = _entries(_token(expires_in_seconds=-600))

    assert tracker.check(tokens, now=NOW) == ()
    assert tracker.emitted == frozenset()


def test_a_token_without_an_expiry_is_not_flagged() -> None:
    tracker = TokenAlertTracker(warn_within_seconds=3600)
    tokens = _entries(_token(expires_in_seconds=None))

    assert tracker.check(tokens, now=NOW) == ()
    assert tracker.emitted == frozenset()


def test_a_new_day_re_emits_the_same_token() -> None:
    tracker = TokenAlertTracker(warn_within_seconds=2 * 24 * 60 * 60)
    tokens = _entries(_token(expires_in_seconds=1.5 * 24 * 60 * 60))

    assert len(tracker.check(tokens, now=NOW)) == 1
    assert tracker.check(tokens, now=NOW) == ()

    alerts = tracker.check(tokens, now=TOMORROW)

    assert [alert.platform for alert in alerts] == ["youtube"]
    assert {alert.period_key for alert in alerts} == {"2026-06-02"}
    assert tracker.check(tokens, now=TOMORROW) == ()


def test_peek_reports_without_consuming_and_check_arms() -> None:
    tracker = TokenAlertTracker(warn_within_seconds=3600)
    tokens = _entries(_token(expires_in_seconds=600))

    peeked = tracker.peek(tokens, now=NOW)

    assert len(peeked) == 1
    assert tracker.emitted == frozenset()

    checked = tracker.check(tokens, now=NOW)

    assert len(checked) == 1
    assert tracker.check(tokens, now=NOW) == ()

    tracker.reset()

    assert len(tracker.check(tokens, now=NOW)) == 1


def test_multiple_tokens_are_each_flagged_once_per_day() -> None:
    tracker = TokenAlertTracker(warn_within_seconds=3600)
    tokens = _entries(
        _token("youtube", "acct-1", expires_in_seconds=600),
        _token("tiktok", "acct-2", expires_in_seconds=1200),
    )

    alerts = tracker.check(tokens, now=NOW)

    assert {alert.key for alert in alerts} == {
        ("2026-06-01", "youtube", "acct-1"),
        ("2026-06-01", "tiktok", "acct-2"),
    }
    assert tracker.check(tokens, now=NOW) == ()


def test_invalid_warn_within_seconds_fails_clearly() -> None:
    with pytest.raises(ConfigError):
        TokenAlertTracker(warn_within_seconds=-1)
    with pytest.raises(ConfigError):
        TokenAlertTracker(warn_within_seconds=cast(float, True))
    with pytest.raises(ConfigError):
        TokenAlertTracker(warn_within_seconds=cast(float, "3600"))


# ------------------------------------------------------------------ notifier


def test_notifier_sends_one_message_per_due_token() -> None:
    tracker = TokenAlertTracker(warn_within_seconds=3600)
    target = MockNotifier()
    notifier = TokenAlertNotifier(target, tracker=tracker)
    tokens = _entries(_token(expires_in_seconds=600))

    receipts = asyncio.run(notifier.check_and_notify(tokens, now=NOW))

    assert len(receipts) == 1
    assert len(target.calls) == 1
    message = cast(NotificationMessage, target.calls[0].request)
    assert message.title == "Publish token about to expire: youtube"
    assert message.level is NotificationLevel.WARNING
    assert "youtube" in message.text
    assert "acct-1" in message.text
    assert message.metadata["platform"] == "youtube"
    assert message.metadata["account"] == "acct-1"
    assert message.metadata["seconds_until_expiry"] == pytest.approx(600.0)

    assert asyncio.run(notifier.check_and_notify(tokens, now=NOW)) == ()
    assert len(target.calls) == 1


def test_notifier_uses_the_console_when_none_is_configured() -> None:
    notifier = TokenAlertNotifier(warn_within_seconds=3600)

    assert isinstance(notifier.notifier, ConsoleNotifier)
    assert notifier.tracker.warn_within_seconds == pytest.approx(3600.0)


def test_message_includes_the_reauth_hint_text() -> None:
    alert = TokenAlertTracker(warn_within_seconds=3600).check(
        _entries(_token(expires_in_seconds=600)), now=NOW
    )[0]

    message = TokenAlertNotifier.message_for(alert, hint="https://example.com/auth")

    assert "Re-auth hint: https://example.com/auth" in message.text
    assert message.has_buttons
    button = message.buttons[0]
    assert button.label == "Re-authenticate"
    assert button.url == "https://example.com/auth"


def test_message_without_a_hint_has_no_button() -> None:
    alert = TokenAlertTracker(warn_within_seconds=3600).check(
        _entries(_token(expires_in_seconds=600)), now=NOW
    )[0]

    message = TokenAlertNotifier.message_for(alert)

    assert "Re-auth hint" not in message.text
    assert not message.has_buttons


def test_a_non_url_hint_is_text_only() -> None:
    alert = TokenAlertTracker(warn_within_seconds=3600).check(
        _entries(_token(expires_in_seconds=600)), now=NOW
    )[0]

    message = TokenAlertNotifier.message_for(alert, hint="run aivia publish login")

    assert "run aivia publish login" in message.text
    assert not message.has_buttons


def test_reauth_hint_callable_receives_platform_and_account() -> None:
    seen: list[tuple[str, str]] = []

    def hint(platform: str, account: str) -> str | None:
        seen.append((platform, account))
        return f"https://example.com/auth/{platform}/{account}"

    tracker = TokenAlertTracker(warn_within_seconds=3600)
    target = MockNotifier()
    notifier = TokenAlertNotifier(target, tracker=tracker, reauth_hint=hint)
    tokens = _entries(_token(expires_in_seconds=600))

    asyncio.run(notifier.check_and_notify(tokens, now=NOW))

    assert seen == [("youtube", "acct-1")]
    message = cast(NotificationMessage, target.calls[0].request)
    assert "https://example.com/auth/youtube/acct-1" in message.text


def test_notifier_accepts_a_shared_tracker_and_rejects_conflicts() -> None:
    tracker = TokenAlertTracker(warn_within_seconds=3600)
    notifier = TokenAlertNotifier(MockNotifier(), tracker=tracker)

    assert notifier.tracker is tracker
    with pytest.raises(ConfigError):
        TokenAlertNotifier(MockNotifier(), tracker=tracker, state_file="/tmp/x.json")
    with pytest.raises(ConfigError):
        TokenAlertNotifier(MockNotifier(), warn_within_seconds=100, tracker=tracker)
    with pytest.raises(ConfigError):
        TokenAlertNotifier(cast(Notifier, object()))


def test_message_for_rejects_non_alert() -> None:
    with pytest.raises(ConfigError):
        TokenAlertNotifier.message_for(cast(TokenAlert, "alert"))


# ------------------------------------------------------------------ state


def test_emitted_keys_survive_a_restart_with_the_same_day(tmp_path: Path) -> None:
    state = tmp_path / "alerts.json"
    first = TokenAlertTracker(warn_within_seconds=2 * 24 * 60 * 60, state_file=state)
    tokens = _entries(_token(expires_in_seconds=1.5 * 24 * 60 * 60))

    assert len(first.check(tokens, now=NOW)) == 1
    assert state.exists()

    restarted = TokenAlertTracker(
        warn_within_seconds=2 * 24 * 60 * 60, state_file=state
    )

    assert restarted.emitted == first.emitted
    assert restarted.check(tokens, now=NOW) == ()
    assert len(restarted.check(tokens, now=TOMORROW)) == 1


def test_a_check_prunes_stale_keys_from_earlier_days(tmp_path: Path) -> None:
    state = tmp_path / "alerts.json"
    tracker = TokenAlertTracker(warn_within_seconds=3600, state_file=state)
    tokens = _entries(_token(expires_in_seconds=600))

    assert len(tracker.check(tokens, now=NOW)) == 1

    # write a stale key from yesterday directly into the state file
    tracker._emitted.add(("2026-05-31", "youtube", "acct-1"))
    tracker._persist()

    reloaded = TokenAlertTracker(warn_within_seconds=3600, state_file=state)

    assert ("2026-05-31", "youtube", "acct-1") in reloaded.emitted
    reloaded.check(tokens, now=NOW)

    assert ("2026-05-31", "youtube", "acct-1") not in reloaded.emitted


def test_module_helper_emits_through_a_shared_tracker() -> None:
    tracker = TokenAlertTracker(warn_within_seconds=3600)
    target = MockNotifier()
    tokens = _entries(_token(expires_in_seconds=600))

    first = asyncio.run(
        emit_token_alerts(notifier=target, tracker=tracker, tokens=tokens, now=NOW)
    )
    second = asyncio.run(
        emit_token_alerts(notifier=target, tracker=tracker, tokens=tokens, now=NOW)
    )

    assert len(first) == 1
    assert second == ()
    assert len(target.calls) == 1


# ------------------------------------------------------------------ receipts


def test_notify_returns_receipts() -> None:
    tracker = TokenAlertTracker(warn_within_seconds=3600)
    target = MockNotifier()
    notifier = TokenAlertNotifier(target, tracker=tracker)
    tokens = _entries(_token(expires_in_seconds=600))

    receipts = asyncio.run(notifier.notify(tracker.check(tokens, now=NOW)))

    assert len(receipts) == 1
    assert isinstance(receipts[0], NotificationReceipt)
    assert receipts[0].delivered is True


def test_default_warn_window_is_one_week() -> None:
    assert DEFAULT_WARN_WINDOW == 7 * 24 * 60 * 60
