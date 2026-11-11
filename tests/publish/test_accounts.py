"""Publish accounts: credentials keyed by platform/language plus daily slots.

Covers mapping a profile, platform and language to the credentials an account
needs, and applying daily posting slots with jitter and deferral notices.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from aivia.core.config.schedule import PublishConfig
from aivia.core.errors import ConfigError
from aivia.publish.accounts import (
    AccountCredentials,
    AccountMap,
    DeferralNotice,
    PostingSlot,
    SlotPlan,
    apply_jitter,
)


def _publish(**accounts: object) -> PublishConfig:
    return PublishConfig.from_mapping({"accounts": dict(accounts)})


# ---------------------------------------------------------------------------
# AccountMap: profile, platform and language -> credentials
# ---------------------------------------------------------------------------


def test_resolve_maps_platform_and_language_to_credentials() -> None:
    publish = _publish(
        **{
            "youtube/en": {
                "profile": "default",
                "credential_refs": ["YOUTUBE_TOKEN", "YOUTUBE_REFRESH"],
            },
            "telegram/tr": {"credential_ref": "TELEGRAM_TOKEN"},
        }
    )
    accounts = AccountMap(publish)

    youtube = accounts.resolve("youtube", "en")
    assert youtube == AccountCredentials(
        platform="youtube",
        language="en",
        profile="default",
        credential_refs=("YOUTUBE_TOKEN", "YOUTUBE_REFRESH"),
    )
    assert youtube.key == "youtube/en"

    telegram = accounts.resolve("telegram", "tr")
    assert telegram.credential_refs == ("TELEGRAM_TOKEN",)
    assert telegram.profile == ""


def test_credential_ref_accepts_a_single_reference() -> None:
    publish = _publish(**{"x/en": {"credential_ref": "X_TOKEN"}})
    accounts = AccountMap(publish)
    assert accounts.resolve("x", "en").credential_refs == ("X_TOKEN",)


def test_resolve_validates_the_requested_profile() -> None:
    publish = _publish(
        **{"youtube/en": {"profile": "default", "credential_refs": ["T"]}}
    )
    accounts = AccountMap(publish)

    accounts.resolve("youtube", "en", profile="default")

    with pytest.raises(ConfigError) as caught:
        accounts.resolve("youtube", "en", profile="other")
    assert "bound to profile" in str(caught.value)
    assert caught.value.details["account_profile"] == "default"


def test_resolve_rejects_a_missing_account() -> None:
    accounts = AccountMap(_publish(**{"youtube/en": {"credential_refs": ["T"]}}))

    with pytest.raises(ConfigError) as caught:
        accounts.resolve("tiktok", "en")
    assert "no publish account configured for tiktok/en" in str(caught.value)
    assert caught.value.details["platform"] == "tiktok"
    assert caught.value.details["language"] == "en"


def test_resolve_rejects_a_disabled_account() -> None:
    publish = _publish(**{"telegram/tr": {"enabled": False, "credential_refs": ["T"]}})
    accounts = AccountMap(publish)

    with pytest.raises(ConfigError) as caught:
        accounts.resolve("telegram", "tr")
    assert "is disabled" in str(caught.value)


def test_by_profile_lists_accounts_bound_to_a_profile() -> None:
    publish = _publish(
        **{
            "youtube/en": {"profile": "default", "credential_refs": ["YT"]},
            "tiktok/en": {"profile": "shorts", "credential_refs": ["TK"]},
            "youtube/tr": {"profile": "default", "credential_refs": ["YT_TR"]},
        }
    )
    accounts = AccountMap(publish)

    defaults = accounts.by_profile("default")
    assert [c.key for c in defaults] == ["youtube/en", "youtube/tr"]
    assert defaults[0].credential_refs == ("YT",)
    assert defaults[1].credential_refs == ("YT_TR",)

    assert accounts.by_profile("shorts")[0].key == "tiktok/en"
    assert accounts.by_profile("missing") == []


def test_names_and_accounts_expose_the_configured_keys() -> None:
    publish = _publish(**{"youtube/en": {}, "telegram/tr": {}})
    accounts = AccountMap(publish)

    assert accounts.names() == ("youtube/en", "telegram/tr")
    assert set(accounts.accounts) == {"youtube/en", "telegram/tr"}


def test_account_key_matches_platform_and_language() -> None:
    publish = PublishConfig.from_mapping(
        {
            "accounts": {
                "youtube/en": {"platform": "youtube", "language": "en"},
            }
        }
    )
    accounts = AccountMap(publish)
    creds = accounts.resolve("youtube", "en")
    assert creds.platform == "youtube"
    assert creds.language == "en"


# ---------------------------------------------------------------------------
# PostingSlot parsing and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, hour, minute",
    [
        ("09:00", 9, 0),
        ("9:00", 9, 0),
        ("9", 9, 0),
        ("00:00", 0, 0),
        ("23:59", 23, 59),
        ("  14:30  ", 14, 30),
    ],
)
def test_posting_slot_parse(text: str, hour: int, minute: int) -> None:
    slot = PostingSlot.parse(text)
    assert slot.hour == hour
    assert slot.minute == minute
    assert str(slot) == f"{hour:02d}:{minute:02d}"


@pytest.mark.parametrize("text", ["", "   ", "25:00", "9:60", "-1:00", "9:-1"])
def test_posting_slot_rejects_invalid_values(text: str) -> None:
    with pytest.raises(ValueError):
        PostingSlot.parse(text)


def test_posting_slot_as_time() -> None:
    assert PostingSlot(17, 30).as_time().hour == 17
    assert PostingSlot(17, 30).as_time().minute == 30


# ---------------------------------------------------------------------------
# SlotPlan: daily posting slots with jitter
# ---------------------------------------------------------------------------


def test_posting_times_apply_jitter_deterministically() -> None:
    plan = SlotPlan(
        date=date(2026, 1, 15),
        slots=(PostingSlot(9, 0), PostingSlot(17, 0)),
        jitter_seconds=60,
        timezone="UTC",
    )
    # rng = 0.0 -> full negative offset (-60s), rng = 1.0 -> full positive (+60s)
    early = plan.posting_times(rng=lambda: 0.0)
    late = plan.posting_times(rng=lambda: 1.0)
    assert early[0] == datetime(2026, 1, 15, 8, 59, 0, tzinfo=UTC)
    assert early[1] == datetime(2026, 1, 15, 16, 59, 0, tzinfo=UTC)
    assert late[0] == datetime(2026, 1, 15, 9, 1, 0, tzinfo=UTC)
    assert late[1] == datetime(2026, 1, 15, 17, 1, 0, tzinfo=UTC)


def test_posting_times_without_jitter_return_the_slot_times() -> None:
    plan = SlotPlan(
        date=date(2026, 1, 15),
        slots=(PostingSlot(9, 0), PostingSlot(17, 0)),
        jitter_seconds=0,
        timezone="UTC",
    )
    times = plan.posting_times()
    assert times == [
        datetime(2026, 1, 15, 9, 0, tzinfo=UTC),
        datetime(2026, 1, 15, 17, 0, tzinfo=UTC),
    ]


def test_posting_times_honour_the_configured_timezone() -> None:
    plan = SlotPlan(
        date=date(2026, 1, 15),
        slots=(PostingSlot(9, 0),),
        jitter_seconds=0,
        timezone="Europe/Berlin",
    )
    times = plan.posting_times()
    assert times[0].tzinfo is not None
    assert times[0].utcoffset() is not None


def test_apply_jitter_shifts_within_the_window() -> None:
    base = datetime(2026, 1, 15, 9, 0, tzinfo=UTC)
    assert apply_jitter(base, 0) == base
    assert apply_jitter(base, 60, rng=lambda: 0.0) == datetime(
        2026, 1, 15, 8, 59, 0, tzinfo=UTC
    )
    assert apply_jitter(base, 60, rng=lambda: 0.5) == base
    assert apply_jitter(base, 60, rng=lambda: 1.0) == datetime(
        2026, 1, 15, 9, 1, 0, tzinfo=UTC
    )


# ---------------------------------------------------------------------------
# Deferral notices
# ---------------------------------------------------------------------------


def test_defer_moves_a_slot_to_the_next_later_slot_and_records_a_notice() -> None:
    plan = SlotPlan(
        date=date(2026, 1, 15),
        slots=(PostingSlot(9, 0), PostingSlot(17, 0)),
        timezone="UTC",
    )
    new_plan, notice = plan.defer(PostingSlot(9, 0), "rate limited", "youtube/en")

    assert isinstance(notice, DeferralNotice)
    assert notice.slot == PostingSlot(9, 0)
    assert notice.deferred_to == datetime(2026, 1, 15, 17, 0, tzinfo=UTC)
    assert notice.reason == "rate limited"
    assert notice.target == "youtube/en"
    assert new_plan.notices == (notice,)
    # the original plan is unchanged
    assert plan.notices == ()


def test_defer_wraps_the_last_slot_to_the_following_day() -> None:
    plan = SlotPlan(
        date=date(2026, 1, 15),
        slots=(PostingSlot(23, 0),),
        timezone="UTC",
    )
    new_plan, notice = plan.defer(PostingSlot(23, 0), "account asleep", "telegram/tr")

    assert notice.deferred_to == datetime(2026, 1, 16, 23, 0, tzinfo=UTC)
    assert len(new_plan.notices) == 1


def test_defer_rejects_a_slot_not_in_the_plan() -> None:
    plan = SlotPlan(date=date(2026, 1, 15), slots=(PostingSlot(9, 0),))
    with pytest.raises(ValueError, match="not part of this plan"):
        plan.defer(PostingSlot(17, 0), "nope", "x/en")


def test_defer_accumulates_notices_without_changing_slots() -> None:
    plan = SlotPlan(
        date=date(2026, 1, 15),
        slots=(PostingSlot(9, 0), PostingSlot(17, 0)),
        timezone="UTC",
    )
    plan, _ = plan.defer(PostingSlot(9, 0), "first", "youtube/en")
    plan, _ = plan.defer(PostingSlot(17, 0), "second", "tiktok/en")

    assert len(plan.notices) == 2
    assert plan.notices[1].reason == "second"
    assert plan.notices[1].deferred_to == datetime(2026, 1, 16, 9, 0, tzinfo=UTC)
    assert plan.slots == (PostingSlot(9, 0), PostingSlot(17, 0))
