"""Session model: assets, current composition, plan history and versions.

Every session round trips through :meth:`Session.model_dump` and
:meth:`Session.model_validate` without losing information.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from aivia.core.models import CompositionReference, Plan, Session, SessionVersion
from aivia.core.models.asset import Asset, AssetKind


def _asset(label: str, kind: AssetKind = AssetKind.IMAGE) -> Asset:
    return Asset(
        label=label,
        kind=kind,
        path=f"/registry/{label}.png",
        content_hash="h" * 64,
    )


def _plan(plan_id: str, instruction: str = "build") -> Plan:
    return Plan(id=plan_id, instruction=instruction)


def test_session_defaults() -> None:
    session = Session()

    assert session.id.startswith("session_")
    assert session.session_id == session.id
    assert session.title == ""
    assert session.run_id == ""
    assert session.profile == ""
    assert session.assets == []
    assert session.asset_count == 0
    assert isinstance(session.current, CompositionReference)
    assert session.history == []
    assert session.plan_count == 0
    assert session.versions == []
    assert session.version_count == 0
    assert session.latest_version is None
    assert session.active_version == 1
    assert session.schema_version >= 1
    assert isinstance(session.created_at, datetime)
    assert isinstance(session.updated_at, datetime)


def test_session_holds_assets_and_composition() -> None:
    session = Session(
        title="intro",
        assets=[_asset("hero"), _asset("logo", AssetKind.AUDIO)],
        run_id="run_abc",
        profile="default",
    )

    assert session.title == "intro"
    assert session.run_id == "run_abc"
    assert session.profile == "default"
    assert [asset.label for asset in session.assets] == ["hero", "logo"]
    assert session.asset_count == 2
    assert session.assets[0].kind is AssetKind.IMAGE
    assert session.assets[1].kind is AssetKind.AUDIO

    session.current.content_hash = "c" * 64
    session.current.renderer_version = "ffmpeg-7"
    assert session.current.content_hash == "c" * 64
    assert session.current.renderer == "ffmpeg-7"


def test_session_records_plan_history() -> None:
    session = Session(
        history=[
            _plan("plan_1", "first draft"),
            _plan("plan_2", "revised"),
        ]
    )

    assert session.plan_count == 2
    assert [plan.id for plan in session.history] == ["plan_1", "plan_2"]
    assert session.plan_by_id("plan_2") is session.history[1]
    assert session.plan_by_id("missing") is None


def test_session_records_versions() -> None:
    session = Session(
        versions=[
            SessionVersion(version=1, note="initial"),
            SessionVersion(version=3, note="added overlay"),
            SessionVersion(version=2, note="tuned audio"),
        ],
        active_version=3,
    )

    assert session.version_count == 3
    assert session.active_version == 3
    latest = session.latest_version
    assert latest is not None
    assert latest.version == 3
    assert latest.note == "added overlay"
    assert session.version_by_number(2) is not None
    assert session.version_by_number(2).note == "tuned audio"
    assert session.version_by_number(9) is None


def test_session_aliases() -> None:
    session = Session.model_validate(
        {
            "session_id": "session_alias",
            "name": "titled",
            "run": "run_x",
            "profile_name": "prof",
            "current_composition": {"content_hash": "x" * 64},
        }
    )

    assert session.id == "session_alias"
    assert session.title == "titled"
    assert session.run_id == "run_x"
    assert session.profile == "prof"
    assert session.current.content_hash == "x" * 64


def test_session_rejects_invalid_active_version() -> None:
    with pytest.raises(ValidationError):
        Session(active_version=0)


def test_session_coerces_single_asset_plan_and_version() -> None:
    session = Session.model_validate(
        {
            "assets": _asset("solo").model_dump(mode="json"),
            "history": _plan("plan_solo").model_dump(mode="json"),
            "versions": SessionVersion(version=1).model_dump(mode="json"),
        }
    )

    assert session.asset_count == 1
    assert session.plan_count == 1
    assert session.history[0].id == "plan_solo"
    assert session.version_count == 1
    assert session.versions[0].version == 1


def test_session_round_trips_through_model_dump() -> None:
    original = Session(
        title="round trip",
        run_id="run_rt",
        profile="prof",
        assets=[_asset("a"), _asset("b", AssetKind.VIDEO)],
        history=[_plan("p1"), _plan("p2")],
        versions=[
            SessionVersion(version=1, note="start"),
            SessionVersion(version=2, note="edit"),
        ],
        active_version=2,
    )
    original.current.content_hash = "digest" * 8

    dumped = original.model_dump(mode="json")
    restored = Session.model_validate(dumped)

    assert restored == original
    assert restored.model_dump(mode="json") == dumped
    assert restored.id == original.id
    assert restored.title == "round trip"
    assert restored.run_id == "run_rt"
    assert restored.profile == "prof"
    assert [asset.label for asset in restored.assets] == ["a", "b"]
    assert restored.asset_count == 2
    assert [plan.id for plan in restored.history] == ["p1", "p2"]
    assert restored.plan_count == 2
    assert [version.version for version in restored.versions] == [1, 2]
    assert restored.version_count == 2
    assert restored.active_version == 2
    assert restored.latest_version is not None
    assert restored.latest_version.version == 2
    assert restored.current.content_hash == "digest" * 8


def test_session_round_trip_preserves_timestamps_and_schema_version() -> None:
    created = datetime(2026, 3, 5, 8, 30, tzinfo=UTC)
    updated = datetime(2026, 3, 5, 9, 15, tzinfo=UTC)
    original = Session(
        schema_version=3,
        created_at=created,
        updated_at=updated,
    )

    restored = Session.model_validate(original.model_dump(mode="json"))

    assert restored.schema_version == 3
    assert restored.created_at == created
    assert restored.updated_at == updated


def test_session_importable_from_assistant() -> None:
    from aivia.assistant import Session as AssistantSession

    assert AssistantSession is Session


def test_session_version_aliases_and_defaults() -> None:
    version = SessionVersion.model_validate(
        {"revision": 5, "message": "final cut", "labels": ["hero", "logo"]}
    )

    assert version.version == 5
    assert version.revision == 5
    assert version.note == "final cut"
    assert version.asset_labels == ["hero", "logo"]


def test_session_unknown_field_is_kept_and_warned() -> None:
    import warnings

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        session = Session.model_validate({"title": "t", "not_a_real_field": 42})

    assert session.not_a_real_field == 42
    assert any("not_a_real_field" in str(w.message) for w in recorded)
