"""Post, delivery, plan and composition reference contracts."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from aivia.core.ids import is_ulid
from aivia.core.models import (
    Attestation,
    AttestationModel,
    Composition,
    CompositionRef,
    CompositionReference,
    Delivery,
    DeliveryMethod,
    DeliveryRecord,
    DeliveryState,
    DeliveryStatus,
    DisclosureAttestation,
    Plan,
    PlanApprovalMode,
    PlanRecord,
    PlanState,
    PlanStatus,
    PlanStep,
    PlanStepState,
    Post,
    PostRecord,
    PostState,
    PostStatus,
    PublishAttestation,
    RightsAttestation,
    ToolchainFingerprint,
    UnknownFieldWarning,
)

_NOW = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
_SHA = "a" * 64


def _attestation(**overrides: object) -> Attestation:
    values: dict[str, object] = {
        "confirmed_by": "ada",
        "rights_confirmed": True,
        "disclosure_confirmed": True,
        "confirmed_at": _NOW,
        "disclosure_text": "AI generated",
        "license_summary": "CC BY 4.0",
    }
    values.update(overrides)
    return Attestation.model_validate(values)


def test_attestation_defaults_to_empty_and_incomplete() -> None:
    attestation = Attestation()

    assert attestation.confirmed_by == ""
    assert attestation.rights_confirmed is False
    assert attestation.disclosure_confirmed is False
    assert attestation.confirmed_at is None
    assert attestation.disclosure_text == ""
    assert attestation.license_summary == ""
    assert attestation.strict is False
    assert attestation.reason == ""
    assert attestation.is_complete is False
    assert attestation.confirmed is False
    assert attestation.is_blocking is False
    assert attestation.missing == [
        "confirmed_by",
        "rights_confirmed",
        "disclosure_confirmed",
        "confirmed_at",
        "disclosure_text",
        "license_summary",
    ]


def test_attestation_records_who_rights_disclosure_when_text_and_license() -> None:
    attestation = _attestation()

    assert attestation.confirmed_by == attestation.who == "ada"
    assert attestation.rights_confirmed is True
    assert attestation.disclosure_confirmed is True
    assert attestation.confirmed_at == attestation.when == _NOW
    assert attestation.disclosure_text == attestation.disclosure == "AI generated"
    assert attestation.license_summary == "CC BY 4.0"
    assert attestation.missing == []
    assert attestation.is_complete is True
    assert attestation.confirmed is True
    assert attestation.is_blocking is False


def test_attestation_accepts_aliases_and_a_short_confirmed_flag() -> None:
    attestation = Attestation.model_validate(
        {
            "who": "  ada  ",
            "confirmed": True,
            "when": _NOW,
            "disclosure": "  AI generated  ",
            "license": {"name": "CC0", "commercial_use": True},
            "strict_mode": True,
            "skip_reason": "  module absent  ",
        }
    )

    assert attestation.confirmed_by == "ada"
    assert attestation.rights_confirmed is True
    assert attestation.disclosure_confirmed is True
    assert attestation.confirmed_at == _NOW
    assert attestation.disclosure_text == "AI generated"
    assert attestation.license_summary == {"name": "CC0", "commercial_use": True}
    assert attestation.strict is True
    assert attestation.reason == "module absent"
    assert attestation.is_complete is True


def test_strict_attestation_blocks_only_when_incomplete() -> None:
    assert Attestation(strict=True).is_blocking is True
    values = _attestation().model_dump()
    values["strict"] = True
    assert Attestation(**values).is_blocking is False


def test_attestation_normalizes_naive_timestamps_to_utc() -> None:
    naive = datetime(2026, 10, 28, 12, 0)

    assert Attestation(confirmed_at=naive).confirmed_at == _NOW


def test_post_defaults_to_a_draft_with_no_media_or_attestation() -> None:
    post = Post()

    assert post.id.startswith("post_")
    assert is_ulid(post.id.removeprefix("post_"))
    assert post.state is PostState.DRAFT
    assert post.platform == ""
    assert post.account == ""
    assert post.caption == ""
    assert post.hashtags == []
    assert post.media == []
    assert post.remote_id == ""
    assert post.url == ""
    assert post.scheduled_for is None
    assert post.published_at is None
    assert post.dry_run is False
    assert post.attestation.is_complete is False
    assert post.has_media is False
    assert post.is_published is False
    assert post.is_terminal is False
    assert post.attestation_complete is False


def test_post_ids_are_unique_and_time_sortable() -> None:
    first = Post()
    second = Post()

    assert first.id != second.id
    assert first.id < second.id


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("draft", PostState.DRAFT),
        ("pending", PostState.PENDING),
        ("queued", PostState.PENDING),
        ("  SCHEDULED  ", PostState.SCHEDULED),
        ("uploading", PostState.PUBLISHING),
        ("posted", PostState.PUBLISHED),
        ("succeeded", PostState.PUBLISHED),
        ("error", PostState.FAILED),
        ("canceled", PostState.CANCELLED),
    ],
)
def test_post_state_accepts_aliases_and_is_normalized(
    raw: str, expected: PostState
) -> None:
    assert Post.model_validate({"state": raw}).state is expected


def test_post_rejects_an_unknown_state() -> None:
    with pytest.raises(ValidationError):
        Post.model_validate({"state": "teleporting"})


def test_post_carries_platform_media_and_platform_ids() -> None:
    post = Post.model_validate(
        {
            "post_id": "post_placeholder",
            "platform_name": "  youtube  ",
            "account_id": "  youtube/en  ",
            "profile": "  daily  ",
            "lang": "  en  ",
            "format": "  short  ",
            "title": "  A bridge  ",
            "text": "  Look at this bridge  ",
            "tags": ["#bridge", " #bridge ", "#dusk"],
            "mentions": ["@ada"],
            "media_paths": ["  /tmp/a.mp4  ", {"path": "/tmp/b.mp4"}],
            "content_type": "  video  ",
            "privacy": "  public  ",
            "status": "posted",
            "platform_post_id": "  yt_123  ",
            "link": "  https://example.com/p/1  ",
            "scheduled_at": _NOW,
            "posted_at": _NOW + timedelta(minutes=1),
            "idempotency_key": "  key-1  ",
            "run": "  run_1  ",
            "composition_id": "  composition_1  ",
            "disclosure_text": "  AI generated  ",
            "cost_usd": 0.25,
            "details": {"views": 0},
        }
    )

    assert post.id == "post_placeholder"
    assert post.platform == "youtube"
    assert post.account == "youtube/en"
    assert post.profile == "daily"
    assert post.language == "en"
    assert post.format == "short"
    assert post.title == "A bridge"
    assert post.caption == post.text == post.body == "Look at this bridge"
    assert post.hashtags == post.tags == ["#bridge", "#dusk"]
    assert post.mentions == ["@ada"]
    assert post.media == ["/tmp/a.mp4", "/tmp/b.mp4"]
    assert post.media_type == "video"
    assert post.visibility == "public"
    assert post.state is PostState.PUBLISHED
    assert post.remote_id == post.external_id == post.platform_post_id == "yt_123"
    assert post.url == post.link == "https://example.com/p/1"
    assert post.scheduled_for == _NOW
    assert post.published_at == _NOW + timedelta(minutes=1)
    assert post.idempotency_key == "key-1"
    assert post.run_id == "run_1"
    assert post.composition_id == "composition_1"
    assert post.disclosure_text == "AI generated"
    assert post.attestation.disclosure_text == "AI generated"
    assert post.cost_usd == 0.25
    assert post.metadata == {"views": 0}
    assert post.media_count == 2
    assert post.is_terminal is True


def test_post_accepts_nested_and_flat_attestation_fields() -> None:
    nested = Post(attestation=_attestation())
    flat = Post.model_validate(
        {
            "confirmed_by": "ada",
            "rights_confirmed": True,
            "disclosure_confirmed": True,
            "confirmed_at": _NOW,
            "license_summary": "CC BY 4.0",
            "disclosure_text": "AI generated",
        }
    )

    assert nested.attestation.is_complete is True
    assert nested.has_attestation is True
    assert flat.attestation.confirmed_by == "ada"
    assert flat.attestation.is_complete is True
    assert flat.disclosure_text == "AI generated"


def test_post_serializes_canonically_and_round_trips() -> None:
    post = Post(
        platform="telegram",
        caption="Hello",
        media=["/tmp/a.png"],
        state=PostState.PUBLISHED,
        remote_id="tg_1",
        url="https://example.com/tg/1",
        attestation=_attestation(),
        metadata={"b": 2, "a": 1},
    )

    payload: dict[str, Any] = json.loads(post.to_json())

    assert list(payload) == sorted(payload)
    assert payload["state"] == "published"
    assert payload["attestation"]["confirmed_by"] == "ada"
    assert payload["metadata"] == {"a": 1, "b": 2}

    restored = Post.model_validate_json(post.to_json())

    assert restored == post
    assert restored.to_json() == post.to_json()


def test_delivery_defaults_to_pending_with_retries_left() -> None:
    delivery = Delivery()

    assert delivery.id.startswith("delivery_")
    assert is_ulid(delivery.id.removeprefix("delivery_"))
    assert delivery.state is DeliveryState.PENDING
    assert delivery.provider == ""
    assert delivery.order_id == ""
    assert delivery.target == ""
    assert delivery.files == []
    assert delivery.attempts == 0
    assert delivery.max_attempts == 3
    assert delivery.attempts_remaining == 3
    assert delivery.has_attempts_left is True
    assert delivery.can_retry is True
    assert delivery.is_terminal is False
    assert delivery.is_delivered is False
    assert delivery.is_due() is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("pending", DeliveryState.PENDING),
        ("queued", DeliveryState.PENDING),
        ("  SCHEDULED  ", DeliveryState.SCHEDULED),
        ("sending", DeliveryState.DELIVERING),
        ("sent", DeliveryState.DELIVERED),
        ("succeeded", DeliveryState.DELIVERED),
        ("error", DeliveryState.FAILED),
        ("canceled", DeliveryState.CANCELLED),
    ],
)
def test_delivery_state_accepts_aliases_and_is_normalized(
    raw: str, expected: DeliveryState
) -> None:
    assert Delivery.model_validate({"state": raw}).state is expected


def test_delivery_rejects_an_unknown_state() -> None:
    with pytest.raises(ValidationError):
        Delivery.model_validate({"state": "teleporting"})


def test_delivery_carries_target_files_and_confirmation() -> None:
    delivery = Delivery.model_validate(
        {
            "delivery_id": "delivery_placeholder",
            "order": "order_1",
            "type": "  EMAIL  ",
            "destination": "  ada@example.com  ",
            "outputs": ["  /tmp/a.mp4  ", {"path": "/tmp/b.mp4"}],
            "scheduled_at": _NOW,
            "delivered_at": _NOW + timedelta(seconds=5),
            "expiry": _NOW + timedelta(days=7),
            "attempts": 1,
            "retries": 4,
            "confirmation_id": "  smtp-1  ",
            "idempotency_key": "  key-1  ",
            "cost_usd": 0.1,
            "details": {"provider": "smtp"},
        }
    )

    assert delivery.id == "delivery_placeholder"
    assert delivery.order_id == "order_1"
    assert delivery.provider == delivery.method == delivery.type == "email"
    assert delivery.target == delivery.destination == "ada@example.com"
    assert delivery.files == delivery.outputs == ["/tmp/a.mp4", "/tmp/b.mp4"]
    assert delivery.scheduled_for == _NOW
    assert delivery.delivered_at == _NOW + timedelta(seconds=5)
    assert delivery.expires_at == _NOW + timedelta(days=7)
    assert delivery.attempts == 1
    assert delivery.max_attempts == 4
    assert delivery.attempts_remaining == 3
    assert delivery.confirmation == "smtp-1"
    assert delivery.has_confirmation is True
    assert delivery.idempotency_key == "key-1"
    assert delivery.cost_usd == 0.1
    assert delivery.metadata == {"provider": "smtp"}
    assert delivery.file_count == 2
    assert delivery.has_files is True


def test_delivery_infers_target_and_supports_methods() -> None:
    folder = Delivery(provider="folder", path="/tmp/out")
    link = Delivery(provider="link", url="https://example.com/out")
    email = Delivery(provider="email", email="ada@example.com")

    assert folder.target == folder.folder == "/tmp/out"
    assert link.target == link.link == "https://example.com/out"
    assert email.target == email.recipient == "ada@example.com"
    assert DeliveryMethod("email") is DeliveryMethod.EMAIL


def test_delivery_rejects_more_attempts_than_the_limit() -> None:
    with pytest.raises(ValidationError):
        Delivery(attempts=4, max_attempts=3)
    with pytest.raises(ValidationError):
        Delivery(attempts=-1)


def test_delivery_schedule_and_terminal_helpers() -> None:
    future = Delivery(scheduled_for=_NOW + timedelta(hours=1))
    due = Delivery(scheduled_for=_NOW - timedelta(hours=1))
    delivered = Delivery(state=DeliveryState.DELIVERED)
    failed = Delivery(state=DeliveryState.FAILED, attempts=3)

    assert future.is_due(now=_NOW) is False
    assert due.is_due(now=_NOW) is True
    assert delivered.is_delivered is True
    assert delivered.delivered is True
    assert delivered.is_terminal is True
    assert failed.is_terminal is True
    assert failed.can_retry is False


def test_delivery_round_trips_through_json() -> None:
    delivery = Delivery(
        order_id="order_1",
        provider="webhook",
        target="https://example.com/hook",
        files=["/tmp/a.mp4"],
        state=DeliveryState.DELIVERED,
        delivered_at=_NOW,
        confirmation="hook-1",
        attestation=_attestation(),
    )

    restored = Delivery.model_validate_json(delivery.to_json())

    assert restored == delivery
    assert restored.to_json() == delivery.to_json()


def test_plan_defaults_to_a_draft_with_no_steps() -> None:
    plan = Plan()

    assert plan.id.startswith("plan_")
    assert is_ulid(plan.id.removeprefix("plan_"))
    assert plan.state is PlanState.DRAFT
    assert plan.instruction == ""
    assert plan.steps == []
    assert plan.assumptions == []
    assert plan.approval_mode is PlanApprovalMode.ASK
    assert plan.approved is False
    assert plan.estimate_usd == 0.0
    assert plan.cost_usd == 0.0
    assert plan.budget_usd is None
    assert plan.step_count == 0
    assert plan.total_estimate_usd == 0.0
    assert plan.needs_approval is True
    assert plan.is_terminal is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("draft", PlanState.DRAFT),
        ("queued", PlanState.PENDING),
        ("awaiting_approval", PlanState.PENDING_APPROVAL),
        ("accepted", PlanState.APPROVED),
        ("denied", PlanState.REJECTED),
        ("executing", PlanState.RUNNING),
        ("completed", PlanState.SUCCEEDED),
        ("error", PlanState.FAILED),
        ("canceled", PlanState.CANCELLED),
    ],
)
def test_plan_state_accepts_aliases_and_is_normalized(
    raw: str, expected: PlanState
) -> None:
    assert Plan.model_validate({"state": raw}).state is expected


def test_plan_rejects_an_unknown_state() -> None:
    with pytest.raises(ValidationError):
        Plan.model_validate({"state": "teleporting"})


def test_plan_carries_steps_assumptions_and_estimates() -> None:
    plan = Plan.model_validate(
        {
            "plan_id": "plan_placeholder",
            "prompt": "  Make a card  ",
            "status": "running",
            "tool_calls": [
                {
                    "step_id": "step_1",
                    "tool_name": "  image_gen  ",
                    "args": {"prompt": "a bridge"},
                    "dependencies": [],
                    "estimate": 0.2,
                    "status": "succeeded",
                },
                {
                    "id": "step_2",
                    "name": "  render  ",
                    "params": {"composition": "c1"},
                    "needs": ["step_1"],
                    "cost_estimate": 0.1,
                },
            ],
            "assumptions": ["  portrait  ", "portrait"],
            "approval": "always",
            "approved": True,
            "approved_by": "  ada  ",
            "approval_time": _NOW,
            "estimated_cost_usd": 0.3,
            "cost_usd": 0.25,
            "budget": 1.0,
            "run": "run_1",
            "format": "  card  ",
            "idempotency_key": "  key-1  ",
            "details": {"source": "cli"},
        }
    )

    assert plan.id == "plan_placeholder"
    assert plan.instruction == "Make a card"
    assert plan.state is PlanState.RUNNING
    assert plan.step_count == 2
    assert plan.step_ids == ["step_1", "step_2"]
    assert plan.steps[0].tool == "image_gen"
    assert plan.steps[0].arguments == {"prompt": "a bridge"}
    assert plan.steps[0].estimate_usd == 0.2
    assert plan.steps[0].succeeded is True
    assert plan.steps[1].tool == "render"
    assert plan.steps[1].depends_on == plan.steps[1].dependencies == ["step_1"]
    assert plan.total_estimate_usd == pytest.approx(0.3)
    assert plan.assumptions == ["portrait"]
    assert plan.approval_mode is PlanApprovalMode.ALWAYS
    assert plan.is_approved is True
    assert plan.needs_approval is False
    assert plan.approved_by == "ada"
    assert plan.approved_at == _NOW
    assert plan.estimate_usd == 0.3
    assert plan.estimate == 0.3
    assert plan.cost_usd == 0.25
    assert plan.budget_usd == 1.0
    assert plan.run_id == "run_1"
    assert plan.format == "card"
    assert plan.idempotency_key == "key-1"
    assert plan.metadata == {"source": "cli"}
    assert plan.step_by_id("step_1") is plan.steps[0]
    assert plan.step_by_id("missing") is None


def test_plan_approval_mode_accepts_bools_and_aliases() -> None:
    always = Plan.model_validate({"approval_mode": True})
    never = Plan.model_validate({"approval_mode": False})
    automatic = Plan.model_validate({"approval_mode": "automatic"})

    assert always.approval_mode is PlanApprovalMode.ALWAYS
    assert never.approval_mode is PlanApprovalMode.NEVER
    assert automatic.approval_mode is PlanApprovalMode.NEVER


def test_plan_detects_a_dependency_cycle() -> None:
    acyclic = Plan(
        steps=[
            PlanStep(id="a"),
            PlanStep(id="b", depends_on=["a"]),
        ]
    )
    cyclic = Plan(
        steps=[
            PlanStep(id="a", depends_on=["b"]),
            PlanStep(id="b", depends_on=["a"]),
        ]
    )

    assert acyclic.has_cycles() is False
    assert cyclic.has_cycles() is True


def test_plan_step_defaults_and_terminal_helpers() -> None:
    step = PlanStep()

    assert step.id.startswith("step_")
    assert is_ulid(step.id.removeprefix("step_"))
    assert step.tool == ""
    assert step.arguments == {}
    assert step.depends_on == []
    assert step.status is PlanStepState.PENDING
    assert step.estimate_usd == 0.0
    assert step.is_terminal is False
    assert PlanStep.model_validate({"status": "done"}).succeeded is True
    assert PlanStep.model_validate({"status": "error"}).failed is True
    assert PlanStep.model_validate({"status": "skipped"}).is_terminal is True


def test_plan_round_trips_through_json() -> None:
    plan = Plan(
        instruction="Make a card",
        state=PlanState.APPROVED,
        steps=[PlanStep(id="step_1", tool="render", arguments={"x": 1})],
        assumptions=["portrait"],
        approved=True,
        approved_at=_NOW,
        estimate_usd=0.5,
        metadata={"b": 2, "a": 1},
    )

    payload: dict[str, Any] = json.loads(plan.to_json())

    assert list(payload) == sorted(payload)
    assert payload["state"] == "approved"
    assert payload["steps"][0]["tool"] == "render"

    restored = Plan.model_validate_json(plan.to_json())

    assert restored == plan
    assert restored.to_json() == plan.to_json()


def test_composition_reference_defaults_and_helpers() -> None:
    reference = CompositionReference()

    assert reference.id.startswith("composition_")
    assert is_ulid(reference.id.removeprefix("composition_"))
    assert reference.path == ""
    assert reference.content_hash == ""
    assert reference.format == ""
    assert reference.version == 1
    assert reference.toolchain_fingerprint.is_known is False
    assert reference.duration_seconds is None
    assert reference.assets == []
    assert reference.pinned is False
    assert reference.asset_count == 0
    assert reference.has_assets is False
    assert reference.is_rendered is False
    assert reference.aspect_ratio is None
    assert reference.is_pinned is False


def test_composition_reference_carries_hash_toolchain_and_media_fields() -> None:
    reference = CompositionReference.model_validate(
        {
            "composition_id": "composition_placeholder",
            "file": "  /runs/run_1/composition.yaml  ",
            "sha256": f"  {_SHA}  ",
            "output_type": "  video  ",
            "revision": 3,
            "renderer": "  renderer-1  ",
            "toolchain": {"renderer_version": "renderer-1", "ffmpeg_version": "6.1"},
            "duration": 12.5,
            "width": 1080,
            "height": 1920,
            "fps": 30,
            "asset_ids": ["  asset_1  ", "asset_2"],
            "is_pinned": True,
            "run": "  run_1  ",
            "lang": "  en  ",
            "platform_name": "  youtube  ",
            "details": {"source": "assistant"},
        }
    )

    assert reference.id == "composition_placeholder"
    assert reference.path == reference.file == reference.uri
    assert reference.path == "/runs/run_1/composition.yaml"
    assert reference.content_hash == reference.hash == reference.sha256 == _SHA
    assert reference.composition_hash == _SHA
    assert reference.format == "video"
    assert reference.version == 3
    assert reference.renderer_version == "renderer-1"
    assert reference.toolchain.ffmpeg_version == "6.1"
    assert reference.fingerprint == reference.toolchain.digest
    assert reference.duration_seconds == 12.5
    assert reference.assets == ["asset_1", "asset_2"]
    assert reference.asset_count == 2
    assert reference.has_assets is True
    assert reference.is_rendered is True
    assert reference.aspect_ratio == 1080 / 1920
    assert reference.is_pinned is True
    assert reference.run_id == "run_1"
    assert reference.language == "en"
    assert reference.platform == "youtube"
    assert reference.metadata == {"source": "assistant"}


def test_composition_reference_round_trips_through_json() -> None:
    reference = CompositionReference(
        path="/tmp/composition.yaml",
        content_hash=_SHA,
        format="video",
        renderer_version="renderer-1",
        toolchain_fingerprint=ToolchainFingerprint(
            renderer_version="renderer-1", ffmpeg_version="6.1"
        ),
        assets=["asset_1"],
        pinned=True,
    )

    restored = CompositionReference.model_validate_json(reference.to_json())

    assert restored == reference
    assert restored.to_json() == reference.to_json()


def test_io_models_keep_unknown_fields_with_a_warning() -> None:
    with pytest.warns(UnknownFieldWarning, match="future_field"):
        post = Post.model_validate({"future_field": "kept"})
    with pytest.warns(UnknownFieldWarning, match="future_field"):
        delivery = Delivery.model_validate({"future_field": "kept"})
    with pytest.warns(UnknownFieldWarning, match="future_field"):
        plan = Plan.model_validate({"future_field": "kept"})
    with pytest.warns(UnknownFieldWarning, match="future_field"):
        reference = CompositionReference.model_validate({"future_field": "kept"})

    for record in (post, delivery, plan, reference):
        assert record.model_extra is not None
        assert record.model_extra["future_field"] == "kept"


def test_io_model_aliases_are_exported() -> None:
    assert AttestationModel is Attestation
    assert PublishAttestation is Attestation
    assert RightsAttestation is Attestation
    assert DisclosureAttestation is Attestation
    assert PostRecord is Post
    assert PostStatus is PostState
    assert DeliveryRecord is Delivery
    assert DeliveryStatus is DeliveryState
    assert PlanRecord is Plan
    assert PlanStatus is PlanState
    assert CompositionRef is CompositionReference
    assert Composition is CompositionReference
