"""Run and stage result contracts: kinds, states, deduplication and provenance."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from aivia.core.ids import is_ulid
from aivia.core.models import (
    Run,
    RunKind,
    RunRecord,
    RunState,
    RunStatus,
    StageResult,
    StageResultModel,
    StageRun,
    StageRunResult,
    StageState,
    StageStatus,
    ToolchainFingerprint,
    ToolchainIdentity,
)


def test_run_defaults_to_a_queued_pipeline_with_an_empty_fingerprint() -> None:
    run = Run()

    assert run.id.startswith("run_")
    assert is_ulid(run.id.removeprefix("run_"))
    assert run.kind is RunKind.PIPELINE
    assert run.state is RunState.QUEUED
    assert run.idempotency_key == ""
    assert run.toolchain_fingerprint.is_known is False
    assert run.profile == ""
    assert run.format == ""
    assert run.run_dir == ""
    assert run.config_snapshot == {}
    assert run.cost_usd == 0.0
    assert run.estimated_cost_usd is None
    assert run.error_code == ""
    assert run.error_message == ""
    assert run.started_at is None
    assert run.finished_at is None
    assert run.paused_at is None
    assert run.pause_reason == ""
    assert run.is_active is True
    assert run.is_terminal is False
    assert run.is_paused is False
    assert run.duration_seconds is None


def test_run_ids_are_unique_and_time_sortable() -> None:
    first = Run()
    second = Run()

    assert first.id != second.id
    assert first.id < second.id


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("pipeline", RunKind.PIPELINE),
        ("content_pipeline", RunKind.PIPELINE),
        ("order", RunKind.ORDER),
        ("instruction", RunKind.INSTRUCTION),
        ("  MEDIA  ", RunKind.MEDIA),
        ("generation", RunKind.MEDIA),
        ("benchmark", RunKind.BENCHMARK),
        ("bench", RunKind.BENCHMARK),
    ],
)
def test_run_kind_accepts_aliases_and_is_normalized(
    raw: str, expected: RunKind
) -> None:
    assert Run.model_validate({"kind": raw}).kind is expected


def test_run_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        Run.model_validate({"kind": "teleport"})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("queued", RunState.QUEUED),
        ("pending", RunState.QUEUED),
        ("running", RunState.RUNNING),
        ("in_progress", RunState.RUNNING),
        ("paused_budget", RunState.PAUSED_BUDGET),
        ("paused", RunState.PAUSED_BUDGET),
        ("budget_paused", RunState.PAUSED_BUDGET),
        ("succeeded", RunState.SUCCEEDED),
        ("completed", RunState.SUCCEEDED),
        ("failed", RunState.FAILED),
        ("error", RunState.FAILED),
        ("cancelled", RunState.CANCELLED),
        ("canceled", RunState.CANCELLED),
    ],
)
def test_run_state_accepts_aliases_and_is_normalized(
    raw: str, expected: RunState
) -> None:
    assert Run.model_validate({"state": raw}).state is expected


def test_run_rejects_an_unknown_state() -> None:
    with pytest.raises(ValidationError):
        Run.model_validate({"state": "reticulating"})


def test_run_state_helpers_mark_paused_budget_and_terminal_states() -> None:
    paused = Run(state=RunState.PAUSED_BUDGET)
    succeeded = Run(state=RunState.SUCCEEDED)
    failed = Run(state=RunState.FAILED)
    cancelled = Run(state=RunState.CANCELLED)

    assert paused.is_paused is True
    assert paused.is_budget_paused is True
    assert paused.is_terminal is False
    assert paused.is_active is False
    assert succeeded.is_terminal is True
    assert failed.is_terminal is True
    assert cancelled.is_terminal is True
    assert Run(state=RunState.RUNNING).is_active is True


def test_run_idempotency_key_is_stripped_and_aliased() -> None:
    run = Run.model_validate({"idempotency": "  request-42  "})

    assert run.idempotency_key == "request-42"
    assert run.has_idempotency_key is True
    assert Run().has_idempotency_key is False
    assert Run.model_validate({"dedupe_key": "another"}).idempotency_key == "another"


def test_run_carries_execution_metadata_and_aliases() -> None:
    stamp = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)
    finished = stamp + timedelta(seconds=90)
    run = Run.model_validate(
        {
            "run_id": "run_explicit",
            "run_kind": "order",
            "status": "completed",
            "idempotency_key": " order-7 ",
            "toolchain": {
                "renderer": "renderer-1",
                "ffmpeg": "ffmpeg-7",
                "fonts": "fonts-abc",
            },
            "profile": "  daily  ",
            "format": "  short  ",
            "run_dir": "  /runs/run_explicit  ",
            "config": {"video": {"fps": 30}},
            "cost": 1.25,
            "estimate": 2.5,
            "failure_code": "provider_exhausted",
            "failure_message": "  no provider left  ",
            "started_at": stamp,
            "finished_at": finished,
            "paused_at": stamp,
            "paused_reason": "  budget  ",
        }
    )

    assert run.run_id == run.id == "run_explicit"
    assert run.run_kind is RunKind.ORDER
    assert run.type is RunKind.ORDER
    assert run.status is RunState.SUCCEEDED
    assert run.run_state is RunState.SUCCEEDED
    assert run.idempotency_key == "order-7"
    assert run.profile == "daily"
    assert run.format == "short"
    assert run.run_dir == "/runs/run_explicit"
    assert run.config_snapshot == {"video": {"fps": 30}}
    assert run.cost_usd == 1.25
    assert run.estimated_cost_usd == 2.5
    assert run.error_code == "provider_exhausted"
    assert run.error_message == "no provider left"
    assert run.paused_at == stamp
    assert run.pause_reason == "budget"
    assert run.duration_seconds == 90.0
    assert run.toolchain.renderer == "renderer-1"
    assert run.fingerprint == run.toolchain.digest


def test_run_timestamps_are_normalized_to_utc() -> None:
    local = timezone(timedelta(hours=2))
    supplied = datetime(2026, 1, 2, 12, 30, tzinfo=local)

    run = Run(started_at=supplied, finished_at=supplied)

    assert run.started_at == datetime(2026, 1, 2, 10, 30, tzinfo=UTC)
    assert run.finished_at == datetime(2026, 1, 2, 10, 30, tzinfo=UTC)
    assert run.started_at.tzinfo is UTC
    assert Run(started_at=datetime(2026, 1, 2, 10, 30)).started_at == datetime(
        2026, 1, 2, 10, 30, tzinfo=UTC
    )


def test_run_rejects_a_finish_before_the_start() -> None:
    start = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)

    with pytest.raises(ValidationError, match="finished_at"):
        Run(started_at=start, finished_at=start - timedelta(seconds=1))


def test_run_treats_a_missing_config_snapshot_as_empty() -> None:
    assert Run.model_validate({"config_snapshot": None}).config_snapshot == {}


def test_toolchain_fingerprint_defaults_are_empty_but_valid() -> None:
    fingerprint = ToolchainFingerprint()

    assert fingerprint.renderer_version == ""
    assert fingerprint.ffmpeg_version == ""
    assert fingerprint.font_set == ""
    assert fingerprint.is_known is False
    assert len(fingerprint.digest) == 64


def test_toolchain_fingerprint_accepts_aliases_and_strips_text() -> None:
    fingerprint = ToolchainFingerprint.model_validate(
        {
            "renderer": "  renderer-1  ",
            "ffmpeg_build": "  ffmpeg-7  ",
            "fonts_hash": "  fonts-abc  ",
        }
    )

    assert fingerprint.renderer_version == "renderer-1"
    assert fingerprint.ffmpeg_version == "ffmpeg-7"
    assert fingerprint.font_set == "fonts-abc"
    assert fingerprint.renderer == "renderer-1"
    assert fingerprint.ffmpeg == "ffmpeg-7"
    assert fingerprint.fonts == "fonts-abc"
    assert fingerprint.is_known is True


def test_toolchain_fingerprint_accepts_an_opaque_string() -> None:
    fingerprint = ToolchainFingerprint.model_validate("fonts-abc")

    assert fingerprint.font_set == "fonts-abc"
    assert fingerprint.is_known is True


def test_toolchain_fingerprint_identity_is_canonical_and_order_independent() -> None:
    first = ToolchainFingerprint(
        renderer_version="renderer-1",
        ffmpeg_version="ffmpeg-7",
        font_set="fonts-abc",
    )
    second = ToolchainFingerprint.model_validate(
        {
            "font_set": "fonts-abc",
            "ffmpeg_version": "ffmpeg-7",
            "renderer_version": "renderer-1",
        }
    )

    assert first.identity == second.identity
    assert first.digest == second.digest == first.fingerprint == first.hash
    assert first.identity == (
        '{"ffmpeg_version":"ffmpeg-7","font_set":"fonts-abc",'
        '"renderer_version":"renderer-1"}'
    )
    assert json.loads(str(first)) == {
        "ffmpeg_version": "ffmpeg-7",
        "font_set": "fonts-abc",
        "renderer_version": "renderer-1",
    }


def test_toolchain_fingerprint_digest_changes_with_any_component() -> None:
    base = ToolchainFingerprint(
        renderer_version="renderer-1",
        ffmpeg_version="ffmpeg-7",
        font_set="fonts-abc",
    )

    assert base.digest != base.model_copy(update={"renderer_version": "2"}).digest
    assert base.digest != base.model_copy(update={"ffmpeg_version": "8"}).digest
    assert base.digest != base.model_copy(update={"font_set": "other"}).digest


def test_stage_result_requires_a_non_empty_stage_name() -> None:
    with pytest.raises(ValidationError):
        StageResult.model_validate({"stage": ""})
    with pytest.raises(ValidationError):
        StageResult.model_validate({"stage": "   "})
    with pytest.raises(ValidationError):
        StageResult.model_validate({})


def test_stage_result_defaults_to_pending_with_no_outputs() -> None:
    result = StageResult(stage="write")

    assert result.stage == "write"
    assert result.run_id == ""
    assert result.status is StageState.PENDING
    assert result.index is None
    assert result.input_hash == ""
    assert result.output_hash == ""
    assert result.artifacts == []
    assert result.error_code == ""
    assert result.error_message == ""
    assert result.cost_usd == 0.0
    assert result.estimated_cost_usd is None
    assert result.toolchain_fingerprint.is_known is False
    assert result.metadata == {}
    assert result.started_at is None
    assert result.finished_at is None
    assert result.is_terminal is False
    assert result.duration_seconds is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("pending", StageState.PENDING),
        ("queued", StageState.PENDING),
        ("running", StageState.RUNNING),
        ("paused_budget", StageState.PAUSED_BUDGET),
        ("paused", StageState.PAUSED_BUDGET),
        ("succeeded", StageState.SUCCEEDED),
        ("done", StageState.SUCCEEDED),
        ("failed", StageState.FAILED),
        ("error", StageState.FAILED),
        ("skipped", StageState.SKIPPED),
        ("skip", StageState.SKIPPED),
        ("cancelled", StageState.CANCELLED),
        ("canceled", StageState.CANCELLED),
    ],
)
def test_stage_status_accepts_aliases_and_is_normalized(
    raw: str, expected: StageState
) -> None:
    assert (
        StageResult.model_validate({"stage": "write", "status": raw}).status is expected
    )


def test_stage_result_rejects_an_unknown_status() -> None:
    with pytest.raises(ValidationError):
        StageResult.model_validate({"stage": "write", "status": "exploded"})


def test_stage_result_status_helpers() -> None:
    succeeded = StageResult(stage="write", status=StageState.SUCCEEDED)
    failed = StageResult(stage="write", status=StageState.FAILED)
    skipped = StageResult(stage="write", status=StageState.SKIPPED)
    paused = StageResult(stage="write", status=StageState.PAUSED_BUDGET)

    assert succeeded.succeeded is True
    assert succeeded.is_terminal is True
    assert failed.failed is True
    assert failed.is_terminal is True
    assert skipped.skipped is True
    assert skipped.is_terminal is True
    assert paused.is_paused is True
    assert paused.is_terminal is False


def test_stage_result_carries_resume_artifacts_error_and_cost() -> None:
    stamp = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)
    result = StageResult.model_validate(
        {
            "run": "run_1",
            "stage_name": "  write  ",
            "state": "completed",
            "order": 3,
            "input_digest": "  input-abc  ",
            "output_digest": "  output-def  ",
            "outputs": ["  /runs/run_1/script.json  ", "/runs/run_1/script.json"],
            "failure_code": "",
            "cost": 0.25,
            "estimate": 0.5,
            "toolchain": {"renderer": "renderer-1"},
            "details": {"beats": 4},
            "started_at": stamp,
            "finished_at": stamp + timedelta(seconds=12),
        }
    )

    assert result.run_id == "run_1"
    assert result.stage == result.name == result.stage_name == "write"
    assert result.status is StageState.SUCCEEDED
    assert result.state is StageState.SUCCEEDED
    assert result.stage_status is StageState.SUCCEEDED
    assert result.index == 3
    assert result.input_hash == result.input_digest == "input-abc"
    assert result.output_hash == result.output_digest == "output-def"
    assert result.artifacts == ["/runs/run_1/script.json"]
    assert result.cost_usd == 0.25
    assert result.estimated_cost_usd == 0.5
    assert result.metadata == {"beats": 4}
    assert result.toolchain.renderer == "renderer-1"
    assert result.fingerprint == result.toolchain.digest
    assert result.duration_seconds == 12.0


def test_stage_result_coerces_a_single_artifact_and_keeps_order() -> None:
    assert StageResult.model_validate(
        {"stage": "write", "artifacts": "a.json"}
    ).artifacts == ["a.json"]
    assert (
        StageResult.model_validate({"stage": "write", "artifacts": None}).artifacts
        == []
    )
    assert StageResult(stage="write", artifacts=["b", "a", "b"]).artifacts == [
        "b",
        "a",
    ]


def test_stage_result_timestamps_are_normalized_to_utc() -> None:
    local = timezone(timedelta(hours=2))
    supplied = datetime(2026, 1, 2, 12, 30, tzinfo=local)

    result = StageResult(stage="write", started_at=supplied, finished_at=supplied)

    assert result.started_at == datetime(2026, 1, 2, 10, 30, tzinfo=UTC)
    assert result.finished_at == datetime(2026, 1, 2, 10, 30, tzinfo=UTC)
    assert result.started_at.tzinfo is UTC


def test_stage_result_rejects_a_finish_before_the_start() -> None:
    start = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)

    with pytest.raises(ValidationError, match="finished_at"):
        StageResult(
            stage="write",
            started_at=start,
            finished_at=start - timedelta(seconds=1),
        )


def test_stage_result_treats_missing_metadata_as_empty() -> None:
    assert (
        StageResult.model_validate({"stage": "write", "metadata": None}).metadata == {}
    )


def test_run_and_stage_models_round_trip_through_canonical_json() -> None:
    run = Run(
        idempotency_key="run-key",
        toolchain_fingerprint=ToolchainFingerprint(
            renderer_version="renderer-1",
            ffmpeg_version="ffmpeg-7",
            font_set="fonts-abc",
        ),
    )
    stage = StageResult(stage="write", run_id=run.id, status=StageState.SUCCEEDED)

    restored_run = Run.model_validate_json(run.to_json())
    restored_stage = StageResult.model_validate_json(stage.to_json())

    assert restored_run.to_json() == run.to_json()
    assert restored_stage.to_json() == stage.to_json()


def test_model_aliases_point_at_the_same_classes() -> None:
    assert RunRecord is Run
    assert RunStatus is RunState
    assert StageResultModel is StageResult
    assert StageRun is StageResult
    assert StageRunResult is StageResult
    assert StageStatus is StageState
    assert ToolchainIdentity is ToolchainFingerprint
