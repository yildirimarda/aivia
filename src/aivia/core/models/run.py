"""Run and stage result records.

A :class:`Run` is one execution of the pipeline, an order, an instruction, a
media request or a benchmark. It carries the identity used for deduplication,
the state that a later state machine will move through, the toolchain
fingerprint needed to reproduce rendered outputs and the per-run cost totals.

A :class:`StageResult` is what one idempotent stage reports back to the
orchestrator: its status, the input hash used to decide whether a resumed run
may skip it, the artifacts it registered and any failure code. The models only
describe data; they never execute a stage, touch the filesystem or call a
provider.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import AliasChoices, Field, field_validator, model_validator

from aivia.core.ids import new_run_id
from aivia.core.models.base import ModelBase, canonical_json

__all__ = [
    "Run",
    "RunKind",
    "RunRecord",
    "RunState",
    "RunStatus",
    "StageResult",
    "StageResultModel",
    "StageRun",
    "StageRunResult",
    "StageState",
    "StageStatus",
    "ToolchainFingerprint",
    "ToolchainIdentity",
]


class RunKind(StrEnum):
    """The entry point that created a run."""

    PIPELINE = "pipeline"
    ORDER = "order"
    INSTRUCTION = "instruction"
    MEDIA = "media"
    BENCHMARK = "benchmark"


_KIND_ALIASES = {
    "content": RunKind.PIPELINE,
    "content_pipeline": RunKind.PIPELINE,
    "pipeline_run": RunKind.PIPELINE,
    "personalized_order": RunKind.ORDER,
    "instruction_run": RunKind.INSTRUCTION,
    "assistant": RunKind.INSTRUCTION,
    "media_generation": RunKind.MEDIA,
    "generation": RunKind.MEDIA,
    "bench": RunKind.BENCHMARK,
    "benchmark_run": RunKind.BENCHMARK,
}


class RunState(StrEnum):
    """The lifecycle state of a run.

    ``PAUSED_BUDGET`` is the explicit paused state used when a budget cap is
    reached and the configured overrun behaviour is to pause. It is distinct
    from a terminal failure so a resume can continue the same run later.
    """

    QUEUED = "queued"
    RUNNING = "running"
    PAUSED_BUDGET = "paused_budget"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


_STATE_ALIASES = {
    "pending": RunState.QUEUED,
    "created": RunState.QUEUED,
    "scheduled": RunState.QUEUED,
    "active": RunState.RUNNING,
    "in_progress": RunState.RUNNING,
    "started": RunState.RUNNING,
    "pause": RunState.PAUSED_BUDGET,
    "paused": RunState.PAUSED_BUDGET,
    "budget_paused": RunState.PAUSED_BUDGET,
    "budget": RunState.PAUSED_BUDGET,
    "complete": RunState.SUCCEEDED,
    "completed": RunState.SUCCEEDED,
    "success": RunState.SUCCEEDED,
    "done": RunState.SUCCEEDED,
    "error": RunState.FAILED,
    "failure": RunState.FAILED,
    "canceled": RunState.CANCELLED,
    "aborted": RunState.CANCELLED,
}

_TERMINAL_RUN_STATES = frozenset(
    {RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED}
)


class StageState(StrEnum):
    """The lifecycle state of one stage result."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED_BUDGET = "paused_budget"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


_STAGE_STATE_ALIASES = {
    "queued": StageState.PENDING,
    "created": StageState.PENDING,
    "waiting": StageState.PENDING,
    "active": StageState.RUNNING,
    "in_progress": StageState.RUNNING,
    "started": StageState.RUNNING,
    "pause": StageState.PAUSED_BUDGET,
    "paused": StageState.PAUSED_BUDGET,
    "budget_paused": StageState.PAUSED_BUDGET,
    "complete": StageState.SUCCEEDED,
    "completed": StageState.SUCCEEDED,
    "success": StageState.SUCCEEDED,
    "done": StageState.SUCCEEDED,
    "error": StageState.FAILED,
    "failure": StageState.FAILED,
    "skip": StageState.SKIPPED,
    "canceled": StageState.CANCELLED,
    "aborted": StageState.CANCELLED,
}

_TERMINAL_STAGE_STATES = frozenset(
    {
        StageState.SUCCEEDED,
        StageState.FAILED,
        StageState.SKIPPED,
        StageState.CANCELLED,
    }
)


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _normalize_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _coerce_toolchain(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, ToolchainFingerprint):
        return value
    if isinstance(value, str):
        return {"font_set": value}
    return value


def _coerce_artifacts(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return value


class ToolchainFingerprint(ModelBase):
    """The tool versions that decide whether a render is reproducible.

    ``renderer_version``, ``ffmpeg_version`` and ``font_set`` are the three
    inputs named by the reproducibility invariant. :attr:`identity` is their
    canonical representation and :attr:`digest` is its SHA-256, suitable for a
    render cache key. An empty fingerprint is allowed for runs that do not
    render, and :attr:`is_known` reports whether any component was recorded.
    """

    renderer_version: str = Field(
        default="",
        validation_alias=AliasChoices(
            "renderer_version",
            "renderer",
            "renderer_ver",
            "renderer_build",
        ),
    )
    ffmpeg_version: str = Field(
        default="",
        validation_alias=AliasChoices(
            "ffmpeg_version",
            "ffmpeg",
            "ffmpeg_build",
            "ffmpeg_ver",
        ),
    )
    font_set: str = Field(
        default="",
        validation_alias=AliasChoices(
            "font_set",
            "fonts",
            "font_set_identity",
            "font_set_hash",
            "font_identity",
            "fonts_hash",
        ),
    )

    @field_validator("renderer_version", "ffmpeg_version", "font_set", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @model_validator(mode="before")
    @classmethod
    def _accept_opaque_string(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"font_set": data}
        if data is None:
            return {}
        return data

    @property
    def is_known(self) -> bool:
        """Whether at least one toolchain component was recorded."""
        return bool(self.renderer_version or self.ffmpeg_version or self.font_set)

    @property
    def identity(self) -> str:
        """The canonical, human readable representation of the components."""
        return canonical_json(
            {
                "ffmpeg_version": self.ffmpeg_version,
                "font_set": self.font_set,
                "renderer_version": self.renderer_version,
            }
        )

    @property
    def digest(self) -> str:
        """The SHA-256 of :attr:`identity`, stable across field order."""
        return hashlib.sha256(self.identity.encode("utf-8")).hexdigest()

    @property
    def fingerprint(self) -> str:
        """Alias for :attr:`digest`."""
        return self.digest

    @property
    def hash(self) -> str:
        """Alias for :attr:`digest`."""
        return self.digest

    @property
    def renderer(self) -> str:
        """Alias for :attr:`renderer_version`."""
        return self.renderer_version

    @property
    def ffmpeg(self) -> str:
        """Alias for :attr:`ffmpeg_version`."""
        return self.ffmpeg_version

    @property
    def fonts(self) -> str:
        """Alias for :attr:`font_set`."""
        return self.font_set

    def __str__(self) -> str:
        return self.identity


class Run(ModelBase):
    """One execution with its kind, state, deduplication key and provenance.

    ``idempotency_key`` is empty when the caller did not supply one; when it is
    set a repeated request can return the existing run instead of paying twice.
    ``toolchain_fingerprint`` is empty for runs that never render. The run
    directory, config snapshot and cost totals are recorded on the run so a
    later replay or resume does not have to reconstruct them.
    """

    id: str = Field(
        default_factory=new_run_id,
        min_length=1,
        validation_alias=AliasChoices("id", "run_id"),
    )
    kind: RunKind = Field(
        default=RunKind.PIPELINE,
        validation_alias=AliasChoices("kind", "run_kind", "type"),
    )
    state: RunState = Field(
        default=RunState.QUEUED,
        validation_alias=AliasChoices("state", "status", "run_state"),
    )
    idempotency_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "idempotency_key",
            "idempotency",
            "idempotent_key",
            "dedupe_key",
            "request_key",
        ),
    )
    toolchain_fingerprint: ToolchainFingerprint = Field(
        default_factory=ToolchainFingerprint,
        validation_alias=AliasChoices(
            "toolchain_fingerprint",
            "toolchain",
            "fingerprint",
        ),
    )
    profile: str = ""
    format: str = Field(
        default="",
        validation_alias=AliasChoices("format", "format_name", "recipe"),
    )
    run_dir: str = Field(
        default="",
        validation_alias=AliasChoices(
            "run_dir",
            "run_directory",
            "directory",
            "path",
        ),
    )
    config_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices(
            "config_snapshot",
            "config",
            "snapshot",
            "settings",
        ),
    )
    cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("cost_usd", "cost", "total_cost_usd"),
    )
    estimated_cost_usd: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices(
            "estimated_cost_usd",
            "estimate_usd",
            "estimated_cost",
            "estimate",
        ),
    )
    error_code: str = Field(
        default="",
        validation_alias=AliasChoices("error_code", "failure_code"),
    )
    error_message: str = Field(
        default="",
        validation_alias=AliasChoices("error_message", "error", "failure_message"),
    )
    started_at: datetime | None = None
    finished_at: datetime | None = None
    paused_at: datetime | None = None
    pause_reason: str = Field(
        default="",
        validation_alias=AliasChoices("pause_reason", "paused_reason"),
    )

    @field_validator("kind", mode="before")
    @classmethod
    def _validate_kind(cls, value: Any) -> Any:
        if isinstance(value, RunKind):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            return _KIND_ALIASES.get(text, text)
        return value

    @field_validator("state", mode="before")
    @classmethod
    def _validate_state(cls, value: Any) -> Any:
        if isinstance(value, RunState):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            return _STATE_ALIASES.get(text, text)
        return value

    @field_validator(
        "idempotency_key",
        "profile",
        "format",
        "run_dir",
        "error_code",
        "error_message",
        "pause_reason",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("toolchain_fingerprint", mode="before")
    @classmethod
    def _validate_toolchain(cls, value: Any) -> Any:
        return _coerce_toolchain(value)

    @field_validator("config_snapshot", mode="before")
    @classmethod
    def _validate_config_snapshot(cls, value: Any) -> Any:
        return {} if value is None else value

    @field_validator("started_at", "finished_at", "paused_at")
    @classmethod
    def _normalize_optional_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _normalize_timestamp(value)

    @model_validator(mode="after")
    def _check_times(self) -> Self:
        if (
            self.started_at is not None
            and self.finished_at is not None
            and self.finished_at < self.started_at
        ):
            raise ValueError("finished_at must not be before started_at")
        return self

    @property
    def run_id(self) -> str:
        """Alias for :attr:`id`."""
        return self.id

    @property
    def run_kind(self) -> RunKind:
        """Alias for :attr:`kind`."""
        return self.kind

    @property
    def type(self) -> RunKind:
        """Alias for :attr:`kind`."""
        return self.kind

    @property
    def status(self) -> RunState:
        """Alias for :attr:`state`."""
        return self.state

    @property
    def run_state(self) -> RunState:
        """Alias for :attr:`state`."""
        return self.state

    @property
    def toolchain(self) -> ToolchainFingerprint:
        """Alias for :attr:`toolchain_fingerprint`."""
        return self.toolchain_fingerprint

    @property
    def fingerprint(self) -> str:
        """The digest of :attr:`toolchain_fingerprint`."""
        return self.toolchain_fingerprint.digest

    @property
    def has_idempotency_key(self) -> bool:
        """Whether the run can be found by a repeated idempotency key."""
        return bool(self.idempotency_key)

    @property
    def is_paused(self) -> bool:
        """Whether the run is paused and can be resumed."""
        return self.state is RunState.PAUSED_BUDGET

    @property
    def is_budget_paused(self) -> bool:
        """Alias for :attr:`is_paused`."""
        return self.is_paused

    @property
    def is_terminal(self) -> bool:
        """Whether the run reached a final state."""
        return self.state in _TERMINAL_RUN_STATES

    @property
    def is_active(self) -> bool:
        """Whether the run is queued or currently running."""
        return self.state in {RunState.QUEUED, RunState.RUNNING}

    @property
    def duration_seconds(self) -> float | None:
        """Elapsed seconds between start and finish, or ``None`` if unfinished."""
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()


class StageResult(ModelBase):
    """What one stage reports to the orchestrator after it runs.

    ``stage`` names the recipe step and ``run_id`` links it to its run. On
    resume, a ``SUCCEEDED`` result whose ``input_hash`` is unchanged may be
    skipped. ``artifacts`` holds the paths registered under the run directory,
    ``error_code`` and ``error_message`` explain a failure and the cost fields
    let a caller sum stage spend without joining the cost events table.
    """

    run_id: str = Field(
        default="",
        validation_alias=AliasChoices("run_id", "run"),
    )
    stage: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "stage",
            "name",
            "stage_name",
            "step",
        ),
    )
    status: StageState = Field(
        default=StageState.PENDING,
        validation_alias=AliasChoices("status", "state", "stage_status"),
    )
    index: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices(
            "index",
            "order",
            "position",
            "sequence",
        ),
    )
    input_hash: str = Field(
        default="",
        validation_alias=AliasChoices(
            "input_hash",
            "input_digest",
            "inputs_hash",
        ),
    )
    output_hash: str = Field(
        default="",
        validation_alias=AliasChoices(
            "output_hash",
            "output_digest",
            "result_hash",
        ),
    )
    artifacts: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "artifacts",
            "outputs",
            "files",
            "paths",
        ),
    )
    error_code: str = Field(
        default="",
        validation_alias=AliasChoices("error_code", "failure_code"),
    )
    error_message: str = Field(
        default="",
        validation_alias=AliasChoices("error_message", "error", "failure_message"),
    )
    cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("cost_usd", "cost", "total_cost_usd"),
    )
    estimated_cost_usd: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices(
            "estimated_cost_usd",
            "estimate_usd",
            "estimated_cost",
            "estimate",
        ),
    )
    toolchain_fingerprint: ToolchainFingerprint = Field(
        default_factory=ToolchainFingerprint,
        validation_alias=AliasChoices(
            "toolchain_fingerprint",
            "toolchain",
            "fingerprint",
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices(
            "metadata",
            "data",
            "details",
            "result",
        ),
    )
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @field_validator("stage", mode="before")
    @classmethod
    def _validate_stage(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError("stage must be a non-empty name")
            return text
        return value

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, value: Any) -> Any:
        if isinstance(value, StageState):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            return _STAGE_STATE_ALIASES.get(text, text)
        return value

    @field_validator(
        "run_id",
        "input_hash",
        "output_hash",
        "error_code",
        "error_message",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("artifacts", mode="before")
    @classmethod
    def _validate_artifacts(cls, value: Any) -> Any:
        artifacts = _coerce_artifacts(value)
        if not isinstance(artifacts, list):
            return artifacts
        normalized: list[str] = []
        for raw in artifacts:
            if not isinstance(raw, str):
                normalized.append(raw)
                continue
            artifact = raw.strip()
            if artifact and artifact not in normalized:
                normalized.append(artifact)
        return normalized

    @field_validator("toolchain_fingerprint", mode="before")
    @classmethod
    def _validate_toolchain(cls, value: Any) -> Any:
        return _coerce_toolchain(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: Any) -> Any:
        return {} if value is None else value

    @field_validator("started_at", "finished_at")
    @classmethod
    def _normalize_optional_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _normalize_timestamp(value)

    @model_validator(mode="after")
    def _check_times(self) -> Self:
        if (
            self.started_at is not None
            and self.finished_at is not None
            and self.finished_at < self.started_at
        ):
            raise ValueError("finished_at must not be before started_at")
        return self

    @property
    def name(self) -> str:
        """Alias for :attr:`stage`."""
        return self.stage

    @property
    def stage_name(self) -> str:
        """Alias for :attr:`stage`."""
        return self.stage

    @property
    def state(self) -> StageState:
        """Alias for :attr:`status`."""
        return self.status

    @property
    def stage_status(self) -> StageState:
        """Alias for :attr:`status`."""
        return self.status

    @property
    def toolchain(self) -> ToolchainFingerprint:
        """Alias for :attr:`toolchain_fingerprint`."""
        return self.toolchain_fingerprint

    @property
    def fingerprint(self) -> str:
        """The digest of :attr:`toolchain_fingerprint`."""
        return self.toolchain_fingerprint.digest

    @property
    def input_digest(self) -> str:
        """Alias for :attr:`input_hash`."""
        return self.input_hash

    @property
    def output_digest(self) -> str:
        """Alias for :attr:`output_hash`."""
        return self.output_hash

    @property
    def succeeded(self) -> bool:
        """Whether the stage completed successfully."""
        return self.status is StageState.SUCCEEDED

    @property
    def failed(self) -> bool:
        """Whether the stage failed."""
        return self.status is StageState.FAILED

    @property
    def skipped(self) -> bool:
        """Whether the stage was skipped."""
        return self.status is StageState.SKIPPED

    @property
    def is_paused(self) -> bool:
        """Whether the stage stopped because the run paused on budget."""
        return self.status is StageState.PAUSED_BUDGET

    @property
    def is_terminal(self) -> bool:
        """Whether the stage reached a final state."""
        return self.status in _TERMINAL_STAGE_STATES

    @property
    def duration_seconds(self) -> float | None:
        """Elapsed seconds between start and finish, or ``None`` if unfinished."""
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()


RunRecord = Run
RunStatus = RunState
StageResultModel = StageResult
StageRun = StageResult
StageRunResult = StageResult
StageStatus = StageState
ToolchainIdentity = ToolchainFingerprint
