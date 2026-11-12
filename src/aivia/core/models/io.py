"""Records for publishing, delivery, assistant plans and composition references.

These models describe the data that leaves the generation core: a post sent to
a platform, a delivery handed to an order recipient, a validated plan of tool
calls and a reference to a stored composition. They deliberately contain no
provider or platform code; adapters and services fill them in and persist them.

Publishing and delivery carry an :class:`Attestation` so a rights and AI
disclosure confirmation travels with the output. Attestation fields are kept
in one small record instead of being copied onto every model, and the
``missing``/``is_complete`` helpers let strict callers refuse an incomplete
confirmation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from aivia.core.platforms import Platform

from pydantic import AliasChoices, Field, field_validator, model_validator

from aivia.core.ids import (
    new_composition_id,
    new_delivery_id,
    new_plan_id,
    new_post_id,
    new_step_id,
)
from aivia.core.models.base import ModelBase
from aivia.core.models.run import ToolchainFingerprint

__all__ = [
    "ApprovalMode",
    "Attestation",
    "AttestationModel",
    "AttestationRecord",
    "Composition",
    "CompositionModel",
    "CompositionRecord",
    "CompositionRef",
    "CompositionReference",
    "CompositionReferenceModel",
    "CompositionReferenceRecord",
    "Delivery",
    "DeliveryConfirmation",
    "DeliveryMethod",
    "DeliveryModel",
    "DeliveryProviderType",
    "DeliveryRecord",
    "DeliveryState",
    "DeliveryStatus",
    "DeliveryType",
    "DisclosureAttestation",
    "Plan",
    "PlanApproval",
    "PlanApprovalMode",
    "PlanModel",
    "PlanRecord",
    "PlanState",
    "PlanStatus",
    "PlanStep",
    "PlanStepModel",
    "PlanStepState",
    "PlanStepStatus",
    "Post",
    "PostAttestation",
    "PostModel",
    "PostRecord",
    "PostState",
    "PostStatus",
    "PublishAttestation",
    "RightsAttestation",
    "Step",
    "StepModel",
    "StepState",
    "ToolCall",
    "new_composition_id",
    "new_delivery_id",
    "new_plan_id",
    "new_post_id",
    "new_step_id",
]


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _normalize_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _coerce_mapping(value: Any) -> Any:
    return {} if value is None else value


def _coerce_names(value: Any) -> Any:
    """Accept a bare string, a sequence or a mapping of records."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return list(value)
    return value


def _normalize_names(value: Any) -> Any:
    entries = _coerce_names(value)
    if not isinstance(entries, list):
        return entries
    names: list[str] = []
    for raw in entries:
        if isinstance(raw, str):
            name = raw.strip()
            if name and name not in names:
                names.append(name)
            continue
        if isinstance(raw, Mapping):
            candidate = raw.get(
                "path",
                raw.get("file", raw.get("id", raw.get("asset_id", raw.get("url")))),
            )
            if isinstance(candidate, str):
                name = candidate.strip()
                if name and name not in names:
                    names.append(name)
                continue
        names.append(raw)
    return names


def _enum_value(
    value: Any,
    enum: type[StrEnum],
    aliases: Mapping[str, Any],
) -> Any:
    if isinstance(value, enum):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        return aliases.get(key, key)
    return value


class Attestation(ModelBase):
    """A rights and AI disclosure confirmation recorded with an output.

    ``confirmed_by`` names the person who accepted responsibility. The two
    confirmation flags say rights and disclosure were checked, ``confirmed_at``
    records when, and ``disclosure_text`` and ``license_summary`` record what
    was actually applied. ``strict`` asks callers to block when anything is
    missing; the model itself never raises for an incomplete attestation.
    """

    confirmed_by: str = Field(
        default="",
        validation_alias=AliasChoices(
            "confirmed_by",
            "who",
            "attested_by",
            "confirmed_user",
        ),
    )
    rights_confirmed_by: str = Field(
        default="",
        validation_alias=AliasChoices("rights_confirmed_by", "rights_by"),
    )
    disclosure_confirmed_by: str = Field(
        default="",
        validation_alias=AliasChoices("disclosure_confirmed_by", "disclosure_by"),
    )
    rights_confirmed: bool = Field(
        default=False,
        validation_alias=AliasChoices("rights_confirmed", "rights_ok", "rights"),
    )
    disclosure_confirmed: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "disclosure_confirmed",
            "disclosure_ok",
            "disclosure_checked",
        ),
    )
    confirmed_at: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "confirmed_at",
            "attested_at",
            "when",
            "confirmed_time",
        ),
    )
    disclosure_text: str = Field(
        default="",
        validation_alias=AliasChoices(
            "disclosure_text",
            "disclosure",
            "ai_disclosure",
            "applied_disclosure",
        ),
    )
    license_summary: str | dict[str, Any] = Field(
        default="",
        validation_alias=AliasChoices(
            "license_summary",
            "licenses",
            "license",
            "rights_summary",
        ),
    )
    strict: bool = Field(
        default=False,
        validation_alias=AliasChoices("strict", "strict_mode", "required"),
    )
    reason: str = Field(
        default="",
        validation_alias=AliasChoices(
            "reason",
            "skip_reason",
            "recorded_reason",
            "note",
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthand(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        confirmed = values.pop("confirmed", None)
        if confirmed is not None:
            values.setdefault("rights_confirmed", bool(confirmed))
            values.setdefault("disclosure_confirmed", bool(confirmed))
        return values

    @field_validator(
        "confirmed_by",
        "rights_confirmed_by",
        "disclosure_confirmed_by",
        "disclosure_text",
        "reason",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("confirmed_at")
    @classmethod
    def _normalize_optional_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _normalize_timestamp(value)

    @property
    def who(self) -> str:
        """Alias for :attr:`confirmed_by`."""
        return self.confirmed_by

    @property
    def when(self) -> datetime | None:
        """Alias for :attr:`confirmed_at`."""
        return self.confirmed_at

    @property
    def disclosure(self) -> str:
        """Alias for :attr:`disclosure_text`."""
        return self.disclosure_text

    @property
    def missing(self) -> list[str]:
        """The names of the required values that were not recorded."""
        missing: list[str] = []
        if not (self.confirmed_by or self.rights_confirmed_by):
            missing.append("confirmed_by")
        if not self.rights_confirmed:
            missing.append("rights_confirmed")
        if not self.disclosure_confirmed:
            missing.append("disclosure_confirmed")
        if self.confirmed_at is None:
            missing.append("confirmed_at")
        if not self.disclosure_text:
            missing.append("disclosure_text")
        if not self.license_summary:
            missing.append("license_summary")
        return missing

    @property
    def is_complete(self) -> bool:
        """Whether every required attestation value is present."""
        return not self.missing

    @property
    def confirmed(self) -> bool:
        """Alias for :attr:`is_complete`."""
        return self.is_complete

    @property
    def is_blocking(self) -> bool:
        """Whether strict mode should refuse this incomplete attestation."""
        return self.strict and not self.is_complete

    @property
    def missing_fields(self) -> list[str]:
        """Alias for :attr:`missing`."""
        return self.missing

    @property
    def is_valid(self) -> bool:
        """Alias for :attr:`is_complete`."""
        return self.is_complete

    @property
    def complete(self) -> bool:
        """Alias for :attr:`is_complete`."""
        return self.is_complete

    @property
    def blocks(self) -> bool:
        """Alias for :attr:`is_blocking`."""
        return self.is_blocking

    @property
    def license(self) -> str | dict[str, Any]:
        """Alias for :attr:`license_summary`."""
        return self.license_summary


class PostState(StrEnum):
    """The lifecycle state of a post."""

    DRAFT = "draft"
    PENDING = "pending"
    SCHEDULED = "scheduled"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    SUCCEEDED = "published"
    FAILED = "failed"
    CANCELLED = "cancelled"


_POST_STATE_ALIASES = {
    "draft": PostState.DRAFT,
    "new": PostState.DRAFT,
    "pending": PostState.PENDING,
    "queued": PostState.PENDING,
    "ready": PostState.PENDING,
    "scheduled": PostState.SCHEDULED,
    "deferred": PostState.SCHEDULED,
    "publishing": PostState.PUBLISHING,
    "uploading": PostState.PUBLISHING,
    "running": PostState.PUBLISHING,
    "in_progress": PostState.PUBLISHING,
    "published": PostState.PUBLISHED,
    "posted": PostState.PUBLISHED,
    "succeeded": PostState.PUBLISHED,
    "success": PostState.PUBLISHED,
    "complete": PostState.PUBLISHED,
    "completed": PostState.PUBLISHED,
    "failed": PostState.FAILED,
    "error": PostState.FAILED,
    "cancelled": PostState.CANCELLED,
    "canceled": PostState.CANCELLED,
    "aborted": PostState.CANCELLED,
}

_TERMINAL_POST_STATES = frozenset(
    {PostState.PUBLISHED, PostState.FAILED, PostState.CANCELLED}
)


class Post(ModelBase):
    """One post request or result, with its platform, media and attestation.

    A post starts as a draft or scheduled request and ends published or failed.
    ``remote_id`` and ``url`` hold the platform's identifiers once a publish
    succeeds, while ``idempotency_key`` prevents a retry from posting twice.
    """

    id: str = Field(
        default_factory=new_post_id,
        min_length=1,
        validation_alias=AliasChoices("id", "post_id", "post"),
    )
    platform: str = Field(
        default="",
        validation_alias=AliasChoices(
            "platform",
            "platform_name",
            "service",
            "target",
        ),
    )
    account: str = Field(
        default="",
        validation_alias=AliasChoices(
            "account",
            "account_id",
            "account_name",
            "channel",
        ),
    )
    profile: str = Field(
        default="",
        validation_alias=AliasChoices("profile", "profile_name", "profile_id"),
    )
    language: str = Field(
        default="",
        validation_alias=AliasChoices("language", "lang", "locale"),
    )
    format: str = Field(
        default="",
        validation_alias=AliasChoices("format", "format_name", "recipe"),
    )
    title: str = ""
    caption: str = Field(
        default="",
        validation_alias=AliasChoices("caption", "text", "body", "content"),
    )
    hashtags: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("hashtags", "tags", "tag_list"),
    )
    mentions: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("mentions", "handles", "at_handles"),
    )
    media: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "media",
            "media_paths",
            "files",
            "artifacts",
            "assets",
            "outputs",
        ),
    )
    media_type: str = Field(
        default="",
        validation_alias=AliasChoices(
            "media_type",
            "content_type",
            "kind",
            "type",
        ),
    )
    visibility: str = Field(
        default="",
        validation_alias=AliasChoices("visibility", "privacy", "audience"),
    )
    state: PostState = Field(
        default=PostState.DRAFT,
        validation_alias=AliasChoices("state", "status", "post_state"),
    )
    remote_id: str = Field(
        default="",
        validation_alias=AliasChoices(
            "remote_id",
            "remote_post_id",
            "platform_post_id",
            "external_id",
            "external_post_id",
            "published_id",
            "platform_id",
        ),
    )
    url: str = Field(
        default="",
        validation_alias=AliasChoices("url", "link", "permalink", "post_url"),
    )
    scheduled_for: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "scheduled_for",
            "scheduled_at",
            "publish_at",
            "due_at",
        ),
    )
    published_at: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices("published_at", "posted_at", "posted"),
    )
    dry_run: bool = Field(
        default=False,
        validation_alias=AliasChoices("dry_run", "is_dry_run", "preview"),
    )
    idempotency_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "idempotency_key",
            "idempotency",
            "dedupe_key",
            "request_key",
        ),
    )
    run_id: str = Field(
        default="",
        validation_alias=AliasChoices("run_id", "run"),
    )
    composition_id: str = Field(
        default="",
        validation_alias=AliasChoices(
            "composition_id",
            "composition_ref",
            "composition_reference",
        ),
    )
    disclosure_text: str = Field(
        default="",
        validation_alias=AliasChoices(
            "disclosure_text",
            "applied_disclosure",
            "disclosure_label",
        ),
    )
    disclosure_required: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "disclosure_required",
            "requires_disclosure",
        ),
    )
    disclosure_applied: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "disclosure_applied",
            "has_disclosure",
        ),
    )
    on_screen_label: str = Field(
        default="",
        validation_alias=AliasChoices(
            "on_screen_label",
            "ai_disclosure_label",
        ),
    )
    attestation: Attestation = Field(
        default_factory=Attestation,
        validation_alias=AliasChoices("attestation", "publish_attestation"),
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
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("metadata", "meta", "details"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_shapes(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        attestation = values.get("attestation")
        attestation = (
            dict(attestation) if isinstance(attestation, Mapping) else attestation
        )
        for key in (
            "confirmed_by",
            "rights_confirmed_by",
            "disclosure_confirmed_by",
            "rights_confirmed",
            "disclosure_confirmed",
            "confirmed_at",
            "license_summary",
            "strict",
            "reason",
        ):
            if key in values:
                value = values.pop(key)
                if isinstance(attestation, dict):
                    attestation.setdefault(key, value)
                elif attestation is None:
                    attestation = {key: value}
        if attestation is not None:
            values["attestation"] = attestation
        return values

    @field_validator(
        "platform",
        "account",
        "profile",
        "language",
        "format",
        "title",
        "caption",
        "media_type",
        "visibility",
        "remote_id",
        "url",
        "idempotency_key",
        "run_id",
        "composition_id",
        "disclosure_text",
        "error_code",
        "error_message",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("hashtags", "mentions", mode="before")
    @classmethod
    def _normalize_tags(cls, value: Any) -> Any:
        return _normalize_names(value)

    @field_validator("media", mode="before")
    @classmethod
    def _normalize_media(cls, value: Any) -> Any:
        return _normalize_names(value)

    @field_validator("state", mode="before")
    @classmethod
    def _normalize_state(cls, value: Any) -> Any:
        return _enum_value(value, PostState, _POST_STATE_ALIASES)

    @field_validator("scheduled_for", "published_at")
    @classmethod
    def _normalize_optional_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _normalize_timestamp(value)

    @field_validator("attestation", mode="before")
    @classmethod
    def _coerce_attestation(cls, value: Any) -> Any:
        return {} if value is None else value

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, value: Any) -> Any:
        return _coerce_mapping(value)

    @model_validator(mode="after")
    def _sync_disclosure(self) -> Self:
        if not self.disclosure_text and self.attestation.disclosure_text:
            self.disclosure_text = self.attestation.disclosure_text
        if not self.attestation.disclosure_text and self.disclosure_text:
            self.attestation.disclosure_text = self.disclosure_text
        return self

    def apply_disclosure(self, platform: Platform) -> Self:
        """Apply *platform*'s AI disclosure rules to this post.

        Stores the disclosure text on the post and mirrors it onto the
        attestation, records the disclosure metadata under
        ``platform.disclosure.metadata_field`` in ``metadata`` when the platform
        supports it, keeps the optional on-screen label on the post, and sets
        ``disclosure_required`` / ``disclosure_applied``.
        """
        rules = platform.disclosure
        self.disclosure_required = rules.required
        if rules.required:
            self.disclosure_text = rules.text or self.disclosure_text
            self.on_screen_label = rules.on_screen_label or self.on_screen_label
            self.disclosure_applied = True
            if rules.metadata and rules.metadata_field and self.disclosure_text:
                self.metadata[rules.metadata_field] = True
        if not self.attestation.disclosure_text and self.disclosure_text:
            self.attestation.disclosure_text = self.disclosure_text
        return self

    @property
    def post_id(self) -> str:
        """Alias for :attr:`id`."""
        return self.id

    @property
    def status(self) -> PostState:
        """Alias for :attr:`state`."""
        return self.state

    @property
    def post_state(self) -> PostState:
        """Alias for :attr:`state`."""
        return self.state

    @property
    def text(self) -> str:
        """Alias for :attr:`caption`."""
        return self.caption

    @property
    def body(self) -> str:
        """Alias for :attr:`caption`."""
        return self.caption

    @property
    def tags(self) -> list[str]:
        """Alias for :attr:`hashtags`."""
        return self.hashtags

    @property
    def files(self) -> list[str]:
        """Alias for :attr:`media`."""
        return self.media

    @property
    def assets(self) -> list[str]:
        """Alias for :attr:`media`."""
        return self.media

    @property
    def media_count(self) -> int:
        """The number of media files attached to the post."""
        return len(self.media)

    @property
    def has_media(self) -> bool:
        """Whether the post has at least one media file."""
        return bool(self.media)

    @property
    def link(self) -> str:
        """Alias for :attr:`url`."""
        return self.url

    @property
    def external_id(self) -> str:
        """Alias for :attr:`remote_id`."""
        return self.remote_id

    @property
    def platform_post_id(self) -> str:
        """Alias for :attr:`remote_id`."""
        return self.remote_id

    @property
    def is_published(self) -> bool:
        """Whether the post reached the platform."""
        return self.state is PostState.PUBLISHED

    @property
    def published(self) -> bool:
        """Alias for :attr:`is_published`."""
        return self.is_published

    @property
    def is_scheduled(self) -> bool:
        """Whether the post is waiting for its scheduled time."""
        return self.state is PostState.SCHEDULED

    @property
    def is_failed(self) -> bool:
        """Whether publishing failed."""
        return self.state is PostState.FAILED

    @property
    def is_terminal(self) -> bool:
        """Whether the post reached a final state."""
        return self.state in _TERMINAL_POST_STATES

    @property
    def has_attestation(self) -> bool:
        """Whether an attestation was recorded."""
        return self.attestation.is_complete

    @property
    def attestation_complete(self) -> bool:
        """Alias for :attr:`Attestation.is_complete`."""
        return self.attestation.is_complete

    @property
    def confirmed_by(self) -> str:
        """Alias for :attr:`Attestation.confirmed_by`."""
        return self.attestation.confirmed_by

    @property
    def rights_confirmed(self) -> bool:
        """Alias for :attr:`Attestation.rights_confirmed`."""
        return self.attestation.rights_confirmed

    @property
    def disclosure_confirmed(self) -> bool:
        """Alias for :attr:`Attestation.disclosure_confirmed`."""
        return self.attestation.disclosure_confirmed

    @property
    def confirmed_at(self) -> datetime | None:
        """Alias for :attr:`Attestation.confirmed_at`."""
        return self.attestation.confirmed_at

    @property
    def license_summary(self) -> str | dict[str, Any]:
        """Alias for :attr:`Attestation.license_summary`."""
        return self.attestation.license_summary

    @property
    def scheduled_at(self) -> datetime | None:
        """Alias for :attr:`scheduled_for`."""
        return self.scheduled_for

    @property
    def posted_at(self) -> datetime | None:
        """Alias for :attr:`published_at`."""
        return self.published_at

    @property
    def remote_url(self) -> str:
        """Alias for :attr:`url`."""
        return self.url


class DeliveryState(StrEnum):
    """The lifecycle state of a delivery."""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    DELIVERING = "delivering"
    DELIVERED = "delivered"
    SUCCEEDED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


_DELIVERY_STATE_ALIASES = {
    "pending": DeliveryState.PENDING,
    "queued": DeliveryState.PENDING,
    "ready": DeliveryState.PENDING,
    "scheduled": DeliveryState.SCHEDULED,
    "deferred": DeliveryState.SCHEDULED,
    "delivering": DeliveryState.DELIVERING,
    "sending": DeliveryState.DELIVERING,
    "running": DeliveryState.DELIVERING,
    "in_progress": DeliveryState.DELIVERING,
    "delivered": DeliveryState.DELIVERED,
    "sent": DeliveryState.DELIVERED,
    "succeeded": DeliveryState.DELIVERED,
    "success": DeliveryState.DELIVERED,
    "complete": DeliveryState.DELIVERED,
    "completed": DeliveryState.DELIVERED,
    "failed": DeliveryState.FAILED,
    "error": DeliveryState.FAILED,
    "cancelled": DeliveryState.CANCELLED,
    "canceled": DeliveryState.CANCELLED,
    "aborted": DeliveryState.CANCELLED,
}

_TERMINAL_DELIVERY_STATES = frozenset(
    {DeliveryState.DELIVERED, DeliveryState.FAILED, DeliveryState.CANCELLED}
)


class DeliveryMethod(StrEnum):
    """The delivery transports the model understands."""

    FOLDER = "folder"
    LINK = "link"
    EMAIL = "email"
    WEBHOOK = "webhook"
    WEB_CARD = "web_card"


class DeliveryConfirmation(ModelBase):
    """A record proving a delivery provider accepted the output."""

    reference: str = Field(
        default="",
        validation_alias=AliasChoices("reference", "id", "confirmation_id"),
    )
    message: str = ""
    received_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reference", "message", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("received_at")
    @classmethod
    def _normalize_optional_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _normalize_timestamp(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, value: Any) -> Any:
        return _coerce_mapping(value)


class Delivery(ModelBase):
    """One attempt to hand a finished output to an order recipient.

    ``provider`` names the configured transport and ``target`` the destination
    (a folder, URL, email address or webhook). ``attempts`` and
    ``max_attempts`` drive retries, ``scheduled_for`` supports timed delivery
    and ``confirmation`` records the provider's acknowledgement.
    """

    id: str = Field(
        default_factory=new_delivery_id,
        min_length=1,
        validation_alias=AliasChoices("id", "delivery_id", "delivery"),
    )
    order_id: str = Field(
        default="",
        validation_alias=AliasChoices("order_id", "order"),
    )
    run_id: str = Field(
        default="",
        validation_alias=AliasChoices("run_id", "run"),
    )
    provider: DeliveryMethod | str = Field(
        default="",
        union_mode="left_to_right",
        validation_alias=AliasChoices(
            "provider",
            "provider_name",
            "method",
            "type",
            "kind",
            "transport",
            "channel",
        ),
    )
    state: DeliveryState = Field(
        default=DeliveryState.PENDING,
        validation_alias=AliasChoices("state", "status", "delivery_state"),
    )
    target: str = Field(
        default="",
        validation_alias=AliasChoices(
            "target",
            "destination",
            "to",
            "recipient",
            "address",
            "endpoint",
            "location",
        ),
    )
    path: str = Field(
        default="",
        validation_alias=AliasChoices("path", "folder", "directory", "file"),
    )
    url: str = Field(
        default="",
        validation_alias=AliasChoices(
            "url",
            "link",
            "webhook_url",
            "signed_url",
        ),
    )
    email: str = Field(
        default="",
        validation_alias=AliasChoices("email", "to_email", "recipient_email"),
    )
    files: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "files",
            "artifacts",
            "outputs",
            "media",
            "paths",
        ),
    )
    scheduled_for: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "scheduled_for",
            "scheduled_at",
            "deliver_at",
            "requested_at",
        ),
    )
    delivered_at: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices("delivered_at", "sent_at", "completed_at"),
    )
    expires_at: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices("expires_at", "expiry", "expires"),
    )
    attempts: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices("attempts", "attempt", "attempt_count"),
    )
    max_attempts: int = Field(
        default=3,
        ge=1,
        validation_alias=AliasChoices(
            "max_attempts",
            "max_attempt",
            "retries",
            "attempt_limit",
        ),
    )
    idempotency_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "idempotency_key",
            "idempotency",
            "dedupe_key",
        ),
    )
    confirmation: str = Field(
        default="",
        validation_alias=AliasChoices(
            "confirmation",
            "confirmation_id",
            "receipt",
            "reference",
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
    attestation: Attestation = Field(
        default_factory=Attestation,
        validation_alias=AliasChoices("attestation", "delivery_attestation"),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("metadata", "meta", "details"),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_shapes(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        attestation = values.get("attestation")
        attestation = (
            dict(attestation) if isinstance(attestation, Mapping) else attestation
        )
        for key in (
            "confirmed_by",
            "rights_confirmed_by",
            "disclosure_confirmed_by",
            "rights_confirmed",
            "disclosure_confirmed",
            "confirmed_at",
            "license_summary",
            "strict",
            "reason",
        ):
            if key in values:
                value = values.pop(key)
                if isinstance(attestation, dict):
                    attestation.setdefault(key, value)
                elif attestation is None:
                    attestation = {key: value}
        if attestation is not None:
            values["attestation"] = attestation
        return values

    @field_validator(
        "order_id",
        "run_id",
        "provider",
        "target",
        "path",
        "url",
        "email",
        "idempotency_key",
        "confirmation",
        "error_code",
        "error_message",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: Any) -> Any:
        stripped = _strip(value)
        if not isinstance(stripped, str):
            return stripped
        key = stripped.lower()
        try:
            return DeliveryMethod(key)
        except ValueError:
            return key

    @field_validator("files", mode="before")
    @classmethod
    def _normalize_files(cls, value: Any) -> Any:
        return _normalize_names(value)

    @field_validator("state", mode="before")
    @classmethod
    def _normalize_state(cls, value: Any) -> Any:
        return _enum_value(value, DeliveryState, _DELIVERY_STATE_ALIASES)

    @field_validator("scheduled_for", "delivered_at", "expires_at")
    @classmethod
    def _normalize_optional_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _normalize_timestamp(value)

    @field_validator("attestation", mode="before")
    @classmethod
    def _coerce_attestation(cls, value: Any) -> Any:
        return {} if value is None else value

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, value: Any) -> Any:
        return _coerce_mapping(value)

    @model_validator(mode="after")
    def _infer_target(self) -> Self:
        if not self.target:
            inferred = self.path or self.url or self.email
            if inferred:
                self.target = inferred
        return self

    @model_validator(mode="after")
    def _check_attempts(self) -> Self:
        if self.attempts > self.max_attempts:
            raise ValueError("attempts must not exceed max_attempts")
        return self

    @property
    def delivery_id(self) -> str:
        """Alias for :attr:`id`."""
        return self.id

    @property
    def status(self) -> DeliveryState:
        """Alias for :attr:`state`."""
        return self.state

    @property
    def delivery_state(self) -> DeliveryState:
        """Alias for :attr:`state`."""
        return self.state

    @property
    def method(self) -> DeliveryMethod | str:
        """Alias for :attr:`provider`."""
        return self.provider

    @property
    def type(self) -> DeliveryMethod | str:
        """Alias for :attr:`provider`."""
        return self.provider

    @property
    def kind(self) -> DeliveryMethod | str:
        """Alias for :attr:`provider`."""
        return self.provider

    @property
    def destination(self) -> str:
        """Alias for :attr:`target`."""
        return self.target

    @property
    def recipient(self) -> str:
        """The email recipient or the generic target."""
        return self.email or self.target

    @property
    def folder(self) -> str:
        """Alias for :attr:`path`."""
        return self.path

    @property
    def link(self) -> str:
        """Alias for :attr:`url`."""
        return self.url

    @property
    def outputs(self) -> list[str]:
        """Alias for :attr:`files`."""
        return self.files

    @property
    def file_count(self) -> int:
        """The number of files in the delivery."""
        return len(self.files)

    @property
    def has_files(self) -> bool:
        """Whether the delivery carries at least one file."""
        return bool(self.files)

    @property
    def attempts_remaining(self) -> int:
        """How many attempts are still allowed, never negative."""
        return max(0, self.max_attempts - self.attempts)

    @property
    def has_attempts_left(self) -> bool:
        """Whether another attempt may be started."""
        return self.attempts < self.max_attempts

    @property
    def is_delivered(self) -> bool:
        """Whether the delivery reached its target."""
        return self.state is DeliveryState.DELIVERED

    @property
    def delivered(self) -> bool:
        """Alias for :attr:`is_delivered`."""
        return self.is_delivered

    @property
    def is_failed(self) -> bool:
        """Whether the last attempt failed."""
        return self.state is DeliveryState.FAILED

    @property
    def is_scheduled(self) -> bool:
        """Whether the delivery is waiting for its scheduled time."""
        return self.state is DeliveryState.SCHEDULED

    @property
    def is_terminal(self) -> bool:
        """Whether the delivery reached a final state."""
        return self.state in _TERMINAL_DELIVERY_STATES

    @property
    def can_retry(self) -> bool:
        """Whether the delivery is not terminal and has attempts left."""
        return self.has_attempts_left and not self.is_terminal

    @property
    def has_confirmation(self) -> bool:
        """Whether a confirmation reference was recorded."""
        return bool(self.confirmation)

    @property
    def confirmed_by(self) -> str:
        """Alias for :attr:`Attestation.confirmed_by`."""
        return self.attestation.confirmed_by

    @property
    def rights_confirmed(self) -> bool:
        """Alias for :attr:`Attestation.rights_confirmed`."""
        return self.attestation.rights_confirmed

    @property
    def disclosure_confirmed(self) -> bool:
        """Alias for :attr:`Attestation.disclosure_confirmed`."""
        return self.attestation.disclosure_confirmed

    @property
    def confirmed_at(self) -> datetime | None:
        """Alias for :attr:`Attestation.confirmed_at`."""
        return self.attestation.confirmed_at

    @property
    def license_summary(self) -> str | dict[str, Any]:
        """Alias for :attr:`Attestation.license_summary`."""
        return self.attestation.license_summary

    @property
    def scheduled_at(self) -> datetime | None:
        """Alias for :attr:`scheduled_for`."""
        return self.scheduled_for

    @property
    def sent_at(self) -> datetime | None:
        """Alias for :attr:`delivered_at`."""
        return self.delivered_at

    @property
    def expiry(self) -> datetime | None:
        """Alias for :attr:`expires_at`."""
        return self.expires_at

    @property
    def target_url(self) -> str:
        """Alias for :attr:`url`."""
        return self.url

    @property
    def output_paths(self) -> list[str]:
        """Alias for :attr:`files`."""
        return self.files

    def is_due(self, *, now: datetime | None = None) -> bool:
        """Whether :attr:`scheduled_for` has passed, ignoring the state."""
        if self.scheduled_for is None:
            return True
        current = _normalize_timestamp(now) if now is not None else datetime.now(UTC)
        return current >= self.scheduled_for


class PlanApprovalMode(StrEnum):
    """Whether a plan needs human approval before it executes."""

    ALWAYS = "always"
    ASK = "ask"
    NEVER = "never"


_PLAN_APPROVAL_ALIASES = {
    "always": PlanApprovalMode.ALWAYS,
    "manual": PlanApprovalMode.ALWAYS,
    "required": PlanApprovalMode.ALWAYS,
    "ask": PlanApprovalMode.ASK,
    "confirm": PlanApprovalMode.ASK,
    "prompt": PlanApprovalMode.ASK,
    "never": PlanApprovalMode.NEVER,
    "auto": PlanApprovalMode.NEVER,
    "automatic": PlanApprovalMode.NEVER,
    "no": PlanApprovalMode.NEVER,
}


class PlanStepState(StrEnum):
    """The execution state of one plan step."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


_PLAN_STEP_STATE_ALIASES = {
    "pending": PlanStepState.PENDING,
    "queued": PlanStepState.PENDING,
    "waiting": PlanStepState.PENDING,
    "ready": PlanStepState.PENDING,
    "running": PlanStepState.RUNNING,
    "active": PlanStepState.RUNNING,
    "in_progress": PlanStepState.RUNNING,
    "succeeded": PlanStepState.SUCCEEDED,
    "success": PlanStepState.SUCCEEDED,
    "complete": PlanStepState.SUCCEEDED,
    "completed": PlanStepState.SUCCEEDED,
    "done": PlanStepState.SUCCEEDED,
    "failed": PlanStepState.FAILED,
    "error": PlanStepState.FAILED,
    "skipped": PlanStepState.SKIPPED,
    "cancelled": PlanStepState.CANCELLED,
    "canceled": PlanStepState.CANCELLED,
}

_TERMINAL_PLAN_STEP_STATES = frozenset(
    {
        PlanStepState.SUCCEEDED,
        PlanStepState.FAILED,
        PlanStepState.SKIPPED,
        PlanStepState.CANCELLED,
    }
)


class PlanStep(ModelBase):
    """One tool call inside a plan, with its arguments and dependencies.

    ``depends_on`` lists the ids of steps that must finish first. ``estimate_usd``
    is the pre-execution quote and ``status`` records how far the step got.
    """

    id: str = Field(
        default_factory=new_step_id,
        min_length=1,
        validation_alias=AliasChoices("id", "step_id", "step"),
    )
    index: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("index", "order", "position"),
    )
    tool: str = Field(
        default="",
        validation_alias=AliasChoices(
            "tool",
            "name",
            "tool_name",
            "action",
            "capability",
        ),
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices(
            "arguments",
            "args",
            "params",
            "inputs",
            "options",
        ),
    )
    depends_on: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "depends_on",
            "dependencies",
            "needs",
            "after",
        ),
    )
    fallbacks: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "fallbacks",
            "fallback_tools",
            "alternatives",
        ),
    )
    status: PlanStepState = Field(
        default=PlanStepState.PENDING,
        validation_alias=AliasChoices("status", "state", "step_state"),
    )
    estimate_usd: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "estimate_usd",
            "estimated_cost_usd",
            "estimate",
            "cost_estimate",
        ),
    )
    cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("cost_usd", "cost", "actual_cost_usd"),
    )
    result: Any = None
    error_code: str = Field(
        default="",
        validation_alias=AliasChoices("error_code", "failure_code"),
    )
    error_message: str = Field(
        default="",
        validation_alias=AliasChoices("error_message", "error", "failure_message"),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("metadata", "meta", "details"),
    )

    @field_validator(
        "id",
        "tool",
        "error_code",
        "error_message",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("arguments", "metadata", mode="before")
    @classmethod
    def _coerce_mapping(cls, value: Any) -> Any:
        return _coerce_mapping(value)

    @field_validator("depends_on", mode="before")
    @classmethod
    def _normalize_depends_on(cls, value: Any) -> Any:
        return _normalize_names(value)

    @field_validator("fallbacks", mode="before")
    @classmethod
    def _normalize_fallbacks(cls, value: Any) -> Any:
        return _normalize_names(value)

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: Any) -> Any:
        return _enum_value(value, PlanStepState, _PLAN_STEP_STATE_ALIASES)

    @property
    def step_id(self) -> str:
        """Alias for :attr:`id`."""
        return self.id

    @property
    def name(self) -> str:
        """Alias for :attr:`tool`."""
        return self.tool

    @property
    def tool_name(self) -> str:
        """Alias for :attr:`tool`."""
        return self.tool

    @property
    def args(self) -> dict[str, Any]:
        """Alias for :attr:`arguments`."""
        return self.arguments

    @property
    def dependencies(self) -> list[str]:
        """Alias for :attr:`depends_on`."""
        return self.depends_on

    @property
    def fallback_tools(self) -> list[str]:
        """Alias for :attr:`fallbacks`."""
        return self.fallbacks

    @property
    def state(self) -> PlanStepState:
        """Alias for :attr:`status`."""
        return self.status

    @property
    def estimate(self) -> float:
        """Alias for :attr:`estimate_usd`."""
        return self.estimate_usd

    @property
    def succeeded(self) -> bool:
        """Whether the step completed successfully."""
        return self.status is PlanStepState.SUCCEEDED

    @property
    def failed(self) -> bool:
        """Whether the step failed."""
        return self.status is PlanStepState.FAILED

    @property
    def is_terminal(self) -> bool:
        """Whether the step reached a final state."""
        return self.status in _TERMINAL_PLAN_STEP_STATES


class PlanState(StrEnum):
    """The lifecycle state of an assistant plan."""

    DRAFT = "draft"
    PENDING = "pending"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


_PLAN_STATE_ALIASES = {
    "draft": PlanState.DRAFT,
    "pending": PlanState.PENDING,
    "queued": PlanState.PENDING,
    "proposed": PlanState.PENDING_APPROVAL,
    "pending_approval": PlanState.PENDING_APPROVAL,
    "awaiting_approval": PlanState.PENDING_APPROVAL,
    "approved": PlanState.APPROVED,
    "accepted": PlanState.APPROVED,
    "rejected": PlanState.REJECTED,
    "denied": PlanState.REJECTED,
    "running": PlanState.RUNNING,
    "executing": PlanState.RUNNING,
    "in_progress": PlanState.RUNNING,
    "succeeded": PlanState.SUCCEEDED,
    "success": PlanState.SUCCEEDED,
    "complete": PlanState.SUCCEEDED,
    "completed": PlanState.SUCCEEDED,
    "done": PlanState.SUCCEEDED,
    "failed": PlanState.FAILED,
    "error": PlanState.FAILED,
    "cancelled": PlanState.CANCELLED,
    "canceled": PlanState.CANCELLED,
}

_TERMINAL_PLAN_STATES = frozenset(
    {
        PlanState.SUCCEEDED,
        PlanState.FAILED,
        PlanState.REJECTED,
        PlanState.CANCELLED,
    }
)


class Plan(ModelBase):
    """A validated plan of ordered tool calls produced from an instruction.

    ``steps`` hold the calls, ``assumptions`` records what the planner had to
    infer and ``estimate_usd`` is the total quote. ``approval_mode`` and the
    approval fields decide whether the plan may run without a human.
    """

    id: str = Field(
        default_factory=new_plan_id,
        min_length=1,
        validation_alias=AliasChoices("id", "plan_id", "plan"),
    )
    instruction: str = Field(
        default="",
        validation_alias=AliasChoices(
            "instruction",
            "prompt",
            "request",
            "text",
        ),
    )
    state: PlanState = Field(
        default=PlanState.DRAFT,
        validation_alias=AliasChoices("state", "status", "plan_state"),
    )
    steps: list[PlanStep] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "steps",
            "tool_calls",
            "calls",
            "actions",
        ),
    )
    assumptions: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("assumptions", "notes"),
    )
    approval_mode: PlanApprovalMode = Field(
        default=PlanApprovalMode.ASK,
        validation_alias=AliasChoices(
            "approval_mode",
            "approval",
            "approval_policy",
        ),
    )
    approved: bool = Field(
        default=False,
        validation_alias=AliasChoices("approved", "is_approved"),
    )
    approved_by: str = Field(
        default="",
        validation_alias=AliasChoices("approved_by", "approver"),
    )
    approved_at: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices("approved_at", "approval_time"),
    )
    estimate_usd: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "estimate_usd",
            "estimated_cost_usd",
            "estimate",
        ),
    )
    cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("cost_usd", "cost", "actual_cost_usd"),
    )
    budget_usd: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices(
            "budget_usd",
            "budget",
            "max_cost_usd",
        ),
    )
    run_id: str = Field(
        default="",
        validation_alias=AliasChoices("run_id", "run"),
    )
    profile: str = Field(
        default="",
        validation_alias=AliasChoices("profile", "profile_name"),
    )
    format: str = Field(
        default="",
        validation_alias=AliasChoices("format", "format_name", "recipe"),
    )
    idempotency_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "idempotency_key",
            "idempotency",
            "dedupe_key",
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
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("metadata", "meta", "details"),
    )

    @field_validator(
        "instruction",
        "approved_by",
        "run_id",
        "profile",
        "format",
        "idempotency_key",
        "error_code",
        "error_message",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("steps", mode="before")
    @classmethod
    def _coerce_steps(cls, value: Any) -> Any:
        return _coerce_names(value)

    @field_validator("assumptions", mode="before")
    @classmethod
    def _normalize_assumptions(cls, value: Any) -> Any:
        return _normalize_names(value)

    @field_validator("state", mode="before")
    @classmethod
    def _normalize_state(cls, value: Any) -> Any:
        return _enum_value(value, PlanState, _PLAN_STATE_ALIASES)

    @field_validator("approval_mode", mode="before")
    @classmethod
    def _normalize_approval(cls, value: Any) -> Any:
        if isinstance(value, bool):
            return PlanApprovalMode.ALWAYS if value else PlanApprovalMode.NEVER
        return _enum_value(value, PlanApprovalMode, _PLAN_APPROVAL_ALIASES)

    @field_validator("approved_at")
    @classmethod
    def _normalize_optional_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _normalize_timestamp(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, value: Any) -> Any:
        return _coerce_mapping(value)

    @property
    def plan_id(self) -> str:
        """Alias for :attr:`id`."""
        return self.id

    @property
    def status(self) -> PlanState:
        """Alias for :attr:`state`."""
        return self.state

    @property
    def plan_state(self) -> PlanState:
        """Alias for :attr:`state`."""
        return self.state

    @property
    def tool_calls(self) -> list[PlanStep]:
        """Alias for :attr:`steps`."""
        return self.steps

    @property
    def step_count(self) -> int:
        """The number of steps in the plan."""
        return len(self.steps)

    @property
    def step_ids(self) -> list[str]:
        """The identifiers of every step in order."""
        return [step.id for step in self.steps]

    @property
    def total_estimate_usd(self) -> float:
        """The sum of the step estimates, treating unknown as zero."""
        return sum(step.estimate_usd for step in self.steps)

    @property
    def estimate(self) -> float:
        """Alias for :attr:`estimate_usd`."""
        return self.estimate_usd

    @property
    def approval(self) -> PlanApprovalMode:
        """Alias for :attr:`approval_mode`."""
        return self.approval_mode

    @property
    def is_approved(self) -> bool:
        """Whether the plan has been approved."""
        return self.approved or self.state is PlanState.APPROVED

    @property
    def needs_approval(self) -> bool:
        """Whether the plan still waits for a human decision."""
        if self.approval_mode is PlanApprovalMode.NEVER:
            return False
        return not self.is_approved

    @property
    def is_running(self) -> bool:
        """Whether the plan is executing."""
        return self.state is PlanState.RUNNING

    @property
    def is_terminal(self) -> bool:
        """Whether the plan reached a final state."""
        return self.state in _TERMINAL_PLAN_STATES

    @property
    def plan_steps(self) -> list[PlanStep]:
        """Alias for :attr:`steps`."""
        return self.steps

    @property
    def instruction_text(self) -> str:
        """Alias for :attr:`instruction`."""
        return self.instruction

    @property
    def estimated_cost_usd(self) -> float:
        """Alias for :attr:`estimate_usd`."""
        return self.estimate_usd

    @property
    def approval_required(self) -> bool:
        """Alias for :attr:`needs_approval`."""
        return self.needs_approval

    def step_by_id(self, step_id: str) -> PlanStep | None:
        """Return the step with *step_id*, or ``None`` when it is absent."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def has_cycles(self) -> bool:
        """Whether the step dependency graph contains a cycle."""
        dependencies = {step.id: set(step.depends_on) for step in self.steps}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            for dependency in dependencies.get(node, set()):
                if visit(dependency):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(visit(node) for node in dependencies)


class CompositionReference(ModelBase):
    """A pointer to a stored composition, not the composition itself.

    The full composition schema lives in the compose package. Records that only
    need to remember which composition produced an output store this reference:
    its id, path, content hash and the renderer toolchain fingerprint used.
    """

    id: str = Field(
        default_factory=new_composition_id,
        min_length=1,
        validation_alias=AliasChoices(
            "id",
            "composition_id",
            "composition",
        ),
    )
    path: str = Field(
        default="",
        validation_alias=AliasChoices(
            "path",
            "file",
            "file_path",
            "uri",
            "url",
            "location",
        ),
    )
    content_hash: str = Field(
        default="",
        validation_alias=AliasChoices(
            "content_hash",
            "hash",
            "sha256",
            "digest",
            "checksum",
            "composition_hash",
        ),
    )
    format: str = Field(
        default="",
        validation_alias=AliasChoices(
            "format",
            "output_type",
            "type",
            "kind",
            "media_type",
        ),
    )
    version: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("version", "schema_version_number", "revision"),
    )
    renderer_version: str = Field(
        default="",
        validation_alias=AliasChoices(
            "renderer_version",
            "renderer",
            "renderer_id",
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
    duration_seconds: float | None = Field(
        default=None,
        ge=0.0,
        validation_alias=AliasChoices(
            "duration_seconds",
            "duration",
            "length_seconds",
        ),
    )
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    fps: float | None = Field(default=None, gt=0)
    assets: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "assets",
            "asset_ids",
            "asset_paths",
            "inputs",
        ),
    )
    pinned: bool = Field(
        default=False,
        validation_alias=AliasChoices("pinned", "is_pinned", "final"),
    )
    run_id: str = Field(
        default="",
        validation_alias=AliasChoices("run_id", "run"),
    )
    language: str = Field(
        default="",
        validation_alias=AliasChoices("language", "lang", "locale"),
    )
    platform: str = Field(
        default="",
        validation_alias=AliasChoices("platform", "platform_name"),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("metadata", "meta", "details"),
    )

    @field_validator(
        "path",
        "content_hash",
        "format",
        "renderer_version",
        "run_id",
        "language",
        "platform",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("toolchain_fingerprint", mode="before")
    @classmethod
    def _coerce_toolchain(cls, value: Any) -> Any:
        if value is None:
            return {}
        return value

    @field_validator("assets", mode="before")
    @classmethod
    def _normalize_assets(cls, value: Any) -> Any:
        return _normalize_names(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, value: Any) -> Any:
        return _coerce_mapping(value)

    @property
    def composition_id(self) -> str:
        """Alias for :attr:`id`."""
        return self.id

    @property
    def file(self) -> str:
        """Alias for :attr:`path`."""
        return self.path

    @property
    def uri(self) -> str:
        """Alias for :attr:`path`."""
        return self.path

    @property
    def hash(self) -> str:
        """Alias for :attr:`content_hash`."""
        return self.content_hash

    @property
    def sha256(self) -> str:
        """Alias for :attr:`content_hash`."""
        return self.content_hash

    @property
    def composition_hash(self) -> str:
        """Alias for :attr:`content_hash`."""
        return self.content_hash

    @property
    def toolchain(self) -> ToolchainFingerprint:
        """Alias for :attr:`toolchain_fingerprint`."""
        return self.toolchain_fingerprint

    @property
    def fingerprint(self) -> str:
        """The digest of :attr:`toolchain_fingerprint`."""
        return self.toolchain_fingerprint.digest

    @property
    def asset_count(self) -> int:
        """The number of assets referenced by the composition."""
        return len(self.assets)

    @property
    def has_assets(self) -> bool:
        """Whether the composition references at least one asset."""
        return bool(self.assets)

    @property
    def is_pinned(self) -> bool:
        """Alias for :attr:`pinned`."""
        return self.pinned

    @property
    def is_rendered(self) -> bool:
        """Whether a stored file and its hash were recorded."""
        return bool(self.path and self.content_hash)

    @property
    def aspect_ratio(self) -> float | None:
        """The width divided by the height, if both are known."""
        if self.width is None or self.height is None:
            return None
        return self.width / self.height

    @property
    def composition_file(self) -> str:
        """Alias for :attr:`path`."""
        return self.path

    @property
    def composition_path(self) -> str:
        """Alias for :attr:`path`."""
        return self.path

    @property
    def output_hash(self) -> str:
        """Alias for :attr:`content_hash`."""
        return self.content_hash

    @property
    def renderer(self) -> str:
        """Alias for :attr:`renderer_version`."""
        return self.renderer_version


AttestationModel = Attestation
AttestationRecord = Attestation
PublishAttestation = Attestation
PostAttestation = Attestation
RightsAttestation = Attestation
DisclosureAttestation = Attestation

PostModel = Post
PostRecord = Post
PostStatus = PostState

DeliveryModel = Delivery
DeliveryRecord = Delivery
DeliveryStatus = DeliveryState
DeliveryType = DeliveryMethod
DeliveryProviderType = DeliveryMethod

PlanModel = Plan
PlanRecord = Plan
PlanStatus = PlanState
PlanApproval = PlanApprovalMode
ApprovalMode = PlanApprovalMode
PlanStepModel = PlanStep
PlanStepStatus = PlanStepState
Step = PlanStep
StepModel = PlanStep
StepState = PlanStepState
ToolCall = PlanStep

CompositionModel = CompositionReference
CompositionRecord = CompositionReference
CompositionRef = CompositionReference
CompositionReferenceModel = CompositionReference
CompositionReferenceRecord = CompositionReference
Composition = CompositionReference
