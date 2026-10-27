"""Operational configuration: review, retention, jobs and storage.

These sections describe how produced media is reviewed, how long it is kept,
how background jobs run and where assets are stored. They are deliberately
separate from the pipeline sections so a deployment can tune operations without
touching content settings.

Storage credentials are referenced by name and resolved through the secrets
loader, never written into the configuration file. Selecting the S3 backend
therefore requires the bucket, endpoint, region and both credential references
to be configured explicitly; the local backend requires none of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, model_validator

from aivia.core.config.base import ConfigModel

__all__ = [
    "JobsConfig",
    "RetentionConfig",
    "ReviewConfig",
    "StorageConfig",
]

S3_REQUIRED_FIELDS: tuple[str, ...] = (
    "bucket",
    "endpoint",
    "region",
    "access_key_ref",
    "secret_key_ref",
)

_REVIEW_ALIASES = {
    "approval": "approval_mode",
    "timeout_action": "on_timeout",
    "regenerate_limit": "max_regenerations",
    "max_retries": "max_regenerations",
    "show_preview": "preview",
    "send_preview": "preview",
}


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError("timeout must be a number") from None


class ReviewConfig(ConfigModel):
    """Approval policy for the optional human review stage.

    ``approval_mode`` is ``always`` (a reviewer must approve every run),
    ``ask`` (the default: approve unless a profile says otherwise) or ``never``
    (skip the review stage). When a review is not answered before
    ``timeout_seconds``, ``on_timeout`` decides whether it is rejected,
    approved or only escalated with a notification.
    """

    enabled: bool = True
    approval_mode: Literal["always", "ask", "never"] = "ask"
    timeout_seconds: int = Field(default=86_400, gt=0)
    on_timeout: Literal["reject", "approve", "notify"] = "reject"
    allow_regenerate: bool = True
    max_regenerations: int = Field(default=3, ge=0)
    preview: bool = True
    notify: bool = True

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)

        for alias, canonical in _REVIEW_ALIASES.items():
            if alias in values:
                value = values.pop(alias)
                values.setdefault(canonical, value)

        if "require_approval" in values:
            required = values.pop("require_approval")
            values.setdefault("approval_mode", "ask" if required else "never")
        if "require_review" in values:
            required = values.pop("require_review")
            values.setdefault("approval_mode", "ask" if required else "never")
        if "auto_approve" in values:
            auto = values.pop("auto_approve")
            values.setdefault("approval_mode", "never" if auto else "ask")

        if "timeout_hours" in values:
            hours = values.pop("timeout_hours")
            values.setdefault("timeout_seconds", round(_number(hours) * 3600))
        if "timeout_minutes" in values:
            minutes = values.pop("timeout_minutes")
            values.setdefault("timeout_seconds", round(_number(minutes) * 60))
        return values

    @property
    def approval(self) -> str:
        """Alias for :attr:`approval_mode`."""
        return self.approval_mode

    @property
    def require_approval(self) -> bool:
        """Whether a reviewer has to approve a run."""
        return self.approval_mode != "never"

    @property
    def timeout_hours(self) -> float:
        """The review timeout expressed in hours."""
        return self.timeout_seconds / 3600

    @property
    def timeout_minutes(self) -> float:
        """The review timeout expressed in minutes."""
        return self.timeout_seconds / 60

    @property
    def timeout_action(self) -> str:
        """Alias for :attr:`on_timeout`."""
        return self.on_timeout

    @property
    def regenerate_limit(self) -> int:
        """Alias for :attr:`max_regenerations`."""
        return self.max_regenerations


class RetentionConfig(ConfigModel):
    """How long intermediates, cache, logs and personal data are kept.

    Final outputs are pinned and are never removed unless
    ``delete_final_outputs`` is explicitly enabled.
    """

    intermediates_days: int = Field(default=7, ge=0)
    cache_days: int = Field(default=14, ge=0)
    logs_days: int = Field(default=30, ge=0)
    personal_data_days: int = Field(default=30, ge=0)
    orders_days: int = Field(default=90, ge=0)
    deliveries_days: int = Field(default=30, ge=0)
    web_card_hits_days: int = Field(default=30, ge=0)
    delete_final_outputs: bool = False

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias, canonical in (
            ("hits_days", "web_card_hits_days"),
            ("web_card_days", "web_card_hits_days"),
            ("expired_orders_days", "orders_days"),
            ("expired_deliveries_days", "deliveries_days"),
        ):
            if alias in values:
                value = values.pop(alias)
                values.setdefault(canonical, value)
        return values


class JobsConfig(ConfigModel):
    """Worker concurrency, retry and heartbeat settings for the job queue."""

    concurrency: int = Field(default=1, ge=1)
    max_attempts: int = Field(default=3, ge=1)
    heartbeat_seconds: int = Field(default=30, gt=0)
    poll_seconds: int = Field(default=2, gt=0)
    stale_after_seconds: int = Field(default=120, gt=0)
    max_queue_depth: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _stale_after_heartbeat(self) -> JobsConfig:
        if self.stale_after_seconds <= self.heartbeat_seconds:
            raise ValueError(
                "jobs.stale_after_seconds must be greater than jobs.heartbeat_seconds"
            )
        return self


class StorageConfig(ConfigModel):
    """Where generated media and intermediates are stored.

    Credentials are referenced by name and resolved through the secrets
    loader, never written into the configuration file.
    """

    backend: Literal["local", "s3"] = "local"
    local_dir: str = Field(default="media", min_length=1)
    bucket: str | None = None
    endpoint: str | None = None
    region: str | None = None
    access_key_ref: str | None = None
    secret_key_ref: str | None = None
    prefix: str = ""
    presign_seconds: int = Field(default=3600, gt=0)
    use_path_style: bool = True

    @model_validator(mode="before")
    @classmethod
    def _accept_nested_s3(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        nested = values.pop("s3", None)
        if isinstance(nested, Mapping):
            for key, value in nested.items():
                values.setdefault(key, value)
        return values

    @model_validator(mode="after")
    def _require_s3_fields(self) -> StorageConfig:
        if self.backend != "s3":
            return self
        missing = [
            name
            for name in S3_REQUIRED_FIELDS
            if not str(getattr(self, name) or "").strip()
        ]
        if missing:
            raise ValueError("S3 backend requires: " + ", ".join(missing))
        return self
