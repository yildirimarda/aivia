"""Session record: assets, current composition, plan history and versions.

A session is the top level record of one assistant run. It keeps the assets
used or produced in the run, a pointer to the current composition, the ordered
history of plans that were executed and a list of version snapshots so the
session can be rolled back or compared across edits. Records carry a schema
version and timestamps like every other persisted record, and serialize to
canonical JSON so a session round trips through ``model_dump`` and
``model_validate`` without losing information.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from pydantic import AliasChoices, Field, field_validator

from aivia.core.ids import new_session_id
from aivia.core.models.asset import Asset
from aivia.core.models.base import ModelBase, utcnow
from aivia.core.models.io import CompositionReference, Plan

__all__ = [
    "Session",
    "SessionModel",
    "SessionRecord",
    "SessionVersion",
]

_REASONABLE_TEXT_FIELDS = (
    "title",
    "run_id",
    "profile",
)


def _coerce_plan_list(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, Plan):
        return [value]
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


class SessionVersion(ModelBase):
    """One snapshot of a session at a given version.

    ``version`` is the sequential revision number. ``note`` records what
    changed, ``composition_id`` and ``plan_ids`` reference the composition and
    plans that were current at that point and ``asset_labels`` lists the
    labels of the assets captured with the snapshot.
    """

    version: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("version", "revision", "schema_version"),
    )
    created_at: datetime = Field(default_factory=utcnow)
    note: str = Field(
        default="",
        validation_alias=AliasChoices("note", "message", "summary", "comment"),
    )
    composition_id: str = Field(
        default="",
        validation_alias=AliasChoices(
            "composition_id",
            "composition",
            "current_composition",
        ),
    )
    plan_ids: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("plan_ids", "plans", "plan"),
    )
    asset_labels: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("asset_labels", "assets", "labels"),
    )

    @field_validator("composition_id", "note", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("plan_ids", "asset_labels", mode="before")
    @classmethod
    def _normalize_names(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, Mapping):
            return [str(value)]
        if isinstance(value, (list, tuple)):
            names: list[str] = []
            for raw in value:
                if isinstance(raw, str):
                    name = raw.strip()
                    if name and name not in names:
                        names.append(name)
                else:
                    names.append(str(raw))
            return names
        return value

    @property
    def revision(self) -> int:
        """Alias for :attr:`version`."""
        return self.version


class Session(ModelBase):
    """One assistant session.

    ``assets`` holds the files used or produced in the session, ``current``
    points at the composition being built, ``history`` is the ordered list of
    plans executed so far and ``versions`` records snapshots taken as the
    session evolves.
    """

    id: str = Field(
        default_factory=new_session_id,
        min_length=1,
        validation_alias=AliasChoices("id", "session_id", "session"),
    )
    title: str = Field(
        default="",
        validation_alias=AliasChoices("title", "name", "label", "slug"),
    )
    run_id: str = Field(
        default="",
        validation_alias=AliasChoices("run_id", "run"),
    )
    profile: str = Field(
        default="",
        validation_alias=AliasChoices("profile", "profile_name"),
    )
    assets: list[Asset] = Field(
        default_factory=list,
        validation_alias=AliasChoices("assets", "asset_records", "files"),
    )
    current: CompositionReference = Field(
        default_factory=CompositionReference,
        validation_alias=AliasChoices(
            "current",
            "current_composition",
            "composition",
            "composition_ref",
        ),
    )
    history: list[Plan] = Field(
        default_factory=list,
        validation_alias=AliasChoices("history", "plans", "plan_history"),
    )
    versions: list[SessionVersion] = Field(
        default_factory=list,
        validation_alias=AliasChoices("versions", "version_history", "revisions"),
    )
    active_version: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("active_version", "current_version", "head"),
    )

    @field_validator(*_REASONABLE_TEXT_FIELDS, mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("assets", mode="before")
    @classmethod
    def _coerce_assets(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, Asset):
            return [value]
        if isinstance(value, Mapping):
            return [value]
        if isinstance(value, (list, tuple)):
            return list(value)
        return value

    @field_validator("current", mode="before")
    @classmethod
    def _coerce_current(cls, value: Any) -> Any:
        if value is None or isinstance(value, CompositionReference):
            return value
        if isinstance(value, Mapping):
            return value
        return value

    @field_validator("history", mode="before")
    @classmethod
    def _coerce_history(cls, value: Any) -> Any:
        return _coerce_plan_list(value)

    @field_validator("versions", mode="before")
    @classmethod
    def _coerce_versions(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, SessionVersion):
            return [value]
        if isinstance(value, Mapping):
            return [value]
        if isinstance(value, (list, tuple)):
            return list(value)
        return value

    @field_validator("active_version", mode="after")
    @classmethod
    def _check_active_version(cls, value: int) -> int:
        if value < 1:
            raise ValueError("active_version must be at least 1")
        return value

    @property
    def session_id(self) -> str:
        """Alias for :attr:`id`."""
        return self.id

    @property
    def plan_count(self) -> int:
        """The number of plans recorded in the session history."""
        return len(self.history)

    @property
    def asset_count(self) -> int:
        """The number of assets held by the session."""
        return len(self.assets)

    @property
    def version_count(self) -> int:
        """The number of version snapshots recorded."""
        return len(self.versions)

    @property
    def latest_version(self) -> SessionVersion | None:
        """The highest numbered version snapshot, if any were recorded."""
        if not self.versions:
            return None
        return max(self.versions, key=lambda item: item.version)

    def plan_by_id(self, plan_id: str) -> Plan | None:
        """Return the plan with *plan_id* from the history, or ``None``."""
        for plan in self.history:
            if plan.id == plan_id:
                return plan
        return None

    def version_by_number(self, version: int) -> SessionVersion | None:
        """Return the version snapshot with *version*, or ``None``."""
        for item in self.versions:
            if item.version == version:
                return item
        return None


SessionModel = Session
SessionRecord = Session
