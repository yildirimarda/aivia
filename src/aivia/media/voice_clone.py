"""Register, list and delete cloned voices behind explicit consent.

Voice cloning is a rights-sensitive operation, so the local index refuses to
store a provider voice id unless the caller supplies an explicit consent grant.
The record keeps the exact provider voice id, the owner whose voice it is and
the consent metadata beside it, which lets later speech requests pin the id and
prove why it was allowed.

The service owns the local registry only. The provider adapter owns creating and
deleting the voice at the vendor; this module never calls a provider, so a
cloned voice id can be recorded once and reused offline.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import Engine
from sqlalchemy.exc import IntegrityError

from aivia.core.db import VoiceCloneRepository
from aivia.core.errors import (
    ContentPolicyError,
    InvalidRequestError,
    ValidationError,
)
from aivia.core.models.base import utcnow
from aivia.core.models.voice import VoiceClone

__all__ = [
    "ConsentInput",
    "VoiceCloneRequest",
    "VoiceCloneService",
    "normalize_consent",
]

ConsentInput = bool | Mapping[str, Any] | None
"""The consent values a caller may pass to :meth:`VoiceCloneService.create`."""

_MISSING = object()
_GRANT_KEYS = ("granted", "consent", "allowed", "approved", "voice_cloning")


def _strip(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            f"{field} must be a non-empty string",
            details={"field": field},
        )
    return value.strip()


def normalize_consent(value: ConsentInput) -> dict[str, Any]:
    """Return consent metadata after requiring an explicit grant.

    ``True`` grants consent. A mapping must carry a truthy ``granted`` (or one
    of its documented aliases); a missing or false grant raises
    :class:`~aivia.core.errors.ContentPolicyError`, because the safe default is
    "no". The returned mapping always has ``granted`` set to ``True`` and a
    ``granted_at`` timestamp when the caller did not supply one.
    """
    if isinstance(value, bool):
        if not value:
            raise ContentPolicyError(
                "voice cloning requires explicit consent",
                details={"consent": False},
            )
        return {"granted": True, "granted_at": utcnow().isoformat()}
    if isinstance(value, Mapping):
        metadata = dict(value)
        granted: Any = None
        for key in _GRANT_KEYS:
            if key in metadata:
                granted = metadata[key]
                break
        if not isinstance(granted, bool) or not granted:
            raise ContentPolicyError(
                "voice cloning requires explicit consent",
                details={"consent": "missing or false"},
            )
        metadata["granted"] = True
        metadata.setdefault("granted_at", utcnow().isoformat())
        return metadata
    raise ContentPolicyError(
        "voice cloning requires explicit consent",
        details={"consent": "missing"},
    )


class VoiceCloneRequest(BaseModel):
    """The values needed to register one provider voice id.

    ``consent`` is optional in the model so the service can raise the rights
    error itself instead of a generic validation failure. ``provider`` and
    ``provider_voice_id`` name the vendor and the exact id a later speech
    request must pin; ``owner`` names whose voice it is.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str = ""
    provider: str = Field(min_length=1)
    provider_voice_id: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    consent: bool | dict[str, Any] | None = None
    language: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", "provider_voice_id", "owner", mode="before")
    @classmethod
    def _strip_required(cls, value: Any) -> Any:
        return _strip(value)

    @field_validator("name", "language", mode="before")
    @classmethod
    def _strip_optional(cls, value: Any) -> Any:
        return _strip(value) or ""

    @field_validator("language")
    @classmethod
    def _normalize_language(cls, value: str) -> str:
        return value.casefold()

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, value: Any) -> Any:
        return {} if value is None else value


def _request_data(
    request: VoiceCloneRequest | VoiceClone | Mapping[str, Any] | str | None,
    values: Mapping[str, Any],
) -> dict[str, Any]:
    if request is None:
        data: dict[str, Any] = {}
    elif isinstance(request, (VoiceClone, VoiceCloneRequest)):
        data = request.model_dump(mode="python")
    elif isinstance(request, Mapping):
        data = dict(request)
    elif isinstance(request, str):
        data = {"name": request}
    else:
        raise TypeError(
            "request must be a VoiceCloneRequest, VoiceClone, mapping, string or None"
        )
    data.update(values)
    return data


class VoiceCloneService:
    """Create, list and delete the local index of cloned voices.

    The service is bound to one database engine, so a clone created by one
    process is visible to the next. Every create requires explicit consent;
    list and delete accept an optional owner so a caller can scope the view to
    the person whose voice it is.
    """

    def __init__(
        self,
        engine: Engine | VoiceCloneRepository | None = None,
        *,
        repository: VoiceCloneRepository | None = None,
    ) -> None:
        if isinstance(engine, VoiceCloneRepository):
            if repository is not None:
                raise TypeError("pass either a repository or an engine, not both")
            repository = engine
            engine = None
        if repository is None:
            if not isinstance(engine, Engine):
                raise TypeError("engine must be a SQLAlchemy Engine")
            repository = VoiceCloneRepository(engine)
        elif engine is not None and not isinstance(engine, Engine):
            raise TypeError("engine must be a SQLAlchemy Engine")
        self._repository = repository

    @property
    def repository(self) -> VoiceCloneRepository:
        """The repository this service reads and writes through."""
        return self._repository

    @property
    def engine(self) -> Engine:
        """The database engine behind the repository."""
        return self._repository.engine

    def create(
        self,
        request: VoiceCloneRequest | VoiceClone | Mapping[str, Any] | str | None = None,
        **values: Any,
    ) -> VoiceClone:
        """Register one provider voice id after checking explicit consent.

        A missing or false consent raises
        :class:`~aivia.core.errors.ContentPolicyError`. The stored record always
        keeps the provider voice id, owner and consent metadata together.
        """
        data = _request_data(request, values)
        consent = data.pop("consent", _MISSING)
        if consent is _MISSING:
            raise ContentPolicyError(
                "voice cloning requires explicit consent",
                details={"consent": "missing"},
            )
        data["consent"] = normalize_consent(consent)
        for field in ("provider", "provider_voice_id", "owner"):
            data[field] = _required_text(data.get(field), field=field)
        if not data.get("name"):
            data["name"] = data["provider_voice_id"]
        record = VoiceClone.model_validate(data)
        try:
            return self._repository.create(record)
        except IntegrityError as exc:
            raise InvalidRequestError(
                "a clone for this provider voice id already exists",
                details={
                    "provider": record.provider,
                    "provider_voice_id": record.provider_voice_id,
                },
            ) from exc

    def list(
        self,
        owner: str | None = None,
        *,
        provider: str | None = None,
        language: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[VoiceClone]:
        """Return clones, optionally scoped to an owner, provider or language."""
        filters: dict[str, Any] = {}
        if owner is not None:
            filters["owner"] = _required_text(owner, field="owner")
        if provider is not None:
            filters["provider"] = _required_text(provider, field="provider")
        if language is not None:
            filters["language"] = _required_text(language, field="language").casefold()
        return self._repository.list(limit=limit, offset=offset, **filters)

    def get(self, clone_id: str, *, owner: str | None = None) -> VoiceClone | None:
        """Return one clone, or ``None`` when it is absent or owned by another."""
        identifier = _required_text(clone_id, field="clone_id")
        record = self._repository.get(identifier)
        if record is None:
            return None
        if owner is not None and record.owner != _required_text(owner, field="owner"):
            return None
        return record

    def delete(self, clone_id: str, *, owner: str | None = None) -> VoiceClone | None:
        """Delete one clone and return it, or ``None`` when it is not found.

        When *owner* is supplied, a clone owned by someone else is treated as
        not found so a caller cannot remove another owner's voice.
        """
        record = self.get(clone_id, owner=owner)
        if record is None:
            return None
        return self._repository.delete(record.id)
