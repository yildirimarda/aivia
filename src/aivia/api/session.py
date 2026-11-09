"""Session HTTP routes: create, message, preview and history.

Every route is scoped to one session. A session's assets, plans and current
composition are only ever visible through the session that owns them, so a
caller cannot reach another session's assets or history. Sessions live in an
in-memory store on the application state; there is no persistent backend for
them yet.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import AliasChoices, BaseModel, Field, field_validator

from aivia.core.models import Asset, Plan, Session

__all__ = ["register_session_routes"]


class _SessionCreateRequest(BaseModel):
    """The fields accepted when creating one session."""

    id: str | None = Field(
        default=None,
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
    assets: list[dict[str, Any]] = Field(
        default_factory=list,
        validation_alias=AliasChoices("assets", "asset_records", "files"),
    )

    @field_validator("id", "title", "run_id", "profile", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class _MessageRequest(BaseModel):
    """One message recorded into a session's plan history."""

    message: str = Field(
        default="",
        validation_alias=AliasChoices("message", "text", "instruction", "prompt"),
    )

    @field_validator("message", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


def _session_store(app: FastAPI) -> dict[str, Session]:
    existing = getattr(app.state, "sessions", None)
    if existing is None:
        existing = {}
        app.state.sessions = existing
    return existing


def _require_session(app: FastAPI, session_id: str) -> Session:
    session = _session_store(app).get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"session {session_id!r} not found")
    return session


def _session_payload(session: Session) -> dict[str, Any]:
    data = session.model_dump(mode="json")
    data["session_id"] = session.id
    data["plan_count"] = session.plan_count
    data["asset_count"] = session.asset_count
    data["version_count"] = session.version_count
    return data


def register_session_routes(app: FastAPI) -> None:
    """Attach the session routes to *app*.

    Routes are scoped to one session: assets and history are only ever visible
    through the session that owns them.
    """

    def create(request: _SessionCreateRequest) -> dict[str, Any]:
        session = Session(
            title=request.title,
            run_id=request.run_id,
            profile=request.profile,
            assets=[Asset.model_validate(item) for item in request.assets],
        )
        if request.id is not None:
            session.id = request.id
        _session_store(app)[session.id] = session
        return _session_payload(session)

    @app.post(
        "/api/v1/sessions",
        status_code=201,
        tags=["sessions"],
        summary="Create a session",
    )
    def post_create_session(request: _SessionCreateRequest) -> dict[str, Any]:
        """Create one session and return it."""
        return create(request)

    @app.post(
        "/api/v1/sessions/{session_id}/message",
        tags=["sessions"],
        summary="Add a message to a session",
    )
    def post_message(
        session_id: str,
        request: _MessageRequest,
    ) -> dict[str, Any]:
        """Record *message* as a plan in the session history."""
        if not request.message:
            raise HTTPException(status_code=422, detail="message must not be empty")
        session = _require_session(app, session_id)
        session.history.append(Plan(instruction=request.message))
        _session_store(app)[session.id] = session
        return _session_payload(session)

    @app.get(
        "/api/v1/sessions/{session_id}/preview",
        tags=["sessions"],
        summary="Preview a session's current composition",
    )
    def get_preview(session_id: str) -> dict[str, Any]:
        """Return the session's current composition reference."""
        session = _require_session(app, session_id)
        return {
            "session_id": session.id,
            "current": session.current.model_dump(mode="json"),
        }

    @app.get(
        "/api/v1/sessions/{session_id}/history",
        tags=["sessions"],
        summary="List a session's plan history",
    )
    def get_history(session_id: str) -> dict[str, Any]:
        """Return the session's ordered plan history."""
        session = _require_session(app, session_id)
        return {
            "session_id": session.id,
            "history": [plan.model_dump(mode="json") for plan in session.history],
            "plan_count": session.plan_count,
        }

    @app.get(
        "/api/v1/sessions/{session_id}/assets",
        tags=["sessions"],
        summary="List a session's assets",
    )
    def get_assets(session_id: str) -> dict[str, Any]:
        """Return the assets scoped to *session_id*."""
        session = _require_session(app, session_id)
        return {
            "session_id": session.id,
            "assets": [asset.model_dump(mode="json") for asset in session.assets],
            "asset_count": session.asset_count,
        }

    @app.get(
        "/api/v1/sessions/{session_id}/assets/{asset_label}",
        tags=["sessions"],
        summary="Get one asset scoped to a session",
    )
    def get_asset(session_id: str, asset_label: str) -> dict[str, Any]:
        """Return one asset belonging to *session_id*, or 404."""
        session = _require_session(app, session_id)
        for asset in session.assets:
            if asset.label == asset_label:
                return {
                    "session_id": session.id,
                    "asset": asset.model_dump(mode="json"),
                }
        raise HTTPException(
            status_code=404,
            detail=(f"session {session_id!r} has no asset labeled {asset_label!r}"),
        )
