"""Order HTTP routes: create, get and list personalized orders.

Every route is protected by the authentication, rate limit and body limit
middleware installed in :func:`aivia.api.app.create_app`, so order creation is
fail-closed without a valid API key. Uploaded filenames are reduced to safe,
single-component path components with :func:`aivia.api.uploads.sanitize_filename`
before they touch the store, so an attacker cannot traverse directories or
smuggle dangerous names through a multipart order.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from sqlalchemy import Engine

from aivia.api.uploads import sanitize_filename
from aivia.core.db import (
    OrderRepository,
    create_database_engine,
    upgrade_database,
)

__all__ = ["register_order_routes"]

_JSON_OBJECT_FIELDS = ("variables", "pricing", "metadata")
_JSON_ARRAY_FIELDS = ("outputs",)
_MAX_FILES = 8
_MAX_FILE_BYTES = 32 * 1024 * 1024


def _parse_json_field(value: str, field: str) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{field} must be valid JSON",
        ) from exc
    return parsed


def register_order_routes(
    app: FastAPI,
    *,
    engine: Engine | None,
    environ: Mapping[str, str] | None,
    database_path: str | Path | None,
    data_dir: str | Path | None,
) -> None:
    """Attach the order routes to *app*.

    Routes are protected by the auth, rate limit and body limit middleware the
    application installs, so none of them is reachable without a valid API key.
    *engine* may be injected; otherwise the API opens the configured local
    database lazily on the first order request.
    """

    def database() -> Engine:
        existing = getattr(app.state, "engine", None)
        if isinstance(existing, Engine):
            return existing
        created = create_database_engine(
            path=database_path,
            data_dir=data_dir,
            environ=environ,
        )
        upgrade_database(engine=created)
        app.state.engine = created
        app.state.api_engine_owned = True
        return created

    @app.post(
        "/api/v1/orders",
        status_code=201,
        tags=["orders"],
        summary="Create a personalized order",
    )
    def create_order(
        occasion: str = Form(""),
        name: str = Form(""),
        message: str = Form(""),
        language: str = Form(""),
        profile: str = Form(""),
        level: str = Form(""),
        timezone: str = Form("UTC"),
        currency: str = Form("USD"),
        amount_usd: float = Form(0.0),
        budget_usd: str | None = Form(None),
        variables: str = Form("{}"),
        outputs: str = Form("[]"),
        pricing: str = Form("{}"),
        metadata: str = Form("{}"),
        scheduled_for: str | None = Form(None),
        expires_at: str | None = Form(None),
        idempotency_key: str = Form(""),
        files: Annotated[list[UploadFile] | None, File()] = None,
    ) -> dict[str, Any]:
        """Create one durable order and return its stored record."""
        target = database()
        repository = OrderRepository(target)

        key = idempotency_key.strip()
        if key:
            existing = repository.find_by_idempotency_key(key)
            if existing is not None:
                return existing

        form: dict[str, str] = {
            "variables": variables,
            "outputs": outputs,
            "pricing": pricing,
            "metadata": metadata,
        }
        parsed: dict[str, Any] = {}
        for field in _JSON_OBJECT_FIELDS:
            parsed[field] = _parse_json_field(form[field], field)
            if not isinstance(parsed[field], dict):
                raise HTTPException(
                    status_code=422, detail=f"{field} must be an object"
                )
        parsed["outputs"] = _parse_json_field(form["outputs"], "outputs")
        if not isinstance(parsed["outputs"], list):
            raise HTTPException(status_code=422, detail="outputs must be a list")

        budget = None
        if budget_usd is not None and budget_usd.strip():
            try:
                budget = float(budget_usd)
            except ValueError as exc:
                raise HTTPException(
                    status_code=422, detail="budget_usd must be a number"
                ) from exc

        uploads: list[dict[str, Any]] = []
        if files:
            if len(files) > _MAX_FILES:
                raise HTTPException(
                    status_code=413,
                    detail=f"too many uploads (maximum {_MAX_FILES})",
                )
            for upload in files:
                if upload is None:
                    continue
                filename = sanitize_filename(upload.filename)
                data = upload.file.read() if upload.file is not None else b""
                if len(data) > _MAX_FILE_BYTES:
                    raise HTTPException(
                        status_code=413, detail="uploaded file too large"
                    )
                uploads.append(
                    {
                        "filename": filename,
                        "size": len(data),
                        "content_type": upload.content_type
                        or "application/octet-stream",
                    }
                )

        values: dict[str, Any] = {
            "occasion": occasion.strip(),
            "name": name.strip(),
            "message": message.strip(),
            "language": language.strip(),
            "profile": profile.strip(),
            "level": level.strip(),
            "timezone": timezone.strip() or "UTC",
            "variables": parsed["variables"],
            "outputs": parsed["outputs"],
            "pricing": parsed["pricing"],
            "currency": currency.strip().upper() or "USD",
            "amount_usd": float(amount_usd),
            "budget_usd": budget,
            "scheduled_for": scheduled_for.strip() if scheduled_for else None,
            "expires_at": expires_at.strip() if expires_at else None,
            "metadata": parsed["metadata"],
            "idempotency_key": key,
        }
        if uploads:
            values["metadata"] = dict(parsed["metadata"])
            values["metadata"]["uploads"] = uploads

        try:
            return repository.create(values)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get(
        "/api/v1/orders/{order_id}",
        tags=["orders"],
        summary="Get an order",
    )
    def get_order(order_id: str) -> dict[str, Any]:
        """Return one order's record, or ``404`` when it is unknown."""
        target = database()
        order = OrderRepository(target).get(order_id)
        if order is None:
            raise HTTPException(status_code=404, detail=f"order {order_id!r} not found")
        return order

    @app.get(
        "/api/v1/orders",
        tags=["orders"],
        summary="List orders",
    )
    def list_orders(
        occasion: str | None = None,
        state: str | None = None,
        profile: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Return orders matching the optional filters."""
        target = database()
        filters: dict[str, Any] = {}
        if occasion:
            filters["occasion"] = occasion
        if state:
            filters["state"] = state
        if profile:
            filters["profile"] = profile
        orders = OrderRepository(target).list(limit=limit, **filters)
        return {"orders": orders, "count": len(orders)}
