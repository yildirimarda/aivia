"""The local HTTP API application factory.

The API is deliberately small and secure by default: it binds to loopback
only, authenticates every non-public request, enforces request body limits and
rate limits before routing, and exposes its OpenAPI document for local clients.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi
from sqlalchemy import Engine

from aivia import __version__
from aivia.api.jobs import register_job_routes
from aivia.api.limits import (
    ApiKeyAuthMiddleware,
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestRateLimiter,
)
from aivia.api.security import validate_bind_host
from aivia.api.uploads import sanitize_filename
from aivia.core.config import ApiConfig, resolve_config
from aivia.core.secrets import SecretsLoader

__all__ = ["create_app", "serve"]

API_TITLE = "AIVIA local API"
API_DESCRIPTION = (
    "Local, authenticated API for generating and inspecting AIVIA media. "
    "Every route except health and the OpenAPI document requires an API key."
)


def create_app(
    config: ApiConfig | None = None,
    engine: Engine | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    secrets: SecretsLoader | None = None,
    database_path: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> FastAPI:
    """Build the local API application from *config*.

    When *config* is omitted the layered configuration is resolved. The
    returned app validates the bind host before it can be served, so a
    non-loopback address fails here rather than exposing the API by accident.
    Pass an *engine* to share an existing database, or let the job routes open
    the configured local database lazily on their first request.
    """
    api = _resolve_api_config(config, environ)
    _validate_api_config(api)

    expected_key = ""
    if api.require_auth:
        loader = secrets if secrets is not None else SecretsLoader(environ=environ)
        expected_key = loader.get(api.api_key_ref)

    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=__version__,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.state.api_config = api
    app.state.engine = engine
    app.state.api_engine_owned = False

    app.add_middleware(
        BodySizeLimitMiddleware,
        max_body_bytes=api.max_body_bytes,
    )
    if api.require_auth:
        app.add_middleware(ApiKeyAuthMiddleware, expected_key=expected_key)
    app.add_middleware(
        RateLimitMiddleware,
        limiter=RequestRateLimiter(api.rate_limit_per_minute),
    )

    @app.get(
        "/health",
        tags=["system"],
        summary="Liveness probe",
        openapi_extra={"security": []},
    )
    async def health() -> dict[str, str]:
        """Return the process status without requiring a key."""
        return {"status": "ok", "version": __version__}

    @app.get("/api/v1/ping", tags=["system"], summary="Authenticated smoke check")
    async def ping() -> dict[str, str]:
        """Confirm that the caller is authenticated."""
        return {"status": "ok"}

    @app.post("/api/v1/uploads", tags=["uploads"], summary="Validate an upload")
    async def upload(file: Annotated[UploadFile, File()]) -> dict[str, object]:
        """Return safe metadata for an uploaded file without storing it."""
        filename = sanitize_filename(file.filename)
        try:
            data = await file.read(api.max_body_bytes + 1)
        finally:
            await file.close()
        if len(data) > api.max_body_bytes:
            raise HTTPException(status_code=413, detail="request body too large")
        return {
            "filename": filename,
            "size": len(data),
            "content_type": file.content_type,
        }

    register_job_routes(
        app,
        engine=engine,
        environ=environ,
        database_path=database_path,
        data_dir=data_dir,
    )

    if api.require_auth:

        def custom_openapi() -> dict[str, Any]:
            if app.openapi_schema is not None:
                return app.openapi_schema
            schema = get_openapi(
                title=app.title,
                description=app.description,
                version=app.version,
                routes=app.routes,
            )
            components = schema.setdefault("components", {})
            schemes = components.setdefault("securitySchemes", {})
            schemes["ApiKeyAuth"] = {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
            }
            schema["security"] = [{"ApiKeyAuth": []}]
            app.openapi_schema = schema
            return schema

        app.openapi = custom_openapi  # type: ignore[method-assign]

    return app


def serve(
    config: ApiConfig | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    secrets: SecretsLoader | None = None,
) -> None:
    """Run the API with uvicorn on the configured loopback address."""
    api = _resolve_api_config(config, environ)
    _validate_api_config(api)
    app = create_app(api, environ=environ, secrets=secrets)

    import uvicorn

    uvicorn.run(app, host=api.host, port=api.port)


def _resolve_api_config(
    config: ApiConfig | None,
    environ: Mapping[str, str] | None,
) -> ApiConfig:
    if config is not None:
        return config
    resolved = resolve_config(environ=environ)
    section = getattr(resolved.app_config, "api", None)
    return section if isinstance(section, ApiConfig) else ApiConfig()


def _validate_api_config(api: ApiConfig) -> None:
    validate_bind_host(api.host)
