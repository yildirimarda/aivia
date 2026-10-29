"""Local API skeleton: health, OpenAPI, binding, auth, limits and uploads."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aivia.api import create_app, is_loopback_host, sanitize_filename
from aivia.api.limits import RequestRateLimiter
from aivia.core.config import ApiConfig
from aivia.core.errors import ConfigError
from aivia.core.secrets import SecretsLoader

KEY = "test-api-key"
AUTH = {"X-API-Key": KEY}


def make_app(**overrides: object) -> TestClient:
    values: dict[str, object] = {
        "require_auth": True,
        "api_key_ref": "AIVA_API_KEY",
    }
    values.update(overrides)
    config = ApiConfig.model_validate(values)
    app = create_app(config, environ={"AIVA_API_KEY": KEY})
    return TestClient(app)


def test_health_is_public_and_reports_the_version() -> None:
    client = make_app()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["version"]


def test_openapi_documents_the_routes_and_api_key_scheme() -> None:
    client = make_app()

    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert set(schema["paths"]) >= {
        "/health",
        "/api/v1/ping",
        "/api/v1/uploads",
    }
    scheme = schema["components"]["securitySchemes"]["ApiKeyAuth"]
    assert scheme == {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    assert schema["security"] == [{"ApiKeyAuth": []}]
    assert schema["paths"]["/health"]["get"]["security"] == []


def test_docs_are_public() -> None:
    client = make_app()

    assert client.get("/docs").status_code == 200


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "example.test"])
def test_a_non_loopback_bind_address_is_rejected(host: str) -> None:
    with pytest.raises(ConfigError, match="localhost"):
        create_app(
            ApiConfig(host=host, require_auth=False),
            environ={},
        )


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "[::1]"])
def test_loopback_bind_addresses_are_accepted(host: str) -> None:
    assert is_loopback_host(host)
    app = create_app(ApiConfig(host=host, require_auth=False), environ={})

    assert app.state.api_config.host == host


def test_authentication_is_fail_closed_without_a_configured_key(
    tmp_path: Path,
) -> None:
    secrets_file = tmp_path / "secrets.env"
    secrets_file.write_text("", encoding="utf-8")
    loader = SecretsLoader(secrets_file=secrets_file, environ={})

    with pytest.raises(ConfigError):
        create_app(ApiConfig(require_auth=True), secrets=loader)


def test_a_protected_route_requires_the_api_key() -> None:
    client = make_app()

    assert client.get("/api/v1/ping").status_code == 401
    assert client.get("/api/v1/ping", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/v1/ping", headers=AUTH).json() == {"status": "ok"}
    bearer = {"Authorization": f"Bearer {KEY}"}
    assert client.get("/api/v1/ping", headers=bearer).status_code == 200


def test_authentication_can_be_disabled_explicitly() -> None:
    client = make_app(require_auth=False)

    assert client.get("/api/v1/ping").status_code == 200


def test_rate_limit_returns_429_with_retry_after() -> None:
    client = make_app(rate_limit_per_minute=2)

    assert client.get("/api/v1/ping", headers=AUTH).status_code == 200
    assert client.get("/api/v1/ping", headers=AUTH).status_code == 200
    response = client.get("/api/v1/ping", headers=AUTH)

    assert response.status_code == 429
    assert response.json()["detail"] == "rate limit exceeded"
    assert int(response.headers["Retry-After"]) >= 1


def test_health_is_exempt_from_rate_limiting() -> None:
    client = make_app(rate_limit_per_minute=1)

    assert client.get("/api/v1/ping", headers=AUTH).status_code == 200
    for _ in range(3):
        assert client.get("/health").status_code == 200
    assert client.get("/api/v1/ping", headers=AUTH).status_code == 429


def test_rate_limiter_refills_after_the_window() -> None:
    now = [0.0]
    limiter = RequestRateLimiter(1, window_seconds=60, clock=lambda: now[0])

    assert limiter.hit("client") == (True, 0.0)
    allowed, retry_after = limiter.hit("client")
    assert allowed is False
    assert retry_after == pytest.approx(60)

    now[0] = 60.0
    assert limiter.hit("client") == (True, 0.0)


def test_body_limit_rejects_a_declared_oversized_body() -> None:
    client = make_app(max_body_bytes=256)

    response = client.post(
        "/api/v1/uploads",
        files={"file": ("big.txt", b"x" * 512)},
        headers=AUTH,
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "request body too large"


def test_body_limit_counts_a_streamed_body_without_content_length() -> None:
    client = make_app(max_body_bytes=256)
    boundary = "testboundary"
    body = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="big.bin"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + b"x" * 512
        + f"\r\n--{boundary}--\r\n".encode()
    )
    headers = {
        **AUTH,
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }

    response = client.post(
        "/api/v1/uploads",
        content=iter([body[:128], body[128:]]),
        headers=headers,
    )

    assert response.status_code == 413


def test_upload_sanitizes_a_path_traversal_filename() -> None:
    client = make_app()

    response = client.post(
        "/api/v1/uploads",
        files={"file": ("../../etc/passwd", b"secret", "text/plain")},
        headers=AUTH,
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "passwd"
    assert response.json()["size"] == 6


def test_upload_sanitizes_windows_separators_and_control_characters() -> None:
    client = make_app()

    response = client.post(
        "/api/v1/uploads",
        files={"file": ("..\\..\\evil<script>.png", b"x", "image/png")},
        headers=AUTH,
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "evil_script_.png"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "upload"),
        ("", "upload"),
        ("..", "upload"),
        (".env", "env"),
        ("CON.txt", "_CON.txt"),
        ("evil\x00<script>.png", "evil_script_.png"),
        ("résumé.png", "r_sum_.png"),
        ("a" * 300 + ".png", "a" * 124 + ".png"),
    ],
)
def test_sanitize_filename_handles_edge_cases(raw: str | None, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_upload_requires_authentication() -> None:
    client = make_app()

    response = client.post(
        "/api/v1/uploads",
        files={"file": ("a.txt", b"data", "text/plain")},
    )

    assert response.status_code == 401
