"""Order HTTP routes: auth, limits, upload sanitization and order records."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from aivia.api import create_app
from aivia.core.config import ApiConfig
from aivia.core.db import (
    OrderRepository,
    create_database_engine,
    upgrade_database,
)

KEY = "test-api-key"
AUTH = {"X-API-Key": KEY}


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    created = create_database_engine(path=tmp_path / "aivia.db")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def client(engine: Engine) -> TestClient:
    return TestClient(create_app(ApiConfig(require_auth=False), engine, environ={}))


def make_client(
    *,
    require_auth: bool = True,
    engine: Engine | None = None,
    **overrides: object,
) -> TestClient:
    values: dict[str, object] = {
        "require_auth": require_auth,
        "api_key_ref": "AIVA_API_KEY",
    }
    values.update(overrides)
    config = ApiConfig.model_validate(values)
    app = create_app(config, engine, environ={"AIVA_API_KEY": KEY})
    return TestClient(app)


def order_data() -> dict[str, str]:
    return {
        "occasion": "birthday",
        "name": "Ada",
        "message": "Happy birthday!",
        "language": "en",
        "profile": "standard",
        "level": "silver",
        "variables": '{"theme": "flowers"}',
        "outputs": '["card.mp4"]',
        "pricing": '{"amount": 10.00}',
        "metadata": '{"source": "api"}',
        "currency": "USD",
        "amount_usd": "10.00",
        "timezone": "UTC",
    }


def test_creating_an_order_requires_authentication() -> None:
    client = make_client(require_auth=True)

    response = client.post("/api/v1/orders", data=order_data())

    assert response.status_code == 401
    assert "invalid or missing API key" in response.json()["detail"]


def test_bearer_token_authenticates_order_creation(engine: Engine) -> None:
    client = make_client(require_auth=True, engine=engine)

    response = client.post(
        "/api/v1/orders",
        data=order_data(),
        headers={"Authorization": f"Bearer {KEY}"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["id"].startswith("order_")
    assert body["occasion"] == "birthday"
    assert body["name"] == "Ada"
    assert body["state"] == "draft"
    assert body["variables"] == {"theme": "flowers"}
    assert body["amount_usd"] == 10.0

    stored = OrderRepository(engine).get(body["id"])
    assert stored is not None
    assert stored["name"] == "Ada"


def test_creating_an_order_stores_the_uploaded_filenames_safely(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/orders",
        data={**order_data(), "occasion": "birthday"},
        files={"files": ("../../etc/passwd", b"secret", "text/plain")},
    )

    assert response.status_code == 201
    body = response.json()
    uploads = body["metadata"]["uploads"]
    assert len(uploads) == 1
    assert uploads[0]["filename"] == "passwd"
    assert uploads[0]["size"] == 6
    assert uploads[0]["content_type"] == "text/plain"


def test_upload_sanitizes_windows_separators_and_control_characters(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/orders",
        data=order_data(),
        files={"files": ("..\\..\\evil<script>.png", b"x", "image/png")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["metadata"]["uploads"][0]["filename"] == "evil_script_.png"


def test_getting_an_order_returns_the_record(client: TestClient) -> None:
    created = client.post("/api/v1/orders", data=order_data()).json()
    order_id = created["id"]

    response = client.get(f"/api/v1/orders/{order_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == order_id
    assert body["occasion"] == "birthday"
    assert body["name"] == "Ada"
    assert body["state"] == "draft"


def test_an_unknown_order_is_not_found(client: TestClient) -> None:
    response = client.get("/api/v1/orders/order_missing")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_listing_orders_returns_created_orders(client: TestClient) -> None:
    client.post("/api/v1/orders", data={**order_data(), "occasion": "birthday"})
    client.post("/api/v1/orders", data={**order_data(), "occasion": "anniversary"})

    response = client.get("/api/v1/orders")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert {order["occasion"] for order in body["orders"]} == {
        "birthday",
        "anniversary",
    }


def test_listing_orders_filters_by_occasion(client: TestClient) -> None:
    client.post("/api/v1/orders", data={**order_data(), "occasion": "birthday"})
    client.post("/api/v1/orders", data={**order_data(), "occasion": "anniversary"})

    response = client.get("/api/v1/orders", params={"occasion": "birthday"})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["orders"][0]["occasion"] == "birthday"


def test_rate_limit_applies_to_order_routes(engine: Engine) -> None:
    client = TestClient(
        create_app(
            ApiConfig(require_auth=False, rate_limit_per_minute=2),
            engine,
            environ={},
        )
    )

    assert client.post("/api/v1/orders", data=order_data()).status_code == 201
    assert client.post("/api/v1/orders", data=order_data()).status_code == 201
    response = client.post("/api/v1/orders", data=order_data())

    assert response.status_code == 429
    assert response.json()["detail"] == "rate limit exceeded"
    assert int(response.headers["Retry-After"]) >= 1


def test_body_limit_applies_to_order_routes(engine: Engine) -> None:
    client = TestClient(
        create_app(
            ApiConfig(require_auth=False, max_body_bytes=64),
            engine,
            environ={},
        )
    )

    response = client.post(
        "/api/v1/orders",
        data=order_data(),
        files={"files": ("big.txt", b"x" * 256)},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "request body too large"


def test_invalid_json_fields_are_rejected(client: TestClient) -> None:
    data = order_data()
    data["variables"] = "not json"

    response = client.post("/api/v1/orders", data=data)

    assert response.status_code == 422


def test_idempotency_key_returns_the_existing_order(
    client: TestClient, engine: Engine
) -> None:
    submission = {**order_data(), "idempotency_key": "order-1"}

    first = client.post("/api/v1/orders", data=submission)
    second = client.post("/api/v1/orders", data=submission)

    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["id"] == first.json()["id"]
    assert len(OrderRepository(engine).list()) == 1


def test_order_routes_appear_in_openapi() -> None:
    client = make_client()

    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert "/api/v1/orders" in schema["paths"]
    assert "/api/v1/orders/{order_id}" in schema["paths"]
