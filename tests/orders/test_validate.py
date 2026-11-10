"""Validation and sanitization of personalized order input.

Every order field is treated as untrusted data: it is normalized to NFKC,
checked for length, screened for profanity and rejected when it contains null
bytes, control characters or prompt-injection tokens.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from aivia.api import create_app
from aivia.api.orders import sanitize_order_input, validate_order_input
from aivia.core.config import ApiConfig
from aivia.core.db import (
    OrderRepository,
    create_database_engine,
    upgrade_database,
)
from aivia.core.errors import ValidationError

KEY = "test-api-key"


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


def test_clean_order_input_passes_through_unchanged() -> None:
    values = {
        "occasion": "birthday",
        "name": "Ada",
        "message": "Happy birthday!",
        "language": "en",
        "currency": "USD",
    }
    assert sanitize_order_input(values) == values


def test_sanitize_order_input_normalises_unicode_to_nfkc() -> None:
    values = {"name": "ﬁnn", "message": "e\u0301nte"}
    cleaned = sanitize_order_input(values)
    assert cleaned["name"] == "finn"
    assert cleaned["message"] == "énte"


def test_sanitize_order_input_rejects_null_byte() -> None:
    with pytest.raises(ValidationError) as caught:
        sanitize_order_input({"name": "Ada\x00"})
    assert caught.value.details["field"] == "name"
    assert caught.value.details["token"] == "\x00"


def test_sanitize_order_input_rejects_control_characters() -> None:
    with pytest.raises(ValidationError):
        sanitize_order_input({"message": "Ada\x01"})
    # tabs, newlines and carriage returns are permitted
    assert sanitize_order_input({"message": "Ada\t!\n"})["message"] == "Ada\t!\n"


@pytest.mark.parametrize("token", ["<<<", ">>>", "{{", "}}", "${", "<%", "%>"])
def test_sanitize_order_input_rejects_injection_tokens(token: str) -> None:
    with pytest.raises(ValidationError):
        sanitize_order_input({"name": f"Ada{token}"})


def test_sanitize_order_input_treats_every_field_as_data() -> None:
    with pytest.raises(ValidationError):
        sanitize_order_input({"language": "en${system}"})


def test_sanitize_order_input_rejects_non_string_text_field() -> None:
    with pytest.raises(ValidationError):
        sanitize_order_input({"name": 123})


def test_sanitize_order_input_enforces_name_length() -> None:
    with pytest.raises(ValidationError):
        sanitize_order_input({"name": "A" * 81})
    assert sanitize_order_input({"name": "A" * 80})["name"] == "A" * 80


def test_sanitize_order_input_enforces_message_length() -> None:
    with pytest.raises(ValidationError):
        sanitize_order_input({"message": "A" * 1001})
    assert sanitize_order_input({"message": "A" * 1000})["message"] == "A" * 1000


def test_sanitize_order_input_enforces_occasion_length() -> None:
    with pytest.raises(ValidationError):
        sanitize_order_input({"occasion": "A" * 129})
    assert sanitize_order_input({"occasion": "A" * 128})["occasion"] == "A" * 128


def test_sanitize_order_input_enforces_language_length() -> None:
    with pytest.raises(ValidationError):
        sanitize_order_input({"language": "x" * 33})


def test_sanitize_order_input_rejects_profanity_in_name() -> None:
    with pytest.raises(ValidationError):
        sanitize_order_input({"name": "fuck"})


def test_sanitize_order_input_rejects_profanity_in_message() -> None:
    with pytest.raises(ValidationError):
        sanitize_order_input({"message": "you are a shit"})


def test_sanitize_order_input_rejects_profanity_in_occasion() -> None:
    with pytest.raises(ValidationError):
        sanitize_order_input({"occasion": "damn"})


def test_sanitize_order_input_passes_json_fields_through() -> None:
    values = {
        "occasion": "birthday",
        "name": "Ada",
        "message": "Happy birthday!",
        "variables": {"theme": "flowers"},
        "outputs": ["card.mp4"],
        "pricing": {"amount": 10.0},
        "metadata": {"source": "api"},
    }
    assert sanitize_order_input(values) == values


def test_validate_order_input_is_an_alias() -> None:
    assert validate_order_input is sanitize_order_input


def test_creating_an_order_with_profanity_is_rejected(
    client: TestClient,
) -> None:
    response = client.post("/api/v1/orders", data={**order_data(), "name": "fuck"})
    assert response.status_code == 422


def test_creating_an_order_with_a_null_byte_is_rejected(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/orders", data={**order_data(), "message": "Ada\x00!"}
    )
    assert response.status_code == 422


def test_creating_an_order_with_an_overlong_name_is_rejected(
    client: TestClient,
) -> None:
    response = client.post("/api/v1/orders", data={**order_data(), "name": "A" * 81})
    assert response.status_code == 422


def test_creating_a_clean_order_still_succeeds(
    client: TestClient, engine: Engine
) -> None:
    response = client.post("/api/v1/orders", data=order_data())
    assert response.status_code == 201
    body = response.json()
    assert body["id"].startswith("order_")
    assert OrderRepository(engine).get(body["id"]) is not None
