"""Redaction contract for logs, data and formatted exceptions."""

# secret-scan: allow-file -- this module intentionally contains fake tokens.

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from typing import Any, cast

import pytest

from aivia.core.errors import ConfigError
from aivia.core.logging import JsonFormatter, run_id
from aivia.core.redact import (
    REDACTED,
    clear_secrets,
    redact,
    redact_data,
    redact_exception,
    register_secret,
    register_secrets,
    registered_secrets,
    unregister_secret,
)


@pytest.fixture(autouse=True)
def _empty_secret_registry() -> Iterator[None]:
    clear_secrets()
    yield
    clear_secrets()


def test_registered_secrets_are_masked_everywhere() -> None:
    register_secret("super-secret-value")

    assert (
        redact("key=super-secret-value and super-secret-value")
        == f"key={REDACTED} and {REDACTED}"
    )


def test_longer_registered_secrets_win_over_their_prefixes() -> None:
    register_secrets("abc123", "abc123456789")

    assert redact("abc123456789") == REDACTED


def test_empty_and_whitespace_secrets_are_ignored() -> None:
    register_secret("")
    register_secret("   ")

    assert registered_secrets() == ()
    assert redact("unchanged") == "unchanged"


def test_secrets_can_be_unregistered_and_cleared() -> None:
    register_secrets("one", "two")
    unregister_secret("one")

    assert registered_secrets() == ("two",)

    clear_secrets()
    assert registered_secrets() == ()


def test_secret_values_are_matched_literally_not_as_patterns() -> None:
    register_secret("a.b*c[1]")

    assert redact("x a.b*c[1] y") == f"x {REDACTED} y"


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("Authorization: Bearer abcdefghijklmnop", "abcdefghijklmnop"),
        ("key sk-abcdefghijklmnopqrstuvwxyz", "sk-abcdefghijklmnopqrstuvwxyz"),
        ("aws AKIAABCDEFGHIJKLMNOP", "AKIAABCDEFGHIJKLMNOP"),
        (
            "github ghp_abcdefghijklmnopqrstuvwxyz012345",
            "ghp_abcdefghijklmnopqrstuvwxyz012345",
        ),
        ("slack xoxb-1234567890-abcdefghij", "xoxb-1234567890-abcdefghij"),
        ("google AIza" + "a" * 35, "AIza" + "a" * 35),
        (
            "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature123",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature123",
        ),
        (
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----",
            "MIIEow",
        ),
    ],
)
def test_token_like_strings_are_masked(text: str, secret: str) -> None:
    masked = redact(text)

    assert secret not in masked
    assert REDACTED in masked


@pytest.mark.parametrize(
    "text",
    [
        "api_key=abcdef123456",
        "token: abcdef123456",
        '"password": "hunter2hunter2"',
        "client_secret=abcdef123456",
        "access_token=abcdef123456",
    ],
)
def test_named_credential_values_are_masked(text: str) -> None:
    masked = redact(text)

    assert "abcdef123456" not in masked
    assert "hunter2hunter2" not in masked
    assert REDACTED in masked


def test_url_credentials_are_masked() -> None:
    masked = redact("fetch https://user:password123@example.com/path")

    assert "password123" not in masked
    assert masked == f"fetch https://user:{REDACTED}@example.com/path"


def test_ordinary_text_is_not_changed() -> None:
    text = "The token budget is 42 and the secret sauce recipe is good."

    assert redact(text) == text


def test_redaction_is_idempotent() -> None:
    register_secret("secret-value")
    once = redact("secret-value sk-abcdefghijklmnop")

    assert redact(once) == once


def test_redact_data_masks_nested_strings_and_keys() -> None:
    register_secret("nested-secret")
    data = {
        "token": "nested-secret",
        "items": ["nested-secret", 3, {"password": "nested-secret"}],
        "nested-secret": "value",
    }

    assert redact_data(data) == {
        REDACTED: "value",
        "token": REDACTED,
        "items": [REDACTED, 3, {"password": REDACTED}],
    }


def test_redact_exception_masks_the_message_and_its_cause_chain() -> None:
    register_secret("secret-in-error")
    error = RuntimeError("failed with secret-in-error")
    error.__cause__ = ValueError("token=abcdef123456")

    formatted = redact_exception(error)

    assert "secret-in-error" not in formatted
    assert "abcdef123456" not in formatted
    assert "RuntimeError" in formatted
    assert "ValueError" in formatted
    assert REDACTED in formatted


def test_error_formatting_redacts_the_message_and_details() -> None:
    register_secret("error-secret")
    error = ConfigError(
        "failed with error-secret",
        details={"token": "error-secret", "items": ["error-secret"]},
    )

    assert "error-secret" not in str(error)

    payload = error.to_dict()
    assert payload["message"] == f"failed with {REDACTED}"
    assert payload["details"] == {"token": REDACTED, "items": [REDACTED]}


def _record(
    *,
    msg: str = "hello",
    exc_info: tuple[Any, Any, Any] | None = None,
) -> logging.LogRecord:
    return logging.LogRecord(
        "aivia.test", logging.INFO, __file__, 1, msg, None, exc_info
    )


def _payload(record: logging.LogRecord) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(JsonFormatter().format(record)))


def test_json_logging_redacts_messages_context_and_token_patterns() -> None:
    register_secret("log-secret")
    token = run_id.set("log-secret")
    try:
        payload = _payload(_record(msg="using log-secret and sk-abcdefghijklmnop"))
    finally:
        run_id.reset(token)

    assert "log-secret" not in payload["msg"]
    assert "sk-abcdefghijklmnop" not in payload["msg"]
    assert payload["msg"] == f"using {REDACTED} and {REDACTED}"
    assert payload["run_id"] == REDACTED


def test_json_logging_redacts_the_formatted_exception() -> None:
    register_secret("exc-secret")
    try:
        raise ValueError("failed with exc-secret")
    except ValueError:
        record = _record(msg="boom", exc_info=sys.exc_info())

    payload = _payload(record)

    assert "exc-secret" not in payload["exc_info"]
    assert REDACTED in payload["exc_info"]
