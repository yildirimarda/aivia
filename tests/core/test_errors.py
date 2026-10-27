"""Error taxonomy contract."""

from __future__ import annotations

import json

import pytest

from aivia.core.errors import (
    AiviaError,
    AuthError,
    BudgetError,
    ConfigError,
    ContentPolicyError,
    InvalidRequestError,
    PermanentError,
    ProviderExhaustedError,
    QuotaError,
    RateLimitError,
    RetryableError,
    ValidationError,
)
from aivia.core.errors import TimeoutError as AiviaTimeoutError

ERROR_CASES: list[tuple[type[AiviaError], str, bool]] = [
    (AiviaError, "aivia_error", False),
    (RetryableError, "retryable", True),
    (PermanentError, "permanent", False),
    (BudgetError, "budget", False),
    (AuthError, "auth", False),
    (RateLimitError, "rate_limit", True),
    (AiviaTimeoutError, "timeout", True),
    (InvalidRequestError, "invalid_request", False),
    (ContentPolicyError, "content_policy", False),
    (ProviderExhaustedError, "provider_exhausted", True),
    (ConfigError, "config", False),
    (ValidationError, "validation", False),
    (QuotaError, "quota", False),
]


@pytest.mark.parametrize(("error_type", "code", "retryable"), ERROR_CASES)
def test_each_error_has_a_stable_code_and_retryability(
    error_type: type[AiviaError], code: str, retryable: bool
) -> None:
    error = error_type("boom")

    assert isinstance(error, AiviaError)
    assert error.code == code
    assert error.retryable is retryable
    assert error.permanent is (not retryable)


def test_error_codes_are_unique() -> None:
    codes = [error_type.code for error_type, _, _ in ERROR_CASES]

    assert len(codes) == len(set(codes))


def test_base_error_defaults() -> None:
    error = AiviaError("boom")

    assert str(error) == "boom"
    assert error.message == "boom"
    assert error.details == {}


def test_details_are_copied_on_construction_and_serialization() -> None:
    details = {"provider": "mock", "attempt": 1}
    error = AiviaError("boom", details=details)

    details["attempt"] = 2
    payload = error.to_dict()
    payload["details"]["attempt"] = 3

    assert error.details == {"provider": "mock", "attempt": 1}


def test_to_dict_is_json_serializable() -> None:
    error = InvalidRequestError("bad prompt", details={"field": "prompt"})

    assert error.to_dict() == {
        "type": "InvalidRequestError",
        "code": "invalid_request",
        "message": "bad prompt",
        "retryable": False,
        "permanent": True,
        "details": {"field": "prompt"},
    }
    assert json.loads(json.dumps(error.to_dict())) == error.to_dict()


def test_to_dict_omits_empty_details() -> None:
    assert "details" not in AiviaError("boom").to_dict()


def test_rate_limit_error_records_retry_after() -> None:
    error = RateLimitError("slow down", retry_after=2.5)

    assert error.retry_after == 2.5
    assert error.to_dict()["retry_after"] == 2.5
    assert "retry_after" not in RateLimitError("slow down").to_dict()


def test_errors_can_be_raised_and_caught_by_the_base_class() -> None:
    with pytest.raises(AiviaError) as caught:
        raise BudgetError("over budget")

    assert isinstance(caught.value, BudgetError)
    assert caught.value.code == "budget"
