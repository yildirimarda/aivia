"""Error taxonomy shared by the core, providers and modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from aivia.core.redact import redact, redact_data

__all__ = [
    "AiviaError",
    "AuthError",
    "BudgetError",
    "ConfigError",
    "ContentPolicyError",
    "InvalidRequestError",
    "PermanentError",
    "ProviderExhaustedError",
    "QuotaError",
    "RateLimitError",
    "RetryableError",
    "TimeoutError",
    "ValidationError",
]


class AiviaError(Exception):
    """Base class for every error raised by AIVIA.

    ``code`` is a stable, machine-readable identifier and ``retryable`` tells
    callers whether the same operation may succeed when attempted again.
    ``details`` carries structured context that is safe to log and serialize.
    """

    code: ClassVar[str] = "aivia_error"
    retryable: ClassVar[bool] = False

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = dict(details) if details else {}

    def __str__(self) -> str:
        return redact(self.message)

    @property
    def permanent(self) -> bool:
        """Whether retrying the unchanged operation cannot succeed."""
        return not self.retryable

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for logs and responses."""
        payload: dict[str, Any] = {
            "type": type(self).__name__,
            "code": self.code,
            "message": redact(self.message),
            "retryable": self.retryable,
            "permanent": self.permanent,
        }
        if self.details:
            payload["details"] = redact_data(dict(self.details))
        return payload


class RetryableError(AiviaError):
    """An error that may succeed when the operation is retried."""

    code = "retryable"
    retryable = True


class PermanentError(AiviaError):
    """An error that cannot succeed by retrying the unchanged operation."""

    code = "permanent"
    retryable = False


class BudgetError(PermanentError):
    """The call would exceed a configured spending limit."""

    code = "budget"


class AuthError(PermanentError):
    """Credentials are missing, invalid or unauthorized."""

    code = "auth"


class RateLimitError(RetryableError):
    """A provider rejected the call because a rate limit was reached."""

    code = "rate_limit"

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.retry_after = retry_after

    def to_dict(self) -> dict[str, Any]:
        """Return the base representation plus the optional retry delay."""
        payload = super().to_dict()
        if self.retry_after is not None:
            payload["retry_after"] = self.retry_after
        return payload


class TimeoutError(RetryableError):
    """The operation exceeded its deadline."""

    code = "timeout"


class InvalidRequestError(PermanentError):
    """The provider rejected the request as malformed or unsupported."""

    code = "invalid_request"


class ContentPolicyError(PermanentError):
    """The request or its result violates a content policy."""

    code = "content_policy"


class ProviderExhaustedError(RetryableError):
    """Every provider in the capability chain failed."""

    code = "provider_exhausted"


class ConfigError(PermanentError):
    """Configuration is missing, invalid or inconsistent."""

    code = "config"


class ValidationError(PermanentError):
    """Input, output or structured data failed validation."""

    code = "validation"


class QuotaError(PermanentError):
    """A hard quota or queue limit was reached."""

    code = "quota"
