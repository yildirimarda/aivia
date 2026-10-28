"""Retry, backoff and timeout settings for the shared HTTP client.

The client itself lives in :mod:`aivia.core.http`; this module only holds the
policy values so the core contains no tuning literals. ``max_attempts`` counts
the first try, ``backoff_base_seconds`` is the first exponential step and
``backoff_max_seconds`` caps every wait. ``jitter_ratio`` is the fraction of a
step that may be removed at random, so retries from many workers do not line
up; ``0`` disables jitter and ``1`` allows the full step.

``timeout_seconds`` is the total deadline for one attempt. The optional
connect, read, write and pool timeouts override the corresponding phase of the
underlying transport when set. ``retry_statuses`` lists the HTTP status codes
worth retrying, and ``respect_retry_after`` honours a server supplied
``Retry-After`` header instead of the computed backoff.
"""

from __future__ import annotations

import math
from typing import Self

from pydantic import Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel

__all__ = ["HttpClientConfig", "HttpConfig"]

DEFAULT_RETRY_STATUSES: tuple[int, ...] = (408, 425, 429, 500, 502, 503, 504)


class HttpClientConfig(ConfigModel):
    """Retry, backoff and timeout policy for :class:`~aivia.core.http.HttpClient`."""

    timeout_seconds: float = Field(default=30.0, gt=0)
    connect_timeout_seconds: float | None = Field(default=None, gt=0)
    read_timeout_seconds: float | None = Field(default=None, gt=0)
    write_timeout_seconds: float | None = Field(default=None, gt=0)
    pool_timeout_seconds: float | None = Field(default=None, gt=0)
    max_attempts: int = Field(default=3, ge=1)
    backoff_base_seconds: float = Field(default=0.5, ge=0)
    backoff_max_seconds: float = Field(default=30.0, ge=0)
    jitter_ratio: float = Field(default=0.5, ge=0, le=1)
    retry_statuses: tuple[int, ...] = DEFAULT_RETRY_STATUSES
    respect_retry_after: bool = True
    follow_redirects: bool = True

    @field_validator(
        "timeout_seconds",
        "connect_timeout_seconds",
        "read_timeout_seconds",
        "write_timeout_seconds",
        "pool_timeout_seconds",
        "backoff_base_seconds",
        "backoff_max_seconds",
    )
    @classmethod
    def _finite_numbers(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("HTTP timeout and backoff values must be finite")
        return value

    @field_validator("retry_statuses")
    @classmethod
    def _valid_statuses(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        statuses: list[int] = []
        for status in value:
            if not isinstance(status, int) or isinstance(status, bool):
                raise ValueError("retry statuses must be integers")
            if not 100 <= status <= 599:
                raise ValueError("retry statuses must be valid HTTP status codes")
            if status not in statuses:
                statuses.append(status)
        return tuple(statuses)

    @model_validator(mode="after")
    def _ordered_backoff(self) -> Self:
        if self.backoff_base_seconds > self.backoff_max_seconds:
            raise ValueError("backoff_base_seconds must not exceed backoff_max_seconds")
        return self


HttpConfig = HttpClientConfig
