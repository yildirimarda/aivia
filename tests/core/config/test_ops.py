"""Operational config: review approval, retention, job workers and storage."""

from __future__ import annotations

from typing import Any, cast

import pytest
import yaml

from aivia.core import paths
from aivia.core.config import (
    JobsConfig,
    RetentionConfig,
    ReviewConfig,
    SectionRegistry,
    StorageConfig,
    register_core_sections,
)
from aivia.core.config.ops import S3_REQUIRED_FIELDS
from aivia.core.errors import ConfigError


def example_section(name: str) -> dict[str, Any]:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    payload = cast(dict[str, Any], yaml.safe_load(text))
    return cast(dict[str, Any], payload[name])


def test_review_defaults_require_approval_with_a_timeout() -> None:
    review = ReviewConfig.from_mapping({})

    assert review.enabled is True
    assert review.approval_mode == "ask"
    assert review.require_approval is True
    assert review.timeout_seconds == 86_400
    assert review.timeout_hours == 24
    assert review.on_timeout == "reject"
    assert review.allow_regenerate is True
    assert review.max_regenerations == 3
    assert review.preview is True
    assert review.notify is True


def test_review_accepts_the_approval_aliases() -> None:
    never = ReviewConfig.from_mapping({"require_approval": False})
    always = ReviewConfig.from_mapping({"approval": "always"})
    automatic = ReviewConfig.from_mapping({"auto_approve": True})

    assert never.approval_mode == "never"
    assert never.require_approval is False
    assert always.approval_mode == "always"
    assert automatic.approval_mode == "never"


def test_review_accepts_timeouts_in_hours_and_minutes() -> None:
    hours = ReviewConfig.from_mapping({"timeout_hours": 2})
    minutes = ReviewConfig.from_mapping({"timeout_minutes": 90})

    assert hours.timeout_seconds == 7200
    assert hours.timeout_hours == 2
    assert minutes.timeout_seconds == 5400
    assert minutes.timeout_minutes == 90


@pytest.mark.parametrize(
    ("values", "field"),
    [
        ({"approval_mode": "sometimes"}, "approval_mode"),
        ({"on_timeout": "explode"}, "on_timeout"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"max_regenerations": -1}, "max_regenerations"),
    ],
)
def test_review_rejects_invalid_values(values: dict[str, Any], field: str) -> None:
    with pytest.raises(ConfigError) as caught:
        ReviewConfig.from_mapping(values)

    assert field in str(caught.value)


def test_retention_defaults_never_delete_final_outputs() -> None:
    retention = RetentionConfig.from_mapping({})

    assert retention.intermediates_days == 7
    assert retention.cache_days == 14
    assert retention.logs_days == 30
    assert retention.personal_data_days == 30
    assert retention.orders_days == 90
    assert retention.deliveries_days == 30
    assert retention.web_card_hits_days == 30
    assert retention.delete_final_outputs is False


def test_retention_allows_explicit_final_output_deletion() -> None:
    retention = RetentionConfig.from_mapping({"delete_final_outputs": True})

    assert retention.delete_final_outputs is True


@pytest.mark.parametrize(
    "field",
    [
        "intermediates_days",
        "cache_days",
        "logs_days",
        "personal_data_days",
        "orders_days",
        "deliveries_days",
        "web_card_hits_days",
    ],
)
def test_retention_days_must_not_be_negative(field: str) -> None:
    with pytest.raises(ConfigError) as caught:
        RetentionConfig.from_mapping({field: -1})

    assert field in str(caught.value)


def test_jobs_defaults_match_a_single_careful_worker() -> None:
    jobs = JobsConfig.from_mapping({})

    assert jobs.concurrency == 1
    assert jobs.max_attempts == 3
    assert jobs.heartbeat_seconds == 30
    assert jobs.poll_seconds == 2
    assert jobs.stale_after_seconds == 120
    assert jobs.max_queue_depth == 0


@pytest.mark.parametrize(
    ("values", "field"),
    [
        ({"concurrency": 0}, "concurrency"),
        ({"max_attempts": 0}, "max_attempts"),
        ({"heartbeat_seconds": 0}, "heartbeat_seconds"),
        ({"poll_seconds": 0}, "poll_seconds"),
        ({"stale_after_seconds": 0}, "stale_after_seconds"),
        ({"max_queue_depth": -1}, "max_queue_depth"),
    ],
)
def test_jobs_rejects_non_positive_or_negative_values(
    values: dict[str, Any], field: str
) -> None:
    with pytest.raises(ConfigError) as caught:
        JobsConfig.from_mapping(values)

    assert field in str(caught.value)


def test_jobs_stale_after_must_exceed_the_heartbeat() -> None:
    with pytest.raises(ConfigError) as caught:
        JobsConfig.from_mapping(
            {"heartbeat_seconds": 30, "stale_after_seconds": 30}
        )

    assert "stale_after_seconds" in str(caught.value)


def test_local_storage_defaults_need_no_s3_settings() -> None:
    storage = StorageConfig.from_mapping({})

    assert storage.backend == "local"
    assert storage.local_dir == "media"
    assert storage.bucket is None
    assert storage.prefix == ""
    assert storage.presign_seconds == 3600


def test_s3_storage_accepts_a_complete_configuration() -> None:
    storage = StorageConfig.from_mapping(
        {
            "backend": "s3",
            "bucket": "media",
            "endpoint": "https://s3.example.test",
            "region": "eu-central-1",
            "access_key_ref": "AIVA_S3_ACCESS_KEY",
            "secret_key_ref": "AIVA_S3_SECRET_KEY",
        }
    )

    assert storage.backend == "s3"
    assert storage.bucket == "media"
    assert storage.endpoint == "https://s3.example.test"
    assert storage.region == "eu-central-1"
    assert storage.access_key_ref == "AIVA_S3_ACCESS_KEY"
    assert storage.secret_key_ref == "AIVA_S3_SECRET_KEY"


def test_s3_storage_accepts_a_nested_s3_mapping() -> None:
    storage = StorageConfig.from_mapping(
        {
            "backend": "s3",
            "s3": {
                "bucket": "media",
                "endpoint": "https://s3.example.test",
                "region": "eu-central-1",
                "access_key_ref": "AIVA_S3_ACCESS_KEY",
                "secret_key_ref": "AIVA_S3_SECRET_KEY",
            },
        }
    )

    assert storage.bucket == "media"


@pytest.mark.parametrize("field", S3_REQUIRED_FIELDS)
def test_s3_storage_requires_every_field(field: str) -> None:
    complete = {
        "backend": "s3",
        "bucket": "media",
        "endpoint": "https://s3.example.test",
        "region": "eu-central-1",
        "access_key_ref": "AIVA_S3_ACCESS_KEY",
        "secret_key_ref": "AIVA_S3_SECRET_KEY",
    }
    del complete[field]

    with pytest.raises(ConfigError) as caught:
        StorageConfig.from_mapping(complete)

    assert field in str(caught.value)


def test_s3_storage_rejects_blank_required_fields() -> None:
    with pytest.raises(ConfigError) as caught:
        StorageConfig.from_mapping(
            {
                "backend": "s3",
                "bucket": "  ",
                "endpoint": "https://s3.example.test",
                "region": "eu-central-1",
                "access_key_ref": "AIVA_S3_ACCESS_KEY",
                "secret_key_ref": "AIVA_S3_SECRET_KEY",
            }
        )

    assert "bucket" in str(caught.value)


def test_unknown_storage_backend_is_rejected() -> None:
    with pytest.raises(ConfigError) as caught:
        StorageConfig.from_mapping({"backend": "ftp"})

    assert "backend" in str(caught.value)


def test_review_and_operations_sections_are_registered() -> None:
    sections = register_core_sections(SectionRegistry())

    parsed = sections.parse(
        {
            "review": {"timeout_hours": 1},
            "retention": {"delete_final_outputs": True},
            "jobs": {"concurrency": 2},
            "storage": {"backend": "local"},
        }
    )

    assert parsed.review.timeout_seconds == 3600  # type: ignore[attr-defined]
    assert parsed.retention.delete_final_outputs is True  # type: ignore[attr-defined]
    assert parsed.jobs.concurrency == 2  # type: ignore[attr-defined]
    assert parsed.storage.backend == "local"  # type: ignore[attr-defined]


def test_the_packaged_example_declares_the_operations_sections() -> None:
    review = ReviewConfig.from_mapping(example_section("review"))
    retention = RetentionConfig.from_mapping(example_section("retention"))
    jobs = JobsConfig.from_mapping(example_section("jobs"))
    storage = StorageConfig.from_mapping(example_section("storage"))

    assert review.require_approval is True
    assert retention.delete_final_outputs is False
    assert jobs.stale_after_seconds > jobs.heartbeat_seconds
    assert storage.backend == "local"
