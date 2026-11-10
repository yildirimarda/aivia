"""Privacy controls for personalized orders.

Orders minimize what they keep, redact personal data when it leaves the
process, support placeholder mode and keep a data processing record of the
providers and fields that touched each order.
"""

from __future__ import annotations

from datetime import UTC, datetime

from aivia.core.config.sections import PrivacyConfig
from aivia.core.redact import REDACTED
from aivia.orders import (
    DataProcessingEntry,
    DataProcessingRecord,
    apply_placeholders,
    minimize_order,
    placeholder_for,
    redact_order,
)


def test_placeholder_for_returns_a_stable_token_per_field() -> None:
    assert placeholder_for("name") == "[NAME]"
    assert placeholder_for("message") == "[MESSAGE]"
    assert placeholder_for("occasion") == "[OCCASION]"
    assert placeholder_for("language") == "[LANGUAGE]"
    assert placeholder_for("timezone") == "[TIMEZONE]"
    assert placeholder_for("variables") == "[VARIABLES]"


def test_minimize_order_keeps_standard_order_fields() -> None:
    values = {
        "occasion": "birthday",
        "name": "Ada",
        "message": "Happy birthday",
        "language": "en",
        "amount_usd": 10.0,
        "should_not_be_kept": "secret",
        "also_drop": {"nested": "data"},
    }
    minimized = minimize_order(values)

    assert minimized["occasion"] == "birthday"
    assert minimized["name"] == "Ada"
    assert minimized["message"] == "Happy birthday"
    assert minimized["language"] == "en"
    assert minimized["amount_usd"] == 10.0
    assert "should_not_be_kept" not in minimized
    assert "also_drop" not in minimized
    assert values["should_not_be_kept"] == "secret"


def test_minimize_order_keeps_declared_occasion_variables() -> None:
    values = {"name": "Ada", "theme": "flowers", "extra": "dropped"}
    minimized = minimize_order(values, declared=["theme"])

    assert minimized == {"name": "Ada", "theme": "flowers"}


def test_minimize_order_does_not_mutate_the_input() -> None:
    values = {"name": "Ada", "junk": "drop"}
    before = dict(values)
    minimize_order(values)
    assert values == before


def test_redact_order_with_default_policy_redacts_personal_data() -> None:
    order = {
        "id": "order_1",
        "name": "Ada",
        "message": "Happy birthday",
        "occasion": "birthday",
        "language": "en",
        "timezone": "UTC",
        "amount_usd": 10.0,
    }
    redacted = redact_order(order, PrivacyConfig())

    assert redacted["name"] == REDACTED
    assert redacted["message"] == REDACTED
    assert redacted["occasion"] == REDACTED
    assert redacted["language"] == REDACTED
    assert redacted["timezone"] == REDACTED
    assert redacted["id"] == "order_1"
    assert redacted["amount_usd"] == 10.0
    assert order["name"] == "Ada"


def test_redact_order_placeholder_mode_replaces_personal_data() -> None:
    order = {"name": "Ada", "message": "Happy birthday", "occasion": "birthday"}
    privacy = PrivacyConfig(placeholder_mode=True)

    redacted = redact_order(order, privacy)

    assert redacted["name"] == "[NAME]"
    assert redacted["message"] == "[MESSAGE]"
    assert redacted["occasion"] == "[OCCASION]"


def test_redact_order_keeps_personal_data_when_allowed() -> None:
    order = {"name": "Ada", "message": "Happy birthday"}
    privacy = PrivacyConfig(allow_personal_data=True, redact_logs=False)

    redacted = redact_order(order, privacy)

    assert redacted["name"] == "Ada"
    assert redacted["message"] == "Happy birthday"


def test_redact_order_masks_secrets_in_allowed_personal_data() -> None:
    from aivia.core.redact import clear_secrets, register_secret

    register_secret("session-token-abc")
    try:
        order = {"name": "Ada session-token-abc", "message": "Hi"}
        privacy = PrivacyConfig(allow_personal_data=True, redact_logs=False)

        redacted = redact_order(order, privacy)

        assert REDACTED in redacted["name"]
        assert "session-token-abc" not in redacted["name"]
    finally:
        clear_secrets()


def test_redact_order_redacts_logs_masks_token_like_strings() -> None:
    from aivia.core.redact import clear_secrets, register_secret

    register_secret("session-token-abc")
    try:
        order = {
            "name": "Ada",
            "message": "Happy birthday",
            "api_key": "session-token-abc",
            "amount_usd": 10.0,
        }
        privacy = PrivacyConfig(allow_personal_data=True, redact_logs=True)

        redacted = redact_order(order, privacy)

        assert redacted["name"] == "Ada"
        assert redacted["api_key"] == REDACTED
        assert redacted["amount_usd"] == 10.0
    finally:
        clear_secrets()


def test_redact_order_redacts_variable_values() -> None:
    from aivia.core.redact import clear_secrets, register_secret

    register_secret("session-token-abc")
    try:
        order = {
            "name": "Ada",
            "variables": {"theme": "flowers", "token": "session-token-abc"},
        }
        privacy = PrivacyConfig()

        redacted = redact_order(order, privacy)

        assert redacted["name"] == REDACTED
        assert redacted["variables"]["theme"] == "flowers"
        assert redacted["variables"]["token"] == REDACTED
        assert "session-token-abc" not in redacted["variables"]["token"]
    finally:
        clear_secrets()


def test_apply_placeholders_replaces_personal_fields() -> None:
    order = {"name": "Ada", "message": "Happy birthday", "amount_usd": 10.0}

    replaced = apply_placeholders(order)

    assert replaced["name"] == "[NAME]"
    assert replaced["message"] == "[MESSAGE]"
    assert replaced["amount_usd"] == 10.0
    assert order["name"] == "Ada"


def test_data_processing_record_stores_provider_and_fields() -> None:
    record = DataProcessingRecord()
    entry = record.add(
        "tts",
        ["name", "message"],
        purpose="speak the recipient's name",
        legal_basis="consent",
    )

    assert len(record) == 1
    assert record.entries[0] is entry
    assert entry.provider == "tts"
    assert entry.fields == ("name", "message")
    assert entry.purpose == "speak the recipient's name"
    assert entry.legal_basis == "consent"
    assert isinstance(entry.occurred_at, datetime)


def test_data_processing_record_deduplicates_fields() -> None:
    record = DataProcessingRecord()
    record.add("voice", ["name", "name", "message"])

    assert record.entries[0].fields == ("name", "message")


def test_data_processing_record_records_multiple_providers() -> None:
    record = DataProcessingRecord()
    record.add("tts", ["name"], purpose="name")
    record.add("music", ["message"], purpose="lyrics")
    record.add("visuals", ["occasion"], purpose="card")

    assert len(record) == 3
    assert [entry.provider for entry in record.entries] == ["tts", "music", "visuals"]


def test_data_processing_record_to_dict_is_json_compatible() -> None:
    record = DataProcessingRecord()
    recorded = record.add(
        "tts",
        ["name"],
        occurred_at=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
    )

    payload = record.to_dict()

    assert len(payload) == 1
    assert payload[0] == {
        "occurred_at": "2024-01-02T03:04:05+00:00",
        "provider": "tts",
        "fields": ["name"],
        "purpose": "",
        "legal_basis": "",
    }
    assert payload[0] is not recorded.to_dict()


def test_data_processing_record_clear_removes_all_entries() -> None:
    record = DataProcessingRecord()
    record.add("tts", ["name"])
    record.add("music", ["message"])

    record.clear()

    assert len(record) == 0
    assert record.entries == []
    assert record.to_dict() == []


def test_data_processing_entry_to_dict_includes_all_fields() -> None:
    entry = DataProcessingEntry(
        occurred_at=datetime(2024, 1, 1, tzinfo=UTC),
        provider="voice",
        fields=("name",),
        purpose="verify pronunciation",
        legal_basis="contract",
    )

    assert entry.to_dict()["provider"] == "voice"
    assert entry.to_dict()["fields"] == ["name"]
    assert entry.to_dict()["purpose"] == "verify pronunciation"
    assert entry.to_dict()["legal_basis"] == "contract"
