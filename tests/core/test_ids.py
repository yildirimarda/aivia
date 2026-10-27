"""Identifier contract: canonical ULIDs and their prefixed record kinds."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from aivia.core.ids import (
    CROCKFORD_ALPHABET,
    ID_PREFIXES,
    ULID_LENGTH,
    IdKind,
    encode_ulid,
    is_ulid,
    new_asset_id,
    new_id,
    new_job_id,
    new_order_id,
    new_run_id,
    ulid,
)

_ZERO_ENTROPY = b"\x00" * 10


def test_ulid_has_26_crockford_characters() -> None:
    value = ulid()

    assert len(value) == ULID_LENGTH
    assert value[0] <= "7"
    assert all(char in CROCKFORD_ALPHABET for char in value)


def test_ulid_encodes_the_timestamp_in_the_high_bits() -> None:
    assert encode_ulid(0, _ZERO_ENTROPY) == "0" * ULID_LENGTH
    assert encode_ulid(1, _ZERO_ENTROPY) == "0000000001" + "0" * 16
    assert encode_ulid((1 << 48) - 1, _ZERO_ENTROPY)[0] == "7"


def test_ulids_sort_by_timestamp_regardless_of_entropy() -> None:
    earlier = encode_ulid(1_700_000_000_000, b"\xff" * 10)
    later = encode_ulid(1_700_000_000_001, _ZERO_ENTROPY)

    assert earlier < later


@pytest.mark.parametrize("timestamp_ms", [-1, 1 << 48])
def test_encode_ulid_rejects_an_out_of_range_timestamp(timestamp_ms: int) -> None:
    with pytest.raises(ValueError, match="timestamp_ms"):
        encode_ulid(timestamp_ms, _ZERO_ENTROPY)


@pytest.mark.parametrize("entropy", [b"", b"\x00" * 9, b"\x00" * 11])
def test_encode_ulid_rejects_entropy_of_the_wrong_length(entropy: bytes) -> None:
    with pytest.raises(ValueError, match="entropy"):
        encode_ulid(0, entropy)


def test_generated_ulids_are_unique_and_sortable() -> None:
    values = [ulid() for _ in range(500)]

    assert len(set(values)) == len(values)
    assert values == sorted(values)


def test_ulids_stay_sortable_when_the_clock_does_not_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aivia.core.ids.time.time_ns",
        lambda: 1_700_000_000_000_000_000,
    )

    values = [ulid() for _ in range(100)]

    assert len(set(values)) == len(values)
    assert values == sorted(values)


def test_ulids_stay_sortable_when_the_clock_moves_backwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter([2_000_000_000_000_000_000, 1_000_000_000_000_000_000])
    monkeypatch.setattr("aivia.core.ids.time.time_ns", lambda: next(times))

    first = ulid()
    second = ulid()

    assert first < second


def test_id_prefixes_cover_the_record_kinds() -> None:
    assert ID_PREFIXES == ("run", "job", "asset", "order")
    assert {kind.value for kind in IdKind} == set(ID_PREFIXES)


@pytest.mark.parametrize(
    ("prefix", "factory"),
    [
        ("run", new_run_id),
        ("job", new_job_id),
        ("asset", new_asset_id),
        ("order", new_order_id),
    ],
)
def test_prefixed_ids_carry_the_kind_and_a_valid_ulid(
    prefix: str, factory: Callable[[], str]
) -> None:
    value = factory()
    actual_prefix, separator, suffix = value.partition("_")

    assert separator == "_"
    assert actual_prefix == prefix
    assert is_ulid(suffix)


def test_new_id_accepts_the_enum_and_the_string_form() -> None:
    assert new_id(IdKind.RUN).startswith("run_")
    assert new_id("order").startswith("order_")


def test_new_id_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown id kind: pipeline"):
        new_id("pipeline")


@pytest.mark.parametrize(
    "value",
    [
        "",
        "0" * 25,
        "0" * 27,
        "0" * 25 + "U",
        "8" + "0" * 25,
        "run_" + "0" * 26,
        "0" * 25 + "i",
    ],
)
def test_is_ulid_rejects_non_canonical_values(value: str) -> None:
    assert not is_ulid(value)


def test_is_ulid_accepts_a_generated_value() -> None:
    assert is_ulid(ulid())
