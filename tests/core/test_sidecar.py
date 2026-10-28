"""Cost sidecars written next to every output."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from aivia.core.models import CostEvent, ToolchainFingerprint
from aivia.core.sidecar import (
    SIDECAR_SUFFIX,
    CostSidecar,
    build_cost_sidecar,
    sidecar_path,
    write_cost_sidecar,
    write_sidecar,
)

GENERATED_AT = datetime(2026, 1, 5, 10, 30, tzinfo=UTC)


def _event(**overrides: Any) -> CostEvent:
    values: dict[str, Any] = {
        "provider": "acme",
        "model": "image-1",
        "operation": "image_gen",
        "units": 2.0,
        "unit_kind": "image",
        "amount_usd": 0.08,
        "run_id": "run_1",
        "pricing_version": "2026-01",
        "estimated": False,
        "metadata": {"stage": "visuals"},
    }
    values.update(overrides)
    return CostEvent.model_validate(values)


def _fingerprint() -> ToolchainFingerprint:
    return ToolchainFingerprint(
        renderer_version="renderer-1",
        ffmpeg_version="ffmpeg-7",
        font_set="fonts-abc",
    )


def test_sidecar_path_sits_beside_the_output_and_keeps_its_full_name(
    tmp_path: Path,
) -> None:
    image = tmp_path / "clip.mp4"

    assert sidecar_path(image) == tmp_path / f"clip.mp4{SIDECAR_SUFFIX}"
    assert sidecar_path(str(image)) == tmp_path / "clip.mp4.cost.json"


def test_build_collects_totals_items_providers_models_pricing_and_toolchain() -> None:
    first = _event()
    second = _event(
        provider="other",
        model="video-1",
        operation="video_gen",
        units=3.0,
        unit_kind="second",
        amount_usd=0.30,
        estimated=True,
        metadata={"prompt": "do not copy this"},
    )

    sidecar = build_cost_sidecar(
        "clip.mp4",
        [first, second],
        toolchain_fingerprint=_fingerprint(),
        generated_at=GENERATED_AT,
    )

    assert sidecar.output == "clip.mp4"
    assert sidecar.run_id == "run_1"
    assert sidecar.totals.total_usd == pytest.approx(0.38)
    assert sidecar.totals.estimated_usd == pytest.approx(0.30)
    assert sidecar.totals.actual_usd == pytest.approx(0.08)
    assert sidecar.totals.units == pytest.approx(5.0)
    assert sidecar.totals.count == 2
    assert [item.cost_event_id for item in sidecar.items] == [first.id, second.id]
    assert [item.operation for item in sidecar.items] == ["image_gen", "video_gen"]
    assert sidecar.providers == ("acme", "other")
    assert sidecar.models == ("image-1", "video-1")
    assert sidecar.pricing_version == "2026-01"
    assert sidecar.pricing_versions == ("2026-01",)
    assert sidecar.toolchain_fingerprint.renderer_version == "renderer-1"
    assert sidecar.toolchain_fingerprint.ffmpeg_version == "ffmpeg-7"
    assert sidecar.toolchain_fingerprint.font_set == "fonts-abc"
    assert sidecar.toolchain_digest == _fingerprint().digest


def test_sidecar_never_copies_prompts_or_free_form_event_metadata() -> None:
    sidecar = build_cost_sidecar(
        "clip.mp4",
        [_event(metadata={"prompt": "a secret prompt", "token": "sk-secret"})],
        toolchain_fingerprint=_fingerprint(),
    )

    payload = json.dumps(sidecar.to_dict())

    assert "a secret prompt" not in payload
    assert "sk-secret" not in payload
    assert "metadata" not in payload


def test_mixed_pricing_versions_keep_the_list_but_leave_the_single_empty() -> None:
    sidecar = build_cost_sidecar(
        "clip.mp4",
        [
            _event(pricing_version="2026-01"),
            _event(pricing_version="2026-02"),
        ],
        toolchain_fingerprint=_fingerprint(),
    )

    assert sidecar.pricing_version == ""
    assert sidecar.pricing_versions == ("2026-01", "2026-02")


def test_an_explicit_pricing_version_wins_and_is_added_to_the_list() -> None:
    sidecar = build_cost_sidecar(
        "clip.mp4",
        [_event(pricing_version="2026-01")],
        pricing_version="2026-03",
        toolchain_fingerprint=_fingerprint(),
    )

    assert sidecar.pricing_version == "2026-03"
    assert sidecar.pricing_versions == ("2026-01", "2026-03")


def test_events_may_be_mappings_and_empty_runs_still_get_a_valid_sidecar() -> None:
    empty = build_cost_sidecar(
        "clip.mp4",
        [],
        toolchain_fingerprint=_fingerprint(),
        generated_at=GENERATED_AT,
    )

    assert empty.totals.total_usd == 0.0
    assert empty.totals.count == 0
    assert empty.items == ()
    assert empty.providers == ()
    assert empty.models == ()
    assert empty.pricing_version == ""
    assert empty.pricing_versions == ()

    mapped = build_cost_sidecar(
        "clip.mp4",
        [
            {
                "provider": "acme",
                "model": "image-1",
                "operation": "image_gen",
                "units": 1.0,
                "amount_usd": 0.04,
                "pricing_version": "2026-01",
            }
        ],
        toolchain_fingerprint=_fingerprint(),
    )

    assert mapped.totals.total_usd == pytest.approx(0.04)
    assert mapped.items[0].provider == "acme"


def test_write_puts_canonical_json_next_to_the_output(tmp_path: Path) -> None:
    output = tmp_path / "run" / "clip.mp4"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"media")
    event = _event()

    written = write_cost_sidecar(
        output,
        [event],
        toolchain_fingerprint=_fingerprint(),
        generated_at=GENERATED_AT,
    )

    assert written == tmp_path / "run" / "clip.mp4.cost.json"
    assert written.exists()
    text = written.read_text(encoding="utf-8")
    assert text.endswith("\n")
    payload = json.loads(text)
    assert list(payload) == sorted(payload)
    assert payload["output"] == str(output)
    assert payload["run_id"] == "run_1"
    assert payload["generated_at"] == "2026-01-05T10:30:00+00:00"
    assert payload["totals"] == {
        "total_usd": pytest.approx(0.08),
        "estimated_usd": pytest.approx(0.0),
        "actual_usd": pytest.approx(0.08),
        "units": pytest.approx(2.0),
        "count": 1,
    }
    assert payload["providers"] == ["acme"]
    assert payload["models"] == ["image-1"]
    assert payload["pricing_version"] == "2026-01"
    assert payload["items"][0]["cost_event_id"] == event.id
    assert payload["toolchain_fingerprint"]["digest"] == _fingerprint().digest
    assert payload["toolchain_fingerprint"]["renderer_version"] == "renderer-1"


def test_rewriting_an_output_replaces_its_sidecar_instead_of_adding_another(
    tmp_path: Path,
) -> None:
    output = tmp_path / "clip.mp4"
    output.write_bytes(b"media")

    first = write_sidecar(
        output,
        [_event(amount_usd=0.10)],
        toolchain_fingerprint=_fingerprint(),
        generated_at=GENERATED_AT,
    )
    second = write_sidecar(
        output,
        [_event(amount_usd=0.20), _event(amount_usd=0.05)],
        toolchain_fingerprint=_fingerprint(),
        generated_at=GENERATED_AT,
    )

    assert first == second
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "clip.mp4",
        "clip.mp4.cost.json",
    ]
    payload = json.loads(second.read_text(encoding="utf-8"))
    assert payload["totals"]["total_usd"] == pytest.approx(0.25)
    assert payload["totals"]["count"] == 2


def test_a_sidecar_can_be_built_without_an_explicit_fingerprint() -> None:
    sidecar: CostSidecar = build_cost_sidecar("clip.mp4", [_event()])

    assert sidecar.toolchain_fingerprint.is_known is False
    assert sidecar.toolchain_digest == ToolchainFingerprint().digest
