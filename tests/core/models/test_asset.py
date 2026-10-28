"""Asset, license, cost link and analysis contracts."""

from __future__ import annotations

import json
import warnings
from typing import Any

import pytest
from pydantic import ValidationError

from aivia.core.ids import is_ulid
from aivia.core.models import (
    Asset,
    AssetAnalysis,
    AssetCost,
    AssetCostLink,
    AssetKind,
    AssetLicense,
    AssetModel,
    AssetRecord,
    CostLink,
    Face,
    FaceRegion,
    License,
    UnknownFieldWarning,
    compute_prompt_hash,
)

_SHA = "a" * 64


def _asset(**overrides: Any) -> Asset:
    values: dict[str, Any] = {
        "label": "hero",
        "path": "/runs/run_1/hero.png",
        "content_hash": _SHA,
        "provider": "acme",
        "model": "image-1",
        "prompt": "A bridge at dusk",
        "alt_text": "A stone bridge crossing a river at dusk",
        "pinned": True,
        "run_id": "run_1",
    }
    values.update(overrides)
    return Asset(**values)


def test_asset_defaults_are_empty_but_valid() -> None:
    asset = Asset()

    assert asset.id.startswith("asset_")
    assert is_ulid(asset.id.removeprefix("asset_"))
    assert asset.kind is AssetKind.IMAGE
    assert asset.label == ""
    assert asset.path == ""
    assert asset.content_hash == ""
    assert asset.provider == ""
    assert asset.model == ""
    assert asset.prompt == ""
    assert asset.prompt_hash == ""
    assert asset.license.name == ""
    assert asset.cost_links == []
    assert asset.analysis.width is None
    assert asset.alt_text == ""
    assert asset.pinned is False
    assert asset.quality_score is None
    assert asset.run_id == ""


def test_asset_ids_are_unique_and_time_sortable() -> None:
    first = Asset()
    second = Asset()

    assert first.id != second.id
    assert first.id < second.id


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  PHOTO ", AssetKind.IMAGE),
        ("clip", AssetKind.VIDEO),
        ("sound", AssetKind.AUDIO),
        ("song", AssetKind.MUSIC),
        ("effect", AssetKind.SFX),
        ("voiceover", AssetKind.VOICE),
        ("document", AssetKind.TEXT),
        ("font_file", AssetKind.FONT),
        ("file", AssetKind.OTHER),
    ],
)
def test_asset_kind_accepts_aliases_and_is_normalized(
    raw: str, expected: AssetKind
) -> None:
    assert Asset.model_validate({"kind": raw}).kind is expected


def test_asset_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        Asset.model_validate({"kind": "hologram"})


def test_asset_carries_provenance_and_aliases() -> None:
    asset = Asset.model_validate(
        {
            "asset_id": "asset_placeholder",
            "type": "video",
            "name": "  Hero clip  ",
            "file": "  /tmp/hero.mp4  ",
            "sha256": f"  {_SHA}  ",
            "vendor": "  acme  ",
            "model_id": "  video-1  ",
            "prompt_text": "  A bridge  ",
            "prompt_sha256": "  explicit-hash  ",
            "run": "  run_1  ",
            "score": 0.75,
        }
    )

    assert asset.asset_id == "asset_placeholder"
    assert asset.kind is AssetKind.VIDEO
    assert asset.media_type is AssetKind.VIDEO
    assert asset.type is AssetKind.VIDEO
    assert asset.name == "Hero clip"
    assert asset.file == asset.path == "/tmp/hero.mp4"
    assert asset.uri == asset.path
    assert asset.url == asset.path
    assert asset.hash == asset.content_hash == _SHA
    assert asset.sha256 == asset.content_hash
    assert asset.vendor == asset.provider == "acme"
    assert asset.model_id == asset.model == "video-1"
    assert asset.prompt == "A bridge"
    assert asset.prompt_hash == "explicit-hash"
    assert asset.run_id == "run_1"
    assert asset.quality_score == 0.75
    assert asset.is_ai_generated is True


def test_asset_derives_a_stable_prompt_hash_when_it_is_omitted() -> None:
    asset = Asset(prompt="  A bridge at dusk  ")

    assert asset.prompt == "A bridge at dusk"
    assert asset.prompt_hash == compute_prompt_hash("A bridge at dusk")
    assert asset.prompt_hash == compute_prompt_hash("  A bridge at dusk  ")
    assert len(asset.prompt_hash) == 64
    assert asset.prompt_hash == asset.prompt_hash.lower()
    assert Asset(prompt="").prompt_hash == ""
    assert Asset(prompt="A bridge", prompt_hash="explicit").prompt_hash == "explicit"


def test_prompt_hash_is_order_independent_for_parameters() -> None:
    first = compute_prompt_hash("A bridge", parameters={"seed": 7, "size": "1080x1920"})
    second = compute_prompt_hash(
        "A bridge", parameters={"size": "1080x1920", "seed": 7}
    )

    assert first == second
    assert first != compute_prompt_hash("A bridge")
    assert first != compute_prompt_hash("Another bridge", parameters={"seed": 7})


def test_asset_license_carries_rights_and_attribution() -> None:
    license = AssetLicense.model_validate(
        {
            "license": "  CC BY 4.0  ",
            "license_url": "  https://creativecommons.org/licenses/by/4.0/  ",
            "credit": "  Photo by Ada  ",
            "allow_commercial": True,
            "requires_attribution": True,
            "copyleft": False,
        }
    )

    assert license.name == "CC BY 4.0"
    assert license.url == "https://creativecommons.org/licenses/by/4.0/"
    assert license.attribution == "Photo by Ada"
    assert license.credit == "Photo by Ada"
    assert license.commercial_use is True
    assert license.attribution_required is True
    assert license.share_alike is False
    assert license.is_known is True
    assert license.is_commercial is True
    assert license.requires_attribution is True


def test_license_unknown_commercial_use_is_not_cleared() -> None:
    license = AssetLicense(name="Custom")

    assert license.commercial_use is None
    assert license.is_commercial is False
    assert license.requires_attribution is False
    assert Asset.model_validate({"license": None}).license.name == ""


def test_asset_accepts_a_license_name_shorthand() -> None:
    asset = Asset.model_validate({"license": "  CC0  "})

    assert asset.license.name == "CC0"
    assert asset.license_name == "CC0"
    assert asset.license.is_commercial is False


def test_asset_cost_links_accept_ids_and_records() -> None:
    asset = Asset.model_validate(
        {
            "costs": [
                "cost_1",
                {
                    "event_id": "  cost_2  ",
                    "amount_usd": 0.25,
                    "provider": "acme",
                    "model": "image-1",
                    "stage": "image",
                },
            ]
        }
    )

    assert [link.cost_event_id for link in asset.cost_links] == ["cost_1", "cost_2"]
    assert asset.cost_event_ids == ["cost_1", "cost_2"]
    assert asset.costs == asset.cost_links
    assert asset.total_cost_usd == 0.25
    assert asset.cost_links[0].amount_usd is None
    assert asset.cost_links[1].id == "cost_2"
    assert asset.cost_links[1].amount == 0.25
    assert asset.cost_links[1].cost == 0.25
    assert asset.cost_links[1].provider == "acme"
    assert asset.cost_links[1].operation == "image"


def test_cost_link_requires_an_id_and_non_negative_amount() -> None:
    with pytest.raises(ValidationError):
        AssetCostLink.model_validate({"cost_event_id": "  "})

    with pytest.raises(ValidationError):
        AssetCostLink(cost_event_id="cost_1", amount_usd=-0.01)


def test_asset_analysis_carries_dimensions_colors_text_and_faces() -> None:
    analysis = AssetAnalysis.model_validate(
        {
            "width": 1080,
            "height": 1920,
            "duration": 2.5,
            "dominant_colors": ["#ffffff", "  #000000  ", "#ffffff"],
            "caption": "  A bridge at dusk  ",
            "ocr": ["Line one", "Line two"],
            "face_regions": [
                {
                    "left": 1,
                    "top": 2,
                    "w": 3,
                    "h": 4,
                    "confidence": 0.8,
                }
            ],
        }
    )

    assert analysis.width == 1080
    assert analysis.height == 1920
    assert analysis.duration_seconds == 2.5
    assert analysis.colors == ["#ffffff", "#000000"]
    assert analysis.dominant_colors == analysis.colors
    assert analysis.description == "A bridge at dusk"
    assert analysis.detected_text == "Line one\nLine two"
    assert analysis.ocr_text == analysis.detected_text
    assert analysis.has_faces is True
    assert analysis.face_regions == analysis.faces
    assert analysis.faces[0].box == (1.0, 2.0, 3.0, 4.0)
    assert analysis.faces[0].area == 12.0
    assert analysis.faces[0].confidence == 0.8
    assert analysis.aspect_ratio == 1080 / 1920
    assert analysis.is_portrait is True
    assert analysis.is_landscape is False
    assert analysis.is_square is False
    assert analysis.is_analyzed is True


def test_asset_analysis_defaults_to_unknown() -> None:
    analysis = AssetAnalysis()

    assert analysis.width is None
    assert analysis.height is None
    assert analysis.duration_seconds is None
    assert analysis.colors == []
    assert analysis.description == ""
    assert analysis.detected_text == ""
    assert analysis.faces == []
    assert analysis.aspect_ratio is None
    assert analysis.has_faces is False
    assert analysis.is_analyzed is False


@pytest.mark.parametrize("field", ["width", "height", "duration_seconds"])
def test_asset_analysis_rejects_non_positive_measurements(field: str) -> None:
    with pytest.raises(ValidationError):
        AssetAnalysis.model_validate({field: 0})

    with pytest.raises(ValidationError):
        AssetAnalysis.model_validate({field: -1})


def test_face_region_rejects_negative_coordinates_and_bad_confidence() -> None:
    with pytest.raises(ValidationError):
        FaceRegion(x=-1)

    with pytest.raises(ValidationError):
        FaceRegion(confidence=1.1)


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_asset_quality_score_is_bounded(score: float) -> None:
    with pytest.raises(ValidationError):
        Asset(quality_score=score)


def test_asset_alt_text_and_pinned_flag() -> None:
    asset = _asset()

    assert asset.alt_text == "A stone bridge crossing a river at dusk"
    assert asset.has_alt_text is True
    assert asset.pinned is True
    assert asset.is_pinned is True

    empty = Asset()
    assert empty.has_alt_text is False
    assert empty.is_pinned is False


def test_asset_delegates_analysis_fields() -> None:
    asset = Asset.model_validate(
        {
            "analysis": {
                "width": 100,
                "height": 50,
                "colors": ["red"],
                "description": "A red shape",
                "detected_text": "STOP",
                "faces": [{"x": 1, "y": 2, "width": 3, "height": 4}],
            }
        }
    )

    assert asset.width == 100
    assert asset.height == 50
    assert asset.colors == ["red"]
    assert asset.description == "A red shape"
    assert asset.detected_text == "STOP"
    assert len(asset.faces) == 1
    assert asset.faces[0].box == (1.0, 2.0, 3.0, 4.0)


def test_asset_serializes_canonically_with_nested_records() -> None:
    asset = _asset(
        kind=AssetKind.VIDEO,
        license=AssetLicense(
            name="CC BY 4.0",
            url="https://creativecommons.org/licenses/by/4.0/",
            attribution="Photo by Ada",
            commercial_use=True,
            attribution_required=True,
        ),
        cost_links=[AssetCostLink(cost_event_id="cost_1", amount_usd=0.5)],
        analysis=AssetAnalysis(
            width=1080,
            height=1920,
            colors=["#ffffff"],
            description="A bridge",
            detected_text="Hello",
            faces=[FaceRegion(x=1, y=2, width=3, height=4)],
        ),
        quality_score=0.9,
    )

    payload: dict[str, Any] = json.loads(asset.to_json())

    assert list(payload) == sorted(payload)
    assert payload["kind"] == "video"
    assert payload["provider"] == "acme"
    assert payload["model"] == "image-1"
    assert payload["prompt_hash"] == compute_prompt_hash("A bridge at dusk")
    assert payload["license"]["name"] == "CC BY 4.0"
    assert payload["license"]["commercial_use"] is True
    assert payload["cost_links"][0]["cost_event_id"] == "cost_1"
    assert payload["analysis"]["faces"][0]["width"] == 3
    assert payload["pinned"] is True


def test_asset_round_trips_through_json() -> None:
    asset = _asset(
        license=AssetLicense(name="CC0", commercial_use=True),
        cost_links=[
            AssetCostLink(cost_event_id="cost_1", amount_usd=0.5),
            AssetCostLink(cost_event_id="cost_2", amount_usd=0.25),
        ],
        analysis=AssetAnalysis(
            width=1080,
            height=1920,
            duration_seconds=2.5,
            colors=["#ffffff", "#000000"],
            description="A bridge",
            detected_text="Hello",
            faces=[FaceRegion(x=1, y=2, width=3, height=4, confidence=0.9)],
        ),
        quality_score=0.9,
    )

    restored = Asset.model_validate_json(asset.to_json())

    assert restored == asset
    assert restored.to_json() == asset.to_json()
    assert restored.cost_event_ids == ["cost_1", "cost_2"]
    assert restored.analysis.faces[0].confidence == 0.9


def test_asset_accepts_aliases_without_unknown_field_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnknownFieldWarning)
        asset = Asset.model_validate(
            {
                "asset_id": "asset_placeholder",
                "type": "image",
                "name": "Hero",
                "file": "/tmp/hero.png",
                "hash": _SHA,
                "vendor": "acme",
                "model_id": "image-1",
                "prompt_text": "A bridge",
                "prompt_sha256": "hash",
                "rights": {"license": "CC0", "allow_commercial": True},
                "cost_ids": ["cost_1"],
                "properties": {"width": 10, "height": 20},
                "alt": "A bridge",
                "is_pinned": True,
                "qa_score": 0.5,
                "run": "run_1",
            }
        )

    assert asset.kind is AssetKind.IMAGE
    assert asset.license.name == "CC0"
    assert asset.license.is_commercial is True
    assert asset.cost_event_ids == ["cost_1"]
    assert asset.analysis.width == 10
    assert asset.alt_text == "A bridge"
    assert asset.is_pinned is True
    assert asset.quality_score == 0.5


def test_asset_keeps_unknown_fields_with_a_warning() -> None:
    with pytest.warns(UnknownFieldWarning, match="future_field"):
        asset = Asset.model_validate({"future_field": "kept"})

    assert asset.model_extra is not None
    assert asset.model_extra["future_field"] == "kept"


def test_asset_model_aliases_are_exported() -> None:
    assert AssetModel is Asset
    assert AssetRecord is Asset
    assert AssetCost is AssetCostLink
    assert CostLink is AssetCostLink
    assert License is AssetLicense
    assert Face is FaceRegion
