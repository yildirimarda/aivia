"""License and attribution tracking per asset, with automatic description credit.

Every asset carries a license. When that license requires attribution, the
credit line is appended to the asset's description automatically so a
publisher never has to remember to add it by hand.
"""

from __future__ import annotations

from typing import Any

import pytest

from aivia.core.models import Asset, AssetAnalysis, AssetLicense


def _license(**overrides: object) -> AssetLicense:
    data: dict[str, Any] = {
        "name": "CC BY 4.0",
        "url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Photo by Ada",
        "commercial_use": True,
        "attribution_required": True,
    }
    data.update(overrides)
    return AssetLicense(**data)


def _asset(**overrides: object) -> Asset:
    values: dict[str, Any] = {
        "label": "hero",
        "license": _license(),
        "analysis": AssetAnalysis(description="A bridge at dusk"),
    }
    values.update(overrides)
    return Asset(**values)


def test_asset_tracks_license_name_url_and_attribution() -> None:
    asset = _asset()

    assert asset.license_name == "CC BY 4.0"
    assert asset.license.url == "https://creativecommons.org/licenses/by/4.0/"
    assert asset.attribution == "Photo by Ada"
    assert asset.credit == "Photo by Ada"
    assert asset.requires_attribution is True


def test_asset_without_license_has_no_attribution() -> None:
    asset = Asset(label="hero")

    assert asset.license_name == ""
    assert asset.attribution == ""
    assert asset.credit == ""
    assert asset.requires_attribution is False


def test_license_unknown_commercial_use_is_not_cleared() -> None:
    asset = Asset(license=AssetLicense(name="Custom"))

    assert asset.license.is_commercial is False
    assert asset.requires_attribution is False


def test_description_includes_required_attribution_automatically() -> None:
    asset = _asset()

    assert asset.description == "A bridge at dusk\nPhoto by Ada"
    assert asset.analysis.description == "A bridge at dusk"


def test_description_omits_attribution_when_not_required() -> None:
    asset = Asset(
        label="hero",
        license=_license(attribution_required=False, attribution=""),
        analysis=AssetAnalysis(description="A bridge at dusk"),
    )

    assert asset.description == "A bridge at dusk"


def test_description_omits_attribution_when_license_has_no_credit() -> None:
    asset = Asset(
        label="hero",
        license=_license(attribution_required=True, attribution=""),
        analysis=AssetAnalysis(description="A bridge at dusk"),
    )

    assert asset.description == "A bridge at dusk"


def test_description_does_not_duplicate_attribution() -> None:
    asset = Asset(
        label="hero",
        license=_license(),
        analysis=AssetAnalysis(description="A bridge at dusk\nPhoto by Ada"),
    )

    assert asset.description == "A bridge at dusk\nPhoto by Ada"


def test_description_is_just_attribution_when_analysis_is_empty() -> None:
    asset = Asset(
        label="hero",
        license=_license(),
        analysis=AssetAnalysis(),
    )

    assert asset.description == "Photo by Ada"


def test_license_name_shorthand_carries_no_attribution() -> None:
    asset = Asset.model_validate({"license": "CC BY 4.0"})

    assert asset.license_name == "CC BY 4.0"
    assert asset.attribution == ""
    assert asset.requires_attribution is False


def test_license_accepts_attribution_aliases() -> None:
    license = AssetLicense.model_validate(
        {
            "credit": "Photo by Ada",
            "requires_attribution": True,
        }
    )

    assert license.attribution == "Photo by Ada"
    assert license.requires_attribution is True


def test_attribution_survives_json_round_trip() -> None:
    asset = _asset()
    restored = Asset.model_validate_json(asset.to_json())

    assert restored.attribution == "Photo by Ada"
    assert restored.requires_attribution is True
    assert restored.description == "A bridge at dusk\nPhoto by Ada"


@pytest.mark.parametrize(
    ("name", "commercial", "attribution_required", "attribution"),
    [
        ("CC0", False, False, ""),
        ("CC BY 4.0", True, True, "Photo by Ada"),
        ("Custom", None, False, ""),
    ],
)
def test_license_properties_reflect_declared_rights(
    name: str,
    commercial: bool | None,
    attribution_required: bool,
    attribution: str,
) -> None:
    license = AssetLicense(
        name=name,
        commercial_use=commercial,
        attribution_required=attribution_required,
        attribution=attribution,
    )

    assert license.is_commercial is bool(commercial)
    assert license.requires_attribution is (attribution_required or bool(attribution))
    assert license.credit == attribution
