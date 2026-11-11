"""Choosing the run output that matches each platform media type.

The platform data file declares, per platform, the media types it supports and
the limits (resolution, aspect ratio, duration) each media type must respect.
This module verifies that a run's outputs are matched to those media types.
"""

from __future__ import annotations

from aivia.core.models.asset import Asset, AssetAnalysis, AssetKind
from aivia.core.platforms import load_platforms


def _asset(
    kind: AssetKind,
    path: str,
    *,
    width: int | None = None,
    height: int | None = None,
    duration_seconds: float | None = None,
    has_audio: bool | None = None,
) -> Asset:
    return Asset(
        kind=kind,
        path=path,
        analysis=AssetAnalysis(
            width=width,
            height=height,
            duration_seconds=duration_seconds,
            has_audio=has_audio,
        ),
    )


def test_chooses_one_output_per_supported_media_type() -> None:
    instagram = load_platforms().require("instagram")
    assert instagram.media_types == ("video", "image", "carousel")

    outputs = [
        _asset(
            AssetKind.VIDEO, "reel.mp4", width=1080, height=1920, duration_seconds=60
        ),
        _asset(AssetKind.IMAGE, "cover.jpg", width=1080, height=1350),
        _asset(AssetKind.IMAGE, "square.png", width=1080, height=1080),
    ]

    chosen = instagram.choose_outputs(outputs)

    assert set(chosen) == {"video", "image", "carousel"}
    assert chosen["video"] is not None
    assert chosen["video"].path == "reel.mp4"
    assert chosen["image"] is not None
    assert chosen["image"].path == "cover.jpg"
    # Carousel has no resolution limits, so any image fits.
    assert chosen["carousel"] is not None
    assert chosen["carousel"].path == "cover.jpg"


def test_unsupported_media_type_reports_none() -> None:
    youtube = load_platforms().require("youtube")
    assert youtube.media_types == ("video",)

    outputs = [_asset(AssetKind.IMAGE, "cover.jpg", width=1080, height=1080)]

    chosen = youtube.choose_outputs(outputs)

    assert chosen == {"video": None}


def test_no_outputs_reports_none_for_every_media_type() -> None:
    instagram = load_platforms().require("instagram")

    chosen = instagram.choose_outputs([])

    assert chosen == {"video": None, "image": None, "carousel": None}


def test_output_must_fit_the_declared_resolution() -> None:
    youtube = load_platforms().require("youtube")
    # YouTube Shorts only declares 1080x1920.
    assert youtube.resolutions == ("1080x1920",)

    too_small = _asset(
        AssetKind.VIDEO, "small.mp4", width=720, height=1280, duration_seconds=60
    )
    fits = _asset(
        AssetKind.VIDEO, "fit.mp4", width=1080, height=1920, duration_seconds=60
    )

    assert youtube.choose_outputs([too_small]) == {"video": None}
    chosen = youtube.choose_outputs([fits])
    assert chosen["video"] is not None
    assert chosen["video"].path == "fit.mp4"


def test_output_must_fit_the_declared_duration_range() -> None:
    youtube = load_platforms().require("youtube")
    assert youtube.min_duration_seconds == 1
    assert youtube.max_duration_seconds == 180

    too_long = _asset(
        AssetKind.VIDEO, "long.mp4", width=1080, height=1920, duration_seconds=300
    )
    fits = _asset(
        AssetKind.VIDEO, "short.mp4", width=1080, height=1920, duration_seconds=60
    )

    assert youtube.choose_outputs([too_long]) == {"video": None}
    chosen = youtube.choose_outputs([fits])
    assert chosen["video"] is not None
    assert chosen["video"].path == "short.mp4"


def test_image_output_has_no_duration_check() -> None:
    instagram = load_platforms().require("instagram")
    # Images carry no duration, so they are not duration-filtered.
    outputs = [_asset(AssetKind.IMAGE, "cover.jpg", width=1080, height=1080)]

    chosen = instagram.choose_outputs(outputs)

    assert chosen["image"] is not None
    assert chosen["image"].path == "cover.jpg"


def test_first_fitting_output_wins() -> None:
    youtube = load_platforms().require("youtube")
    outputs = [
        _asset(
            AssetKind.VIDEO, "first.mp4", width=1080, height=1920, duration_seconds=60
        ),
        _asset(
            AssetKind.VIDEO, "second.mp4", width=1080, height=1920, duration_seconds=90
        ),
    ]

    chosen = youtube.choose_outputs(outputs)

    assert chosen["video"] is not None
    assert chosen["video"].path == "first.mp4"
