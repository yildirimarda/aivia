"""A web card is one offline document with animation, play, expiry and hits."""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from aivia.compose import (
    WEB_CARD_RENDERER_VERSION,
    Composition,
    WebCardError,
    WebCardRenderResult,
    build_web_card,
    render_web_card,
)


def _composition(**overrides: Any) -> Composition:
    values: dict[str, Any] = {
        "id": "card-1",
        "canvas": {"width": 1080, "height": 1920, "background": "#102030"},
        "duration_seconds": 3.0,
        "fps": 30,
        "layers": [
            {
                "id": "headline",
                "type": "text",
                "text": "Hello <world>",
                "x": 80,
                "y": 120,
                "width": 920,
                "height": 200,
                "font_size": 72,
                "color": "#ffffff",
                "align": "center",
                "entrance": {
                    "type": "slide_up",
                    "duration": 0.8,
                    "distance": 64,
                },
            },
            {
                "id": "body",
                "type": "text",
                "text": "A second line",
                "start_seconds": 0.5,
            },
        ],
    }
    values.update(overrides)
    return Composition.model_validate(values)


def test_the_card_is_one_self_contained_html_document() -> None:
    document = build_web_card(_composition(), title="A card", summary="Summary")

    assert document.startswith("<!doctype html>")
    assert f"web_card/{WEB_CARD_RENDERER_VERSION}" in document
    assert "<style>" in document
    assert "<script>" in document
    assert "background-color:#102030;" in document
    assert 'src="http' not in document
    assert 'href="http' not in document
    assert "url(http" not in document
    assert "sendBeacon" not in document


def test_an_image_background_is_embedded_as_a_data_uri(tmp_path: Path) -> None:
    background = tmp_path / "background.png"
    Image.new("RGB", (16, 16), (12, 34, 56)).save(background)

    document = build_web_card(_composition(), background_image=background)
    payload = base64.b64encode(background.read_bytes()).decode("ascii")

    assert f"data:image/png;base64,{payload}" in document
    assert str(background) not in document
    assert "background-size:cover" in document


def test_an_image_layer_becomes_the_embedded_background(tmp_path: Path) -> None:
    background = tmp_path / "layer.webp"
    Image.new("RGB", (16, 16), (12, 34, 56)).save(background, format="WEBP")
    composition = _composition(
        layers=[
            {
                "id": "background",
                "type": "image",
                "source": str(background),
                "x": 0,
                "y": 0,
                "width": 1080,
                "height": 1920,
            },
            {"id": "headline", "type": "text", "text": "Over the image"},
        ]
    )

    result = render_web_card(composition, tmp_path / "card.html")

    assert result.background_embedded is True
    assert "data:image/webp;base64," in result.html
    assert str(background) not in result.html


def test_text_layers_become_css_animated_and_escaped_elements() -> None:
    document = build_web_card(_composition())

    assert "@keyframes aivia-anim-" in document
    assert "animation-name:aivia-anim-" in document
    assert "animation-duration:0.800s" in document
    assert "translate(" in document
    assert "aivia-animated" in document
    assert "prefers-reduced-motion" in document
    assert "Hello &lt;world&gt;" in document
    assert "Hello <world>" not in document
    assert "A second line" in document


def test_the_play_button_is_explicit_and_animations_start_paused() -> None:
    document = build_web_card(_composition())

    assert '<button class="aivia-play" type="button"' in document
    assert 'aria-pressed="false"' in document
    assert "animation-play-state:paused" in document
    assert "addEventListener('click'" in document
    assert "autoplay" not in document.lower()


def test_optional_expiry_is_embedded_and_reported(tmp_path: Path) -> None:
    expires_at = datetime(2030, 1, 1, 12, 30, tzinfo=UTC)

    document = build_web_card(_composition(), expires_at=expires_at)
    result = render_web_card(
        _composition(),
        tmp_path / "expiring.html",
        expires_at=expires_at,
    )

    assert 'name="aivia-expires-at"' in document
    assert 'data-expires-at="2030-01-01T12:30:00+00:00"' in document
    assert "Date.parse(expires)" in document
    assert result.expires_at == expires_at
    assert result.is_expired(now=expires_at - timedelta(seconds=1)) is False
    assert result.is_expired(now=expires_at) is True


def test_a_card_without_expiry_reports_no_expiry(tmp_path: Path) -> None:
    document = build_web_card(_composition())
    result = render_web_card(_composition(), tmp_path / "plain.html")

    assert 'name="aivia-expires-at"' not in document
    assert 'data-expires-at="' not in document
    assert result.expires_at is None
    assert result.is_expired() is False


def test_the_view_hit_hook_beacons_once_to_the_configured_url() -> None:
    document = build_web_card(
        _composition(),
        hit_url="https://example.test/cards/card-1/hit",
        card_id="card-1",
        order_id="order-1",
    )

    assert 'name="aivia-hit-url"' in document
    assert 'content="https://example.test/cards/card-1/hit"' in document
    assert "navigator.sendBeacon" in document
    assert "JSON.stringify" in document
    assert "event:'view'" in document
    assert 'data-card-id="card-1"' in document
    assert 'data-order-id="order-1"' in document
    assert "addEventListener('load',send,{once:true})" in document


def test_a_relative_hit_url_is_allowed() -> None:
    document = build_web_card(_composition(), hit_url="/api/cards/card-1/hit")

    assert 'content="/api/cards/card-1/hit"' in document


@pytest.mark.parametrize(
    "hit_url",
    [
        "javascript:alert(1)",
        "ftp://example.test/hit",
        "example.test/hit",
        "//example.test/hit",
        " ",
    ],
)
def test_unsafe_hit_urls_are_rejected(hit_url: str) -> None:
    with pytest.raises(WebCardError):
        build_web_card(_composition(), hit_url=hit_url)


def test_unsafe_css_and_missing_backgrounds_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(WebCardError):
        build_web_card(_composition(), background_color="#fff;</style><script>")
    with pytest.raises(WebCardError):
        build_web_card(_composition(), background_image=tmp_path / "missing.png")


def test_render_web_card_writes_atomically_and_returns_the_result(
    tmp_path: Path,
) -> None:
    output = tmp_path / "nested" / "card.html"

    result = render_web_card(_composition(), output, title="A card")

    assert isinstance(result, WebCardRenderResult)
    assert result.output == output
    assert result.path == output
    assert output.is_file()
    assert output.read_text(encoding="utf-8") == result.html
    assert result.size_bytes == len(result.html.encode("utf-8"))
    assert list(output.parent.glob(".card.html.*.tmp")) == []
    assert result.expires_at is None
    assert result.hit_url is None
