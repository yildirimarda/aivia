"""Render a composition as one self-contained, offline web card.

A web card is a single HTML document with no external stylesheet, script,
font or image request. The composition background becomes an inline CSS
background (a colour, gradient or an embedded data URI), text layers become
animated HTML elements, and an explicit play button starts the CSS animations
instead of autoplaying them.

The card can carry an optional expiry time and an optional view hit URL. The
expiry is embedded in the document and checked by the browser; the hit URL is
called once per load with a small JSON body naming the card and order. Neither
feature makes a request while the card is being rendered.
"""

from __future__ import annotations

import base64
import html
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from aivia.core.config.media import DEFAULT_MEDIA_MAX_BYTES

from .errors import RenderError
from .keyframes import Easing
from .schema import Composition, ImageLayer, Layer, TextLayer
from .text_animation import TextAnimation, TextAnimationType

__all__ = [
    "DEFAULT_WEB_CARD_PLAY_LABEL",
    "WEB_CARD_RENDERER_VERSION",
    "WebCardError",
    "WebCardRenderResult",
    "build_web_card",
    "export_web_card",
    "render_web_card",
    "web_card_html",
    "write_web_card",
]

WEB_CARD_RENDERER_VERSION = "1"
"""The version of the web card document written by this release."""

DEFAULT_WEB_CARD_PLAY_LABEL = "Play"
"""The visible label of the explicit play button."""

_MIME_BY_FORMAT: dict[str, str] = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "GIF": "image/gif",
    "WEBP": "image/webp",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
}

_EASING_CSS: dict[Easing, str] = {
    Easing.LINEAR: "linear",
    Easing.STEP: "steps(1, end)",
    Easing.SMOOTHSTEP: "cubic-bezier(.4, 0, .2, 1)",
    Easing.EASE_IN: "cubic-bezier(.42, 0, 1, 1)",
    Easing.EASE_OUT: "cubic-bezier(0, 0, .58, 1)",
    Easing.EASE_IN_OUT: "ease-in-out",
    Easing.EASE_IN_SINE: "cubic-bezier(.12, 0, .39, 0)",
    Easing.EASE_OUT_SINE: "cubic-bezier(.61, 1, .88, 1)",
    Easing.EASE_IN_OUT_SINE: "cubic-bezier(.37, 0, .63, 1)",
    Easing.EASE_IN_QUAD: "cubic-bezier(.11, 0, .5, 0)",
    Easing.EASE_OUT_QUAD: "cubic-bezier(.5, 1, .89, 1)",
    Easing.EASE_IN_OUT_QUAD: "cubic-bezier(.45, 0, .55, 1)",
    Easing.EASE_IN_CUBIC: "cubic-bezier(.32, 0, .67, 0)",
    Easing.EASE_OUT_CUBIC: "cubic-bezier(.33, 1, .68, 1)",
    Easing.EASE_IN_OUT_CUBIC: "cubic-bezier(.65, 0, .35, 1)",
    Easing.EASE_IN_EXPO: "cubic-bezier(.7, 0, .84, 0)",
    Easing.EASE_OUT_EXPO: "cubic-bezier(.16, 1, .3, 1)",
    Easing.EASE_IN_OUT_EXPO: "cubic-bezier(.87, 0, .13, 1)",
}

_FORBIDDEN_CSS = frozenset("<>{};")
_DEFAULT_BACKGROUND = "#111111"
_DEFAULT_FOREGROUND = "#ffffff"


class WebCardError(RenderError):
    """A composition could not be rendered as a self-contained web card."""

    code = "web_card"


@dataclass(frozen=True, slots=True)
class _TextBlock:
    """One piece of card text and the CSS animation that reveals it."""

    text: str
    tag: str
    animation: TextAnimation | None
    delay: float
    color: str | None
    align: str | None
    font_size: float | None


@dataclass(frozen=True, slots=True)
class WebCardRenderResult:
    """A written web card, its HTML and the options embedded in it."""

    output: Path
    html: str
    expires_at: datetime | None
    background_embedded: bool
    hit_url: str | None
    card_id: str
    order_id: str

    @property
    def path(self) -> Path:
        """Alias for :attr:`output`."""
        return self.output

    @property
    def size_bytes(self) -> int:
        """The UTF-8 size of the written document."""
        return len(self.html.encode("utf-8"))

    def is_expired(self, now: datetime | None = None) -> bool:
        """Whether the card is expired at *now* (default: the current time)."""
        if self.expires_at is None:
            return False
        moment = datetime.now(UTC) if now is None else _as_utc(now)
        return moment >= self.expires_at

    def __fspath__(self) -> str:
        return str(self.output)

    def __str__(self) -> str:
        return str(self.output)


def _text(value: Any, name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise WebCardError(f"{name} must be text")
    stripped = value.strip()
    if "\x00" in stripped:
        raise WebCardError(f"{name} must not contain NUL")
    return stripped


def _css(value: Any, name: str) -> str:
    text = _text(value, name)
    if not text:
        raise WebCardError(f"{name} must not be empty")
    if _FORBIDDEN_CSS & set(text):
        raise WebCardError(f"{name} contains characters that are unsafe in CSS")
    if "url(" in text.lower() or "expression(" in text.lower():
        raise WebCardError(f"{name} must not load an external resource")
    return text


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _expiry(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise WebCardError("expires_at must be a datetime")
    return _as_utc(value)


def _hit_url(value: str | None) -> str | None:
    if value is None:
        return None
    text = _text(value, "hit_url")
    if not text:
        raise WebCardError("hit_url must not be empty")
    if any(character.isspace() for character in text):
        raise WebCardError("hit_url must not contain whitespace")
    parts = urlsplit(text)
    if parts.scheme:
        if parts.scheme.lower() not in {"http", "https"}:
            raise WebCardError("hit_url must use http or https")
        if not parts.netloc:
            raise WebCardError("hit_url must include a host")
    elif not text.startswith("/") or text.startswith("//"):
        raise WebCardError("hit_url must be absolute or start with a single /")
    return text


def _animation(value: TextAnimation | TextAnimationType | str | None) -> TextAnimation:
    if value is None:
        return TextAnimation(type=TextAnimationType.FADE)
    if isinstance(value, TextAnimation):
        return value
    return TextAnimation(type=TextAnimationType(value))


def _image_mime(path: Path) -> str:
    try:
        with Image.open(path) as image:
            fmt = image.format
    except (OSError, UnidentifiedImageError) as exc:
        raise WebCardError(f"background image could not be read: {path}") from exc
    if fmt is None or fmt not in _MIME_BY_FORMAT:
        raise WebCardError(f"unsupported background image format: {fmt or path}")
    return _MIME_BY_FORMAT[fmt]


def _embed_image(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise WebCardError(f"background image could not be read: {path}") from exc
    if not data:
        raise WebCardError("background image must not be empty")
    if len(data) > DEFAULT_MEDIA_MAX_BYTES:
        raise WebCardError(
            f"background image is larger than {DEFAULT_MEDIA_MAX_BYTES} bytes"
        )
    mime = _image_mime(path)
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _image_layer(layer: Layer) -> ImageLayer:
    fields = ImageLayer.model_fields
    data = {key: value for key, value in layer.model_dump().items() if key in fields}
    try:
        return ImageLayer.model_validate(data)
    except ValidationError as exc:
        raise WebCardError(f"image layer {layer.id or '(unnamed)'} is invalid") from exc


def _text_layer(layer: Layer) -> TextLayer:
    fields = TextLayer.model_fields
    data = {key: value for key, value in layer.model_dump().items() if key in fields}
    try:
        return TextLayer.model_validate(data)
    except ValidationError as exc:
        raise WebCardError(f"text layer {layer.id or '(unnamed)'} is invalid") from exc


def _background_image(
    composition: Composition,
    explicit: str | Path | None,
) -> Path | None:
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if not candidate.is_file():
            raise WebCardError(f"background image not found: {candidate}")
        return candidate
    for layer in sorted(composition.layers, key=lambda item: item.z_index):
        if layer.type.value != "image" or not layer.visible:
            continue
        source = Path(_image_layer(layer).source).expanduser()
        if source.is_file():
            return source
    return None


def _text_blocks(
    composition: Composition,
    *,
    title: str,
    summary: str,
    animation: TextAnimation,
) -> list[_TextBlock]:
    blocks: list[_TextBlock] = []
    if title:
        blocks.append(
            _TextBlock(
                text=title,
                tag="h1",
                animation=animation,
                delay=0.0,
                color=None,
                align=None,
                font_size=None,
            )
        )
    if summary:
        blocks.append(
            _TextBlock(
                text=summary,
                tag="p",
                animation=animation,
                delay=0.0,
                color=None,
                align=None,
                font_size=None,
            )
        )
    for layer in sorted(composition.layers, key=lambda item: item.z_index):
        if layer.type.value != "text" or not layer.visible:
            continue
        parsed = _text_layer(layer)
        chosen = parsed.entrance if parsed.entrance is not None else animation
        delay = parsed.start_seconds
        if chosen is not None:
            delay += chosen.delay
        blocks.append(
            _TextBlock(
                text=parsed.text,
                tag="p",
                animation=chosen,
                delay=delay,
                color=parsed.color,
                align=parsed.align.value,
                font_size=parsed.font_size,
            )
        )
    return blocks


def _keyframe_name(index: int) -> str:
    return f"aivia-anim-{index}"


def _keyframes(name: str, animation: TextAnimation) -> str:
    preset = animation.type
    start_opacity = animation.opacity_from
    end_opacity = animation.opacity_to
    if preset is TextAnimationType.NONE:
        return ""
    start = f"opacity:{start_opacity:.4f};"
    end = f"opacity:{end_opacity:.4f};"
    if preset is TextAnimationType.FADE:
        return f"@keyframes {name} {{ from {{ {start} }} to {{ {end} }} }}"
    if preset in {
        TextAnimationType.SLIDE,
        TextAnimationType.SLIDE_UP,
        TextAnimationType.SLIDE_DOWN,
        TextAnimationType.SLIDE_LEFT,
        TextAnimationType.SLIDE_RIGHT,
    }:
        x, y = animation.offset_for(entrance=True)
        return (
            f"@keyframes {name} {{ from {{ {start} "
            f"transform:translate({x:.2f}px, {y:.2f}px); }} "
            f"to {{ {end} transform:translate(0, 0); }} }}"
        )
    if preset in {TextAnimationType.ZOOM, TextAnimationType.POP}:
        start_scale = animation.start_scale
        end_scale = animation.end_scale
        return (
            f"@keyframes {name} {{ from {{ {start} "
            f"transform:scale({start_scale:.4f}); }} "
            f"to {{ {end} transform:scale({end_scale:.4f}); }} }}"
        )
    return f"@keyframes {name} {{ from {{ {start} }} to {{ {end} }} }}"


def _style(animation: TextAnimation, name: str, delay: float) -> str:
    if animation.type is TextAnimationType.NONE:
        return ""
    easing = _EASING_CSS.get(animation.easing, "ease")
    return (
        f"animation-name:{name};"
        f"animation-duration:{animation.duration:.3f}s;"
        f"animation-delay:{delay:.3f}s;"
        f"animation-timing-function:{easing};"
        "animation-fill-mode:both;"
    )


def _block_html(block: _TextBlock, index: int) -> str:
    classes = "aivia-text"
    if (
        block.animation is not None
        and block.animation.type is not TextAnimationType.NONE
    ):
        classes += " aivia-animated"
    attributes = [f'class="{classes}"']
    styles: list[str] = []
    if block.animation is not None:
        style = _style(block.animation, _keyframe_name(index), block.delay)
        if style:
            styles.append(style)
    if block.color:
        styles.append(f"color:{_css(block.color, 'text color')};")
    if block.align:
        styles.append(f"text-align:{_css(block.align, 'text align')};")
    if block.font_size:
        styles.append(f"font-size:{block.font_size:.2f}px;")
    if styles:
        attributes.append(f'style="{html.escape("".join(styles), quote=True)}"')
    content = html.escape(block.text)
    return f"<{block.tag} {' '.join(attributes)}>{content}</{block.tag}>"


def _background_style(
    composition: Composition,
    color: str | None,
    image: str | None,
) -> str:
    if image is not None:
        return (
            f"background-color:{_DEFAULT_BACKGROUND};"
            f'background-image:url("{image}");'
            "background-size:cover;background-position:center;"
            "background-repeat:no-repeat;"
        )
    chosen = _css(
        color if color is not None else composition.canvas.background,
        "background",
    )
    return f"background-color:{chosen};"


def build_web_card(
    composition: Composition | Mapping[str, Any],
    *,
    title: str = "",
    summary: str = "",
    language: str = "en",
    play_label: str = DEFAULT_WEB_CARD_PLAY_LABEL,
    animation: TextAnimation | TextAnimationType | str | None = None,
    background_color: str | None = None,
    background_image: str | Path | None = None,
    expires_at: datetime | None = None,
    hit_url: str | None = None,
    card_id: str = "",
    order_id: str = "",
) -> str:
    """Build the self-contained web card document without writing it."""
    if not isinstance(composition, Composition):
        composition = Composition.model_validate(composition)
    chosen_animation = _animation(animation)
    title_text = _text(title, "title")
    summary_text = _text(summary, "summary")
    language_text = _text(language, "language") or "en"
    label = _text(play_label, "play_label") or DEFAULT_WEB_CARD_PLAY_LABEL
    expiry = _expiry(expires_at)
    hit = _hit_url(hit_url)
    card = _text(card_id, "card_id")
    order = _text(order_id, "order_id")

    source = _background_image(composition, background_image)
    embedded: str | None = None
    if source is not None:
        embedded = _embed_image(source)
    background = _background_style(composition, background_color, embedded)

    blocks = _text_blocks(
        composition,
        title=title_text,
        summary=summary_text,
        animation=chosen_animation,
    )
    keyframes: list[str] = []
    body: list[str] = []
    for index, block in enumerate(blocks, start=1):
        if block.animation is not None:
            frames = _keyframes(_keyframe_name(index), block.animation)
            if frames:
                keyframes.append(frames)
        body.append(_block_html(block, index))

    expiry_attribute = ""
    expiry_meta = ""
    if expiry is not None:
        value = expiry.isoformat()
        expiry_attribute = f' data-expires-at="{html.escape(value, quote=True)}"'
        expiry_meta = (
            f'<meta name="aivia-expires-at" content="{html.escape(value, quote=True)}">'
        )
    hit_meta = ""
    if hit is not None:
        hit_meta = (
            f'<meta name="aivia-hit-url" content="{html.escape(hit, quote=True)}">'
        )

    style = (
        ":root{color-scheme:dark light;}"
        "html,body{margin:0;min-height:100%;}"
        "body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;}"
        ".aivia-card{position:relative;min-height:100vh;display:flex;"
        "flex-direction:column;align-items:center;justify-content:center;"
        "padding:2rem;"
        f"color:{_DEFAULT_FOREGROUND};{background}}}"
        ".aivia-card::before{content:'';position:absolute;inset:0;"
        "background:linear-gradient(180deg,rgba(0,0,0,.15),rgba(0,0,0,.55));"
        "pointer-events:none;}"
        ".aivia-content{position:relative;z-index:1;max-width:44rem;width:100%;}"
        ".aivia-text{margin:0 0 1rem;line-height:1.25;"
        "overflow-wrap:anywhere;}"
        ".aivia-text:last-child{margin-bottom:0;}"
        ".aivia-card:not(.is-playing) .aivia-animated{"
        "animation-play-state:paused;}"
        ".aivia-play{position:relative;z-index:2;margin:1.5rem auto 0;"
        "display:inline-flex;align-items:center;gap:.5rem;border:0;"
        "border-radius:999px;padding:.75rem 1.25rem;font:inherit;"
        "font-weight:bold;cursor:pointer;background:#fff;color:#111;}"
        ".aivia-play:focus-visible{outline:3px solid #7dd3fc;outline-offset:3px;}"
        ".aivia-card.is-playing .aivia-play{display:none;}"
        ".aivia-expired{display:none;}"
        ".aivia-card.is-expired .aivia-content,"
        ".aivia-card.is-expired .aivia-play{display:none;}"
        ".aivia-card.is-expired .aivia-expired{display:block;}"
        "@media (prefers-reduced-motion:reduce){"
        ".aivia-animated{animation:none!important;}}" + "".join(keyframes)
    )
    title_markup = ""
    if title_text:
        title_markup = f"<title>{html.escape(title_text)}</title>"
    content = "\n".join(f"      {item}" for item in body)
    script = (
        "(function(){"
        "var card=document.querySelector('.aivia-card');"
        "if(!card){return;}"
        "var play=card.querySelector('.aivia-play');"
        "if(play){play.addEventListener('click',function(){"
        "card.classList.add('is-playing');"
        "play.setAttribute('aria-pressed','true');"
        "});}"
        "var expires=card.getAttribute('data-expires-at');"
        "if(expires){var when=Date.parse(expires);"
        "if(!isNaN(when)&&Date.now()>=when){card.classList.add('is-expired');}}"
    )
    if hit is not None:
        script += (
            "var meta=document.querySelector('meta[name=\"aivia-hit-url\"]');"
            "if(!meta){return;}var url=meta.getAttribute('content');"
            "if(!url){return;}"
            "var payload=JSON.stringify({"
            "event:'view',"
            "card_id:card.getAttribute('data-card-id')||'',"
            "order_id:card.getAttribute('data-order-id')||''"
            "});"
            "function send(){"
            "if(navigator.sendBeacon){"
            "navigator.sendBeacon(url,new Blob([payload],"
            "{type:'application/json'}));"
            "}else if(window.fetch){fetch(url,{method:'POST',"
            "headers:{'Content-Type':'application/json'},body:payload,"
            "keepalive:true});}"
            "}"
            "if(document.readyState==='complete'){send();}"
            "else{window.addEventListener('load',send,{once:true});}"
        )
    script += "})();"
    return (
        "<!doctype html>\n"
        f'<html lang="{html.escape(language_text, quote=True)}">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="aivia-renderer" content="web_card/'
        f'{WEB_CARD_RENDERER_VERSION}">\n'
        f"{title_markup}\n"
        f"{expiry_meta}\n"
        f"{hit_meta}\n"
        f"<style>{style}</style>\n"
        "</head>\n"
        f'<body data-card-id="{html.escape(card, quote=True)}" '
        f'data-order-id="{html.escape(order, quote=True)}">\n'
        f'<main class="aivia-card"{expiry_attribute}>\n'
        '  <div class="aivia-content">\n'
        f"{content}\n"
        "  </div>\n"
        f'  <button class="aivia-play" type="button" aria-pressed="false">'
        f'<span aria-hidden="true">\u25b6</span> {html.escape(label)}</button>\n'
        '  <p class="aivia-expired" role="status">This card has expired.</p>\n'
        "</main>\n"
        "<noscript><style>.aivia-animated{animation-play-state:running!important;}"
        ".aivia-play{display:none;}</style></noscript>\n"
        f"<script>{script}</script>\n"
        "</body>\n"
        "</html>\n"
    )


web_card_html = build_web_card
"""Alias for :func:`build_web_card`."""


def render_web_card(
    composition: Composition | Mapping[str, Any],
    output: str | Path,
    *,
    title: str = "",
    summary: str = "",
    language: str = "en",
    play_label: str = DEFAULT_WEB_CARD_PLAY_LABEL,
    animation: TextAnimation | TextAnimationType | str | None = None,
    background_color: str | None = None,
    background_image: str | Path | None = None,
    expires_at: datetime | None = None,
    hit_url: str | None = None,
    card_id: str = "",
    order_id: str = "",
) -> WebCardRenderResult:
    """Write a self-contained web card to *output* and return the result."""
    card = (
        composition
        if isinstance(composition, Composition)
        else Composition.model_validate(composition)
    )
    target = Path(output).expanduser()
    document = build_web_card(
        card,
        title=title,
        summary=summary,
        language=language,
        play_label=play_label,
        animation=animation,
        background_color=background_color,
        background_image=background_image,
        expires_at=expires_at,
        hit_url=hit_url,
        card_id=card_id,
        order_id=order_id,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(document)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    source = _background_image(card, background_image)
    return WebCardRenderResult(
        output=target,
        html=document,
        expires_at=_expiry(expires_at),
        background_embedded=source is not None,
        hit_url=_hit_url(hit_url),
        card_id=_text(card_id, "card_id"),
        order_id=_text(order_id, "order_id"),
    )


write_web_card = render_web_card
"""Alias for :func:`render_web_card`."""

export_web_card = render_web_card
"""Alias for :func:`render_web_card`."""
