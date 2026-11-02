"""Cover frames and brand-kit watermark, intro and outro application."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from aivia.compose import (
    COVER_BRANDING_VERSION,
    Anchor,
    BrandedVideoResult,
    BrandingError,
    Composition,
    CoverBrandingError,
    CoverFrame,
    CoverRenderResult,
    apply_branding_to_video,
    apply_intro_outro,
    apply_watermark,
    build_watermark_command,
    cover_frame_times,
    frame_score,
    generate_cover_frames,
    render_cover,
    render_thumbnail,
    select_cover_frame,
    watermark_composition,
    watermark_image,
    watermark_video,
)
from aivia.core.config.character import BrandKitConfig
from aivia.core.media_probe import run_ffprobe

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg and ffprobe are required for a real branding render",
)


def _image(tmp_path: Path, name: str, size: tuple[int, int], color: Any) -> Path:
    path = tmp_path / name
    Image.new("RGBA", size, color).save(path)
    return path


def _checkerboard(tmp_path: Path) -> Path:
    path = tmp_path / "checker.png"
    image = Image.new("RGB", (32, 32), (0, 0, 0))
    for y in range(0, 32, 4):
        for x in range(0, 32, 4):
            if (x // 4 + y // 4) % 2 == 0:
                image.paste((255, 255, 255), (x, y, x + 4, y + 4))
    image.save(path)
    return path


def _composition(
    tmp_path: Path,
    *,
    width: int = 128,
    height: int = 128,
    detail_start: float = 0.6,
) -> Composition:
    checker = _checkerboard(tmp_path)
    return Composition.model_validate(
        {
            "id": "cover",
            "canvas": {"width": width, "height": height, "background": "#000000"},
            "duration_seconds": 2.0,
            "fps": 12,
            "layers": [
                {
                    "id": "detail",
                    "type": "image",
                    "source": str(checker),
                    "x": 0,
                    "y": 0,
                    "width": width,
                    "height": height,
                    "start_seconds": detail_start,
                    "end_seconds": 2.0,
                }
            ],
        }
    )


def _branding(tmp_path: Path, **overrides: Any) -> BrandKitConfig:
    values: dict[str, Any] = {
        "name": "studio",
        "watermark": str(_image(tmp_path, "mark.png", (16, 16), (0, 0, 255, 255))),
    }
    values.update(overrides)
    return BrandKitConfig.from_mapping(values)


def test_cover_frame_times_are_evenly_spaced_and_validated(
    tmp_path: Path,
) -> None:
    composition = _composition(tmp_path)

    assert cover_frame_times(composition, 4) == [0.25, 0.75, 1.25, 1.75]
    assert cover_frame_times(composition, 1) == [1.0]
    assert cover_frame_times(composition, 2, start_seconds=0.5, end_seconds=1.5) == [
        0.75,
        1.25,
    ]

    with pytest.raises(CoverBrandingError, match="count"):
        cover_frame_times(composition, 0)
    with pytest.raises(CoverBrandingError, match="start_seconds"):
        cover_frame_times(composition, 2, start_seconds=2.0)
    with pytest.raises(CoverBrandingError, match="end_seconds"):
        cover_frame_times(composition, 2, start_seconds=1.0, end_seconds=0.5)


def test_frame_score_prefers_detail_over_flat_frames(tmp_path: Path) -> None:
    flat = Image.new("RGB", (64, 64), (0, 0, 0))
    detailed = Image.open(_checkerboard(tmp_path)).convert("RGB")

    assert frame_score(flat) < frame_score(detailed)
    with pytest.raises(CoverBrandingError):
        frame_score("not an image")  # type: ignore[arg-type]


def test_generate_cover_frames_writes_scored_candidates(
    tmp_path: Path,
) -> None:
    composition = _composition(tmp_path)
    output = tmp_path / "covers"

    frames = generate_cover_frames(
        composition,
        output,
        times=[0.25, 1.0],
        size=(64, 64),
    )

    assert [frame.output.name for frame in frames] == ["cover-000.png", "cover-001.png"]
    assert all(frame.output.is_file() for frame in frames)
    assert all(frame.size == (64, 64) for frame in frames)
    assert frames[1].score > frames[0].score
    assert frames[1].time_seconds == 1.0


def test_generate_cover_frames_applies_the_branding_watermark(
    tmp_path: Path,
) -> None:
    composition = _composition(tmp_path, detail_start=0.0)
    frames = generate_cover_frames(
        composition,
        tmp_path / "covers",
        times=[0.5],
        branding=_branding(tmp_path),
        margin=0,
    )

    assert frames[0].watermarked is True
    with Image.open(frames[0].output) as image:
        pixel = image.convert("RGBA").getpixel((127, 127))
    assert isinstance(pixel, tuple)
    assert pixel[2] > 200
    assert pixel[0] < 80


def test_render_cover_selects_the_best_candidate_and_returns_a_result(
    tmp_path: Path,
) -> None:
    composition = _composition(tmp_path)
    output = tmp_path / "cover.png"

    result = render_cover(composition, output, count=4)

    assert isinstance(result, CoverRenderResult)
    assert result.output == output
    assert result == output
    assert output.is_file()
    assert result.time_seconds >= 0.75
    assert result.selected.score == pytest.approx(result.score)
    assert len(result.frames) == 4
    assert result.watermark_applied is False
    with Image.open(output) as image:
        assert image.size == composition.canvas.size


def test_render_thumbnail_uses_the_default_width(tmp_path: Path) -> None:
    composition = _composition(tmp_path, width=1000, height=500, detail_start=0.0)

    result = render_thumbnail(composition, tmp_path / "thumb.png", time_seconds=0.5)

    assert isinstance(result, CoverRenderResult)
    assert result.selected.size == (640, 320)


def test_apply_watermark_composites_an_image_at_the_requested_anchor(
    tmp_path: Path,
) -> None:
    base = Image.new("RGBA", (40, 30), (255, 0, 0, 255))
    mark = Image.new("RGBA", (10, 10), (0, 0, 255, 255))

    result = apply_watermark(
        base,
        mark,
        anchor=Anchor.BOTTOM_RIGHT,
        margin=0,
        opacity=1.0,
    )

    assert isinstance(result, Image.Image)
    assert result.size == base.size
    assert result.getpixel((0, 0)) == (255, 0, 0, 255)
    assert result.getpixel((39, 29)) == (0, 0, 255, 255)
    assert base.getpixel((39, 29)) == (255, 0, 0, 255)


def test_watermark_image_accepts_paths_and_rejects_missing_files(
    tmp_path: Path,
) -> None:
    base = _image(tmp_path, "base.png", (20, 20), (255, 255, 255, 255))
    mark = _image(tmp_path, "mark.png", (4, 4), (0, 0, 0, 255))

    result = watermark_image(base, mark, margin=0, opacity=1.0)

    assert result.size == (20, 20)
    assert result.getpixel((19, 19)) == (0, 0, 0, 255)
    with pytest.raises(CoverBrandingError, match="could not be read"):
        watermark_image(base, tmp_path / "missing.png")


def test_watermark_composition_adds_a_logo_layer_once(tmp_path: Path) -> None:
    composition = _composition(tmp_path, detail_start=0.0)
    branding = _branding(tmp_path)

    first = watermark_composition(composition, branding=branding)
    second = watermark_composition(first, branding=branding)

    layers = [layer for layer in first.layers if layer.type.value == "logo"]
    assert len(layers) == 1
    extras = layers[0].model_extra or {}
    assert layers[0].id == "brand-watermark"
    assert extras["source"] == branding.watermark
    assert extras["anchor"] == "bottom_right"
    assert len(second.layers) == len(first.layers) + 1
    assert [layer.id for layer in second.layers][-1] == "brand-watermark-2"
    assert watermark_composition(composition, branding=BrandKitConfig()) == composition


def test_select_cover_frame_prefers_the_highest_score_then_the_earliest(
    tmp_path: Path,
) -> None:
    early = CoverFrame(tmp_path / "early.png", 0.5, 0.9, 10, 10)
    late = CoverFrame(tmp_path / "late.png", 1.5, 0.9, 10, 10)
    low = CoverFrame(tmp_path / "low.png", 0.25, 0.2, 10, 10)

    assert select_cover_frame([low, late, early]) is early
    with pytest.raises(CoverBrandingError, match="at least one"):
        select_cover_frame([])


def test_build_watermark_command_exposes_the_overlay_graph(tmp_path: Path) -> None:
    command = build_watermark_command(
        "video.mp4",
        "mark.png",
        "out.mp4",
        anchor=Anchor.TOP_LEFT,
        margin=(8, 12),
        opacity=0.5,
        scale=0.25,
    )

    assert command[0] == "ffmpeg"
    assert command.inputs[0].source == "video.mp4"
    assert command.inputs[1].source == "mark.png"
    assert "scale=iw*0.25:ih*0.25" in command.filtergraph
    assert "colorchannelmixer=aa=0.5" in command.filtergraph
    assert "overlay=8:12:format=auto" in command.filtergraph
    assert command.output == "out.mp4"

    with pytest.raises(CoverBrandingError, match="opacity"):
        build_watermark_command("v.mp4", "m.png", "o.mp4", opacity=1.5)
    with pytest.raises(CoverBrandingError, match="scale"):
        build_watermark_command("v.mp4", "m.png", "o.mp4", scale=0)


def test_apply_intro_outro_joins_the_branding_clips_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aivia.compose import cover_branding

    calls: list[list[Path]] = []
    options: list[dict[str, Any]] = []

    def fake_concat(sources: Sequence[Path], destination: Path, **kwargs: Any) -> Path:
        calls.append(list(sources))
        options.append(dict(kwargs))
        Path(destination).write_bytes(b"joined")
        return Path(destination)

    monkeypatch.setattr(cover_branding, "concatenate_videos", fake_concat)
    intro = tmp_path / "intro.mp4"
    outro = tmp_path / "outro.mp4"
    video = tmp_path / "main.mp4"
    intro.write_bytes(b"intro")
    outro.write_bytes(b"outro")
    video.write_bytes(b"main")
    branding = BrandKitConfig.from_mapping({"intro": str(intro), "outro": str(outro)})

    result = apply_intro_outro(video, tmp_path / "branded.mp4", branding=branding)

    assert isinstance(result, BrandedVideoResult)
    assert calls == [[intro, video, outro]]
    assert options[0]["normalize"] is True
    assert options[0]["reference"] == 1
    assert result.intro == intro
    assert result.outro == outro
    assert result.sources == (intro, video, outro)
    assert result.output.read_bytes() == b"joined"


def test_apply_intro_outro_normalizes_mixed_clips_to_the_main_video(
    tmp_path: Path,
) -> None:
    intro = tmp_path / "intro.mp4"
    main = tmp_path / "main.mp4"
    outro = tmp_path / "outro.mp4"
    for clip in (intro, main, outro):
        clip.write_bytes(b"video")
    calls: list[list[str]] = []

    def probe(
        *,
        audio: bool,
        width: int,
        height: int,
        frame_rate: str,
        sample_rate: int,
        channels: int,
        pixel_format: str,
    ) -> str:
        streams: list[dict[str, Any]] = [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": width,
                "height": height,
                "duration": "1.0",
                "avg_frame_rate": frame_rate,
                "pix_fmt": pixel_format,
            }
        ]
        if audio:
            streams.append(
                {
                    "index": 1,
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "duration": "1.0",
                    "sample_rate": str(sample_rate),
                    "channels": channels,
                }
            )
        return json.dumps(
            {
                "streams": streams,
                "format": {
                    "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                    "format_long_name": "QuickTime / MOV",
                    "duration": "1.0",
                    "size": "12345",
                },
            }
        )

    formats = {
        str(intro): probe(
            audio=True,
            width=320,
            height=240,
            frame_rate="24/1",
            sample_rate=44100,
            channels=1,
            pixel_format="yuv444p",
        ),
        str(main): probe(
            audio=True,
            width=640,
            height=360,
            frame_rate="30/1",
            sample_rate=48000,
            channels=2,
            pixel_format="yuv420p",
        ),
        str(outro): probe(
            audio=True,
            width=1280,
            height=720,
            frame_rate="25/1",
            sample_rate=48000,
            channels=6,
            pixel_format="yuv420p",
        ),
    }

    def ffprobe_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            list(argv), 0, stdout=formats[argv[-1]], stderr=""
        )

    def ffmpeg_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        Path(argv[-1]).write_bytes(b"joined")
        return subprocess.CompletedProcess(list(argv), 0, stdout="", stderr="")

    result = apply_intro_outro(
        main,
        tmp_path / "branded.mp4",
        intro=intro,
        outro=outro,
        ffmpeg_runner=ffmpeg_runner,
        ffprobe_runner=ffprobe_runner,
    )

    assert len(calls) == 1
    graph = calls[0][calls[0].index("-filter_complex") + 1]
    intro_filter = (
        "scale=640:360:force_original_aspect_ratio=decrease:flags=lanczos,"
        "pad=640:360:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
        "fps=30.000000,format=pix_fmts=yuv420p"
    )
    assert f"[0:v]{intro_filter}[vn0]" in graph
    assert f"[2:v]{intro_filter.replace(',format=pix_fmts=yuv420p', '')}[vn2]" in graph
    assert "[vn0][1:v][vn2]concat=n=3:v=1:a=0[v]" in graph
    assert "[0:a]aresample=48000,aformat=channel_layouts=stereo[an0]" in graph
    assert "[2:a]aformat=channel_layouts=stereo[an2]" in graph
    assert "[an0][1:a][an2]concat=n=3:v=0:a=1[a]" in graph
    assert result.sources == (intro, main, outro)
    assert result.intro == intro
    assert result.outro == outro
    assert result.output.read_bytes() == b"joined"


def test_apply_intro_outro_drops_audio_when_a_clip_has_none(
    tmp_path: Path,
) -> None:
    intro = tmp_path / "intro.mp4"
    main = tmp_path / "main.mp4"
    for clip in (intro, main):
        clip.write_bytes(b"video")
    calls: list[list[str]] = []

    def probe(audio: bool) -> str:
        streams: list[dict[str, Any]] = [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 640,
                "height": 360,
                "duration": "1.0",
                "avg_frame_rate": "30/1",
                "pix_fmt": "yuv420p",
            }
        ]
        if audio:
            streams.append(
                {
                    "index": 1,
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "duration": "1.0",
                    "sample_rate": "48000",
                    "channels": 2,
                }
            )
        return json.dumps(
            {
                "streams": streams,
                "format": {
                    "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                    "format_long_name": "QuickTime / MOV",
                    "duration": "1.0",
                    "size": "12345",
                },
            }
        )

    formats = {str(intro): probe(False), str(main): probe(True)}

    def ffprobe_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            list(argv), 0, stdout=formats[argv[-1]], stderr=""
        )

    def ffmpeg_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        Path(argv[-1]).write_bytes(b"joined")
        return subprocess.CompletedProcess(list(argv), 0, stdout="", stderr="")

    apply_intro_outro(
        main,
        tmp_path / "branded.mp4",
        intro=intro,
        ffmpeg_runner=ffmpeg_runner,
        ffprobe_runner=ffprobe_runner,
    )

    graph = calls[0][calls[0].index("-filter_complex") + 1]
    assert "concat=n=2:v=1:a=0[v]" in graph
    assert "concat=n=2:v=0:a=1" not in graph
    assert "-an" in calls[0]


def test_apply_intro_outro_copies_when_no_clip_is_configured(
    tmp_path: Path,
) -> None:
    video = tmp_path / "main.mp4"
    video.write_bytes(b"video")
    output = tmp_path / "copy.mp4"

    result = apply_intro_outro(video, output)

    assert result.sources == (video,)
    assert output.read_bytes() == b"video"


def test_watermark_video_runs_an_overlay_command_and_reports_it(
    tmp_path: Path,
) -> None:
    video = tmp_path / "main.mp4"
    video.write_bytes(b"video")
    output = tmp_path / "watermarked.mp4"
    calls: list[list[str]] = []

    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        Path(argv[-1]).write_bytes(b"watermarked")
        return subprocess.CompletedProcess(list(argv), 0, stdout="", stderr="")

    result = watermark_video(
        video,
        output,
        _image(tmp_path, "mark.png", (4, 4), (0, 0, 0, 255)),
        ffmpeg_runner=runner,
    )

    assert result.watermarked is True
    assert output.read_bytes() == b"watermarked"
    assert "-filter_complex" in calls[0]
    assert "overlay=" in calls[0][calls[0].index("-filter_complex") + 1]


def test_apply_branding_to_video_joins_then_watermarks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aivia.compose import cover_branding

    calls: list[str] = []
    video = tmp_path / "main.mp4"
    video.write_bytes(b"video")
    mark = _image(tmp_path, "mark.png", (4, 4), (0, 0, 0, 255))
    branding = BrandKitConfig.from_mapping({"watermark": str(mark)})

    def fake_join(
        source: str | Path, output: str | Path, **options: Any
    ) -> BrandedVideoResult:
        calls.append("join")
        Path(output).write_bytes(b"joined")
        return BrandedVideoResult(Path(output), None, None, False, (Path(source),))

    def fake_watermark(
        source: str | Path,
        output: str | Path,
        watermark: str | Path | None = None,
        **options: Any,
    ) -> BrandedVideoResult:
        calls.append("watermark")
        Path(output).write_bytes(b"branded")
        return BrandedVideoResult(Path(output), None, None, True, (Path(source),))

    monkeypatch.setattr(cover_branding, "apply_intro_outro", fake_join)
    monkeypatch.setattr(cover_branding, "watermark_video", fake_watermark)

    result = apply_branding_to_video(
        video,
        tmp_path / "branded.mp4",
        branding=branding,
        intro=tmp_path / "intro.mp4",
    )

    assert calls == ["join", "watermark"]
    assert result.watermarked is True
    assert result.output.read_bytes() == b"branded"
    assert result.sources == (tmp_path / "intro.mp4", video)
    assert result.intro == tmp_path / "intro.mp4"
    assert result.outro is None


def test_invalid_branding_values_are_rejected() -> None:
    assert COVER_BRANDING_VERSION == "1"
    with pytest.raises(BrandingError):
        apply_watermark(Image.new("RGB", (4, 4)), object())  # type: ignore[arg-type]
    with pytest.raises(CoverBrandingError, match="branding"):
        watermark_composition(
            Composition.model_validate(
                {
                    "canvas": {"width": 10, "height": 10},
                    "duration_seconds": 1,
                }
            ),
            branding=object(),
        )


@requires_ffmpeg
@pytest.mark.slow
def test_a_real_branding_pass_joins_and_watermarks_a_video(
    tmp_path: Path,
) -> None:
    assert FFMPEG is not None

    def make_clip(name: str, color: str) -> Path:
        path = tmp_path / name
        subprocess.run(
            [
                FFMPEG,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=64x48:r=12:d=1",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return path

    main = make_clip("main.mp4", "red")
    intro = make_clip("intro.mp4", "green")
    outro = make_clip("outro.mp4", "blue")
    mark = _image(tmp_path, "mark.png", (8, 8), (255, 255, 0, 255))
    branding = BrandKitConfig.from_mapping(
        {
            "intro": str(intro),
            "outro": str(outro),
            "watermark": str(mark),
        }
    )

    result = apply_branding_to_video(
        main,
        tmp_path / "branded.mp4",
        branding=branding,
        margin=0,
    )

    probe = run_ffprobe(result.output)
    assert probe.duration_seconds == pytest.approx(3.0, abs=0.3)
    assert result.sources == (intro, main, outro)
    assert result.watermarked is True
