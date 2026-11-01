"""Video utilities trim, retime, join and encode without editing the source."""

from __future__ import annotations

import json
import struct
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from aivia.core.errors import ValidationError
from aivia.media.video_utils import (
    VideoOutputFormat,
    VideoSizeError,
    change_speed,
    concatenate_videos,
    convert_video,
    extract_audio,
    extract_frame,
    make_seamless_loop,
    trim_video,
)

MP4 = struct.pack(">I", 24) + b"ftypisom" + b"\x00" * 12
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
AUDIO = b"ID3" + b"\x00" * 16


def writer(
    *,
    calls: list[list[str]] | None = None,
    payload: bytes = MP4,
    sizes: Sequence[int] | None = None,
    concat_lists: list[str] | None = None,
) -> Callable[[Sequence[str]], subprocess.CompletedProcess[Any]]:
    remaining = list(sizes) if sizes is not None else None

    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[Any]:
        if calls is not None:
            calls.append(list(argv))
        target = Path(argv[-1])
        if remaining is not None:
            target.write_bytes(b"x" * remaining.pop(0))
        else:
            target.write_bytes(payload)
        if concat_lists is not None and "concat" in argv:
            list_path = Path(argv[argv.index("-i") + 1])
            concat_lists.append(list_path.read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(list(argv), 0, stdout=b"", stderr=b"")

    return run


def probe_json(
    duration: float,
    *,
    audio: bool = True,
    width: int = 1280,
    height: int = 720,
    frame_rate: str = "30/1",
) -> str:
    streams: list[dict[str, Any]] = [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "width": width,
            "height": height,
            "duration": f"{duration:.3f}",
            "avg_frame_rate": frame_rate,
        }
    ]
    if audio:
        streams.append(
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "duration": f"{duration:.3f}",
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
                "duration": f"{duration:.3f}",
                "size": "12345",
            },
        }
    )


def probe_runner(
    durations: Sequence[float],
    *,
    audio: bool | Sequence[bool] = True,
    dimensions: Sequence[tuple[int, int]] | None = None,
    frame_rates: Sequence[str] | None = None,
    calls: list[list[str]] | None = None,
) -> Callable[[Sequence[str]], subprocess.CompletedProcess[str]]:
    remaining = iter(durations)
    audio_values = (
        list(audio) if isinstance(audio, Sequence) else [bool(audio)] * len(durations)
    )
    audio_iter = iter(audio_values)
    dimension_values = (
        list(dimensions) if dimensions is not None else [(1280, 720)] * len(durations)
    )
    dimension_iter = iter(dimension_values)
    rate_values = (
        list(frame_rates) if frame_rates is not None else ["30/1"] * len(durations)
    )
    rate_iter = iter(rate_values)

    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append(list(argv))
        width, height = next(dimension_iter)
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout=probe_json(
                next(remaining),
                audio=next(audio_iter),
                width=width,
                height=height,
                frame_rate=next(rate_iter),
            ),
            stderr="",
        )

    return run


def source_file(tmp_path: Path, name: str = "source.mp4") -> Path:
    path = tmp_path / name
    path.write_bytes(MP4)
    return path


def option(argv: Sequence[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def filter_graph(argv: Sequence[str]) -> str:
    return option(argv, "-filter_complex")


def test_trim_video_writes_the_requested_range_and_keeps_the_source(
    tmp_path: Path,
) -> None:
    source = source_file(tmp_path)
    before = source.read_bytes()
    calls: list[list[str]] = []
    destination = tmp_path / "trimmed.mp4"

    result = trim_video(
        source,
        destination,
        start_seconds=1.5,
        end_seconds=4.0,
        ffmpeg_runner=writer(calls=calls),
    )

    assert result == destination
    assert destination.read_bytes() == MP4
    assert source.read_bytes() == before
    assert len(calls) == 1
    assert option(calls[0], "-i") == str(source)
    assert option(calls[0], "-ss") == "1.500000"
    assert option(calls[0], "-t") == "2.500000"
    assert calls[0][-1] != str(destination)


def test_trim_video_accepts_duration_and_an_inline_source(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    destination = tmp_path / "trimmed.mp4"

    trim_video(
        MP4,
        destination,
        start_seconds=2,
        duration_seconds=3,
        ffmpeg_runner=writer(calls=calls),
    )

    assert destination.exists()
    assert option(calls[0], "-ss") == "2.000000"
    assert option(calls[0], "-t") == "3.000000"


def test_trim_video_rejects_conflicting_or_invalid_ranges(tmp_path: Path) -> None:
    source = source_file(tmp_path)

    with pytest.raises(ValidationError, match="not both"):
        trim_video(
            source,
            tmp_path / "out.mp4",
            end_seconds=2,
            duration_seconds=1,
        )
    with pytest.raises(ValidationError, match="greater than"):
        trim_video(source, tmp_path / "out.mp4", start_seconds=2, end_seconds=2)
    with pytest.raises(ValidationError, match="not be negative"):
        trim_video(source, tmp_path / "out.mp4", start_seconds=-1)


def test_change_speed_stretches_video_and_chains_atempo(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    calls: list[list[str]] = []

    change_speed(
        source,
        tmp_path / "fast.mp4",
        speed=4.0,
        ffmpeg_runner=writer(calls=calls),
    )

    graph = filter_graph(calls[0])
    assert "setpts=(PTS-STARTPTS)/4.000000" in graph
    assert option(calls[0], "-af") == "atempo=2.000000,atempo=2.000000"
    assert "-map" in calls[0]


def test_change_speed_slows_audio_and_can_drop_it(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    slow_calls: list[list[str]] = []
    silent_calls: list[list[str]] = []

    change_speed(
        source,
        tmp_path / "slow.mp4",
        speed=0.25,
        ffmpeg_runner=writer(calls=slow_calls),
    )
    change_speed(
        source,
        tmp_path / "silent.mp4",
        speed=2.0,
        keep_audio=False,
        ffmpeg_runner=writer(calls=silent_calls),
    )

    assert option(slow_calls[0], "-af") == "atempo=0.500000,atempo=0.500000"
    assert "-af" not in silent_calls[0]
    assert "-an" in silent_calls[0]


def test_change_speed_rejects_a_non_positive_factor(tmp_path: Path) -> None:
    source = source_file(tmp_path)

    with pytest.raises(ValidationError, match="positive"):
        change_speed(source, tmp_path / "out.mp4", speed=0)


def test_concatenate_without_crossfade_uses_a_concat_list(tmp_path: Path) -> None:
    first = source_file(tmp_path, "first.mp4")
    second = source_file(tmp_path, "second.mp4")
    calls: list[list[str]] = []
    lists: list[str] = []

    concatenate_videos(
        [first, second],
        tmp_path / "joined.mp4",
        ffmpeg_runner=writer(calls=calls, concat_lists=lists),
    )

    assert len(calls) == 1
    assert option(calls[0], "-f") == "concat"
    assert str(first.resolve()) in lists[0]
    assert str(second.resolve()) in lists[0]


def test_concatenate_with_crossfade_builds_offsets_and_audio_fades(
    tmp_path: Path,
) -> None:
    first = source_file(tmp_path, "first.mp4")
    second = source_file(tmp_path, "second.mp4")
    calls: list[list[str]] = []
    probes: list[list[str]] = []

    concatenate_videos(
        [first, second],
        tmp_path / "faded.mp4",
        crossfade_seconds=1.0,
        ffmpeg_runner=writer(calls=calls),
        ffprobe_runner=probe_runner([4.0, 6.0], calls=probes),
    )

    graph = filter_graph(calls[0])
    assert "xfade=transition=fade:duration=1.000000:offset=3.000000" in graph
    assert "acrossfade=d=1.000000" in graph
    assert "-map" in calls[0]
    assert "[v1]" in calls[0]
    assert "[a1]" in calls[0]
    assert len(probes) == 2


def test_concatenate_crossfade_normalizes_mixed_dimensions_and_frame_rates(
    tmp_path: Path,
) -> None:
    first = source_file(tmp_path, "first.mp4")
    second = source_file(tmp_path, "second.mp4")
    third = source_file(tmp_path, "third.mp4")
    calls: list[list[str]] = []

    concatenate_videos(
        [first, second, third],
        tmp_path / "mixed.mp4",
        crossfade_seconds=1.0,
        ffmpeg_runner=writer(calls=calls),
        ffprobe_runner=probe_runner(
            [4.0, 6.0, 8.0],
            dimensions=[(1280, 720), (640, 480), (1920, 1080)],
            frame_rates=["30/1", "24/1", "60/1"],
        ),
    )

    graph = filter_graph(calls[0])
    assert (
        "[1:v]scale=1280:720:force_original_aspect_ratio=decrease:flags=lanczos,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
        "fps=30.000000[vn1]"
    ) in graph
    assert (
        "[2:v]scale=1280:720:force_original_aspect_ratio=decrease:flags=lanczos,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
        "fps=30.000000[vn2]"
    ) in graph
    assert (
        "[0:v][vn1]xfade=transition=fade:duration=1.000000:offset=3.000000[v1]"
    ) in graph
    assert (
        "[v1][vn2]xfade=transition=fade:duration=1.000000:offset=8.000000[v2]"
    ) in graph
    assert "[0:v]scale" not in graph


def test_concatenate_crossfade_skips_normalization_for_matching_geometry(
    tmp_path: Path,
) -> None:
    first = source_file(tmp_path, "first.mp4")
    second = source_file(tmp_path, "second.mp4")
    calls: list[list[str]] = []

    concatenate_videos(
        [first, second],
        tmp_path / "matched.mp4",
        crossfade_seconds=1.0,
        ffmpeg_runner=writer(calls=calls),
        ffprobe_runner=probe_runner([4.0, 6.0]),
    )

    graph = filter_graph(calls[0])
    assert "scale=" not in graph
    assert "fps=" not in graph
    assert "[0:v][1:v]xfade" in graph


def test_concatenate_crossfade_requires_a_shorter_fade_than_every_clip(
    tmp_path: Path,
) -> None:
    first = source_file(tmp_path, "first.mp4")
    second = source_file(tmp_path, "second.mp4")

    with pytest.raises(ValidationError, match="shorter than every clip"):
        concatenate_videos(
            [first, second],
            tmp_path / "bad.mp4",
            crossfade_seconds=1.0,
            ffmpeg_runner=writer(),
            ffprobe_runner=probe_runner([1.0, 6.0]),
        )


def test_concatenate_needs_at_least_two_sources(tmp_path: Path) -> None:
    source = source_file(tmp_path)

    with pytest.raises(ValidationError, match="at least two"):
        concatenate_videos([source], tmp_path / "out.mp4")


def test_concatenate_drops_audio_when_a_clip_has_none(tmp_path: Path) -> None:
    first = source_file(tmp_path, "first.mp4")
    second = source_file(tmp_path, "second.mp4")
    calls: list[list[str]] = []

    concatenate_videos(
        [first, second],
        tmp_path / "silent.mp4",
        crossfade_seconds=0.5,
        ffmpeg_runner=writer(calls=calls),
        ffprobe_runner=probe_runner([4.0, 6.0], audio=[True, False]),
    )

    assert "acrossfade" not in filter_graph(calls[0])
    assert "-an" in calls[0]


def test_extract_frame_writes_the_frame_at_the_timestamp(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    calls: list[list[str]] = []
    destination = tmp_path / "frame.png"

    result = extract_frame(
        source,
        destination,
        timestamp_seconds=2.5,
        ffmpeg_runner=writer(calls=calls, payload=PNG),
    )

    assert result == destination
    assert destination.read_bytes() == PNG
    assert option(calls[0], "-ss") == "2.500000"
    assert option(calls[0], "-frames:v") == "1"


def test_extract_frame_rejects_a_negative_timestamp_or_missing_suffix(
    tmp_path: Path,
) -> None:
    source = source_file(tmp_path)

    with pytest.raises(ValidationError, match="not be negative"):
        extract_frame(source, tmp_path / "frame.png", timestamp_seconds=-1)
    with pytest.raises(ValidationError, match="add a suffix"):
        extract_frame(source, tmp_path / "frame")


def test_extract_audio_infers_the_codec_from_the_suffix(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    mp3_calls: list[list[str]] = []
    wav_calls: list[list[str]] = []

    extract_audio(
        source,
        tmp_path / "audio.mp3",
        ffmpeg_runner=writer(calls=mp3_calls, payload=AUDIO),
    )
    extract_audio(
        source,
        tmp_path / "audio.wav",
        ffmpeg_runner=writer(calls=wav_calls, payload=AUDIO),
    )

    assert "-vn" in mp3_calls[0]
    assert option(mp3_calls[0], "-map") == "0:a:0"
    assert option(mp3_calls[0], "-c:a") == "libmp3lame"
    assert option(mp3_calls[0], "-b:a") == "192k"
    assert option(wav_calls[0], "-c:a") == "pcm_s16le"
    assert "-b:a" not in wav_calls[0]


def test_extract_audio_rejects_an_unknown_suffix(tmp_path: Path) -> None:
    source = source_file(tmp_path)

    with pytest.raises(ValidationError, match="unsupported audio format"):
        extract_audio(source, tmp_path / "audio.xyz")


def test_seamless_loop_reverses_and_concatenates_the_clip(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    calls: list[list[str]] = []

    make_seamless_loop(
        source,
        tmp_path / "loop.mp4",
        ffmpeg_runner=writer(calls=calls),
    )

    graph = filter_graph(calls[0])
    assert "reverse[reversed]" in graph
    assert "concat=n=2:v=1:a=0[v]" in graph
    assert "areverse" not in graph
    assert "-an" in calls[0]


def test_seamless_loop_can_keep_reversed_audio(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    calls: list[list[str]] = []

    make_seamless_loop(
        source,
        tmp_path / "loop.mp4",
        keep_audio=True,
        ffmpeg_runner=writer(calls=calls),
    )

    graph = filter_graph(calls[0])
    assert "asplit=2[af][ar]" in graph
    assert "areverse[arr]" in graph
    assert "[a]" in calls[0]


def test_convert_video_infers_gif_and_uses_a_palette(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    calls: list[list[str]] = []

    convert_video(
        source,
        tmp_path / "out.gif",
        ffmpeg_runner=writer(calls=calls),
    )

    graph = filter_graph(calls[0])
    assert "palettegen" in graph
    assert "paletteuse" in graph
    assert "fps=15.000000" in graph
    assert option(calls[0], "-loop") == "0"
    assert option(calls[0], "-f") == "gif"
    assert "-an" in calls[0]


def test_convert_video_infers_webp_and_uses_the_animated_encoder(
    tmp_path: Path,
) -> None:
    source = source_file(tmp_path)
    calls: list[list[str]] = []

    convert_video(
        source,
        tmp_path / "out.webp",
        ffmpeg_runner=writer(calls=calls),
    )

    assert option(calls[0], "-c:v") == "libwebp_anim"
    assert option(calls[0], "-loop") == "0"
    assert option(calls[0], "-quality") == "80"
    assert option(calls[0], "-f") == "webp"
    assert "-an" in calls[0]


def test_convert_video_infers_mp4_and_encodes_h264_with_audio(
    tmp_path: Path,
) -> None:
    source = source_file(tmp_path)
    calls: list[list[str]] = []

    convert_video(
        source,
        tmp_path / "out.mp4",
        ffmpeg_runner=writer(calls=calls),
    )

    assert option(calls[0], "-c:v") == "libx264"
    assert option(calls[0], "-crf") == "23"
    assert option(calls[0], "-pix_fmt") == "yuv420p"
    assert option(calls[0], "-c:a") == "aac"
    assert option(calls[0], "-f") == "mp4"


def test_convert_video_applies_explicit_size_and_frame_rate(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    calls: list[list[str]] = []

    convert_video(
        source,
        tmp_path / "out.mp4",
        width=640,
        height=360,
        fps=24,
        ffmpeg_runner=writer(calls=calls),
    )

    graph = filter_graph(calls[0])
    assert "scale=trunc(640*1.000000/2)*2:trunc(360*1.000000/2)*2" in graph
    assert "fps=24.000000" in graph


def test_convert_video_retries_until_the_output_fits_the_cap(
    tmp_path: Path,
) -> None:
    source = source_file(tmp_path)
    calls: list[list[str]] = []
    destination = tmp_path / "capped.mp4"

    result = convert_video(
        source,
        destination,
        max_bytes=500,
        ffmpeg_runner=writer(calls=calls, sizes=[1000, 400]),
    )

    assert result == destination
    assert destination.stat().st_size == 400
    assert len(calls) == 2
    assert option(calls[0], "-crf") == "23"
    assert option(calls[1], "-crf") == "27"


@pytest.mark.parametrize(("suffix", "container"), [(".gif", "gif"), (".webp", "webp")])
def test_animated_outputs_retry_quality_and_size_until_the_cap_is_met(
    tmp_path: Path,
    suffix: str,
    container: str,
) -> None:
    source = source_file(tmp_path)
    calls: list[list[str]] = []
    destination = tmp_path / f"capped{suffix}"

    convert_video(
        source,
        destination,
        max_bytes=500,
        ffmpeg_runner=writer(calls=calls, sizes=[1000, 900, 400]),
    )

    assert destination.stat().st_size == 400
    assert len(calls) == 3
    if container == "gif":
        assert "max_colors=205" in filter_graph(calls[0])
        assert "max_colors=154" in filter_graph(calls[1])
        assert "max_colors=102" in filter_graph(calls[2])
    else:
        assert option(calls[0], "-quality") == "80"
        assert option(calls[1], "-quality") == "60"
        assert option(calls[2], "-quality") == "40"
    assert option(calls[2], "-f") == container


def test_convert_video_raises_when_no_attempt_fits_the_cap(
    tmp_path: Path,
) -> None:
    source = source_file(tmp_path)
    destination = tmp_path / "too-big.mp4"

    with pytest.raises(VideoSizeError, match="below 100 bytes") as raised:
        convert_video(
            source,
            destination,
            max_bytes=100,
            ffmpeg_runner=writer(sizes=[1000] * 20),
        )

    assert raised.value.details["max_bytes"] == 100
    assert raised.value.details["last_size_bytes"] == 1000
    assert not destination.exists()


def test_convert_video_rejects_an_invalid_cap(tmp_path: Path) -> None:
    source = source_file(tmp_path)

    with pytest.raises(ValidationError, match="positive integer"):
        convert_video(source, tmp_path / "out.mp4", max_bytes=0)


def test_convert_video_refuses_to_overwrite_the_source(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    before = source.read_bytes()

    with pytest.raises(ValidationError, match="must not overwrite"):
        convert_video(source, source)

    assert source.read_bytes() == before


def test_convert_video_accepts_inline_bytes_and_an_explicit_format(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "output"

    convert_video(
        MP4,
        destination,
        output_format="mp4",
        ffmpeg_runner=writer(),
    )

    assert destination.exists()


def test_video_format_parse_and_path_inference() -> None:
    assert VideoOutputFormat.parse(".GIF") is VideoOutputFormat.GIF
    assert VideoOutputFormat.parse("m4v") is VideoOutputFormat.MP4
    assert VideoOutputFormat.from_path(Path("clip.webp")) is VideoOutputFormat.WEBP
    assert VideoOutputFormat.MP4.supports_audio
    assert not VideoOutputFormat.GIF.supports_audio
    with pytest.raises(ValidationError, match="unsupported video format"):
        VideoOutputFormat.parse("avi")
    with pytest.raises(ValidationError, match="cannot infer"):
        VideoOutputFormat.from_path(Path("clip"))


def test_missing_source_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="does not exist"):
        trim_video(tmp_path / "missing.mp4", tmp_path / "out.mp4")


def test_a_failing_ffmpeg_runner_raises_a_validation_error(tmp_path: Path) -> None:
    source = source_file(tmp_path)

    def failing(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(argv), 1, stdout="", stderr="boom")

    with pytest.raises(ValidationError, match="could not trim"):
        trim_video(source, tmp_path / "out.mp4", ffmpeg_runner=failing)


def test_the_public_aliases_point_at_the_same_functions() -> None:
    from aivia.media import video_utils

    assert video_utils.trim is video_utils.trim_video
    assert video_utils.speed_video is video_utils.change_speed
    assert video_utils.concat_videos is video_utils.concatenate_videos
    assert video_utils.seamless_loop is video_utils.make_seamless_loop
    assert video_utils.export_video is video_utils.convert_video
