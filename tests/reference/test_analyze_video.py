"""A video reference is sampled by scene and analyzed for structure and style."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import subprocess
from collections.abc import Coroutine, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from aivia.core.errors import ValidationError
from aivia.providers.protocols.llm import (
    ImagePart,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    TextPart,
)
from aivia.reference import (
    DEFAULT_REFERENCE_VIDEO_PROMPT,
    CaptionStyle,
    ReferenceAnalysis,
    ReferenceKind,
    ReferenceSpec,
    ShotType,
    VideoReferenceAnalyzer,
    analyze_video_reference,
    analyze_video_reference_offline,
    analyze_video_reference_sync,
    build_video_reference_request,
    parse_video_reference_analysis,
    sample_video_scenes,
)

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
MP4 = struct.pack(">I", 24) + b"ftypisom" + b"\x00" * 12
SCENE_STDERR = (
    "[Parsed_showinfo_1 @ 0x1] n:0 pts:0 pts_time:0.000 pos:0\n"
    "[Parsed_showinfo_1 @ 0x1] n:1 pts:36 pts_time:1.500 pos:100\n"
    "[Parsed_showinfo_1 @ 0x1] n:2 pts:72 pts_time:3.000 pos:200\n"
)


def _run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine)


def _probe_json(
    *,
    width: int = 640,
    height: int = 360,
    duration: str = "5.0",
    video: bool = True,
) -> str:
    streams: list[dict[str, Any]] = []
    if video:
        streams.append(
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": width,
                "height": height,
                "duration": duration,
                "avg_frame_rate": "24/1",
            }
        )
    else:
        streams.append(
            {
                "index": 0,
                "codec_type": "audio",
                "codec_name": "aac",
                "duration": duration,
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
                "duration": duration,
                "size": "12345",
            },
        }
    )


def _ffprobe_runner(payload: str) -> Any:
    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout=payload,
            stderr="",
        )

    return runner


def _ffmpeg_runner(
    *,
    scene_stderr: str = SCENE_STDERR,
    extraction: bytes = PNG_1X1,
    calls: list[list[str]] | None = None,
) -> Any:
    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[Any]:
        if calls is not None:
            calls.append(list(argv))
        if any("showinfo" in part for part in argv):
            return subprocess.CompletedProcess(
                list(argv),
                0,
                stdout="",
                stderr=scene_stderr,
            )
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout=extraction,
            stderr=b"",
        )

    return runner


class FakeProvider:
    """An async LLM provider that returns one canned reference payload."""

    def __init__(self, payload: Mapping[str, Any] | str, *, name: str = "acme") -> None:
        self.payload = payload
        self.name = name
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        content = (
            self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        )
        return LLMResponse(
            content=content,
            provider=self.name,
            model="vision-model",
            usage=LLMUsage(input_tokens=10, output_tokens=5),
        )


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "structure": {
            "scenes": [
                {
                    "start_seconds": 0.0,
                    "end_seconds": 1.5,
                    "shot_type": "wide",
                    "description": "establishing shot",
                    "text": "WELCOME",
                },
                {
                    "start_seconds": 1.5,
                    "end_seconds": 3.0,
                    "shot_type": "close-up",
                    "description": "face fills the frame",
                    "text": "ONE IDEA",
                    "has_caption": True,
                },
                {
                    "start_seconds": 3.0,
                    "end_seconds": 5.0,
                    "shot_type": "medium",
                    "description": "payoff",
                    "has_caption": True,
                },
            ],
            "hook": "Open on a question",
            "hook_seconds": 1.5,
            "narrative_arc": "question to answer",
            "beats": ["hook", "explain", "payoff"],
        },
        "pacing": {
            "style": "fast",
            "energy": 0.7,
            "words_per_second": 2.5,
        },
        "palette": {
            "colors": ["#123456", "#abcdef"],
            "primary": "#123456",
        },
        "typography": {
            "font_family": "Inter",
            "font_weight": "bold",
            "caption_style": "burned-in",
        },
        "subject": "a golden retriever",
        "reproducing_prompt": (
            "A golden retriever runs through a wide editorial frame with "
            "burned-in captions."
        ),
        "detected_text": "WELCOME",
        "confidence": 0.8,
    }
    payload.update(overrides)
    return payload


def test_offline_analysis_reads_cuts_pacing_aspect_and_duration() -> None:
    calls: list[list[str]] = []

    analysis = analyze_video_reference_offline(
        MP4,
        frame_cap=4,
        ffprobe_runner=_ffprobe_runner(_probe_json()),
        ffmpeg_runner=_ffmpeg_runner(calls=calls),
    )

    assert isinstance(analysis, ReferenceAnalysis)
    assert analysis.kind is ReferenceKind.VIDEO
    assert analysis.aspect.width == 640
    assert analysis.aspect.height == 360
    assert analysis.aspect.ratio == "16:9"
    assert analysis.aspect.is_landscape is True
    assert analysis.duration_seconds == 5.0
    assert analysis.structure.scene_count == 3
    assert [scene.start for scene in analysis.structure.scenes] == [0.0, 1.5, 3.0]
    assert [scene.end for scene in analysis.structure.scenes] == [1.5, 3.0, 5.0]
    assert analysis.structure.scene_durations == [1.5, 1.5, 2.0]
    assert analysis.structure.cuts == 2
    assert analysis.structure.hook_seconds == 1.5
    assert analysis.structure.shot_types == []
    assert analysis.pacing.cuts_per_minute == pytest.approx(24.0)
    assert analysis.pacing.average_shot_seconds == pytest.approx(5.0 / 3.0)
    assert analysis.pacing.min_shot_seconds == 1.5
    assert analysis.pacing.max_shot_seconds == 2.0
    assert analysis.detected_text == ""
    assert analysis.reproducing_prompt == ""
    extraction = [argv for argv in calls if "image2pipe" in argv]
    assert len(extraction) == 4


def test_frame_cap_bounds_sampling_without_losing_detected_cuts() -> None:
    scene_stderr = "".join(
        f"[Parsed_showinfo_1 @ 0x1] n:{index} pts:0 "
        f"pts_time:{index * 0.5:.3f} pos:0\n"
        for index in range(1, 10)
    )
    calls: list[list[str]] = []

    analysis = analyze_video_reference_offline(
        MP4,
        frame_cap=3,
        ffprobe_runner=_ffprobe_runner(_probe_json()),
        ffmpeg_runner=_ffmpeg_runner(scene_stderr=scene_stderr, calls=calls),
    )

    assert analysis.structure.cuts == 9
    assert analysis.structure.scene_count == 10
    extraction = [argv for argv in calls if "image2pipe" in argv]
    assert len(extraction) == 3


def test_sample_video_scenes_never_exceeds_the_cap() -> None:
    sample = sample_video_scenes(
        MP4,
        count=20,
        frame_cap=2,
        ffprobe_runner=_ffprobe_runner(_probe_json()),
        ffmpeg_runner=_ffmpeg_runner(),
    )

    assert sample.frame_count == 2
    assert sample.width == 640
    assert sample.height == 360
    assert sample.duration_seconds == 5.0
    assert sample.cuts == (1.5, 3.0)
    assert [frame.scene_change for frame in sample.frames] == [False, True]


def test_vision_analysis_maps_shot_types_captions_text_and_hook() -> None:
    provider = FakeProvider(_payload())

    analysis = _run(
        analyze_video_reference(
            MP4,
            provider=provider,
            frame_cap=4,
            ffprobe_runner=_ffprobe_runner(_probe_json()),
            ffmpeg_runner=_ffmpeg_runner(),
        )
    )

    assert analysis.structure.scenes[0].shot_type is ShotType.WIDE
    assert analysis.structure.scenes[1].shot_type is ShotType.CLOSE_UP
    assert analysis.structure.scenes[2].shot_type is ShotType.MEDIUM
    assert analysis.structure.shot_types == [
        ShotType.WIDE,
        ShotType.CLOSE_UP,
        ShotType.MEDIUM,
    ]
    assert analysis.structure.hook == "Open on a question"
    assert analysis.structure.hook_seconds == 1.5
    assert analysis.structure.narrative_arc == "question to answer"
    assert analysis.caption_style is CaptionStyle.BURNED_IN
    assert analysis.typography.caption_style is CaptionStyle.BURNED_IN
    assert analysis.detected_text == "WELCOME\nONE IDEA"
    assert analysis.subject == "a golden retriever"
    assert "golden retriever" not in analysis.reproducing_prompt.lower()
    assert analysis.provider == "acme"
    assert analysis.model == "vision-model"
    assert analysis.aspect.dimensions == (640, 360)
    assert analysis.duration_seconds == 5.0
    assert analysis.pacing.cuts_per_minute == pytest.approx(24.0)
    assert analysis.pacing.style == "fast"


def test_request_carries_capped_frames_with_timestamps_and_schema() -> None:
    provider = FakeProvider(_payload())

    _run(
        analyze_video_reference(
            MP4,
            provider=provider,
            model="vision-model",
            frame_cap=4,
            ffprobe_runner=_ffprobe_runner(_probe_json()),
            ffmpeg_runner=_ffmpeg_runner(),
        )
    )

    request = provider.requests[0]
    assert request.model == "vision-model"
    assert request.role == "reference_analysis"
    assert request.response_schema is not None
    assert {"structure", "pacing", "typography", "aspect"} <= set(
        request.response_schema["properties"]
    )
    assert request.metadata["reference_kind"] == "video"
    assert request.metadata["frame_count"] == 4
    assert request.metadata["duration_seconds"] == 5.0
    content = request.messages[0].content
    assert isinstance(content, list)
    images = [part for part in content if isinstance(part, ImagePart)]
    assert len(images) == 4
    texts = [part.text for part in content if isinstance(part, TextPart)]
    assert "Frame 1 at 0.00s" in texts
    assert any("detected cut" in text for text in texts)
    assert any("Measured:" in text for text in texts)
    assert base64.b64decode(images[0].data or "") == PNG_1X1


def test_build_request_can_be_inspected_without_a_provider() -> None:
    request = build_video_reference_request(
        [PNG_1X1, PNG_1X1],
        model="vision-model",
    )

    assert isinstance(request, LLMRequest)
    assert request.role == "reference_analysis"
    assert request.model == "vision-model"
    assert request.response_schema is not None
    assert len(request.messages[0].images) == 2


def test_parse_merges_measured_timing_and_strips_the_subject() -> None:
    analysis = parse_video_reference_analysis(
        _payload(),
        source=MP4,
        reference_id="clip-1",
        provider="acme",
        model="vision-model",
        ffprobe_runner=_ffprobe_runner(_probe_json()),
        ffmpeg_runner=_ffmpeg_runner(),
    )

    assert analysis.reference_id == "clip-1"
    assert analysis.kind is ReferenceKind.VIDEO
    assert analysis.aspect.dimensions == (640, 360)
    assert analysis.duration_seconds == 5.0
    assert analysis.structure.scene_count == 3
    assert analysis.structure.scenes[0].shot_type is ShotType.WIDE
    assert analysis.structure.scenes[1].text == "ONE IDEA"
    assert analysis.structure.shot_types == [
        ShotType.WIDE,
        ShotType.CLOSE_UP,
        ShotType.MEDIUM,
    ]
    assert "golden retriever" not in analysis.reproducing_prompt.lower()
    assert analysis.reproducing_prompt


def test_analysis_rejects_a_reference_that_is_not_a_video(
    tmp_path: Path,
) -> None:
    path = tmp_path / "look.png"
    path.write_bytes(PNG_1X1)

    with pytest.raises(ValidationError, match="not a video"):
        analyze_video_reference_offline(
            ReferenceSpec(kind=ReferenceKind.IMAGE, path=str(path))
        )


def test_analysis_rejects_a_remote_reference_before_any_download() -> None:
    with pytest.raises(ValidationError, match="downloaded"):
        analyze_video_reference_offline(
            ReferenceSpec(kind=ReferenceKind.VIDEO, url="https://example.test/a.mp4")
        )


def test_analysis_rejects_a_clip_with_no_video_stream() -> None:
    with pytest.raises(ValidationError, match="no video stream"):
        analyze_video_reference_offline(
            MP4,
            ffprobe_runner=_ffprobe_runner(_probe_json(video=False)),
            ffmpeg_runner=_ffmpeg_runner(),
        )


def test_sync_wrapper_and_reusable_analyzer_share_the_result() -> None:
    provider = FakeProvider(_payload())

    direct = analyze_video_reference_sync(
        MP4,
        provider=provider,
        frame_cap=3,
        ffprobe_runner=_ffprobe_runner(_probe_json()),
        ffmpeg_runner=_ffmpeg_runner(),
    )
    reusable = _run(
        VideoReferenceAnalyzer(
            provider,
            frame_cap=3,
            ffprobe_runner=_ffprobe_runner(_probe_json()),
            ffmpeg_runner=_ffmpeg_runner(),
        ).analyze(MP4)
    )

    assert direct.reproducing_prompt == reusable.reproducing_prompt
    assert reusable.provider == "acme"


def test_the_default_prompt_names_every_required_section() -> None:
    prompt = DEFAULT_REFERENCE_VIDEO_PROMPT.lower()

    for word in (
        "shot_type",
        "caption",
        "on-screen text",
        "hook",
        "aspect",
        "duration",
        "pacing",
    ):
        assert word in prompt
