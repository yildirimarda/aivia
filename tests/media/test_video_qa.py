"""Generated clips are sampled by scene, scored, retried and animated as fallback."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import struct
import subprocess
from collections.abc import Sequence
from typing import Any

import pytest

from aivia.core.config.quality import QualityConfig
from aivia.core.models.asset import Asset, AssetKind
from aivia.core.models.cost import CostEvent
from aivia.media.i2v_flow import I2VOutcome
from aivia.media.video_cache import VideoCacheOutcome
from aivia.media.video_qa import (
    DEFAULT_VIDEO_QA_PROMPT,
    VideoFrame,
    VideoQA,
    VideoQAAttempt,
    VideoQAResult,
    VideoQualityAssurance,
    VideoQualityError,
    VideoQualityScore,
    run_video_qa,
    sample_video_frames,
    score_video_frames,
    select_frame_times,
)
from aivia.media.video_service import VideoOutcome
from aivia.providers.protocols.llm import LLMRequest, LLMResponse, LLMUsage
from aivia.providers.protocols.media import VideoFormat
from aivia.providers.protocols.video import VideoRequest, VideoResponse, VideoResult

PROMPT = "A lighthouse beam sweeping across a stormy sea"
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
MP4 = struct.pack(">I", 24) + b"ftypisom" + b"\x00" * 12
SCENE_STDERR = (
    "[Parsed_showinfo_1 @ 0x1] n:0 pts:0 pts_time:0.000 pos:0\n"
    "[Parsed_showinfo_1 @ 0x1] n:1 pts:45 pts_time:1.500 pos:100\n"
    "[Parsed_showinfo_1 @ 0x1] n:2 pts:90 pts_time:3.000 pos:200\n"
)


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def _asset(provider: str = "acme") -> Asset:
    return Asset(
        kind=AssetKind.VIDEO,
        provider=provider,
        model="mock-video",
        prompt=PROMPT,
        content_hash=hashlib.sha256(MP4).hexdigest(),
    )


def _outcome(provider: str = "acme", *, amount: float = 0.5) -> VideoOutcome:
    request = VideoRequest(prompt=PROMPT)
    return VideoOutcome(
        request=request,
        response=VideoResponse(
            videos=[
                VideoResult(
                    data=MP4,
                    format=VideoFormat.MP4,
                    duration_seconds=5.0,
                )
            ],
            provider=provider,
            model="mock-video",
        ),
        assets=(_asset(provider),),
        events=(
            CostEvent(
                provider=provider,
                model="mock-video",
                operation="video_gen",
                units=1.0,
                unit_kind="clip",
                amount_usd=amount,
            ),
        ),
    )


def _i2v(provider: str = "still-video", *, amount: float = 0.3) -> I2VOutcome:
    request = VideoRequest(prompt=PROMPT)
    event = CostEvent(
        provider=provider,
        model="mock-video",
        operation="video_gen",
        units=1.0,
        unit_kind="clip",
        amount_usd=amount,
    )
    stored = VideoCacheOutcome(
        request=request,
        result=VideoResult(
            data=MP4,
            format=VideoFormat.MP4,
            duration_seconds=5.0,
        ),
        asset=_asset(provider),
        event=event,
    )
    return I2VOutcome(request=request, video=stored, video_event=event)


class ScriptedGenerator:
    """A fake video service with a chain and provider aware generation."""

    def __init__(self, chain: list[str], outcomes: list[VideoOutcome]) -> None:
        self.providers = tuple(chain)
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        request: VideoRequest,
        *,
        exclude: set[str] | None = None,
        provider: str | None = None,
        no_cache: bool = True,
        token: Any = None,
        run_id: str = "",
        metadata: Any = None,
    ) -> VideoOutcome:
        self.calls.append(
            {
                "provider": provider,
                "exclude": set(exclude or ()),
                "no_cache": no_cache,
                "token": token,
                "run_id": run_id,
                "metadata": metadata,
            }
        )
        if self.outcomes:
            return self.outcomes.pop(0)
        return _outcome(provider or "acme")


class ScriptedSampler:
    """Return one sampled frame per call and remember the arguments."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, source: Any, **kwargs: Any) -> list[VideoFrame]:
        self.calls.append({"source": source, **kwargs})
        return [VideoFrame(index=0, timestamp_seconds=0.0, data=PNG_1X1)]


class ScriptedScorer:
    """Return one score per frame from a fixed sequence."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = list(scores)
        self.sources: list[Any] = []

    async def __call__(self, source: Any) -> float:
        self.sources.append(source)
        return self.scores.pop(0) if self.scores else 1.0


class FakeVisionProvider:
    """An async LLM provider returning a canned quality payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            content=json.dumps(self.payload),
            provider="acme",
            model="vision-model",
            usage=LLMUsage(input_tokens=10, output_tokens=5),
        )


class FakeI2V:
    """A still to video fallback returning a ready animated outcome."""

    def __init__(self, outcome: I2VOutcome) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    async def animate(self, still: Any = None, **kwargs: Any) -> I2VOutcome:
        self.calls.append({"still": still, **kwargs})
        return self.outcome


def test_scene_times_are_preferred_and_gaps_are_filled() -> None:
    assert select_frame_times([1.0, 3.0], 5.0, 3) == (0.0, 1.0, 3.0)
    assert select_frame_times([2.0], 6.0, 3) == (0.0, 2.0, 4.0)
    assert select_frame_times([], 5.0, 3) == (0.0, 2.5, 4.999)
    assert select_frame_times([1.0, 2.0, 3.0, 4.0], 5.0, 2) == (0.0, 4.0)


def test_scene_aware_sampling_extracts_the_selected_timestamps() -> None:
    calls: list[list[str]] = []

    def runner(argv: Sequence[str]) -> subprocess.CompletedProcess[Any]:
        calls.append(list(argv))
        if any("showinfo" in part for part in argv):
            return subprocess.CompletedProcess(
                list(argv), 0, stdout="", stderr=SCENE_STDERR
            )
        return subprocess.CompletedProcess(list(argv), 0, stdout=PNG_1X1, stderr=b"")

    frames = sample_video_frames(
        MP4,
        count=3,
        duration_seconds=5.0,
        ffmpeg_runner=runner,
    )

    assert [frame.timestamp_seconds for frame in frames] == [0.0, 1.5, 3.0]
    assert [frame.scene_change for frame in frames] == [False, True, True]
    extraction = [argv for argv in calls if "image2pipe" in argv]
    assert len(extraction) == 3
    assert [argv[argv.index("-ss") + 1] for argv in extraction] == [
        "0.000",
        "1.500",
        "3.000",
    ]


def test_every_sampled_frame_is_scored_and_the_weakest_wins() -> None:
    provider = FakeVisionProvider(
        {
            "overall": 0.82,
            "prompt_adherence": 0.9,
            "technical_quality": 0.8,
            "aesthetic_quality": 0.7,
        }
    )
    frames = [
        VideoFrame(index=0, timestamp_seconds=0.0, data=PNG_1X1),
        VideoFrame(index=1, timestamp_seconds=2.0, data=PNG_1X1),
    ]

    score = _run(
        score_video_frames(
            frames,
            provider=provider,
            model="vision-model",
            original_prompt=PROMPT,
            threshold=0.8,
        )
    )

    assert isinstance(score, VideoQualityScore)
    assert score.frame_count == 2
    assert score.overall == pytest.approx(0.82)
    assert score.passed is True
    assert len(provider.requests) == 2
    request = provider.requests[0]
    assert request.role == "vision_qa"
    assert request.metadata["frame_index"] == 0
    content = request.messages[0].content
    assert isinstance(content, list)
    text = next(part for part in content if hasattr(part, "text"))
    assert DEFAULT_VIDEO_QA_PROMPT.split(".")[0] in text.text


def test_a_low_scoring_clip_is_retried_then_accepted() -> None:
    generator = ScriptedGenerator(
        ["acme"],
        [_outcome("acme"), _outcome("acme")],
    )
    sampler = ScriptedSampler()
    scorer = ScriptedScorer([0.2, 0.9])

    result = _run(
        VideoQualityAssurance(
            generator,
            scorer=scorer,
            sampler=sampler,
            config=QualityConfig(min_score=0.7, max_retries=2, sample_frames=1),
        ).run(VideoRequest(prompt=PROMPT))
    )

    assert isinstance(result, VideoQAResult)
    assert result.passed is True
    assert result.provider == "acme"
    assert len(result.attempts) == 2
    assert [attempt.passed for attempt in result.attempts] == [False, True]
    assert [attempt.retry for attempt in result.attempts] == [False, True]
    assert len(generator.calls) == 2
    assert all(call["no_cache"] is True for call in generator.calls)
    assert len(sampler.calls) == 2
    assert result.total_amount_usd == pytest.approx(1.0)


def test_a_provider_that_never_passes_falls_back_to_the_next_provider() -> None:
    generator = ScriptedGenerator(
        ["first", "second"],
        [_outcome("first"), _outcome("first"), _outcome("second")],
    )
    scorer = ScriptedScorer([0.1, 0.2, 0.9])

    result = _run(
        VideoQualityAssurance(
            generator,
            scorer=scorer,
            sampler=ScriptedSampler(),
            config=QualityConfig(min_score=0.7, max_retries=1, sample_frames=1),
        ).run(VideoRequest(prompt=PROMPT))
    )

    assert result.passed is True
    assert result.provider == "second"
    assert result.fallback == "provider"
    assert [attempt.provider for attempt in result.attempts] == [
        "first",
        "first",
        "second",
    ]
    assert [call["provider"] for call in generator.calls] == [
        "first",
        "first",
        "second",
    ]
    assert generator.calls[0]["exclude"] == set()
    assert generator.calls[1]["exclude"] == set()
    assert generator.calls[2]["exclude"] == {"first"}


def test_every_clip_failing_falls_back_to_animating_a_still() -> None:
    generator = ScriptedGenerator(["first"], [_outcome("first")])
    fallback = FakeI2V(_i2v())

    result = _run(
        VideoQualityAssurance(
            generator,
            scorer=ScriptedScorer([0.1, 0.95]),
            sampler=ScriptedSampler(),
            fallback=fallback,
            still=b"still-bytes",
            config=QualityConfig(min_score=0.7, max_retries=0, sample_frames=1),
        ).run(VideoRequest(prompt=PROMPT), run_id="run-1")
    )

    assert result.passed is True
    assert result.used_i2v is True
    assert result.used_fallback is True
    assert result.fallback == "image_to_video"
    assert result.provider == "still-video"
    assert result.outcome is None
    assert result.i2v is fallback.outcome
    assert [attempt.fallback for attempt in result.attempts] == [
        "",
        "image_to_video",
    ]
    assert fallback.calls[0]["still"] == b"still-bytes"
    assert fallback.calls[0]["run_id"] == "run-1"


def test_disabling_quality_bypasses_sampling_and_scoring() -> None:
    generator = ScriptedGenerator(["acme"], [_outcome("acme")])
    sampler = ScriptedSampler()
    scorer = ScriptedScorer([0.0])

    result = _run(
        VideoQualityAssurance(
            generator,
            scorer=scorer,
            sampler=sampler,
            config=QualityConfig(enabled=False),
        ).run(VideoRequest(prompt=PROMPT))
    )

    assert result.passed is True
    assert isinstance(result.attempts[0], VideoQAAttempt)
    assert result.attempts[0].score is None
    assert sampler.calls == []
    assert scorer.sources == []
    assert len(generator.calls) == 1


def test_an_exhausted_result_can_be_required_and_raises() -> None:
    result = _run(
        VideoQualityAssurance(
            ScriptedGenerator(["acme"], [_outcome("acme")]),
            scorer=ScriptedScorer([0.1]),
            sampler=ScriptedSampler(),
            config=QualityConfig(min_score=0.9, max_retries=0, sample_frames=1),
        ).run(VideoRequest(prompt=PROMPT))
    )

    assert result.passed is False
    assert result.failed is True
    assert result.score is not None
    assert result.score.overall == pytest.approx(0.1)
    assert result.fallback == "exhausted"
    with pytest.raises(VideoQualityError, match="quality threshold"):
        result.require()


def test_the_public_helper_builds_and_runs_the_assurance() -> None:
    result = _run(
        run_video_qa(
            VideoRequest(prompt=PROMPT),
            generate=ScriptedGenerator(["acme"], [_outcome("acme")]),
            scorer=ScriptedScorer([0.95]),
            sampler=ScriptedSampler(),
            config=QualityConfig(min_score=0.7, max_retries=0, sample_frames=1),
        )
    )

    assert result.passed is True
    assert VideoQA is VideoQualityAssurance
    assert result.score is not None
    assert result.score.passed is True
