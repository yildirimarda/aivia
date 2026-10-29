"""Media requests are adapted to provider limits with explicit warnings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from aivia.core.errors import InvalidRequestError
from aivia.core.models.provider import ProviderMeta
from aivia.providers.media import (
    AdaptationAction,
    AdaptationResult,
    AdaptationWarning,
    ImageRequest,
    MediaConstraints,
    MediaKind,
    MediaRequest,
    MusicRequest,
    SpeechRequest,
    UnsupportedOptionError,
    VideoRequest,
    adapt_image_request,
    adapt_media_request,
    adapt_music_request,
    adapt_request,
    adapt_speech_request,
    adapt_video_request,
    clamp_duration,
    snap_size,
)
from aivia.providers.protocols.media import ImageFormat, MediaDimensions, MediaInput

_URL = "https://example.com/reference.png"


def test_snap_size_prefers_the_closest_shape_then_area() -> None:
    sizes = [
        MediaDimensions(width=512, height=512),
        (1024, 1024),
        "1920x1080",
    ]

    assert snap_size(1000, 1000, sizes) == MediaDimensions(width=1024, height=1024)
    assert snap_size(1900, 1000, sizes) == MediaDimensions(width=1920, height=1080)
    assert snap_size(None, None, sizes, aspect_ratio="16:9") == MediaDimensions(
        width=1920,
        height=1080,
    )
    assert snap_size(None, None, sizes, aspect_ratio="1:1") == MediaDimensions(
        width=512,
        height=512,
    )

    with pytest.raises(ValueError, match="at least one"):
        snap_size(1000, 1000, [])
    with pytest.raises(ValueError, match="needs width and height"):
        snap_size(None, None, sizes)


def test_clamp_duration_bounds_both_ends() -> None:
    assert clamp_duration(2.0, minimum=5.0, maximum=10.0) == 5.0
    assert clamp_duration(12.0, minimum=5.0, maximum=10.0) == 10.0
    assert clamp_duration(7.0, minimum=5.0, maximum=10.0) == 7.0
    assert clamp_duration(7.0) == 7.0


def test_image_request_snaps_size_and_keeps_supported_options() -> None:
    constraints = MediaConstraints(
        sizes=(
            MediaDimensions(width=1024, height=1024),
            MediaDimensions(width=512, height=512),
        )
    )
    request = ImageRequest(
        prompt="  a red fox  ",
        width=1000,
        height=1000,
        aspect_ratio="1:1",
        count=2,
        seed=3,
    )

    result = adapt_image_request(request, constraints)

    assert request.width == 1000
    assert result.request.width == 1024
    assert result.request.height == 1024
    assert result.request.aspect_ratio == "1:1"
    assert result.request.prompt == "a red fox"
    assert result.request.count == 2
    assert result.request.seed == 3
    assert result.changed is True
    assert len(result.warnings) == 1
    warning = result.warnings[0]
    assert warning.action is AdaptationAction.SNAPPED_SIZE
    assert warning.option == "size"
    assert warning.requested == "1000x1000"
    assert warning.applied == "1024x1024"
    assert "1000x1000" in result.messages[0]


def test_image_request_snaps_an_aspect_only_request_to_a_size() -> None:
    constraints = MediaConstraints(
        sizes=(
            MediaDimensions(width=1024, height=1024),
            MediaDimensions(width=1920, height=1080),
        )
    )
    request = ImageRequest(prompt="a wave", aspect_ratio="16:9")

    result = adapt_image_request(request, constraints)

    assert result.request.width == 1920
    assert result.request.height == 1080
    assert result.request.aspect_ratio == "16:9"
    assert result.warnings[0].requested == "16:9"
    assert result.warnings[0].applied == "1920x1080"

    square = adapt_image_request(
        ImageRequest(prompt="a wave", aspect_ratio="3:2"),
        MediaConstraints(sizes=(MediaDimensions(width=1024, height=1024),)),
    )
    assert square.request.width == 1024
    assert square.request.height == 1024
    assert square.request.aspect_ratio == "1:1"


def test_aspect_ratio_is_snapped_when_no_sizes_are_declared() -> None:
    constraints = MediaConstraints(aspect_ratios=("1:1", "16:9"))
    request = ImageRequest(prompt="a wave", aspect_ratio="3:2")

    result = adapt_image_request(request, constraints)

    assert result.request.aspect_ratio == "16:9"
    assert result.request.width is None
    assert result.request.height is None
    assert result.warnings[0].action is AdaptationAction.SNAPPED_ASPECT_RATIO

    exact = adapt_image_request(
        ImageRequest(prompt="a wave", aspect_ratio="16:9"),
        constraints,
    )
    assert exact.changed is False
    assert exact.request.aspect_ratio == "16:9"


def test_video_request_clamps_duration_and_count_and_records_both() -> None:
    constraints = MediaConstraints(
        min_duration_seconds=5.0,
        max_duration_seconds=10.0,
        max_count=2,
    )
    request = VideoRequest(
        prompt="a slow pan",
        duration_seconds=15.0,
        count=3,
        fps=24,
        seed=9,
    )

    result = adapt_video_request(request, constraints)

    assert result.request.duration_seconds == 10.0
    assert result.request.count == 2
    assert result.request.fps == 24
    assert result.request.seed == 9
    assert [warning.action for warning in result.warnings] == [
        AdaptationAction.CLAMPED_DURATION,
        AdaptationAction.CLAMPED_COUNT,
    ]
    assert result.warnings[0].requested == 15.0
    assert result.warnings[0].applied == 10.0


def test_video_request_raises_duration_to_the_minimum() -> None:
    constraints = MediaConstraints(min_duration_seconds=5.0, max_duration_seconds=10.0)
    result = adapt_video_request(
        VideoRequest(prompt="a slow pan", duration_seconds=1.0),
        constraints,
    )

    assert result.request.duration_seconds == 5.0
    assert result.warnings[0].action is AdaptationAction.CLAMPED_DURATION


def test_music_request_clamps_duration_without_dropping_lyrics() -> None:
    constraints = MediaConstraints(min_duration_seconds=30.0, max_duration_seconds=60.0)
    request = MusicRequest(
        prompt="a warm synth song",
        lyrics="a verse",
        duration_seconds=90.0,
        count=2,
    )

    result = adapt_music_request(request, constraints)

    assert result.request.duration_seconds == 60.0
    assert result.request.lyrics == "a verse"
    assert result.request.count == 2
    assert result.warnings[0].action is AdaptationAction.CLAMPED_DURATION


def test_supported_audio_options_survive_video_adaptation() -> None:
    constraints = MediaConstraints(
        max_duration_seconds=10.0,
        supports=frozenset({"generate_audio", "audio_prompt", "duration"}),
    )
    request = VideoRequest(
        prompt="a storm",
        duration_seconds=5.0,
        generate_audio=True,
        audio_prompt="thunder",
    )

    result = adapt_video_request(request, constraints)

    assert result.request.generate_audio is True
    assert result.request.audio_prompt == "thunder"
    assert result.changed is False


@pytest.mark.parametrize(
    ("media_request", "constraints", "option"),
    [
        (
            ImageRequest(prompt="x", negative_prompt="text"),
            MediaConstraints(supports=frozenset({"size"})),
            "negative_prompt",
        ),
        (
            ImageRequest(prompt="x", reference_images=[MediaInput(url=_URL)]),
            MediaConstraints(supports=frozenset({"size"})),
            "reference_images",
        ),
        (
            ImageRequest(prompt="x", style="cinematic"),
            MediaConstraints(supports=frozenset({"size"})),
            "style",
        ),
        (
            VideoRequest(prompt="x", generate_audio=True),
            MediaConstraints(supports=frozenset({"duration"})),
            "generate_audio",
        ),
        (
            VideoRequest(
                prompt="x",
                first_frame=MediaInput(url=_URL),
            ),
            MediaConstraints(supports=frozenset({"duration"})),
            "first_frame",
        ),
        (
            MusicRequest(prompt="x", lyrics="words"),
            MediaConstraints(supports=frozenset({"duration"})),
            "lyrics",
        ),
        (
            MusicRequest(prompt="x", tempo_bpm=120.0),
            MediaConstraints(supports=frozenset({"duration"})),
            "tempo_bpm",
        ),
        (
            SpeechRequest(
                text="hello",
                voice="narrator",
                pronunciation={"aivia": "ah-vee-ah"},
            ),
            MediaConstraints(supports=frozenset({"text"})),
            "pronunciation",
        ),
        (
            SpeechRequest(text="hello", voice="narrator", speed=1.5),
            MediaConstraints(supports=frozenset()),
            "speed",
        ),
    ],
)
def test_unsupported_options_are_rejected_instead_of_dropped(
    media_request: ImageRequest | VideoRequest | MusicRequest | SpeechRequest,
    constraints: MediaConstraints,
    option: str,
) -> None:
    with pytest.raises(UnsupportedOptionError) as caught:
        adapt_request(media_request, constraints)

    assert isinstance(caught.value, InvalidRequestError)
    assert caught.value.option == option
    assert caught.value.details["option"] == option
    assert "does not support" in str(caught.value)


def test_value_options_formats_models_and_fps_are_validated() -> None:
    quality = MediaConstraints(
        supports=frozenset({"quality"}),
        supported_options={"quality": ("high", "medium")},
    )
    with pytest.raises(UnsupportedOptionError, match="supported values: high, medium"):
        adapt_image_request(ImageRequest(prompt="x", quality="low"), quality)
    kept = adapt_image_request(ImageRequest(prompt="x", quality="high"), quality)
    assert kept.request.quality == "high"

    with pytest.raises(UnsupportedOptionError) as caught:
        adapt_image_request(
            ImageRequest(prompt="x", format=ImageFormat.JPEG),
            MediaConstraints(formats=("webp",)),
        )
    assert caught.value.option == "format"

    with pytest.raises(UnsupportedOptionError) as caught:
        adapt_video_request(
            VideoRequest(prompt="x", fps=25),
            MediaConstraints(fps=(24, 30)),
        )
    assert caught.value.option == "fps"

    with pytest.raises(UnsupportedOptionError) as caught:
        adapt_image_request(
            ImageRequest(prompt="x", model="model-b"),
            MediaConstraints(models=("model-a",)),
        )
    assert caught.value.option == "model"

    kept_model = adapt_image_request(
        ImageRequest(prompt="x", model="model-a"),
        MediaConstraints(models=("model-a",)),
    )
    assert kept_model.request.model == "model-a"


def test_prompt_is_truncated_at_a_word_boundary_and_recorded() -> None:
    constraints = MediaConstraints(max_prompt_chars=12)
    request = ImageRequest(prompt="one two three four five")

    result = adapt_image_request(request, constraints)

    assert result.request.prompt == "one two"
    warning = result.warnings[0]
    assert warning.action is AdaptationAction.TRUNCATED_PROMPT
    assert warning.option == "prompt"
    assert warning.requested == len("one two three four five")
    assert warning.applied == len("one two")


def test_speech_text_uses_the_same_prompt_limit() -> None:
    result = adapt_speech_request(
        SpeechRequest(text="one two three four five", voice="narrator"),
        MediaConstraints(max_prompt_chars=12),
    )

    assert result.request.text == "one two"
    assert result.warnings[0].option == "text"


def test_media_request_dispatch_preserves_kind_and_original() -> None:
    media = MediaRequest.for_video(
        VideoRequest(prompt="a wave", duration_seconds=15.0)
    )
    constraints = MediaConstraints(max_duration_seconds=10.0)

    result = adapt_media_request(media, constraints)

    assert isinstance(result, AdaptationResult)
    assert result.request.kind is MediaKind.VIDEO
    assert result.request.video is not None
    assert result.request.video.duration_seconds == 10.0
    assert media.video is not None
    assert media.video.duration_seconds == 15.0

    dispatched = adapt_request(media, constraints)
    assert dispatched.request.video is not None
    assert dispatched.request.video.duration_seconds == 10.0


def test_from_metadata_parses_declared_option_tokens() -> None:
    metadata = ProviderMeta(
        name="example-image",
        capabilities=["image_gen"],
        supported_options={
            "image_gen": [
                "1024x1024",
                "16:9",
                "negative_prompt",
                "seed",
                "quality:high|medium",
                "count:4",
                "prompt:100",
                "format:png,webp",
                "duration:5-10",
                "fps:24,30",
                "models:model-a,model-b",
            ]
        },
    )

    constraints = MediaConstraints.from_metadata(metadata, "image_gen")

    assert constraints.sizes == (MediaDimensions(width=1024, height=1024),)
    assert constraints.aspect_ratios == ("16:9",)
    assert constraints.min_duration_seconds == 5.0
    assert constraints.max_duration_seconds == 10.0
    assert constraints.fps == (24, 30)
    assert constraints.formats == ("png", "webp")
    assert constraints.models == ("model-a", "model-b")
    assert constraints.max_count == 4
    assert constraints.max_prompt_chars == 100
    assert constraints.supported_options["quality"] == ("high", "medium")
    assert constraints.supports == frozenset(
        {
            "size",
            "aspect_ratio",
            "negative_prompt",
            "seed",
            "quality",
            "count",
            "prompt",
            "format",
            "duration",
            "fps",
            "model",
        }
    )

    missing = MediaConstraints.from_metadata(metadata, "video_gen")
    assert missing.supports is None
    assert missing.sizes == ()


def test_constraints_reject_invalid_ranges_and_values() -> None:
    with pytest.raises(PydanticValidationError, match="must not exceed"):
        MediaConstraints(min_duration_seconds=10.0, max_duration_seconds=5.0)
    with pytest.raises(PydanticValidationError):
        MediaConstraints.model_validate({"sizes": ("wide",)})
    with pytest.raises(PydanticValidationError):
        MediaConstraints(aspect_ratios=("wide",))
    with pytest.raises(PydanticValidationError):
        MediaConstraints(max_count=0)
    with pytest.raises(PydanticValidationError):
        MediaConstraints(max_prompt_chars=0)


def test_empty_constraints_leave_every_request_untouched() -> None:
    constraints = MediaConstraints()
    requests: list[
        ImageRequest | VideoRequest | MusicRequest | SpeechRequest
    ] = [
        ImageRequest(prompt="x", width=1000, height=1000, seed=1),
        VideoRequest(prompt="x", duration_seconds=3.0, generate_audio=True),
        MusicRequest(prompt="x", duration_seconds=3.0),
        SpeechRequest(text="x", voice="narrator", speed=2.0),
    ]

    for request in requests:
        result = adapt_request(request, constraints)
        assert result.request == request
        assert result.changed is False


def test_warning_serializes_for_sidecars() -> None:
    warning = AdaptationWarning(
        action=AdaptationAction.SNAPPED_SIZE,
        option="size",
        message="size changed",
        requested="1000x1000",
        applied="1024x1024",
    )

    assert warning.to_dict() == {
        "action": "snapped_size",
        "option": "size",
        "message": "size changed",
        "requested": "1000x1000",
        "applied": "1024x1024",
    }
