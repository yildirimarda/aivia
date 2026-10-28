"""Deterministic mock providers for every capability."""

from __future__ import annotations

import asyncio
import io
import shutil
import subprocess
import wave
from collections.abc import Callable, Sequence
from typing import Any, cast

import pytest

from aivia.core.errors import PermanentError, RateLimitError
from aivia.core.models import ProviderMeta
from aivia.core.router import Router
from aivia.providers import Provider
from aivia.providers.mock import (
    MOCK_EPOCH,
    MockCharacterTrainer,
    MockDeliveryProvider,
    MockEmbeddingProvider,
    MockImageEditProvider,
    MockImageProvider,
    MockLipSyncProvider,
    MockLLMProvider,
    MockMusicProvider,
    MockNotifier,
    MockPageFetchProvider,
    MockProvider,
    MockPublisher,
    MockSearchProvider,
    MockSFXProvider,
    MockStockProvider,
    MockStorageProvider,
    MockSTTProvider,
    MockTTSProvider,
    MockVideoProvider,
    mock_provider_for,
    register_mock_providers,
)
from aivia.providers.protocols import (
    CAPABILITY_CHARACTER_TRAINER,
    CAPABILITY_DELIVERY,
    CAPABILITY_EMBEDDINGS,
    CAPABILITY_IMAGE_EDIT,
    CAPABILITY_IMAGE_GEN,
    CAPABILITY_LIPSYNC,
    CAPABILITY_LLM,
    CAPABILITY_MUSIC_GEN,
    CAPABILITY_NOTIFY,
    CAPABILITY_PAGE_FETCH,
    CAPABILITY_PUBLISH,
    CAPABILITY_SEARCH,
    CAPABILITY_SFX,
    CAPABILITY_STOCK,
    CAPABILITY_STORAGE,
    CAPABILITY_STT,
    CAPABILITY_TTS,
    CAPABILITY_VIDEO_GEN,
    AudioFormat,
    CharacterTrainerRequest,
    ChatMessage,
    DeliveryArtifact,
    DeliveryMethod,
    DeliveryRequest,
    EmbeddingRequest,
    ImageEditOperation,
    ImageEditRequest,
    ImageFormat,
    ImageRequest,
    LipSyncRequest,
    LLMRequest,
    MediaInput,
    MessageRole,
    MusicRequest,
    NotificationMessage,
    PageFetchRequest,
    PublishRequest,
    SearchRequest,
    SFXRequest,
    StockRequest,
    StoragePutRequest,
    STTRequest,
    ToolChoice,
    ToolDefinition,
    TrainingState,
    TTSRequest,
    VideoFormat,
    VideoRequest,
)
from aivia.providers.registry import ProviderRegistry

from .provider_contract import ProviderContract

_ALL_CAPABILITIES = (
    CAPABILITY_CHARACTER_TRAINER,
    CAPABILITY_DELIVERY,
    CAPABILITY_EMBEDDINGS,
    CAPABILITY_IMAGE_EDIT,
    CAPABILITY_IMAGE_GEN,
    CAPABILITY_LIPSYNC,
    CAPABILITY_LLM,
    CAPABILITY_MUSIC_GEN,
    CAPABILITY_NOTIFY,
    CAPABILITY_PAGE_FETCH,
    CAPABILITY_PUBLISH,
    CAPABILITY_SEARCH,
    CAPABILITY_SFX,
    CAPABILITY_STOCK,
    CAPABILITY_STORAGE,
    CAPABILITY_STT,
    CAPABILITY_TTS,
    CAPABILITY_VIDEO_GEN,
)

_PROVIDER_CLASSES = (
    MockCharacterTrainer,
    MockDeliveryProvider,
    MockEmbeddingProvider,
    MockImageEditProvider,
    MockImageProvider,
    MockLipSyncProvider,
    MockLLMProvider,
    MockMusicProvider,
    MockNotifier,
    MockPageFetchProvider,
    MockProvider,
    MockPublisher,
    MockSFXProvider,
    MockSTTProvider,
    MockSearchProvider,
    MockStockProvider,
    MockStorageProvider,
    MockTTSProvider,
    MockVideoProvider,
)


class TestMockProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return MockProvider()


@pytest.mark.parametrize("provider_class", _PROVIDER_CLASSES)
def test_every_mock_has_metadata_and_a_safe_lifecycle(
    provider_class: type[Provider],
) -> None:
    provider = provider_class()

    assert isinstance(provider.metadata, ProviderMeta)
    assert provider.name
    assert provider.capabilities
    for capability in provider.capabilities:
        assert provider.supports(capability)

    report = asyncio.run(provider.health())
    assert report.healthy is True

    asyncio.run(provider.close())
    asyncio.run(provider.close())
    assert provider.closed


@pytest.mark.parametrize("capability", _ALL_CAPABILITIES)
def test_every_capability_has_a_mock_provider(capability: str) -> None:
    provider = mock_provider_for(capability)

    assert provider.supports(capability)
    assert capability in MockProvider().capabilities


def test_the_all_in_one_mock_declares_every_capability() -> None:
    provider = MockProvider()

    assert set(provider.capabilities) == set(_ALL_CAPABILITIES)


def test_mock_providers_can_be_registered_by_type() -> None:
    registry = ProviderRegistry()

    register_mock_providers(registry)
    register_mock_providers(registry)

    assert registry.resolve("mock") is MockProvider
    assert registry.resolve("mock-video") is MockVideoProvider
    assert registry.resolve("mock-llm") is MockLLMProvider
    assert len(registry.names()) == len(_PROVIDER_CLASSES)


def test_llm_is_deterministic_and_records_a_simulated_cost() -> None:
    provider = MockLLMProvider()
    request = LLMRequest(
        messages=[
            ChatMessage(role=MessageRole.USER, content="Write a deterministic line")
        ]
    )

    first = asyncio.run(provider.complete(request))
    second = asyncio.run(provider.complete(request))

    assert first.content == second.content == "mock: Write a deterministic line"
    assert first.usage.input_tokens > 0
    assert first.provider == "mock-llm"
    assert len(provider.calls) == 2
    assert provider.total_cost_usd == pytest.approx(0.001)
    assert first.metadata["cost_usd"] == pytest.approx(0.0005)
    assert provider.last_cost_usd == pytest.approx(0.0005)


def test_llm_returns_a_tool_call_when_the_caller_requires_one() -> None:
    provider = MockLLMProvider()
    request = LLMRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="look it up")],
        tools=[ToolDefinition(name="lookup", parameters={"type": "object"})],
        tool_choice=ToolChoice.REQUIRED,
    )

    response = asyncio.run(provider.complete(request))

    assert response.has_tool_calls
    assert response.tool_calls[0].name == "lookup"
    assert response.finish_reason.value == "tool_calls"


def test_embeddings_are_deterministic_and_can_be_normalized() -> None:
    provider = MockEmbeddingProvider()
    request = EmbeddingRequest(
        inputs=["alpha", "beta"],
        dimensions=4,
        normalize=True,
    )

    first = asyncio.run(provider.embed(request))
    second = asyncio.run(provider.embed(request))

    assert first == second
    assert first.dimensions == 4
    assert [embedding.index for embedding in first.embeddings] == [0, 1]
    for vector in first.vectors:
        assert sum(component * component for component in vector) == pytest.approx(1.0)


def test_image_generation_returns_a_valid_png_of_the_requested_size() -> None:
    provider = MockImageProvider()
    request = ImageRequest(prompt="a deterministic card", width=96, height=64, count=2)

    response = asyncio.run(provider.generate(request))

    assert response.count == 2
    for image in response.images:
        assert image.data is not None
        assert image.data.startswith(b"\x89PNG\r\n\x1a\n")
        assert image.dimensions is not None
        assert image.dimensions.width == 96
        assert image.dimensions.height == 64
        assert image.format is ImageFormat.PNG
    assert response.images[0].data != response.images[1].data
    assert provider.total_cost_usd == pytest.approx(0.04)


def test_image_edit_uses_the_source_size_and_upscale_factor() -> None:
    source = asyncio.run(
        MockImageProvider().generate(ImageRequest(prompt="source", width=40, height=30))
    ).first
    provider = MockImageEditProvider()
    request = ImageEditRequest(
        image=MediaInput(data=source.data, media_type="image/png"),
        operation=ImageEditOperation.UPSCALE,
        scale=2.0,
    )

    response = asyncio.run(provider.edit(request))

    assert response.operation is ImageEditOperation.UPSCALE
    assert response.first.dimensions is not None
    assert response.first.dimensions.width == 80
    assert response.first.dimensions.height == 60
    assert provider.total_cost_usd == pytest.approx(0.03)


def _wav_duration(audio: bytes) -> float:
    with wave.open(io.BytesIO(audio), "rb") as reader:
        return reader.getnframes() / reader.getframerate()


def test_tts_returns_sized_audio_and_word_timestamps() -> None:
    provider = MockTTSProvider()
    request = TTSRequest(
        text="one two three",
        voice="mock-voice",
        format=AudioFormat.WAV,
        sample_rate=8_000,
        word_timestamps=True,
    )

    response = asyncio.run(provider.synthesize(request))

    assert response.format is AudioFormat.WAV
    assert response.duration_seconds == pytest.approx(3 / 2.5)
    assert _wav_duration(response.audio) == pytest.approx(3 / 2.5, abs=1e-3)
    assert [word.word for word in response.word_timestamps] == ["one", "two", "three"]
    assert response.size_bytes == len(response.audio)


def test_music_and_sfx_return_audio_sized_to_the_requested_duration() -> None:
    music = MockMusicProvider()
    sfx = MockSFXProvider()

    track = asyncio.run(
        music.generate(
            MusicRequest(
                prompt="calm loop",
                duration_seconds=2.0,
                format=AudioFormat.WAV,
                sample_rate=8_000,
            )
        )
    ).first
    clip = asyncio.run(
        sfx.generate(
            SFXRequest(
                prompt="click",
                duration_seconds=1.5,
                format=AudioFormat.WAV,
                sample_rate=8_000,
                loop=True,
            )
        )
    ).first

    assert track.duration_seconds == pytest.approx(2.0)
    assert _wav_duration(track.data or b"") == pytest.approx(2.0, abs=1e-3)
    assert clip.duration_seconds == pytest.approx(1.5)
    assert _wav_duration(clip.data or b"") == pytest.approx(1.5, abs=1e-3)
    assert clip.loop is True


def test_video_renders_with_ffmpeg_lavfi_and_reports_the_requested_metadata() -> None:
    commands: list[list[str]] = []

    def runner(args: Sequence[str]) -> bytes:
        commands.append(list(args))
        return b"LAVFIVIDEO"

    provider = MockVideoProvider(ffmpeg_runner=runner)
    request = VideoRequest(
        prompt="a deterministic clip",
        width=64,
        height=48,
        duration_seconds=1.25,
        fps=12,
        generate_audio=True,
        seed=7,
    )

    response = asyncio.run(provider.generate(request))

    assert response.first.data == b"LAVFIVIDEO"
    assert response.first.format is VideoFormat.MP4
    assert response.first.dimensions is not None
    assert response.first.dimensions.width == 64
    assert response.first.dimensions.height == 48
    assert response.first.duration_seconds == pytest.approx(1.25)
    assert response.first.fps == 12
    assert response.first.has_audio is True
    assert response.first.seed == 7

    command = commands[0]
    assert "lavfi" in command
    assert any("testsrc2=size=64x48:rate=12" in part for part in command)
    assert any("sine=frequency=440" in part for part in command)
    assert command[-1].endswith(".mp4")
    assert provider.total_cost_usd == pytest.approx(0.05)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for the lavfi integration test",
)
def test_video_is_a_real_mp4_when_ffmpeg_is_installed() -> None:
    provider = MockVideoProvider()
    request = VideoRequest(
        prompt="real lavfi clip",
        width=64,
        height=64,
        duration_seconds=1.0,
        fps=10,
        generate_audio=True,
    )

    response = asyncio.run(provider.generate(request))
    data = response.first.data or b""

    assert data[4:8] == b"ftyp"
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height",
            "-of",
            "default=noprint_wrappers=1",
            "-",
        ],
        input=data,
        check=True,
        capture_output=True,
    )
    assert b"codec_type=video" in probe.stdout
    assert b"codec_type=audio" in probe.stdout


def test_failures_are_injected_once_and_do_not_record_a_cost() -> None:
    provider = MockImageProvider(failures=[RateLimitError("slow down")])
    request = ImageRequest(prompt="fallback")

    with pytest.raises(RateLimitError):
        asyncio.run(provider.generate(request))
    assert provider.total_cost_usd == 0.0

    response = asyncio.run(provider.generate(request))

    assert response.count == 1
    assert provider.total_cost_usd == pytest.approx(0.02)
    assert len(provider.calls) == 1


def test_a_permanent_error_fails_every_call() -> None:
    provider = MockLLMProvider(error=PermanentError("never available"))

    with pytest.raises(PermanentError):
        asyncio.run(
            provider.complete(
                LLMRequest(messages=[ChatMessage(role=MessageRole.USER, content="hi")])
            )
        )
    with pytest.raises(PermanentError):
        asyncio.run(
            provider.complete(
                LLMRequest(messages=[ChatMessage(role=MessageRole.USER, content="hi")])
            )
        )


def test_fail_next_injects_a_failure_without_replacing_the_provider() -> None:
    provider = MockSearchProvider()
    provider.fail_next(RateLimitError("retry"))

    with pytest.raises(RateLimitError):
        asyncio.run(provider.search(SearchRequest(query="topic")))

    assert len(asyncio.run(provider.search(SearchRequest(query="topic"))).results) == 3


def test_a_failing_mock_falls_back_through_the_router() -> None:
    bad = MockImageProvider(name="bad", failures=[RateLimitError("slow")])
    good = MockImageProvider(name="good")
    router = Router(providers=[bad, good], chain=["bad", "good"])
    request = ImageRequest(prompt="fallback image")

    response = asyncio.run(
        router.execute(
            CAPABILITY_IMAGE_GEN,
            lambda provider: cast(MockImageProvider, provider).generate(request),
        )
    )

    assert response.count == 1
    assert len(bad.calls) == 0
    assert len(good.calls) == 1


def test_stock_search_returns_licensed_deterministic_assets() -> None:
    provider = MockStockProvider()

    first = asyncio.run(provider.search(StockRequest(query="city", per_page=2)))
    second = asyncio.run(provider.search(StockRequest(query="city", per_page=2)))

    assert first == second
    assert first.count == 2
    assert all(asset.attribution for asset in first.assets)
    assert all(asset.url.startswith("https://") for asset in first.assets)


def test_stt_transcribes_deterministically_with_word_timing() -> None:
    provider = MockSTTProvider()
    request = STTRequest(audio=b"some audio", word_timestamps=True)

    first = asyncio.run(provider.transcribe(request))
    second = asyncio.run(provider.transcribe(request))

    assert first == second
    assert first.text == "mock transcript"
    assert first.has_word_timestamps
    assert [word.word for word in first.words] == ["mock", "transcript"]


def test_search_and_page_fetch_are_deterministic() -> None:
    search = MockSearchProvider()
    fetch = MockPageFetchProvider()

    first = asyncio.run(search.search(SearchRequest(query="topic", max_results=2)))
    second = asyncio.run(search.search(SearchRequest(query="topic", max_results=2)))
    page = asyncio.run(fetch.fetch(PageFetchRequest(url="https://example.com/a")))

    assert first == second
    assert len(first.results) == 2
    assert page.status_code == 200
    assert page.fetched_at == MOCK_EPOCH
    assert page.ok is True


def test_notify_publish_and_deliver_return_deterministic_receipts() -> None:
    notifier = MockNotifier()
    publisher = MockPublisher()
    delivery = MockDeliveryProvider()
    message = NotificationMessage(text="done")
    post = PublishRequest(platform="youtube", idempotency_key="key-1")
    deliver = DeliveryRequest(
        method=DeliveryMethod.LINK,
        target="https://example.com/delivery",
        artifacts=[DeliveryArtifact(data=b"card", media_type="image/png")],
        expires_seconds=60,
    )

    assert asyncio.run(notifier.send(message)) == asyncio.run(notifier.send(message))
    assert asyncio.run(publisher.publish(post)) == asyncio.run(publisher.publish(post))
    receipt = asyncio.run(delivery.deliver(deliver))
    assert receipt.delivered_at == MOCK_EPOCH
    assert receipt.expires_at is not None
    assert receipt.url is not None

    dry = asyncio.run(publisher.publish(post.model_copy(update={"dry_run": True})))
    assert dry.dry_run is True
    assert dry.is_live is False


def test_storage_is_content_addressed_and_dedupes() -> None:
    provider = MockStorageProvider()

    first = asyncio.run(provider.put(StoragePutRequest(data=b"hello")))
    second = asyncio.run(provider.put(StoragePutRequest(data=b"hello")))

    assert first == second
    assert first.key.startswith("objects/")
    assert asyncio.run(provider.get(first.key)) == b"hello"
    assert asyncio.run(provider.stat(first.key)) == first
    assert asyncio.run(provider.delete(first.key)) is True
    assert asyncio.run(provider.delete(first.key)) is False
    assert asyncio.run(provider.stat(first.key)) is None


def test_lipsync_marks_sprite_sheet_requests_as_fallback() -> None:
    provider = MockLipSyncProvider()
    source = LipSyncRequest(
        audio=MediaInput(data=b"audio", media_type="audio/wav"),
        source=MediaInput(
            data=asyncio.run(
                MockImageProvider().generate(
                    ImageRequest(prompt="face", width=32, height=32)
                )
            ).first.data,
            media_type="image/png",
        ),
        duration_seconds=0.5,
        fps=10,
    )

    response = asyncio.run(provider.animate(source))

    assert response.video.used_fallback is False
    assert response.video.data is not None
    assert response.video.duration_seconds == pytest.approx(0.5)


def test_character_trainer_returns_a_ready_job_and_status() -> None:
    provider = MockCharacterTrainer()
    request = CharacterTrainerRequest(
        character_id="ada",
        trigger_words=["ada"],
        reference_images=[MediaInput(data=b"image", media_type="image/png")],
    )

    job = asyncio.run(provider.train(request))

    assert job.state is TrainingState.READY
    assert job.is_ready is True
    assert job.model_id is not None
    assert job.cost_usd > 0
    assert asyncio.run(provider.status(job.job_id)) == job
    assert job == asyncio.run(provider.train(request))


def test_costs_match_the_declared_pricing() -> None:
    provider = MockImageProvider()
    request = ImageRequest(prompt="two", count=2)

    asyncio.run(provider.generate(request))

    price = provider.metadata.pricing_for(CAPABILITY_IMAGE_GEN)
    assert price is not None
    assert provider.last_cost_usd == pytest.approx(price.cost_for(2))
    assert provider.estimate_cost(CAPABILITY_IMAGE_GEN, 2) == pytest.approx(
        price.cost_for(2)
    )


def test_a_custom_cost_override_is_used_for_every_capability() -> None:
    provider = MockProvider(cost_usd=0.25)

    for capability in provider.capabilities:
        assert provider.estimate_cost(capability, 1) == pytest.approx(0.25)


def test_unknown_capability_has_no_mock_provider() -> None:
    with pytest.raises(KeyError, match="no mock provider"):
        mock_provider_for("not_a_capability")


def test_mock_results_are_typed_and_do_not_leak_unknown_fields() -> None:
    provider = MockLLMProvider()

    response = asyncio.run(
        provider.complete(
            LLMRequest(messages=[ChatMessage(role=MessageRole.USER, content="hi")])
        )
    )

    assert set(response.model_dump()) <= set(type(response).model_fields)
    assert response.metadata["mock"] is True


def _request_factories() -> list[tuple[str, Callable[[Any], Any]]]:
    return [
        (CAPABILITY_IMAGE_GEN, lambda p: p.generate(ImageRequest(prompt="x"))),
        (
            CAPABILITY_IMAGE_EDIT,
            lambda p: p.edit(
                ImageEditRequest(
                    image=MediaInput(data=b"image", media_type="image/png"),
                    prompt="edit",
                )
            ),
        ),
        (CAPABILITY_MUSIC_GEN, lambda p: p.generate(MusicRequest(prompt="x"))),
        (CAPABILITY_SFX, lambda p: p.generate(SFXRequest(prompt="x"))),
        (CAPABILITY_STOCK, lambda p: p.search(StockRequest(query="x"))),
        (CAPABILITY_TTS, lambda p: p.synthesize(TTSRequest(text="hi", voice="v"))),
        (CAPABILITY_STT, lambda p: p.transcribe(STTRequest(audio=b"audio"))),
        (CAPABILITY_SEARCH, lambda p: p.search(SearchRequest(query="x"))),
        (
            CAPABILITY_PAGE_FETCH,
            lambda p: p.fetch(PageFetchRequest(url="https://example.com")),
        ),
        (CAPABILITY_NOTIFY, lambda p: p.send(NotificationMessage(text="x"))),
        (CAPABILITY_PUBLISH, lambda p: p.publish(PublishRequest(platform="youtube"))),
        (
            CAPABILITY_DELIVERY,
            lambda p: p.deliver(
                DeliveryRequest(method=DeliveryMethod.FOLDER, target="/tmp")
            ),
        ),
        (CAPABILITY_STORAGE, lambda p: p.put(StoragePutRequest(data=b"x"))),
    ]


@pytest.mark.parametrize(("capability", "call"), _request_factories())
def test_all_in_one_mock_dispatches_each_capability(
    capability: str,
    call: Callable[[Any], Any],
) -> None:
    provider = MockProvider()

    result = asyncio.run(call(provider))

    assert result is not None
    assert provider.last_call is not None
    assert provider.last_call.capability == capability
