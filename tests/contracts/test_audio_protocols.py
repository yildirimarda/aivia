"""Embeddings, text to speech and speech to text protocols and fakes."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, cast

import pytest
from pydantic import ValidationError

from aivia.core.errors import RateLimitError
from aivia.core.models import ProviderMeta
from aivia.providers import Provider
from aivia.providers.protocols import (
    CAPABILITY_EMBEDDINGS,
    CAPABILITY_STT,
    CAPABILITY_TTS,
    AudioFormat,
    CapabilityProvider,
    Embedding,
    EmbeddingInputType,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    STTProvider,
    STTRequest,
    STTResponse,
    TranscriptSegment,
    TTSProvider,
    TTSRequest,
    TTSResponse,
    WordTimestamp,
)

from .provider_contract import ProviderContract

_AUDIO_URL = "https://example.com/audio/sample.wav"


class FakeEmbeddingProvider(Provider):
    """A deterministic embeddings fake that records requests."""

    default_metadata = ProviderMeta(
        name="fake-embeddings",
        capabilities=[CAPABILITY_EMBEDDINGS],
    )

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self._error = error
        self.requests: list[EmbeddingRequest] = []

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        dimensions = request.dimensions or 3
        embeddings: list[Embedding] = []
        for index, text in enumerate(request.inputs):
            seed = sum(ord(character) for character in text)
            vector = [
                round(((seed + index + component) % 17) / 17.0, 6)
                for component in range(dimensions)
            ]
            embeddings.append(Embedding(index=index, vector=vector))
        return EmbeddingResponse(
            embeddings=embeddings,
            model=request.model or "fake-embedding-1",
            provider=self.name,
            usage=EmbeddingUsage(
                input_tokens=sum(len(text.split()) for text in request.inputs)
            ),
        )


class FakeTTSProvider(Provider):
    """A deterministic text to speech fake with synthetic word timing."""

    default_metadata = ProviderMeta(
        name="fake-tts",
        capabilities=[CAPABILITY_TTS],
    )

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self._error = error
        self.requests: list[TTSRequest] = []

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        words = request.text.split()
        timestamps: list[WordTimestamp] = []
        cursor = 0.0
        for word in words:
            timestamps.append(
                WordTimestamp(
                    word=word,
                    start_seconds=cursor,
                    end_seconds=cursor + 0.4,
                )
            )
            cursor += 0.4
        return TTSResponse(
            audio=b"FAKEAUDIO:" + request.text.encode("utf-8"),
            format=request.format,
            sample_rate=request.sample_rate or 24_000,
            duration_seconds=cursor,
            voice=request.voice,
            model=request.model or "fake-voice",
            provider=self.name,
            word_timestamps=timestamps if request.word_timestamps else [],
        )


class FakeSTTProvider(Provider):
    """A deterministic speech to text fake with word level timestamps."""

    default_metadata = ProviderMeta(
        name="fake-stt",
        capabilities=[CAPABILITY_STT],
    )

    def __init__(
        self,
        responses: list[STTResponse] | None = None,
        error: Exception | None = None,
    ) -> None:
        super().__init__()
        self._responses = deque(responses or ())
        self._error = error
        self.requests: list[STTRequest] = []

    async def transcribe(self, request: STTRequest) -> STTResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        if self._responses:
            return self._responses.popleft()
        if request.audio is not None:
            text = request.audio.decode("utf-8", errors="replace")
        else:
            text = f"transcript for {request.audio_url}"
        text = text.removeprefix("FAKEAUDIO:").strip()
        words = text.split()
        timestamps: list[WordTimestamp] = []
        cursor = 0.0
        for word in words:
            timestamps.append(
                WordTimestamp(
                    word=word,
                    start_seconds=cursor,
                    end_seconds=cursor + 0.4,
                )
            )
            cursor += 0.4
        segments = (
            [
                TranscriptSegment(
                    text=text,
                    start_seconds=0.0,
                    end_seconds=cursor,
                    words=timestamps,
                )
            ]
            if text
            else []
        )
        return STTResponse(
            text=text,
            language=request.language or "en",
            duration_seconds=cursor,
            segments=segments,
            words=timestamps,
            model=request.model or "fake-whisper",
            provider=self.name,
        )


class TestFakeEmbeddingProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return FakeEmbeddingProvider()


class TestFakeTTSProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return FakeTTSProvider()


class TestFakeSTTProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return FakeSTTProvider()


def test_fakes_satisfy_the_capability_protocols() -> None:
    embeddings = FakeEmbeddingProvider()
    tts = FakeTTSProvider()
    stt = FakeSTTProvider()

    assert isinstance(embeddings, EmbeddingProvider)
    assert isinstance(tts, TTSProvider)
    assert isinstance(stt, STTProvider)
    assert isinstance(embeddings, CapabilityProvider)
    assert isinstance(tts, CapabilityProvider)
    assert isinstance(stt, CapabilityProvider)
    assert not isinstance(tts, EmbeddingProvider)
    assert not isinstance(stt, TTSProvider)
    assert not isinstance(Provider(), EmbeddingProvider)


def test_capability_names_are_stable() -> None:
    assert CAPABILITY_EMBEDDINGS == "embeddings"
    assert CAPABILITY_TTS == "tts"
    assert CAPABILITY_STT == "stt"


def test_embedding_request_normalizes_and_validates_inputs() -> None:
    request = EmbeddingRequest(
        inputs=["  first text  ", "second text"],
        model="  some-model  ",
        dimensions=8,
        input_type=EmbeddingInputType.QUERY,
        normalize=True,
    )

    assert request.inputs == ["first text", "second text"]
    assert request.model == "some-model"
    assert request.dimensions == 8
    assert request.input_type is EmbeddingInputType.QUERY
    assert request.normalize is True
    assert request.truncate is True

    single = EmbeddingRequest(inputs=cast(Any, "one text"))
    assert single.inputs == ["one text"]

    with pytest.raises(ValidationError):
        EmbeddingRequest(inputs=[])
    with pytest.raises(ValidationError, match="must be non-empty"):
        EmbeddingRequest(inputs=["   "])
    with pytest.raises(ValidationError):
        EmbeddingRequest(inputs=["x"], dimensions=0)
    with pytest.raises(ValidationError):
        EmbeddingRequest(inputs=["x"], timeout_seconds=0)


def test_embedding_response_helpers_and_validation() -> None:
    first = Embedding(index=0, vector=[0.1, 0.2, 0.3])
    second = Embedding(index=1, vector=[0.4, 0.5, 0.6])
    response = EmbeddingResponse(
        embeddings=[first, second],
        model="some-model",
        provider="fake-embeddings",
        usage=EmbeddingUsage(input_tokens=12),
    )

    assert response.vectors == [first.vector, second.vector]
    assert response.dimensions == 3
    assert first.dimensions == 3
    assert response.usage.total_tokens == 12
    assert response.usage.prompt_tokens == 12

    assert EmbeddingResponse().dimensions is None

    with pytest.raises(ValidationError):
        Embedding(index=0, vector=[])
    with pytest.raises(ValidationError, match="must be finite"):
        Embedding(index=0, vector=[float("nan")])
    with pytest.raises(ValidationError, match="indices must be unique"):
        EmbeddingResponse(
            embeddings=[
                Embedding(index=0, vector=[0.1]),
                Embedding(index=0, vector=[0.2]),
            ]
        )
    with pytest.raises(ValidationError, match="same dimension"):
        EmbeddingResponse(
            embeddings=[
                Embedding(index=0, vector=[0.1, 0.2]),
                Embedding(index=1, vector=[0.1, 0.2, 0.3]),
            ]
        )


def test_tts_request_validates_text_voice_and_controls() -> None:
    request = TTSRequest(
        text="  Hello brave world  ",
        voice="  narrator  ",
        model="  some-voice-model  ",
        language=" EN ",
        format=AudioFormat.WAV,
        sample_rate=48_000,
        speed=1.25,
        instructions="  warm and calm  ",
        pronunciation={"tomato": "  tuh-MAY-toh  "},
        word_timestamps=True,
    )

    assert request.text == "Hello brave world"
    assert request.voice == "narrator"
    assert request.model == "some-voice-model"
    assert request.language == "en"
    assert request.format is AudioFormat.WAV
    assert request.sample_rate == 48_000
    assert request.speed == 1.25
    assert request.instructions == "warm and calm"
    assert request.pronunciation == {"tomato": "tuh-MAY-toh"}
    assert request.word_timestamps is True

    with pytest.raises(ValidationError):
        TTSRequest(text="   ", voice="narrator")
    with pytest.raises(ValidationError):
        TTSRequest(text="hello", voice="   ")
    with pytest.raises(ValidationError):
        TTSRequest(text="hello", voice="narrator", speed=0.1)
    with pytest.raises(ValidationError):
        TTSRequest(text="hello", voice="narrator", speed=5.0)
    with pytest.raises(ValidationError):
        TTSRequest(text="hello", voice="narrator", sample_rate=0)
    with pytest.raises(ValidationError, match="keys must be non-empty"):
        TTSRequest(text="hello", voice="narrator", pronunciation={"  ": "x"})
    with pytest.raises(ValidationError, match="values must be non-empty"):
        TTSRequest(text="hello", voice="narrator", pronunciation={"x": "  "})


def test_tts_response_carries_audio_and_word_timestamps() -> None:
    timestamps = [
        WordTimestamp(word="hello", start_seconds=0.0, end_seconds=0.5),
        WordTimestamp(word="world", start_seconds=0.5, end_seconds=1.0),
    ]
    response = TTSResponse(
        audio=b"fake-mp3",
        format=AudioFormat.MP3,
        sample_rate=24_000,
        duration_seconds=1.0,
        voice="narrator",
        model="some-voice-model",
        provider="fake-tts",
        word_timestamps=timestamps,
    )

    assert response.size_bytes == 8
    assert response.has_word_timestamps is True
    assert response.word_timestamps[0].duration_seconds == 0.5
    assert response.word_timestamps[1].text == "world"

    empty = TTSResponse(audio=b"fake-mp3")
    assert empty.has_word_timestamps is False

    with pytest.raises(ValidationError):
        TTSResponse(audio=b"")
    with pytest.raises(ValidationError, match="ordered by start_seconds"):
        TTSResponse(
            audio=b"fake-mp3",
            word_timestamps=[timestamps[1], timestamps[0]],
        )


def test_word_timestamp_rejects_an_inverted_span() -> None:
    with pytest.raises(ValidationError, match="end_seconds"):
        WordTimestamp(word="hello", start_seconds=1.0, end_seconds=0.5)
    with pytest.raises(ValidationError):
        WordTimestamp(word="   ", start_seconds=0.0, end_seconds=0.5)
    with pytest.raises(ValidationError):
        WordTimestamp(
            word="hello",
            start_seconds=-0.1,
            end_seconds=0.5,
        )
    with pytest.raises(ValidationError):
        WordTimestamp(
            word="hello",
            start_seconds=0.0,
            end_seconds=0.5,
            confidence=1.5,
        )


def test_stt_request_accepts_inline_audio_or_a_url() -> None:
    inline = STTRequest(
        audio=b"fake-audio",
        media_type=" Audio/WAV ",
        filename="  sample.wav  ",
        model="  some-whisper  ",
        language=" EN ",
        prompt="  names: Ada  ",
        temperature=0.2,
        word_timestamps=True,
        diarize=True,
    )

    assert inline.audio == b"fake-audio"
    assert inline.media_type == "audio/wav"
    assert inline.filename == "sample.wav"
    assert inline.model == "some-whisper"
    assert inline.language == "en"
    assert inline.prompt == "names: Ada"
    assert inline.temperature == 0.2
    assert inline.word_timestamps is True
    assert inline.diarize is True

    remote = STTRequest(audio_url=f"  {_AUDIO_URL}  ")
    assert remote.audio_url == _AUDIO_URL


def test_stt_request_validates_exactly_one_source_and_media_type() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        STTRequest()
    with pytest.raises(ValidationError, match="exactly one"):
        STTRequest(audio=b"fake-audio", audio_url=_AUDIO_URL)
    with pytest.raises(ValidationError, match="audio must be non-empty"):
        STTRequest(audio=b"")
    with pytest.raises(ValidationError, match="media_type"):
        STTRequest(audio=b"fake-audio", media_type="text/plain")
    with pytest.raises(ValidationError):
        STTRequest(audio_url="ftp://example.com/audio.wav")
    with pytest.raises(ValidationError):
        STTRequest(audio=b"fake-audio", temperature=1.5)


def test_stt_response_flattens_word_timestamps_from_segments() -> None:
    words = [
        WordTimestamp(word="hello", start_seconds=0.0, end_seconds=0.4),
        WordTimestamp(word="world", start_seconds=0.4, end_seconds=0.8),
    ]
    segment = TranscriptSegment(
        text="  hello world  ",
        start_seconds=0.0,
        end_seconds=0.8,
        words=words,
        speaker="  narrator  ",
    )
    response = STTResponse(
        text="  hello world  ",
        language=" EN ",
        duration_seconds=0.8,
        segments=[segment],
        model="some-whisper",
        provider="fake-stt",
    )

    assert response.text == "hello world"
    assert response.language == "en"
    assert segment.text == "hello world"
    assert segment.speaker == "narrator"
    assert segment.duration_seconds == 0.8
    assert response.all_words == words
    assert response.word_count == 2
    assert response.has_word_timestamps is True

    direct = STTResponse(
        words=[WordTimestamp(word="one", start_seconds=0.0, end_seconds=0.2)]
    )
    assert direct.all_words == direct.words

    empty = STTResponse(text="")
    assert empty.all_words == []
    assert empty.word_count == 0
    assert empty.has_word_timestamps is False


def test_stt_response_and_segment_reject_inverted_or_unordered_spans() -> None:
    with pytest.raises(ValidationError, match="segment end_seconds"):
        TranscriptSegment(text="hello", start_seconds=1.0, end_seconds=0.5)

    later = WordTimestamp(word="later", start_seconds=0.5, end_seconds=0.8)
    earlier = WordTimestamp(word="earlier", start_seconds=0.0, end_seconds=0.4)

    with pytest.raises(ValidationError, match="segment words"):
        TranscriptSegment(
            text="later earlier",
            start_seconds=0.0,
            end_seconds=0.8,
            words=[later, earlier],
        )
    with pytest.raises(ValidationError, match="ordered by start_seconds"):
        STTResponse(words=[later, earlier])
    with pytest.raises(ValidationError, match="segments must be ordered"):
        STTResponse(
            segments=[
                TranscriptSegment(text="later", start_seconds=0.5, end_seconds=0.8),
                TranscriptSegment(text="earlier", start_seconds=0.0, end_seconds=0.4),
            ]
        )


def test_audio_protocol_models_are_immutable_and_reject_unknown_fields() -> None:
    request = EmbeddingRequest(inputs=["hello"])

    with pytest.raises(ValidationError):
        request.normalize = True
    with pytest.raises(ValidationError):
        EmbeddingRequest.model_validate({"inputs": ["hello"], "unknown": True})
    with pytest.raises(ValidationError):
        TTSRequest.model_validate(
            {"text": "hello", "voice": "narrator", "unknown": True}
        )
    with pytest.raises(ValidationError):
        STTRequest.model_validate({"audio_url": _AUDIO_URL, "unknown": True})


def test_fake_embedding_returns_typed_vectors_and_records_requests() -> None:
    fake = FakeEmbeddingProvider()
    request = EmbeddingRequest(inputs=["alpha beta", "gamma"], dimensions=4)

    response = asyncio.run(fake.embed(request))

    assert isinstance(response, EmbeddingResponse)
    assert response.provider == "fake-embeddings"
    assert response.model == "fake-embedding-1"
    assert len(response.embeddings) == 2
    assert response.dimensions == 4
    assert all(embedding.dimensions == 4 for embedding in response.embeddings)
    assert [embedding.index for embedding in response.embeddings] == [0, 1]
    assert response.usage.input_tokens == 3
    assert fake.requests == [request]


def test_fake_embedding_propagates_a_typed_error() -> None:
    error = RateLimitError("slow down", retry_after=1.5)
    fake = FakeEmbeddingProvider(error=error)

    with pytest.raises(RateLimitError):
        asyncio.run(fake.embed(EmbeddingRequest(inputs=["hello"])))


def test_fake_tts_returns_audio_and_word_timestamps() -> None:
    fake = FakeTTSProvider()
    request = TTSRequest(
        text="hello brave world",
        voice="narrator",
        format=AudioFormat.WAV,
        sample_rate=48_000,
        word_timestamps=True,
    )

    response = asyncio.run(fake.synthesize(request))

    assert isinstance(response, TTSResponse)
    assert response.audio.startswith(b"FAKEAUDIO:")
    assert response.format is AudioFormat.WAV
    assert response.sample_rate == 48_000
    assert response.voice == "narrator"
    assert response.provider == "fake-tts"
    assert [word.word for word in response.word_timestamps] == [
        "hello",
        "brave",
        "world",
    ]
    assert response.word_timestamps[0].start_seconds == 0.0
    assert response.duration_seconds == pytest.approx(1.2)
    assert fake.requests == [request]


def test_fake_tts_omits_timestamps_when_not_requested() -> None:
    fake = FakeTTSProvider()
    response = asyncio.run(fake.synthesize(TTSRequest(text="hello", voice="narrator")))

    assert response.word_timestamps == []
    assert response.has_word_timestamps is False


def test_fake_tts_propagates_a_typed_error() -> None:
    error = RateLimitError("quota", retry_after=2.0)
    fake = FakeTTSProvider(error=error)

    with pytest.raises(RateLimitError):
        asyncio.run(fake.synthesize(TTSRequest(text="hello", voice="narrator")))


def test_fake_stt_returns_a_transcript_with_word_timestamps() -> None:
    fake = FakeSTTProvider()
    request = STTRequest(
        audio=b"FAKEAUDIO:hello brave world",
        media_type="audio/wav",
        language="en",
    )

    response = asyncio.run(fake.transcribe(request))

    assert isinstance(response, STTResponse)
    assert response.text == "hello brave world"
    assert response.language == "en"
    assert response.provider == "fake-stt"
    assert [word.word for word in response.all_words] == [
        "hello",
        "brave",
        "world",
    ]
    assert response.word_count == 3
    assert response.has_word_timestamps is True
    assert len(response.segments) == 1
    assert response.segments[0].text == "hello brave world"
    assert fake.requests == [request]


def test_fake_stt_returns_scripted_responses_and_a_url_fallback() -> None:
    scripted = STTResponse(text="scripted", provider="fake-stt")
    fake = FakeSTTProvider(responses=[scripted])

    first = asyncio.run(fake.transcribe(STTRequest(audio=b"fake-audio")))
    fallback = asyncio.run(fake.transcribe(STTRequest(audio_url=_AUDIO_URL)))

    assert first is scripted
    assert fallback.text == f"transcript for {_AUDIO_URL}"
    assert fallback.language == "en"
    assert len(fake.requests) == 2


def test_fake_stt_propagates_a_typed_error() -> None:
    error = RateLimitError("blocked", retry_after=3.0)
    fake = FakeSTTProvider(error=error)

    with pytest.raises(RateLimitError):
        asyncio.run(fake.transcribe(STTRequest(audio=b"fake-audio")))
