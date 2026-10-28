"""Notifier, publisher, delivery, storage, lip sync and trainer protocols."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from pydantic import ValidationError

from aivia.core.errors import InvalidRequestError, RateLimitError
from aivia.core.models import Attestation, ProviderMeta
from aivia.providers import Provider
from aivia.providers.protocols import (
    CAPABILITY_CHARACTER_TRAINER,
    CAPABILITY_DELIVERY,
    CAPABILITY_LIPSYNC,
    CAPABILITY_NOTIFY,
    CAPABILITY_PUBLISH,
    CAPABILITY_STORAGE,
    CapabilityProvider,
    CharacterTrainer,
    CharacterTrainerRequest,
    CharacterTrainingJob,
    DeliveryArtifact,
    DeliveryMethod,
    DeliveryProvider,
    DeliveryReceipt,
    DeliveryRequest,
    LipSyncProvider,
    LipSyncRequest,
    LipSyncResponse,
    LipSyncResult,
    LipSyncSpriteSheet,
    MediaInput,
    NotificationButton,
    NotificationLevel,
    NotificationMedia,
    NotificationMessage,
    NotificationReceipt,
    Notifier,
    PostVisibility,
    Publisher,
    PublishMedia,
    PublishRequest,
    PublishResult,
    StorageObject,
    StorageProvider,
    StoragePutRequest,
    TrainingState,
    VideoFormat,
)

from .provider_contract import ProviderContract

_IMAGE_URL = "https://example.com/media/image.png"
_AUDIO_URL = "https://example.com/media/audio.wav"
_VIDEO_URL = "https://example.com/media/clip.mp4"


class FakeNotifier(Provider):
    """A notifier fake that records messages and acknowledges each one."""

    default_metadata = ProviderMeta(
        name="fake-notifier",
        capabilities=[CAPABILITY_NOTIFY],
    )

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self._error = error
        self.messages: list[NotificationMessage] = []

    async def send(self, message: NotificationMessage) -> NotificationReceipt:
        self.messages.append(message)
        if self._error is not None:
            raise self._error
        return NotificationReceipt(
            message_id=f"msg-{len(self.messages)}",
            channel="fake-channel",
            provider=self.name,
            delivered_at=datetime.now(UTC),
            metadata={"level": message.level.value},
        )


class FakePublisher(Provider):
    """A publisher fake that records requests and returns stable identifiers."""

    default_metadata = ProviderMeta(
        name="fake-publisher",
        capabilities=[CAPABILITY_PUBLISH],
    )

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self._error = error
        self.requests: list[PublishRequest] = []

    async def publish(self, request: PublishRequest) -> PublishResult:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        number = len(self.requests)
        if request.dry_run:
            return PublishResult(
                platform=request.platform,
                account=request.account,
                provider=self.name,
                dry_run=True,
            )
        return PublishResult(
            platform=request.platform,
            account=request.account,
            provider=self.name,
            remote_id=f"remote-{number}",
            url=f"https://example.com/posts/{number}",
            published_at=datetime.now(UTC),
        )


class FakeDeliveryProvider(Provider):
    """A delivery fake that records requests and acknowledges the target."""

    default_metadata = ProviderMeta(
        name="fake-delivery",
        capabilities=[CAPABILITY_DELIVERY],
    )

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self._error = error
        self.requests: list[DeliveryRequest] = []

    async def deliver(self, request: DeliveryRequest) -> DeliveryReceipt:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        number = len(self.requests)
        url = (
            f"https://example.com/deliveries/{number}"
            if request.method is DeliveryMethod.LINK
            else None
        )
        return DeliveryReceipt(
            method=request.method,
            reference=f"delivery-{number}",
            target=request.target,
            provider=self.name,
            delivered_at=datetime.now(UTC),
            url=url,
            expires_at=(
                datetime.now(UTC) + timedelta(seconds=request.expires_seconds)
                if request.expires_seconds is not None
                else None
            ),
        )


class FakeStorageProvider(Provider):
    """An in-memory storage fake with content addressed keys and dedupe."""

    default_metadata = ProviderMeta(
        name="fake-storage",
        capabilities=[CAPABILITY_STORAGE],
    )

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self._error = error
        self._objects: dict[str, tuple[bytes, StorageObject]] = {}

    async def put(self, request: StoragePutRequest) -> StorageObject:
        if self._error is not None:
            raise self._error
        digest = hashlib.sha256(request.data).hexdigest()
        key = request.key or f"objects/{digest[:2]}/{digest}"
        existing = self._objects.get(key)
        if existing is not None:
            if existing[0] == request.data:
                return existing[1]
            if not request.overwrite:
                raise InvalidRequestError(f"storage key already exists: {key}")
        stored = StorageObject(
            key=key,
            size_bytes=len(request.data),
            content_type=request.content_type,
            sha256=digest,
            url=self.url(key),
            created_at=datetime.now(UTC),
        )
        self._objects[key] = (request.data, stored)
        return stored

    async def get(self, key: str) -> bytes:
        if key not in self._objects:
            raise KeyError(key)
        return self._objects[key][0]

    async def stat(self, key: str) -> StorageObject | None:
        entry = self._objects.get(key)
        return None if entry is None else entry[1]

    async def delete(self, key: str) -> bool:
        return self._objects.pop(key, None) is not None

    def url(self, key: str, *, expires_seconds: int | None = None) -> str:
        base = f"https://storage.example.com/{key}"
        if expires_seconds is not None:
            return f"{base}?expires={expires_seconds}"
        return base


class FakeLipSyncProvider(Provider):
    """A lip sync fake that marks sprite sheet requests as a fallback."""

    default_metadata = ProviderMeta(
        name="fake-lipsync",
        capabilities=[CAPABILITY_LIPSYNC],
    )

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self._error = error
        self.requests: list[LipSyncRequest] = []

    async def animate(self, request: LipSyncRequest) -> LipSyncResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        payload = request.audio.data or b"remote-audio"
        return LipSyncResponse(
            video=LipSyncResult(
                data=b"FAKELIPSYNC:" + payload,
                format=request.format,
                width=512,
                height=512,
                duration_seconds=request.duration_seconds,
                fps=request.fps,
                used_fallback=request.sprite_sheet is not None,
            ),
            model="fake-lipsync-1",
            provider=self.name,
        )


class FakeCharacterTrainer(Provider):
    """A trainer fake that immediately returns a ready adapter."""

    default_metadata = ProviderMeta(
        name="fake-trainer",
        capabilities=[CAPABILITY_CHARACTER_TRAINER],
    )

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self._error = error
        self.requests: list[CharacterTrainerRequest] = []
        self.jobs: dict[str, CharacterTrainingJob] = {}

    async def train(self, request: CharacterTrainerRequest) -> CharacterTrainingJob:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        number = len(self.requests)
        job = CharacterTrainingJob(
            job_id=f"train-{number}",
            character_id=request.character_id,
            state=TrainingState.READY,
            model_id=f"fake-adapter-{number}",
            trigger_words=request.trigger_words,
            cost_usd=1.5,
            created_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
        self.jobs[job.job_id] = job
        return job

    async def status(self, job_id: str) -> CharacterTrainingJob:
        return self.jobs[job_id]


class TestFakeNotifierContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return FakeNotifier()


class TestFakePublisherContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return FakePublisher()


class TestFakeDeliveryProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return FakeDeliveryProvider()


class TestFakeStorageProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return FakeStorageProvider()


class TestFakeLipSyncProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return FakeLipSyncProvider()


class TestFakeCharacterTrainerContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return FakeCharacterTrainer()


def _attestation() -> Attestation:
    return Attestation(
        confirmed_by="Ada",
        rights_confirmed=True,
        disclosure_confirmed=True,
        confirmed_at=datetime.now(UTC),
        disclosure_text="AI generated",
        license_summary="CC0",
    )


def test_fakes_satisfy_the_capability_protocols() -> None:
    notifier = FakeNotifier()
    publisher = FakePublisher()
    delivery = FakeDeliveryProvider()
    storage = FakeStorageProvider()
    lipsync = FakeLipSyncProvider()
    trainer = FakeCharacterTrainer()

    assert isinstance(notifier, Notifier)
    assert isinstance(publisher, Publisher)
    assert isinstance(delivery, DeliveryProvider)
    assert isinstance(storage, StorageProvider)
    assert isinstance(lipsync, LipSyncProvider)
    assert isinstance(trainer, CharacterTrainer)
    for fake in (notifier, publisher, delivery, storage, lipsync, trainer):
        assert isinstance(fake, CapabilityProvider)
    assert not isinstance(notifier, Publisher)
    assert not isinstance(publisher, Notifier)
    assert not isinstance(storage, DeliveryProvider)
    assert not isinstance(Provider(), Notifier)


def test_capability_names_are_stable() -> None:
    assert CAPABILITY_NOTIFY == "notify"
    assert CAPABILITY_PUBLISH == "publish"
    assert CAPABILITY_DELIVERY == "delivery"
    assert CAPABILITY_STORAGE == "storage"
    assert CAPABILITY_LIPSYNC == "lipsync"
    assert CAPABILITY_CHARACTER_TRAINER == "character_trainer"


def test_notification_message_validates_content_buttons_and_media() -> None:
    message = NotificationMessage(
        title="  Review needed  ",
        text="  A run is waiting  ",
        level=NotificationLevel.WARNING,
        media=[
            NotificationMedia(
                data=b"image",
                media_type=" Image/PNG ",
                filename="  preview.png  ",
                caption="  preview  ",
            )
        ],
        buttons=[
            NotificationButton(label="  Approve  ", callback_data=" approve "),
            NotificationButton(label="Open", url=f"  {_IMAGE_URL}  "),
        ],
    )

    assert message.title == "Review needed"
    assert message.text == "A run is waiting"
    assert message.level is NotificationLevel.WARNING
    assert message.media[0].media_type == "image/png"
    assert message.media[0].filename == "preview.png"
    assert message.media[0].caption == "preview"
    assert message.media[0].size_bytes == 5
    assert message.buttons[0].label == "Approve"
    assert message.buttons[0].callback_data == "approve"
    assert message.buttons[1].url == _IMAGE_URL
    assert message.has_media is True
    assert message.has_buttons is True

    with pytest.raises(ValidationError, match="text, title or media"):
        NotificationMessage()
    with pytest.raises(ValidationError, match="exactly one"):
        NotificationButton(label="x")
    with pytest.raises(ValidationError, match="exactly one"):
        NotificationButton(label="x", url=_IMAGE_URL, callback_data="payload")
    with pytest.raises(ValidationError):
        NotificationButton(label="   ", callback_data="payload")
    with pytest.raises(ValidationError):
        NotificationButton(label="x", url="ftp://example.com")
    with pytest.raises(ValidationError, match="exactly one"):
        NotificationMedia(data=b"x", url=_IMAGE_URL)
    with pytest.raises(ValidationError, match="non-empty"):
        NotificationMedia(data=b"")
    with pytest.raises(ValidationError):
        NotificationMedia(data=b"x", media_type="text/plain")


def test_notification_receipt_normalizes_its_fields() -> None:
    naive = datetime(2026, 1, 2, 3, 4, 5)
    receipt = NotificationReceipt(
        message_id="  msg-1  ",
        channel="  fake  ",
        provider="  fake-notifier  ",
        delivered_at=naive,
        url=f"  {_IMAGE_URL}  ",
    )

    assert receipt.message_id == "msg-1"
    assert receipt.channel == "fake"
    assert receipt.provider == "fake-notifier"
    assert receipt.delivered_at == naive.replace(tzinfo=UTC)
    assert receipt.url == _IMAGE_URL
    assert receipt.delivered is True

    with pytest.raises(ValidationError):
        NotificationReceipt(url="ftp://example.com")


def test_publish_request_and_result_validate_platform_fields() -> None:
    request = PublishRequest(
        platform="  youtube  ",
        account="  main  ",
        title="  A title  ",
        caption="  A caption  ",
        hashtags=["  #one  ", "#one", "  #two  "],
        mentions=["  @ada  "],
        media=[PublishMedia(path="  output.mp4  ", media_type="video/mp4")],
        visibility=PostVisibility.PUBLIC,
        scheduled_for=datetime(2026, 1, 2, 3, 4, 5),
        idempotency_key="  key-1  ",
        disclosure_text="  AI generated  ",
        attestation=_attestation(),
    )

    assert request.platform == "youtube"
    assert request.account == "main"
    assert request.title == "A title"
    assert request.caption == "A caption"
    assert request.hashtags == ["#one", "#two"]
    assert request.mentions == ["@ada"]
    assert request.media[0].path == "output.mp4"
    assert request.media[0].media_type == "video/mp4"
    assert request.visibility is PostVisibility.PUBLIC
    assert request.scheduled_for == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert request.idempotency_key == "key-1"
    assert request.disclosure_text == "AI generated"
    assert request.attestation is not None
    assert request.attestation.is_complete is True
    assert request.has_media is True

    result = PublishResult(
        platform="  youtube  ",
        provider="  fake-publisher  ",
        remote_id="  remote-1  ",
        url=f"  {_VIDEO_URL}  ",
        published_at=datetime(2026, 1, 2, 3, 4, 5),
    )
    assert result.platform == "youtube"
    assert result.remote_id == "remote-1"
    assert result.url == _VIDEO_URL
    assert result.published_at == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert result.is_live is True
    assert PublishResult(remote_id="x", dry_run=True).is_live is False

    with pytest.raises(ValidationError):
        PublishRequest(platform="   ")
    with pytest.raises(ValidationError, match="non-empty"):
        PublishRequest(platform="youtube", hashtags=["  "])
    with pytest.raises(ValidationError, match="exactly one"):
        PublishMedia(data=b"x", path="output.png")
    with pytest.raises(ValidationError):
        PublishResult(url="ftp://example.com/post")


def test_delivery_request_and_receipt_validate_the_method() -> None:
    request = DeliveryRequest(
        method=DeliveryMethod.FOLDER,
        target="  /tmp/deliveries  ",
        order_id="  order-1  ",
        subject="  Your card  ",
        message="  Enjoy  ",
        artifacts=[DeliveryArtifact(path="  card.png  ", media_type="image/png")],
        expires_seconds=3600,
        scheduled_for=datetime(2026, 1, 2, 3, 4, 5),
        idempotency_key="  delivery-key  ",
        attestation=_attestation(),
    )

    assert request.method is DeliveryMethod.FOLDER
    assert request.target == "/tmp/deliveries"
    assert request.order_id == "order-1"
    assert request.subject == "Your card"
    assert request.artifacts[0].path == "card.png"
    assert request.artifacts[0].media_type == "image/png"
    assert request.scheduled_for == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert request.has_artifacts is True

    receipt = DeliveryReceipt(
        method=DeliveryMethod.LINK,
        reference="  delivery-1  ",
        target="  https://example.com/delivery  ",
        provider="  fake-delivery  ",
        delivered_at=datetime(2026, 1, 2, 3, 4, 5),
        url=f"  {_IMAGE_URL}  ",
    )
    assert receipt.method is DeliveryMethod.LINK
    assert receipt.reference == "delivery-1"
    assert receipt.target == "https://example.com/delivery"
    assert receipt.delivered_at == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert receipt.url == _IMAGE_URL

    with pytest.raises(ValidationError):
        DeliveryRequest(method=cast(Any, "carrier-pigeon"), target="x")
    with pytest.raises(ValidationError):
        DeliveryRequest(method=DeliveryMethod.FOLDER, target="   ")
    with pytest.raises(ValidationError, match="exactly one"):
        DeliveryArtifact(data=b"x", url=_IMAGE_URL)
    with pytest.raises(ValidationError):
        DeliveryArtifact(url="ftp://example.com/file")


def test_storage_put_request_and_object_validate_content_addressing() -> None:
    request = StoragePutRequest(
        data=b"hello",
        key="  outputs/hello.txt  ",
        content_type="  text/plain  ",
        metadata={"run": "run-1"},
    )

    assert request.key == "outputs/hello.txt"
    assert request.content_type == "text/plain"
    assert request.size_bytes == 5

    digest = hashlib.sha256(b"hello").hexdigest()
    stored = StorageObject(
        key="  objects/ab/hello  ",
        size_bytes=5,
        content_type="  text/plain  ",
        sha256=digest.upper(),
        url=f"  {_IMAGE_URL}  ",
        expires_at=datetime(2026, 1, 2, 3, 4, 5),
        created_at=datetime(2026, 1, 2, 3, 4, 5),
    )
    assert stored.key == "objects/ab/hello"
    assert stored.content_type == "text/plain"
    assert stored.sha256 == digest
    assert stored.url == _IMAGE_URL
    assert stored.expires_at == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

    assert StorageObject(key="local", size_bytes=0, url="file:///tmp/x").url == (
        "file:///tmp/x"
    )

    with pytest.raises(ValidationError, match="non-empty"):
        StoragePutRequest(data=b"")
    with pytest.raises(ValidationError):
        StorageObject(key="   ", size_bytes=1)
    with pytest.raises(ValidationError):
        StorageObject(key="x", size_bytes=-1)
    with pytest.raises(ValidationError, match="64 character"):
        StorageObject(key="x", size_bytes=1, sha256="not-a-digest")
    with pytest.raises(ValidationError, match="http, https or file"):
        StorageObject(key="x", size_bytes=1, url="ftp://example.com/file")


def test_lipsync_request_accepts_a_source_or_a_sprite_sheet() -> None:
    request = LipSyncRequest(
        audio=MediaInput(data=b"audio", media_type=" Audio/WAV "),
        source=MediaInput(url=_IMAGE_URL, media_type="image/png"),
        format=VideoFormat.WEBM,
        fps=24,
        duration_seconds=2.5,
        smooth=False,
    )

    assert request.audio.media_type == "audio/wav"
    assert request.source is not None
    assert request.source.url == _IMAGE_URL
    assert request.format is VideoFormat.WEBM
    assert request.fps == 24
    assert request.duration_seconds == 2.5
    assert request.smooth is False

    sheet = LipSyncSpriteSheet(
        image=MediaInput(url=_IMAGE_URL, media_type="image/png"),
        frame_width=128,
        frame_height=128,
        frame_count=8,
        fps=12,
        mouth_frames={"  closed  ": 0, "open": 1},
        blink_frame=7,
        idle_frames=[0, 2],
    )
    assert sheet.mouth_frames == {"closed": 0, "open": 1}
    assert sheet.blink_frame == 7
    assert sheet.idle_frames == [0, 2]

    fallback = LipSyncRequest(
        audio=MediaInput(url=_AUDIO_URL, media_type="audio/wav"),
        sprite_sheet=sheet,
    )
    assert fallback.source is None
    assert fallback.sprite_sheet is sheet

    with pytest.raises(ValidationError, match="source or a sprite sheet"):
        LipSyncRequest(audio=MediaInput(data=b"audio", media_type="audio/wav"))
    with pytest.raises(ValidationError, match="image or video"):
        LipSyncRequest(
            audio=MediaInput(data=b"audio", media_type="audio/wav"),
            source=MediaInput(data=b"audio", media_type="audio/wav"),
        )
    with pytest.raises(ValidationError, match="audio type"):
        LipSyncRequest(
            audio=MediaInput(data=b"video", media_type="video/mp4"),
            source=MediaInput(url=_IMAGE_URL, media_type="image/png"),
        )
    with pytest.raises(ValidationError, match="frame_count"):
        LipSyncSpriteSheet(
            image=MediaInput(url=_IMAGE_URL, media_type="image/png"),
            frame_width=128,
            frame_height=128,
            frame_count=2,
            mouth_frames={"open": 2},
        )
    with pytest.raises(ValidationError, match="non-empty"):
        LipSyncSpriteSheet(
            image=MediaInput(url=_IMAGE_URL, media_type="image/png"),
            frame_width=128,
            frame_height=128,
            frame_count=2,
            mouth_frames={"  ": 0},
        )


def test_lipsync_result_and_response_carry_the_clip() -> None:
    result = LipSyncResult(
        data=b"clip",
        format=VideoFormat.MP4,
        width=512,
        height=512,
        duration_seconds=2.0,
        fps=30,
        used_fallback=True,
    )
    assert result.is_inline is True
    assert result.size_bytes == 4
    assert result.dimensions is not None
    assert result.dimensions.width == 512
    assert result.used_fallback is True

    response = LipSyncResponse(
        video=result,
        model="  fake-lipsync-1  ",
        provider="  fake-lipsync  ",
        request_id="  req-1  ",
    )
    assert response.video is result
    assert response.model == "fake-lipsync-1"
    assert response.provider == "fake-lipsync"
    assert response.request_id == "req-1"

    remote = LipSyncResult(url=f"  {_VIDEO_URL}  ")
    assert remote.is_inline is False
    assert remote.size_bytes is None
    assert remote.url == _VIDEO_URL

    with pytest.raises(ValidationError, match="exactly one"):
        LipSyncResult()
    with pytest.raises(ValidationError, match="exactly one"):
        LipSyncResult(data=b"clip", url=_VIDEO_URL)
    with pytest.raises(ValidationError):
        LipSyncResult(data=b"")
    with pytest.raises(ValidationError):
        LipSyncResult(data=b"clip", width=512)
    with pytest.raises(ValidationError):
        LipSyncResult(url="ftp://example.com/clip.mp4")


def test_character_trainer_request_and_job_validate_training_fields() -> None:
    request = CharacterTrainerRequest(
        character_id="  ada  ",
        name="  Ada  ",
        trigger_words=["  ada  ", "ada", "  narrator  "],
        reference_images=[
            MediaInput(url=_IMAGE_URL, media_type="image/png"),
            MediaInput(data=b"image", media_type="image/jpeg"),
        ],
        base_model="  base-1  ",
        steps=500,
        learning_rate=0.0001,
    )

    assert request.character_id == "ada"
    assert request.name == "Ada"
    assert request.trigger_words == ["ada", "narrator"]
    assert len(request.reference_images) == 2
    assert request.base_model == "base-1"
    assert request.steps == 500
    assert request.learning_rate == 0.0001

    job = CharacterTrainingJob(
        job_id="  train-1  ",
        character_id="  ada  ",
        state=TrainingState.READY,
        model_id="  adapter-1  ",
        trigger_words=["  ada  "],
        cost_usd=1.25,
        created_at=datetime(2026, 1, 2, 3, 4, 5),
        completed_at=datetime(2026, 1, 2, 4, 5, 6),
    )
    assert job.job_id == "train-1"
    assert job.character_id == "ada"
    assert job.state is TrainingState.READY
    assert job.model_id == "adapter-1"
    assert job.trigger_words == ["ada"]
    assert job.cost_usd == 1.25
    assert job.created_at == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert job.is_ready is True
    assert job.is_terminal is True

    failed = CharacterTrainingJob(
        job_id="train-2",
        character_id="ada",
        state=TrainingState.FAILED,
        error="  quota exceeded  ",
    )
    assert failed.error == "quota exceeded"
    assert failed.is_ready is False
    assert failed.is_terminal is True
    assert (
        CharacterTrainingJob(
            job_id="train-3", character_id="ada", state=TrainingState.TRAINING
        ).is_terminal
        is False
    )

    with pytest.raises(ValidationError):
        CharacterTrainerRequest(
            character_id="ada",
            trigger_words=["ada"],
            reference_images=[],
        )
    with pytest.raises(ValidationError):
        CharacterTrainerRequest(
            character_id="ada",
            trigger_words=[],
            reference_images=[MediaInput(url=_IMAGE_URL, media_type="image/png")],
        )
    with pytest.raises(ValidationError, match="non-empty"):
        CharacterTrainerRequest(
            character_id="ada",
            trigger_words=["  "],
            reference_images=[MediaInput(url=_IMAGE_URL, media_type="image/png")],
        )
    with pytest.raises(ValidationError, match="reference images"):
        CharacterTrainerRequest(
            character_id="ada",
            trigger_words=["ada"],
            reference_images=[MediaInput(data=b"audio", media_type="audio/wav")],
        )
    with pytest.raises(ValidationError):
        CharacterTrainerRequest(
            character_id="ada",
            trigger_words=["ada"],
            reference_images=[MediaInput(url=_IMAGE_URL, media_type="image/png")],
            steps=0,
        )
    with pytest.raises(ValidationError, match="model_id"):
        CharacterTrainingJob(
            job_id="train-4",
            character_id="ada",
            state=TrainingState.READY,
        )


def test_io_protocol_models_are_immutable_and_reject_unknown_fields() -> None:
    message = NotificationMessage(text="hello")
    with pytest.raises(ValidationError):
        message.text = "changed"
    with pytest.raises(ValidationError):
        NotificationMessage.model_validate({"text": "hello", "unknown": True})
    with pytest.raises(ValidationError):
        PublishRequest.model_validate({"platform": "youtube", "unknown": True})
    with pytest.raises(ValidationError):
        DeliveryRequest.model_validate(
            {"method": "folder", "target": "x", "unknown": True}
        )
    with pytest.raises(ValidationError):
        StoragePutRequest.model_validate({"data": b"x", "unknown": True})
    with pytest.raises(ValidationError):
        LipSyncRequest.model_validate(
            {
                "audio": {"url": _AUDIO_URL, "media_type": "audio/wav"},
                "source": {"url": _IMAGE_URL, "media_type": "image/png"},
                "unknown": True,
            }
        )
    with pytest.raises(ValidationError):
        CharacterTrainerRequest.model_validate(
            {
                "character_id": "ada",
                "trigger_words": ["ada"],
                "reference_images": [{"url": _IMAGE_URL, "media_type": "image/png"}],
                "unknown": True,
            }
        )


def test_fake_notifier_records_and_acknowledges_messages() -> None:
    fake = FakeNotifier()
    message = NotificationMessage(
        text="hello",
        buttons=[NotificationButton(label="OK", callback_data="ok")],
    )

    receipt = asyncio.run(fake.send(message))

    assert receipt.message_id == "msg-1"
    assert receipt.channel == "fake-channel"
    assert receipt.provider == "fake-notifier"
    assert receipt.delivered_at is not None
    assert receipt.metadata == {"level": "info"}
    assert fake.messages == [message]


def test_fake_notifier_propagates_a_typed_error() -> None:
    fake = FakeNotifier(error=RateLimitError("slow down", retry_after=1.0))

    with pytest.raises(RateLimitError):
        asyncio.run(fake.send(NotificationMessage(text="hello")))


def test_fake_publisher_publishes_and_honours_dry_run() -> None:
    fake = FakePublisher()
    request = PublishRequest(
        platform="youtube",
        account="main",
        caption="hello",
        idempotency_key="key-1",
    )

    result = asyncio.run(fake.publish(request))

    assert result.platform == "youtube"
    assert result.account == "main"
    assert result.remote_id == "remote-1"
    assert result.url == "https://example.com/posts/1"
    assert result.is_live is True
    assert fake.requests == [request]

    dry = asyncio.run(fake.publish(request.model_copy(update={"dry_run": True})))
    assert dry.dry_run is True
    assert dry.remote_id == ""
    assert dry.is_live is False


def test_fake_delivery_returns_a_receipt_for_the_target() -> None:
    fake = FakeDeliveryProvider()
    request = DeliveryRequest(
        method=DeliveryMethod.LINK,
        target="https://example.com/delivery",
        artifacts=[DeliveryArtifact(data=b"card", media_type="image/png")],
        expires_seconds=600,
    )

    receipt = asyncio.run(fake.deliver(request))

    assert receipt.method is DeliveryMethod.LINK
    assert receipt.reference == "delivery-1"
    assert receipt.target == request.target
    assert receipt.provider == "fake-delivery"
    assert receipt.url == "https://example.com/deliveries/1"
    assert receipt.expires_at is not None
    assert fake.requests == [request]


def test_fake_storage_is_content_addressed_and_dedupes() -> None:
    fake = FakeStorageProvider()
    first = asyncio.run(fake.put(StoragePutRequest(data=b"hello")))

    assert first.key.startswith("objects/")
    assert first.sha256 == hashlib.sha256(b"hello").hexdigest()
    assert first.size_bytes == 5
    assert asyncio.run(fake.get(first.key)) == b"hello"
    assert asyncio.run(fake.stat(first.key)) == first

    second = asyncio.run(fake.put(StoragePutRequest(data=b"hello")))
    assert second == first
    asyncio.run(fake.put(StoragePutRequest(data=b"fixed", key="fixed.txt")))
    with pytest.raises(InvalidRequestError):
        asyncio.run(fake.put(StoragePutRequest(data=b"different", key="fixed.txt")))
    assert asyncio.run(fake.delete(first.key)) is True
    assert asyncio.run(fake.delete(first.key)) is False
    assert asyncio.run(fake.stat(first.key)) is None


def test_fake_lipsync_marks_sprite_sheet_requests_as_fallback() -> None:
    fake = FakeLipSyncProvider()
    source = LipSyncRequest(
        audio=MediaInput(data=b"audio", media_type="audio/wav"),
        source=MediaInput(url=_IMAGE_URL, media_type="image/png"),
        duration_seconds=1.5,
    )
    fallback = LipSyncRequest(
        audio=MediaInput(data=b"audio", media_type="audio/wav"),
        sprite_sheet=LipSyncSpriteSheet(
            image=MediaInput(url=_IMAGE_URL, media_type="image/png"),
            frame_width=128,
            frame_height=128,
            frame_count=4,
        ),
    )

    first = asyncio.run(fake.animate(source))
    second = asyncio.run(fake.animate(fallback))

    assert first.video.used_fallback is False
    assert first.video.data is not None
    assert first.video.data.startswith(b"FAKELIPSYNC:")
    assert first.provider == "fake-lipsync"
    assert second.video.used_fallback is True
    assert fake.requests == [source, fallback]


def test_fake_trainer_returns_a_ready_adapter_and_status() -> None:
    fake = FakeCharacterTrainer()
    request = CharacterTrainerRequest(
        character_id="ada",
        trigger_words=["ada"],
        reference_images=[MediaInput(url=_IMAGE_URL, media_type="image/png")],
    )

    job = asyncio.run(fake.train(request))

    assert job.state is TrainingState.READY
    assert job.model_id == "fake-adapter-1"
    assert job.trigger_words == ["ada"]
    assert job.cost_usd == 1.5
    assert asyncio.run(fake.status(job.job_id)) is job
    assert fake.requests == [request]
