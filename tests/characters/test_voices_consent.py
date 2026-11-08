"""Per language character voices, pinned provider ids and consent gating."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import ValidationError
from sqlalchemy import Engine

from aivia.characters import Character, resolve_character_voice
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import ContentPolicyError, InvalidRequestError
from aivia.core.models import VoiceClone
from aivia.media.voice_clone import VoiceCloneService

PROVIDER = "speech-vendor"
OWNER = "ada@example.test"


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def service(engine: Engine) -> VoiceCloneService:
    return VoiceCloneService(engine)


def _create_clone(
    service: VoiceCloneService,
    *,
    provider_voice_id: str = "voice-clone-1",
    language: str = "en",
    consent: object = True,
) -> VoiceClone:
    return service.create(
        name="Narrator",
        provider=PROVIDER,
        provider_voice_id=provider_voice_id,
        owner=OWNER,
        consent=consent,
        language=language,
    )


def test_character_voice_for_language_resolves_per_language() -> None:
    character = Character.model_validate(
        {
            "name": "Mascot",
            "voice": "default-en",
            "voices": {"en": "voice-en", "de": "voice-de"},
        }
    )

    assert character.voice_for_language("en") == "voice-en"
    assert character.voice_for_language("DE") == "voice-de"
    assert character.voice_for_language("fr") == "default-en"
    assert character.voice_for_language("") == "default-en"


def test_character_voices_reject_blank_keys_and_values() -> None:
    with pytest.raises(ValidationError, match="language keys must be non-empty"):
        Character.model_validate({"name": "Mascot", "voices": {"": "voice-en"}})
    with pytest.raises(ValidationError, match="voice ids must be non-empty"):
        Character.model_validate({"name": "Mascot", "voices": {"en": "   "}})
    with pytest.raises(ValidationError):
        Character.model_validate({"name": "Mascot", "voices": "voice-en"})


def test_create_clone_requires_consent_and_pins_provider_voice_id(
    service: VoiceCloneService,
) -> None:
    with pytest.raises(ContentPolicyError, match="explicit consent"):
        service.create(
            name="Narrator",
            provider=PROVIDER,
            provider_voice_id="voice-1",
            owner=OWNER,
        )

    clone = _create_clone(service)

    assert clone.provider_voice_id == "voice-clone-1"
    assert clone.voice_id == "voice-clone-1"
    assert clone.owner == OWNER
    assert clone.language == "en"
    assert clone.consent_granted is True
    assert service.list() == [clone]


def test_pinned_provider_voice_id_survives_a_new_service(
    engine: Engine, service: VoiceCloneService
) -> None:
    clone = _create_clone(service)

    reopened = VoiceCloneService(engine)

    cloned = reopened.get(clone.id)
    assert cloned is not None
    assert cloned.provider_voice_id == "voice-clone-1"
    assert reopened.list(owner=OWNER, language="en") == [clone]


def test_resolve_character_voice_returns_the_pinned_provider_voice_id(
    service: VoiceCloneService,
) -> None:
    character = Character.model_validate(
        {
            "name": "Mascot",
            "voice": "default-en",
            "voices": {"en": "voice-clone-1", "de": "voice-clone-2"},
        }
    )
    _create_clone(service, provider_voice_id="voice-clone-1", language="en")
    _create_clone(service, provider_voice_id="voice-clone-2", language="de")

    assert (
        resolve_character_voice(character, "en", voice_clones=service, owner=OWNER)
        == "voice-clone-1"
    )
    assert (
        resolve_character_voice(character, "de", voice_clones=service, owner=OWNER)
        == "voice-clone-2"
    )


def test_resolve_character_voice_falls_back_without_a_clone_store() -> None:
    character = Character.model_validate(
        {
            "name": "Mascot",
            "voice": "default-en",
            "voices": {"en": "voice-en"},
        }
    )

    assert resolve_character_voice(character, "en") == "voice-en"
    assert resolve_character_voice(character, "fr") == "default-en"


def test_resolve_character_voice_refuses_an_unregistered_voice(
    service: VoiceCloneService,
) -> None:
    character = Character.model_validate({"name": "Mascot", "voice": "voice-missing"})

    with pytest.raises(InvalidRequestError):
        resolve_character_voice(character, "en", voice_clones=service, owner=OWNER)


def test_resolve_character_voice_refuses_without_consent(
    engine: Engine, service: VoiceCloneService
) -> None:
    character = Character.model_validate(
        {"name": "Mascot", "voice": "voice-1", "voices": {"en": "voice-1"}}
    )
    service.repository.create(
        VoiceClone(
            name="Narrator",
            provider=PROVIDER,
            provider_voice_id="voice-1",
            owner=OWNER,
            language="en",
            consent={"granted": False},
        )
    )

    with pytest.raises(ContentPolicyError, match="explicit consent"):
        resolve_character_voice(character, "en", voice_clones=service, owner=OWNER)
