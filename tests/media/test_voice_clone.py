"""Voice cloning needs explicit consent and keeps provider ids with their owner."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine

from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.errors import (
    ContentPolicyError,
    InvalidRequestError,
    ValidationError,
)
from aivia.core.models import VoiceClone
from aivia.media.voice_clone import (
    VoiceCloneRequest,
    VoiceCloneService,
    normalize_consent,
)

PROVIDER = "speech-vendor"
VOICE_ID = "voice-clone-1"
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


def _create(
    service: VoiceCloneService,
    *,
    name: str = "Narrator",
    voice_id: str = VOICE_ID,
    owner: str = OWNER,
    consent: object = True,
    provider: str = PROVIDER,
    language: str = "",
    metadata: dict[str, object] | None = None,
) -> VoiceClone:
    return service.create(
        name=name,
        provider=provider,
        provider_voice_id=voice_id,
        owner=owner,
        consent=consent,
        language=language,
        metadata=metadata or {},
    )


def test_create_stores_the_provider_voice_id_owner_and_consent(
    service: VoiceCloneService,
) -> None:
    clone = _create(
        service,
        metadata={"source": "studio"},
    )

    assert isinstance(clone, VoiceClone)
    assert clone.id.startswith("voice_")
    assert clone.name == "Narrator"
    assert clone.provider == PROVIDER
    assert clone.provider_voice_id == VOICE_ID
    assert clone.owner == OWNER
    assert clone.metadata == {"source": "studio"}
    assert clone.consent["granted"] is True
    assert clone.consent["granted_at"]
    assert clone.consent_granted is True
    assert clone.clone_id == clone.id
    assert clone.voice_id == VOICE_ID

    listed = service.list()
    assert [item.id for item in listed] == [clone.id]
    assert listed[0].provider_voice_id == VOICE_ID
    assert listed[0].owner == OWNER
    assert listed[0].consent_metadata == clone.consent


def test_create_refuses_missing_false_or_empty_consent(
    service: VoiceCloneService,
) -> None:
    with pytest.raises(ContentPolicyError, match="explicit consent"):
        service.create(
            provider=PROVIDER,
            provider_voice_id=VOICE_ID,
            owner=OWNER,
        )
    with pytest.raises(ContentPolicyError, match="explicit consent"):
        _create(service, consent=False)
    with pytest.raises(ContentPolicyError, match="explicit consent"):
        _create(service, consent={"granted": False})
    with pytest.raises(ContentPolicyError, match="explicit consent"):
        _create(service, consent={"reference": "no grant here"})

    assert service.list() == []


def test_consent_mapping_is_kept_as_metadata_and_normalized(
    service: VoiceCloneService,
) -> None:
    clone = _create(
        service,
        consent={
            "granted": True,
            "granted_by": "Ada",
            "reference": "contract-42",
            "scope": "voice_cloning",
        },
    )

    assert clone.consent["granted"] is True
    assert clone.consent["granted_by"] == "Ada"
    assert clone.consent["reference"] == "contract-42"
    assert clone.consent["scope"] == "voice_cloning"
    assert clone.consent["granted_at"]


def test_normalize_consent_accepts_aliases_and_rejects_a_bad_value() -> None:
    assert normalize_consent({"allowed": True})["granted"] is True
    assert normalize_consent({"voice_cloning": True})["granted"] is True

    with pytest.raises(ContentPolicyError):
        normalize_consent(None)
    with pytest.raises(ContentPolicyError):
        normalize_consent("yes")  # type: ignore[arg-type]


def test_create_requires_a_provider_voice_id_and_owner(
    service: VoiceCloneService,
) -> None:
    with pytest.raises(ValidationError, match="owner"):
        service.create(
            provider=PROVIDER,
            provider_voice_id=VOICE_ID,
            owner="   ",
            consent=True,
        )
    with pytest.raises(ValidationError, match="provider_voice_id"):
        service.create(
            provider=PROVIDER,
            provider_voice_id="",
            owner=OWNER,
            consent=True,
        )
    with pytest.raises(ValidationError, match="provider"):
        service.create(
            provider="",
            provider_voice_id=VOICE_ID,
            owner=OWNER,
            consent=True,
        )


def test_create_accepts_a_request_model_mapping_or_name_string(
    service: VoiceCloneService,
) -> None:
    from_request = service.create(
        VoiceCloneRequest(
            name="Request",
            provider=PROVIDER,
            provider_voice_id="voice-request",
            owner=OWNER,
            consent=True,
        )
    )
    from_mapping = service.create(
        {
            "name": "Mapping",
            "provider": PROVIDER,
            "provider_voice_id": "voice-mapping",
            "owner": OWNER,
            "consent": True,
        }
    )
    from_name = service.create(
        "Named",
        provider=PROVIDER,
        provider_voice_id="voice-named",
        owner=OWNER,
        consent=True,
    )

    assert from_request.name == "Request"
    assert from_mapping.name == "Mapping"
    assert from_name.name == "Named"
    assert from_name.consent["granted"] is True


def test_create_rejects_a_duplicate_provider_voice_id(
    service: VoiceCloneService,
) -> None:
    _create(service)

    with pytest.raises(InvalidRequestError, match="already exists"):
        _create(service, name="Second")


def test_list_filters_by_owner_provider_and_language_and_pages(
    service: VoiceCloneService,
) -> None:
    first = _create(service, name="First", voice_id="voice-1", language="EN")
    second = _create(
        service,
        name="Second",
        voice_id="voice-2",
        owner="bob@example.test",
        language="de",
    )
    third = _create(
        service,
        name="Third",
        voice_id="voice-3",
        owner="carol@example.test",
        provider="other-vendor",
        language="de",
    )

    assert [item.id for item in service.list(OWNER)] == [first.id]
    assert [item.id for item in service.list(provider="other-vendor")] == [third.id]
    assert [item.id for item in service.list(language="DE")] == [
        second.id,
        third.id,
    ]
    assert [item.id for item in service.list(limit=1)] == [first.id]
    assert [item.id for item in service.list(limit=1, offset=1)] == [second.id]


def test_get_and_delete_are_scoped_to_the_owner(service: VoiceCloneService) -> None:
    clone = _create(service)

    assert service.get(clone.id) is not None
    assert service.get(clone.id, owner=OWNER) is not None
    assert service.get(clone.id, owner="bob@example.test") is None

    assert service.delete(clone.id, owner="bob@example.test") is None
    assert service.get(clone.id) is not None

    deleted = service.delete(clone.id, owner=OWNER)
    assert deleted is not None
    assert deleted.id == clone.id
    assert service.get(clone.id) is None
    assert service.list() == []
    assert service.delete(clone.id) is None


def test_records_survive_a_new_service_bound_to_the_same_engine(
    engine: Engine,
    service: VoiceCloneService,
) -> None:
    clone = _create(service)

    reopened = VoiceCloneService(engine)

    assert [item.id for item in reopened.list()] == [clone.id]
    assert reopened.get(clone.id) is not None
