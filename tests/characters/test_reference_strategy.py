"""Character reference images are attached to requests per supporting provider."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.characters.capabilities import CharacterReferenceLimitError
from aivia.characters.reference_strategy import (
    CharacterReferenceAttachError,
    attach_character_references,
    load_character_references,
    prepare_character_image_request,
)
from aivia.characters.schema import Character
from aivia.core.models.provider import ProviderMeta
from aivia.providers import Provider
from aivia.providers.protocols.image import ImageRequest
from aivia.providers.protocols.media import MediaInput


class FakeProvider(Provider):
    def __init__(
        self,
        name: str,
        capabilities: list[str],
        supported_options: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__(
            ProviderMeta(
                name=name,
                capabilities=capabilities,
                supported_options=supported_options or {},
            )
        )


def _character(*, references: int = 2, name: str = "Mascot") -> Character:
    images = [f"ref-{i}.png" for i in range(references)]
    return Character.model_validate({"name": name, "reference_images": images})


def _request() -> ImageRequest:
    return ImageRequest(prompt="a mascot in a scene", width=1000, height=1000)


_PNG = b"\x89PNG\r\n\x1a\n"


def _write_references(directory: Path, count: int) -> None:
    for index in range(count):
        (directory / f"ref-{index}.png").write_bytes(_PNG)


def test_load_character_references_reads_files_as_inline_media(tmp_path: Path) -> None:
    _write_references(tmp_path, 2)

    inputs = load_character_references(_character(), tmp_path)

    assert len(inputs) == 2
    for item, expected in zip(inputs, ("ref-0.png", "ref-1.png"), strict=True):
        assert item.is_inline
        assert item.data == _PNG
        assert item.media_type == "image/png"
        assert item.filename == expected


def test_load_character_references_detects_media_type_from_extension(
    tmp_path: Path,
) -> None:
    character = Character.model_validate(
        {"name": "Mascot", "reference_images": ["ref-0.jpg"]}
    )
    (tmp_path / "ref-0.jpg").write_bytes(b"\xff\xd8\xff")

    inputs = load_character_references(character, tmp_path)

    assert len(inputs) == 1
    assert inputs[0].media_type == "image/jpeg"


def test_load_character_references_raises_when_an_image_is_missing(
    tmp_path: Path,
) -> None:
    (tmp_path / "ref-0.png").write_bytes(_PNG)

    with pytest.raises(CharacterReferenceAttachError) as caught:
        load_character_references(_character(), tmp_path)

    assert caught.value.provider == "Mascot"
    assert "ref-1.png" in caught.value.reason
    assert caught.value.details["reference_index"] == 1


def test_attach_character_references_returns_a_copy_without_mutating() -> None:
    request = _request()
    references = [
        MediaInput(data=b"bytes-one", media_type="image/png", filename="a.png"),
        MediaInput(data=b"bytes-two", media_type="image/png", filename="b.png"),
    ]

    attached = attach_character_references(request, references)

    assert attached is not request
    assert attached.reference_images == references
    assert request.reference_images == []


def test_prepare_returns_only_providers_with_character_capability(
    tmp_path: Path,
) -> None:
    _write_references(tmp_path, 2)

    without = FakeProvider("without", [])
    with_flag = FakeProvider("with-flag", ["character_reference"])

    prepared, warning = prepare_character_image_request(
        _character(),
        _request(),
        [without, with_flag],
        max_reference_images=4,
        base_dir=tmp_path,
    )

    assert warning is None
    assert [provider.name for provider, _, _ in prepared] == ["with-flag"]
    provider, adapted, _constraints = prepared[0]
    assert provider is with_flag
    assert len(adapted.reference_images) == 2
    assert adapted.reference_images[0].data == _PNG


def test_prepare_adapts_each_provider_to_its_own_limits(tmp_path: Path) -> None:
    _write_references(tmp_path, 1)

    snapping = FakeProvider(
        "snapping",
        ["character_reference"],
        supported_options={
            "image_gen": ["size:512x512", "size:1024x1024"],
        },
    )
    unsnapped = FakeProvider(
        "unsnapped",
        ["character_reference"],
        supported_options={},
    )

    prepared, _ = prepare_character_image_request(
        _character(references=1),
        ImageRequest(prompt="a scene", width=900, height=900),
        [snapping, unsnapped],
        max_reference_images=4,
        base_dir=tmp_path,
    )

    sizes = {provider.name: adapted.width for provider, adapted, _ in prepared}
    assert sizes == {"snapping": 1024, "unsnapped": 900}


def test_prepare_enforces_limit_before_filtering_providers(tmp_path: Path) -> None:
    _write_references(tmp_path, 5)
    with_flag = FakeProvider("with-flag", ["character_reference"])

    with pytest.raises(CharacterReferenceLimitError):
        prepare_character_image_request(
            _character(references=5),
            _request(),
            [with_flag],
            max_reference_images=4,
            base_dir=tmp_path,
        )


def test_prepare_falls_back_to_all_providers_without_references(tmp_path: Path) -> None:
    without = FakeProvider("without", [])
    with_flag = FakeProvider("with-flag", ["character_reference"])

    prepared, warning = prepare_character_image_request(
        _character(references=0),
        _request(),
        [without, with_flag],
        max_reference_images=4,
        base_dir=tmp_path,
    )

    assert warning is None
    assert [provider.name for provider, _, _ in prepared] == ["without", "with-flag"]
    for _, adapted, _ in prepared:
        assert adapted.reference_images == []


def test_prepare_falls_back_when_no_provider_supports_references(
    tmp_path: Path,
) -> None:
    _write_references(tmp_path, 2)
    without = FakeProvider("without", [])

    prepared, warning = prepare_character_image_request(
        _character(),
        _request(),
        [without],
        max_reference_images=4,
        base_dir=tmp_path,
    )

    assert warning is not None
    assert "character_reference" in warning
    assert "prompt-only generation" in warning
    assert [provider.name for provider, _, _ in prepared] == ["without"]
    for _, adapted, _ in prepared:
        assert adapted.reference_images == []


def test_prepare_returns_constraints_per_provider(tmp_path: Path) -> None:
    _write_references(tmp_path, 1)

    provider = FakeProvider(
        "sized",
        ["character_reference"],
        supported_options={"image_gen": ["size:512x512"]},
    )

    prepared, _ = prepare_character_image_request(
        _character(references=1),
        ImageRequest(prompt="a scene", width=100, height=100),
        [provider],
        max_reference_images=4,
        base_dir=tmp_path,
    )

    _, _, constraints = prepared[0]
    assert any(size.width == 512 for size in constraints.sizes)


def test_prepare_propagates_adaptation_errors(tmp_path: Path) -> None:
    _write_references(tmp_path, 1)
    provider = FakeProvider(
        "strict",
        ["character_reference"],
        supported_options={"image_gen": ["models:known-model"]},
    )

    with pytest.raises(CharacterReferenceAttachError) as caught:
        prepare_character_image_request(
            _character(references=1),
            ImageRequest(
                prompt="a scene",
                width=100,
                height=100,
                model="unknown-model",
            ),
            [provider],
            max_reference_images=4,
            base_dir=tmp_path,
        )

    assert caught.value.provider == "strict"
    assert caught.value.details["character"] == "Mascot"


def test_prepare_keeps_the_original_request_unmodified(tmp_path: Path) -> None:
    _write_references(tmp_path, 2)
    provider = FakeProvider("with-flag", ["character_reference"])

    request = ImageRequest(prompt="a scene", width=900, height=900)
    original = request.model_copy()

    prepare_character_image_request(
        _character(), request, [provider], max_reference_images=4, base_dir=tmp_path
    )

    assert request == original


def test_prepare_uses_the_character_reference_capability(tmp_path: Path) -> None:
    from aivia.characters.capabilities import (
        CAPABILITY_CHARACTER_REFERENCE,
        supports_character_references,
    )

    _write_references(tmp_path, 1)
    provider = FakeProvider("capable", [CAPABILITY_CHARACTER_REFERENCE])
    incapable = FakeProvider("incapable", [])

    assert supports_character_references(provider) is True
    assert supports_character_references(incapable) is False

    prepared, warning = prepare_character_image_request(
        _character(references=1),
        _request(),
        [incapable, provider],
        max_reference_images=4,
        base_dir=tmp_path,
    )

    assert warning is None
    assert [provider.name for provider, _, _ in prepared] == ["capable"]
