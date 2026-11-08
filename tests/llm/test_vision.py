"""Image input support and vision capable model filtering."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.core import paths
from aivia.core.errors import ConfigError
from aivia.core.model_ids import (
    CATALOG_FILENAME,
    VISION_CAPABILITY,
    ModelCatalog,
    ModelInfo,
    load_model_catalog,
)
from aivia.llm import (
    VisionModelSelector,
    VisionNotSupportedError,
    filter_models_for_request,
    filter_vision_models,
    message_has_images,
    messages_need_vision,
    model_supports_vision,
    request_needs_vision,
    require_vision_model,
    select_vision_model,
)
from aivia.providers.protocols.llm import (
    ChatMessage,
    ContentPart,
    ImagePart,
    LLMRequest,
    MessageRole,
    TextPart,
)

CATALOG = {
    "schema_version": 1,
    "models": {
        "text-only": {"capabilities": ["llm"]},
        "vision-model": {"capabilities": ["llm", "vision"]},
        "other-vision": {"capabilities": ["llm", "vision"]},
        "unknown-capability": {"capabilities": ["llm", "audio"]},
    },
}


@pytest.fixture
def catalog() -> ModelCatalog:
    return ModelCatalog.from_mapping(CATALOG)


def _image() -> ImagePart:
    return ImagePart(url="https://example.com/image.png", media_type="image/png")


def _request(content: str | list[ContentPart] = "describe") -> LLMRequest:
    return LLMRequest(
        messages=[ChatMessage(role=MessageRole.USER, content=content)],
        model="text-only",
    )


def _vision_request() -> LLMRequest:
    content: list[ContentPart] = [
        TextPart(text="describe this"),
        _image(),
    ]
    return _request(content)


def test_a_text_only_request_does_not_need_vision() -> None:
    request = _request()

    assert request.has_images is False
    assert request.images == []
    assert request.requires_vision is False
    assert request.needs_vision is False
    assert request_needs_vision(request) is False


def test_an_image_part_makes_the_request_require_vision() -> None:
    request = _vision_request()

    assert request.has_images is True
    assert request.requires_vision is True
    assert request.needs_vision is True
    assert len(request.images) == 1
    assert request_needs_vision(request) is True


def test_message_helpers_detect_images() -> None:
    text = ChatMessage(role=MessageRole.USER, content="hello")
    image = ChatMessage(
        role=MessageRole.USER,
        content=[TextPart(text="look"), _image()],
    )

    assert message_has_images(text) is False
    assert message_has_images(image) is True
    assert messages_need_vision([text]) is False
    assert messages_need_vision([text, image]) is True


def test_only_models_declaring_vision_support_it(catalog: ModelCatalog) -> None:
    assert model_supports_vision("vision-model", catalog) is True
    assert model_supports_vision("other-vision", catalog) is True
    assert model_supports_vision("text-only", catalog) is False
    assert model_supports_vision("unknown-capability", catalog) is False
    assert model_supports_vision("missing", catalog) is False
    assert model_supports_vision(None, catalog) is False


def test_a_model_info_answers_without_a_catalog() -> None:
    info = ModelInfo(id="vision-model", capabilities=("llm", VISION_CAPABILITY))

    assert model_supports_vision(info) is True
    assert info.supports_vision is True
    assert info.supports("llm") is True
    assert info.supports("audio") is False


def test_the_catalog_exposes_capability_helpers(catalog: ModelCatalog) -> None:
    assert catalog.supports("vision-model", VISION_CAPABILITY) is True
    assert catalog.supports("text-only", VISION_CAPABILITY) is False
    assert catalog.supports_vision("vision-model") is True
    assert catalog.vision_models() == ("other-vision", "vision-model")
    assert catalog.models_with_capability("llm") == (
        "other-vision",
        "text-only",
        "unknown-capability",
        "vision-model",
    )


def test_filtering_drops_models_without_vision_and_preserves_order(
    catalog: ModelCatalog,
) -> None:
    filtered = filter_vision_models(
        ["vision-model", "text-only", "missing", "other-vision"],
        catalog,
    )

    assert filtered == ["vision-model", "other-vision"]


def test_filtering_accepts_one_model_id_and_deduplicates(catalog: ModelCatalog) -> None:
    assert filter_vision_models("vision-model", catalog) == ["vision-model"]
    assert filter_vision_models(
        ["vision-model", " vision-model ", "text-only"],
        catalog,
    ) == ["vision-model"]


def test_a_text_request_keeps_the_candidate_list(catalog: ModelCatalog) -> None:
    models = ["text-only", "vision-model"]

    assert filter_models_for_request(models, _request(), catalog) == models


def test_an_image_request_filters_the_candidate_list(catalog: ModelCatalog) -> None:
    models = ["text-only", "vision-model", "missing"]

    assert filter_models_for_request(models, _vision_request(), catalog) == [
        "vision-model"
    ]


def test_requiring_a_vision_model_accepts_a_capable_one(catalog: ModelCatalog) -> None:
    assert require_vision_model("vision-model", catalog) == "vision-model"
    assert (
        require_vision_model("vision-model", catalog, request=_vision_request())
        == "vision-model"
    )


def test_a_text_request_may_use_any_model(catalog: ModelCatalog) -> None:
    assert require_vision_model("text-only", catalog, request=_request()) == "text-only"


def test_requiring_a_vision_model_rejects_a_text_only_one(
    catalog: ModelCatalog,
) -> None:
    with pytest.raises(VisionNotSupportedError) as caught:
        require_vision_model("text-only", catalog, request=_vision_request())

    assert isinstance(caught.value, ConfigError)
    assert caught.value.model == "text-only"
    assert caught.value.available == ("other-vision", "vision-model")
    assert "does not declare vision support" in str(caught.value)


def test_selecting_returns_the_first_capable_candidate(catalog: ModelCatalog) -> None:
    assert (
        select_vision_model(
            ["text-only", "vision-model", "other-vision"],
            catalog,
        )
        == "vision-model"
    )


def test_selecting_for_a_text_request_uses_the_first_candidate(
    catalog: ModelCatalog,
) -> None:
    assert (
        select_vision_model(
            ["text-only", "vision-model"],
            catalog,
            request=_request(),
        )
        == "text-only"
    )


def test_selecting_without_a_capable_model_fails_clearly(
    catalog: ModelCatalog,
) -> None:
    with pytest.raises(VisionNotSupportedError) as caught:
        select_vision_model(["text-only", "missing"], catalog)

    assert caught.value.model == "text-only"
    assert caught.value.details["candidates"] == ["text-only", "missing"]


def test_the_selector_wraps_filtering_validation_and_selection(
    catalog: ModelCatalog,
) -> None:
    selector = VisionModelSelector(
        catalog,
        ["text-only", "vision-model", "other-vision"],
    )

    assert selector.supports("vision-model") is True
    assert selector.models == ["text-only", "vision-model", "other-vision"]
    assert selector.filter(request=_vision_request()) == [
        "vision-model",
        "other-vision",
    ]
    assert selector.for_request(_request()) == [
        "text-only",
        "vision-model",
        "other-vision",
    ]
    assert selector.ensure("vision-model", request=_vision_request()) == "vision-model"
    assert selector.select(request=_vision_request()) == "vision-model"
    assert selector.select() == "vision-model"


def test_the_packaged_catalog_marks_the_mock_llm_as_vision_capable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    catalog = load_model_catalog()

    assert catalog.source == paths.package_defaults_dir() / CATALOG_FILENAME
    assert catalog.supports_vision("mock-llm") is True
    assert "mock-llm" in catalog.vision_models()


def test_the_active_catalog_is_used_when_none_is_passed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_home = tmp_path
    user_catalog = config_home / "aivia" / CATALOG_FILENAME
    user_catalog.parent.mkdir(parents=True)
    user_catalog.write_text(
        "schema_version: 1\n"
        "models:\n"
        "  custom-vision:\n"
        "    capabilities: [llm, vision]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    assert model_supports_vision("custom-vision") is True
    assert model_supports_vision("mock-llm") is False
