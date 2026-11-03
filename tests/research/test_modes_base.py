"""Topic mode protocol, base class and registry with dotted-path loading."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aivia.core.errors import ConfigError
from aivia.providers.protocols.source import (
    SourceCandidate,
    SourceRequest,
    SourceResponse,
    SourceResult,
)
from aivia.research.modes import (
    TopicMode,
    TopicModeBase,
    TopicModeCandidate,
    TopicModeRegistry,
    TopicModeRequest,
    TopicModeResponse,
    TopicModeResult,
    mode_registry,
    register_mode,
    register_topic_mode,
    resolve_mode,
    resolve_topic_mode,
    topic_mode_registry,
)


class RecordingMode(TopicModeBase):
    """A tiny mode that records requests and returns one candidate."""

    name = "recording"

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.requests: list[SourceRequest] = []

    async def discover(self, request: SourceRequest) -> SourceResponse:
        self.requests.append(request)
        return SourceResponse(
            provider=self.name,
            query=request.query,
            candidates=[
                TopicModeCandidate(
                    title=f"Recorded {request.query}",
                    mode=request.mode,
                )
            ],
        )


def test_protocol_aliases_and_base_mode_are_stable() -> None:
    assert TopicModeRequest is SourceRequest
    assert TopicModeResponse is SourceResponse
    assert TopicModeCandidate is SourceCandidate
    assert TopicModeResult is SourceResult
    assert mode_registry is topic_mode_registry

    mode = RecordingMode()

    assert isinstance(mode, TopicMode)
    assert mode.name == "recording"
    assert mode.mode == "recording"
    assert "recording" in repr(mode)


def test_base_mode_find_and_collect_delegate_to_discover() -> None:
    mode = RecordingMode()
    request = SourceRequest(query="history", mode="trending")

    discovered = asyncio.run(mode.discover(request))
    found = asyncio.run(mode.find(request))
    collected = asyncio.run(mode.collect(request))

    assert discovered.titles == found.titles
    assert discovered.titles == ["Recorded history"]
    assert discovered.provider == "recording"
    assert discovered.first is not None
    assert discovered.first.mode == "trending"
    assert [candidate.title for candidate in collected] == ["Recorded history"]
    assert mode.requests == [request, request, request]


def test_base_mode_allows_an_explicit_name_and_rejects_an_empty_one() -> None:
    assert RecordingMode(name="custom").name == "custom"

    with pytest.raises(ConfigError, match="must not be empty"):
        RecordingMode(name="   ")


def test_registry_registers_resolves_and_rejects_duplicates() -> None:
    registry = TopicModeRegistry()

    registration = registry.register_mode("recording", RecordingMode)

    assert registration.name == "recording"
    assert registration.mode is RecordingMode
    assert registration.provider is RecordingMode
    assert registration.cls is RecordingMode
    assert registry.resolve("recording") is RecordingMode
    assert registry.resolve_mode("recording") is RecordingMode
    assert registry.get("recording") is RecordingMode
    assert registry.get_mode("recording") is RecordingMode
    assert registry["recording"] is RecordingMode
    assert registry.names() == ("recording",)
    assert registry.mode_names() == ("recording",)
    assert registry.keys() == ("recording",)
    assert "recording" in registry
    assert len(registry) == 1
    assert [entry.name for entry in registry] == ["recording"]

    with pytest.raises(ConfigError, match="already registered"):
        registry.register("recording", RecordingMode)


def test_registry_supports_the_decorator_form() -> None:
    registry = TopicModeRegistry()

    @registry.register("decorated")
    class Decorated(RecordingMode):
        name = "decorated"

    @registry.register_mode("also-decorated")
    class AlsoDecorated(RecordingMode):
        name = "also-decorated"

    assert registry.resolve("decorated") is Decorated
    assert registry.resolve("also-decorated") is AlsoDecorated


def test_registry_rejects_a_non_mode_class() -> None:
    registry = TopicModeRegistry()

    with pytest.raises(TypeError, match="subclass TopicModeBase"):
        registry.register("not-a-mode", object)  # type: ignore[arg-type]


def test_registry_accepts_a_structural_mode_class() -> None:
    registry = TopicModeRegistry()

    class StructuralMode:
        name = "structural"

        async def discover(self, request: SourceRequest) -> SourceResponse:
            return SourceResponse()

    registry.register("structural", StructuralMode)  # type: ignore[arg-type]

    assert registry.resolve("structural").__name__ == "StructuralMode"
    assert isinstance(StructuralMode(), TopicMode)


def test_registry_unknown_name_lists_the_registered_modes() -> None:
    registry = TopicModeRegistry()
    registry.register("recording", RecordingMode)

    with pytest.raises(ConfigError) as caught:
        registry.resolve("trending")

    message = str(caught.value)
    assert "unknown topic mode: trending" in message
    assert "registered: recording" in message
    assert caught.value.details["mode"] == "trending"
    assert caught.value.details["available"] == ["recording"]

    with pytest.raises(ConfigError, match="unknown topic mode: trending"):
        registry.get("trending")


def test_registry_loads_a_custom_mode_by_dotted_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = f"custom_mode_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{module_name}.py").write_text(
        "\n".join(
            [
                "from aivia.research.modes import TopicModeBase",
                "from aivia.providers.protocols.source import SourceResponse",
                "",
                "class CustomMode(TopicModeBase):",
                '    name = "custom"',
                "",
                "    async def discover(self, request):",
                "        return SourceResponse(provider=self.name)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    registry = TopicModeRegistry()

    colon_loaded = registry.load(f"{module_name}:CustomMode")
    dot_loaded = registry.load(f"{module_name}.CustomMode")

    assert colon_loaded is dot_loaded
    assert colon_loaded.__name__ == "CustomMode"
    assert registry.resolve(f"{module_name}:CustomMode") is colon_loaded
    assert registry.resolve(f"{module_name}.CustomMode") is colon_loaded


def test_registry_dotted_path_errors_are_clear() -> None:
    registry = TopicModeRegistry()

    with pytest.raises(ConfigError, match="invalid topic mode path"):
        registry.load("missing-attribute:")

    with pytest.raises(ConfigError, match="cannot load topic mode") as missing_module:
        registry.resolve("not_a_real_mode_module:CustomMode")
    assert missing_module.value.details["module"] == "not_a_real_mode_module"

    with pytest.raises(ConfigError, match="has no attribute") as missing_attribute:
        registry.resolve("aivia.research.modes.base:NotAMode")
    assert missing_attribute.value.details["attribute"] == "NotAMode"

    with pytest.raises(ConfigError, match="cannot load topic mode") as wrong_type:
        registry.resolve("aivia.research.modes.base:SourceRequest")
    assert wrong_type.value.details["module"] == "aivia.research.modes.base"


def test_shared_registry_register_and_resolve_helpers_agree() -> None:
    @register_topic_mode("shared-recording")
    class SharedRecording(RecordingMode):
        name = "shared-recording"

    try:
        assert topic_mode_registry.resolve("shared-recording") is SharedRecording
        assert resolve_topic_mode("shared-recording") is SharedRecording
        assert resolve_mode("shared-recording") is SharedRecording
        assert register_mode.__self__ is topic_mode_registry  # type: ignore[attr-defined]
    finally:
        topic_mode_registry.unregister("shared-recording")
