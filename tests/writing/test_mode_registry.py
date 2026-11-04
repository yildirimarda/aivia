"""Writing mode base class, registry and custom modes loaded from files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from aivia.core.errors import ConfigError
from aivia.writing import (
    BaseWritingMode,
    BuiltinWritingMode,
    CustomModeDefinition,
    CustomPayload,
    FactsPayload,
    ModeBase,
    ModeRegistry,
    Script,
    WritingMode,
    WritingModeBase,
    WritingModeProtocol,
    WritingModeRegistration,
    WritingModeRegistry,
    builtin_writing_mode,
    is_known_writing_mode,
    load_writing_mode,
    load_writing_mode_file,
    mode_registry,
    register_mode,
    register_writing_mode,
    resolve_mode,
    resolve_writing_mode,
    writing_mode_registry,
)


class RecordingMode(WritingModeBase):
    """A tiny mode that keeps its instructions and payload model."""

    name = "recording"
    instructions = "Record the input."
    payload_model = FactsPayload


def test_base_class_aliases_and_defaults_are_stable() -> None:
    assert BaseWritingMode is WritingModeBase
    assert ModeBase is WritingModeBase
    assert ModeRegistry is WritingModeRegistry
    assert mode_registry is writing_mode_registry
    assert register_mode.__self__ is writing_mode_registry  # type: ignore[attr-defined]
    assert resolve_mode.__self__ is writing_mode_registry  # type: ignore[attr-defined]
    assert load_writing_mode.__self__ is writing_mode_registry  # type: ignore[attr-defined]
    assert load_writing_mode_file.__self__ is writing_mode_registry  # type: ignore[attr-defined]

    mode = RecordingMode()

    assert isinstance(mode, WritingModeProtocol)
    assert mode.name == "recording"
    assert mode.mode == "recording"
    assert mode.payload_type is FactsPayload
    assert mode.payload_class is FactsPayload
    assert "recording" in repr(mode)


def test_base_class_builds_a_typed_payload() -> None:
    mode = RecordingMode()

    payload = mode.build_payload({"facts": ["A fact."]})

    assert isinstance(payload, FactsPayload)
    assert [fact.text for fact in payload.facts] == ["A fact."]
    assert mode.build({"facts": ["A fact."]}) == payload
    assert mode.payload_for({"facts": ["A fact."]}) == payload
    assert mode.validate_payload({"facts": ["A fact."]}) == payload


def test_base_class_accepts_an_explicit_name_and_rejects_an_empty_one() -> None:
    assert RecordingMode(name="custom").name == "custom"

    with pytest.raises(ConfigError, match="must not be empty"):
        RecordingMode(name="   ")

    with pytest.raises(ConfigError, match="invalid writing mode name"):
        RecordingMode(name="two words")


def test_base_class_to_dict_and_custom_definition_are_plain_data() -> None:
    definition = CustomModeDefinition(
        name="acrostic",
        instructions="Start each line with the next letter.",
        description="A shaped poem.",
        aliases=("acrostic-poem",),
        default_payload={"lines": []},
    )
    mode = WritingModeBase(
        definition.name,
        description=definition.description,
        instructions=definition.instructions,
        aliases=definition.aliases,
        default_payload=definition.default_payload,
    )

    assert mode.to_dict() == {
        "name": "acrostic",
        "description": "A shaped poem.",
        "instructions": "Start each line with the next letter.",
        "aliases": ["acrostic-poem"],
        "default_payload": {"lines": []},
    }
    assert isinstance(mode.build_payload(), CustomPayload)


def test_registry_registers_resolves_and_rejects_duplicates() -> None:
    registry = WritingModeRegistry()

    registration = registry.register_mode("recording", RecordingMode)

    assert isinstance(registration, WritingModeRegistration)
    assert registration.name == "recording"
    assert registration.mode is RecordingMode
    assert registration.provider is RecordingMode
    assert registration.cls is RecordingMode
    assert registration.instructions == "Record the input."
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
    registry = WritingModeRegistry()

    @registry.register("decorated")
    class Decorated(RecordingMode):
        name = "decorated"

    @registry.register_mode("also-decorated")
    class AlsoDecorated(RecordingMode):
        name = "also-decorated"

    assert registry.resolve("decorated") is Decorated
    assert registry.resolve("also-decorated") is AlsoDecorated


def test_registry_accepts_an_instance_and_a_structural_mode() -> None:
    registry = WritingModeRegistry()
    instance = WritingModeBase("instance-mode", instructions="Keep it short.")

    class StructuralMode:
        name = "structural"

        def build_payload(self, value: Any = None) -> CustomPayload:
            return CustomPayload()

    registry.register(instance)
    registry.register("structural", StructuralMode)  # type: ignore[arg-type]

    assert registry.resolve("instance-mode") is instance
    assert registry.resolve("structural") is StructuralMode  # type: ignore[comparison-overlap]


def test_registry_rejects_a_non_mode_class() -> None:
    registry = WritingModeRegistry()

    with pytest.raises(TypeError, match="subclass WritingModeBase"):
        registry.register("not-a-mode", object)  # type: ignore[arg-type]


def test_registry_unknown_name_lists_the_registered_modes() -> None:
    registry = WritingModeRegistry()
    registry.register("recording", RecordingMode)

    with pytest.raises(ConfigError) as caught:
        registry.resolve("interpretive_dance")

    message = str(caught.value)
    assert "unknown writing mode: interpretive_dance" in message
    assert "registered: recording" in message
    assert caught.value.details["mode"] == "interpretive_dance"
    assert caught.value.details["available"] == ["recording"]

    with pytest.raises(ConfigError, match="unknown writing mode"):
        registry.get("interpretive_dance")


def test_built_in_modes_resolve_to_placeholder_classes() -> None:
    registry = WritingModeRegistry()

    facts = cast(type[BuiltinWritingMode], registry.resolve("facts"))

    assert isinstance(facts, type)
    assert issubclass(facts, BuiltinWritingMode)
    assert facts is builtin_writing_mode(WritingMode.FACTS)
    assert facts().name == "facts"
    assert facts().payload_model is FactsPayload
    assert registry.resolve("lyrics") is builtin_writing_mode("lyrics")

    shared = writing_mode_registry
    assert {mode.value for mode in WritingMode} <= set(shared.names())
    assert shared.resolve("joke") is builtin_writing_mode(WritingMode.JOKE)


def test_registry_loads_a_custom_mode_from_a_python_file(tmp_path: Path) -> None:
    path = tmp_path / "custom_mode.py"
    path.write_text(
        "\n".join(
            [
                "from aivia.writing import WritingModeBase",
                "",
                "class CustomMode(WritingModeBase):",
                '    name = "custom-file"',
                '    instructions = "Follow the file."',
                "",
            ]
        ),
        encoding="utf-8",
    )
    registry = WritingModeRegistry()

    loaded = cast(type[WritingModeBase], registry.load_file(path))

    assert issubclass(loaded, WritingModeBase)
    assert loaded.name == "custom-file"
    assert loaded.instructions == "Follow the file."
    assert registry.resolve("custom-file") is loaded
    assert registry.load(str(path)) is loaded
    assert registry.load_file(str(path)) is loaded


def test_registry_names_a_python_mode_after_its_file(tmp_path: Path) -> None:
    path = tmp_path / "nameless_mode.py"
    path.write_text(
        "from aivia.writing import WritingModeBase\n\n"
        "class Nameless(WritingModeBase):\n"
        "    pass\n",
        encoding="utf-8",
    )
    registry = WritingModeRegistry()

    loaded = cast(type[WritingModeBase], registry.load_file(path))

    assert loaded.name == "nameless_mode"
    assert registry.resolve("nameless_mode") is loaded


def test_registry_selects_one_class_from_a_python_file(tmp_path: Path) -> None:
    path = tmp_path / "several.py"
    path.write_text(
        "\n".join(
            [
                "from aivia.writing import WritingModeBase",
                "",
                "class FirstMode(WritingModeBase):",
                '    name = "first"',
                "",
                "class SecondMode(WritingModeBase):",
                '    name = "second"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    registry = WritingModeRegistry()

    with pytest.raises(ConfigError, match="defines several modes"):
        registry.load_file(path)

    assert registry.load(f"{path}:SecondMode").name == "second"


def test_registry_loads_a_custom_mode_from_a_yaml_file(tmp_path: Path) -> None:
    path = tmp_path / "acrostic.yaml"
    path.write_text(
        "\n".join(
            [
                "name: acrostic",
                "description: A shaped poem.",
                "instructions:",
                "  - Start each line with the next letter.",
                "  - Keep the lines short.",
                "aliases: [acrostic-poem]",
                "payload:",
                "  lines: []",
            ]
        ),
        encoding="utf-8",
    )
    registry = WritingModeRegistry()

    loaded = cast(type[WritingModeBase], registry.load_file(path))

    assert loaded.name == "acrostic"
    assert loaded.description == "A shaped poem."
    assert loaded.instructions == (
        "Start each line with the next letter.\nKeep the lines short."
    )
    assert loaded.aliases == ("acrostic-poem",)
    assert loaded.default_payload == {"lines": []}
    assert registry.resolve("acrostic") is loaded
    assert isinstance(loaded().build_payload(), CustomPayload)


def test_registry_loads_a_custom_mode_from_json_and_toml(tmp_path: Path) -> None:
    json_path = tmp_path / "json_mode.json"
    json_path.write_text(
        json.dumps({"name": "json-mode", "instructions": "From JSON."}),
        encoding="utf-8",
    )
    toml_path = tmp_path / "toml_mode.toml"
    toml_path.write_text(
        'name = "toml-mode"\ninstructions = "From TOML."\n',
        encoding="utf-8",
    )
    registry = WritingModeRegistry()

    assert registry.load_file(json_path).instructions == "From JSON."
    assert registry.load_file(toml_path).instructions == "From TOML."
    assert registry.resolve("json-mode").name == "json-mode"
    assert registry.resolve("toml-mode").name == "toml-mode"


def test_registry_loads_a_plain_text_instruction_file(tmp_path: Path) -> None:
    path = tmp_path / "haiku.md"
    path.write_text("Write three short lines.\n", encoding="utf-8")
    registry = WritingModeRegistry()

    loaded = registry.load_file(path)

    assert loaded.name == "haiku"
    assert loaded.instructions == "Write three short lines."
    assert registry.resolve("haiku") is loaded


def test_registry_loads_several_modes_from_one_data_file(tmp_path: Path) -> None:
    path = tmp_path / "modes.yaml"
    path.write_text(
        "\n".join(
            [
                "modes:",
                "  first:",
                "    instructions: First mode.",
                "  second:",
                "    instructions: Second mode.",
            ]
        ),
        encoding="utf-8",
    )
    registry = WritingModeRegistry()

    loaded = registry.load_file(path)

    assert loaded.name == "first"
    assert registry.resolve("first") is loaded
    assert registry.resolve("second").instructions == "Second mode."


def test_registry_register_file_registers_the_file_mode(tmp_path: Path) -> None:
    path = tmp_path / "file_mode.yaml"
    path.write_text("name: file-mode\ninstructions: From a file.\n", encoding="utf-8")
    registry = WritingModeRegistry()

    registration = registry.register_file(path)

    assert registration.name == "file-mode"
    assert registration.mode is registry.resolve("file-mode")
    assert registration.source == str(path)
    assert registry.names() == ("file-mode",)


def test_registry_loads_a_custom_mode_by_dotted_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = f"custom_writing_mode_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{module_name}.py").write_text(
        "\n".join(
            [
                "from aivia.writing import WritingModeBase",
                "",
                "class DottedMode(WritingModeBase):",
                '    name = "dotted"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    registry = WritingModeRegistry()

    colon_loaded = registry.load(f"{module_name}:DottedMode")
    dot_loaded = registry.load(f"{module_name}.DottedMode")

    assert colon_loaded is dot_loaded
    assert colon_loaded.name == "dotted"
    assert registry.resolve(f"{module_name}:DottedMode") is colon_loaded


def test_registry_file_and_dotted_path_errors_are_clear(tmp_path: Path) -> None:
    registry = WritingModeRegistry()

    with pytest.raises(ConfigError, match="invalid writing mode path"):
        registry.load("missing-attribute:")

    with pytest.raises(ConfigError, match="file does not exist"):
        registry.load(str(tmp_path / "missing.yaml"))

    empty = tmp_path / "empty.py"
    empty.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="defines no WritingModeBase"):
        registry.load_file(empty)

    with pytest.raises(ConfigError, match="cannot load writing mode") as missing:
        registry.resolve("not_a_real_mode_module:DottedMode")
    assert missing.value.details["module"] == "not_a_real_mode_module"

    with pytest.raises(ConfigError, match="has no attribute") as attribute:
        registry.resolve("aivia.writing.modes:NotAMode")
    assert attribute.value.details["attribute"] == "NotAMode"


def test_script_accepts_a_registered_custom_mode(tmp_path: Path) -> None:
    path = tmp_path / "script_mode.yaml"
    path.write_text(
        "name: acrostic-script\ninstructions: Shape the lines.\n",
        encoding="utf-8",
    )
    registry = WritingModeRegistry()
    registry.load_file(path)

    script = Script.model_validate(
        {"mode": "acrostic-script", "payload": {"lines": ["A line"]}}
    )

    assert script.mode == "acrostic-script"
    assert isinstance(script.payload, CustomPayload)
    assert script.payload.model_extra == {"lines": ["A line"]}
    assert Script.model_validate_json(script.to_json()) == script
    assert is_known_writing_mode("acrostic-script")
    assert not is_known_writing_mode("interpretive_dance")


def test_shared_registry_register_resolve_and_unregister_helpers_agree() -> None:
    @register_writing_mode("shared-recording")
    class SharedRecording(RecordingMode):
        name = "shared-recording"

    try:
        assert writing_mode_registry.resolve("shared-recording") is SharedRecording
        assert resolve_writing_mode("shared-recording") is SharedRecording
        assert resolve_mode("shared-recording") is SharedRecording
        assert "shared-recording" in writing_mode_registry
    finally:
        writing_mode_registry.unregister("shared-recording")

    with pytest.raises(ConfigError, match="unknown writing mode"):
        writing_mode_registry.resolve("shared-recording")


def test_registry_resolves_custom_mode_aliases(tmp_path: Path) -> None:
    path = tmp_path / "alias_mode.yaml"
    path.write_text(
        "\n".join(
            [
                "name: acrostic",
                "instructions: Shape the lines.",
                "aliases: [acrostic-poem, shaped]",
            ]
        ),
        encoding="utf-8",
    )
    registry = WritingModeRegistry()

    loaded = registry.load_file(path)

    assert registry.resolve("acrostic-poem") is loaded
    assert registry.resolve_mode("shaped") is loaded
    assert registry.get("acrostic-poem") is loaded
    assert registry["shaped"] is loaded
    assert "acrostic-poem" in registry
    assert is_known_writing_mode("acrostic-poem")
    assert is_known_writing_mode("shaped")


def test_registry_resolves_python_mode_aliases() -> None:
    class AliasMode(RecordingMode):
        name = "alias-mode"
        aliases = ("alias-one", "alias-two")

    registry = WritingModeRegistry()
    registry.register(AliasMode)

    assert registry.resolve("alias-one") is AliasMode
    assert registry.resolve("alias-two") is AliasMode


def test_registry_rejects_an_alias_that_collides_with_a_registered_name() -> None:
    class AliasCollision(RecordingMode):
        name = "alias-collision"
        aliases = ("recording",)

    registry = WritingModeRegistry()
    registry.register("recording", RecordingMode)

    with pytest.raises(ConfigError, match="alias") as caught:
        registry.register(AliasCollision)

    assert caught.value.details["alias"] == "recording"
    assert caught.value.details["mode"] == "alias-collision"
    assert registry.resolve("recording") is RecordingMode


def test_registry_rejects_a_name_that_collides_with_an_alias() -> None:
    class AliasOwner(RecordingMode):
        name = "alias-owner"
        aliases = ("taken",)

    registry = WritingModeRegistry()
    registry.register(AliasOwner)

    with pytest.raises(ConfigError, match="alias") as caught:
        registry.register("taken", RecordingMode)

    assert caught.value.details["mode"] == "taken"
    assert caught.value.details["alias_of"] == "alias-owner"
    assert registry.resolve("taken") is AliasOwner


def test_registry_rejects_an_alias_shared_by_two_modes() -> None:
    class FirstAlias(RecordingMode):
        name = "first-alias"
        aliases = ("shared-alias",)

    class SecondAlias(RecordingMode):
        name = "second-alias"
        aliases = ("shared-alias",)

    registry = WritingModeRegistry()
    registry.register(FirstAlias)

    with pytest.raises(ConfigError, match="alias") as caught:
        registry.register(SecondAlias)

    assert caught.value.details["alias"] == "shared-alias"
    assert caught.value.details["existing"] == "first-alias"


def test_registry_rejects_an_alias_colliding_with_a_built_in_mode() -> None:
    class SharedRegistry(WritingModeRegistry):
        include_builtins = True

    class FactsAlias(RecordingMode):
        name = "facts-alias"
        aliases = ("facts",)

    registry = SharedRegistry()

    with pytest.raises(ConfigError, match="built in") as caught:
        registry.register(FactsAlias)

    assert caught.value.details["alias"] == "facts"


def test_registry_rejects_an_alias_collision_while_loading_a_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "loaded_alias.yaml"
    path.write_text(
        "name: loaded-alias\ninstructions: Keep it.\naliases: [recording]\n",
        encoding="utf-8",
    )
    registry = WritingModeRegistry()
    registry.register("recording", RecordingMode)

    with pytest.raises(ConfigError, match="alias") as caught:
        registry.load_file(path)

    assert caught.value.details["alias"] == "recording"


def test_registry_loads_every_mode_from_a_directory(tmp_path: Path) -> None:
    directory = tmp_path / "modes"
    directory.mkdir()
    (directory / "01_first.yaml").write_text(
        "name: first\ninstructions: First mode.\naliases: [first-alias]\n",
        encoding="utf-8",
    )
    (directory / "02_second.json").write_text(
        json.dumps({"name": "second", "instructions": "Second mode."}),
        encoding="utf-8",
    )
    (directory / "03_third.yaml").write_text(
        "name: third\ninstructions: Third mode.\n", encoding="utf-8"
    )
    (directory / "04_fourth.py").write_text(
        "from aivia.writing import WritingModeBase\n\n"
        "class FourthMode(WritingModeBase):\n"
        '    name = "fourth"\n'
        '    instructions = "Fourth mode."\n',
        encoding="utf-8",
    )
    (directory / "ignored.ini").write_text("not a mode\n", encoding="utf-8")
    (directory / ".hidden.yaml").write_text(
        "name: hidden\ninstructions: Hidden.\n", encoding="utf-8"
    )

    registry = WritingModeRegistry()
    loaded = registry.load_directory(directory)

    assert [mode.name for mode in loaded] == ["first", "second", "third", "fourth"]
    assert {mode.name for mode in loaded} == {"first", "second", "third", "fourth"}
    assert registry.resolve("first").instructions == "First mode."
    assert registry.resolve("first-alias") is registry.resolve("first")
    assert registry.resolve("second").instructions == "Second mode."
    assert registry.resolve("third").instructions == "Third mode."
    assert registry.resolve("fourth").instructions == "Fourth mode."
    assert "hidden" not in registry
    assert "ignored" not in registry
    assert registry.load_dir(directory) == loaded
    assert registry.from_directory(directory) == loaded
    assert registry.load_modes(directory) == loaded
    assert registry.load_from_directory(directory) == loaded


def test_registry_loads_the_configured_modes_directory_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "aivia" / "modes"
    directory.mkdir(parents=True)
    (directory / "configured.yaml").write_text(
        "name: configured\ninstructions: From the config directory.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    registry = WritingModeRegistry()
    loaded = registry.load_directory()

    assert [mode.name for mode in loaded] == ["configured"]
    assert registry.resolve("configured").instructions == ("From the config directory.")


def test_registry_directory_rejects_duplicate_names_and_rolls_back(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "modes"
    directory.mkdir()
    (directory / "first.yaml").write_text(
        "name: duplicated\ninstructions: First.\n", encoding="utf-8"
    )
    (directory / "second.yaml").write_text(
        "name: duplicated\ninstructions: Second.\n", encoding="utf-8"
    )
    registry = WritingModeRegistry()

    with pytest.raises(ConfigError, match="duplicate") as caught:
        registry.load_directory(directory)

    assert caught.value.details["mode"] == "duplicated"
    assert caught.value.details["existing"] == str(directory / "first.yaml")
    assert "duplicated" not in registry


def test_registry_directory_rejects_duplicate_aliases_and_rolls_back(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "modes"
    directory.mkdir()
    (directory / "first.yaml").write_text(
        "name: first\ninstructions: First.\naliases: [shared]\n",
        encoding="utf-8",
    )
    (directory / "second.yaml").write_text(
        "name: second\ninstructions: Second.\naliases: [shared]\n",
        encoding="utf-8",
    )
    registry = WritingModeRegistry()

    with pytest.raises(ConfigError, match="alias") as caught:
        registry.load_directory(directory)

    assert caught.value.details["alias"] == "shared"
    assert caught.value.details["existing"] == "first"
    assert "first" not in registry
    assert "second" not in registry


def test_registry_directory_rejects_a_name_already_registered(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "modes"
    directory.mkdir()
    (directory / "taken.yaml").write_text(
        "name: recording\ninstructions: Loaded.\n", encoding="utf-8"
    )
    registry = WritingModeRegistry()
    registry.register("recording", RecordingMode)

    with pytest.raises(ConfigError, match="duplicate") as caught:
        registry.load_directory(directory)

    assert caught.value.details["mode"] == "recording"
    assert registry.resolve("recording") is RecordingMode


def test_registry_rejects_a_duplicate_name_inside_one_data_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicates.yaml"
    path.write_text(
        "\n".join(
            [
                "modes:",
                "  first:",
                "    name: same-name",
                "    instructions: First.",
                "  second:",
                "    name: same-name",
                "    instructions: Second.",
            ]
        ),
        encoding="utf-8",
    )
    registry = WritingModeRegistry()

    with pytest.raises(ConfigError, match="duplicate") as caught:
        registry.load_file(path)

    assert caught.value.details["mode"] == "same-name"
    assert "same-name" not in registry


def test_registry_directory_must_exist_and_be_a_directory(tmp_path: Path) -> None:
    registry = WritingModeRegistry()

    with pytest.raises(ConfigError, match="does not exist"):
        registry.load_directory(tmp_path / "missing")

    file_path = tmp_path / "a_file.yaml"
    file_path.write_text("name: a-file\ninstructions: Text.\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not a directory"):
        registry.load_directory(file_path)
