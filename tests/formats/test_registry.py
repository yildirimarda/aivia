"""Stage registry: typed contracts, lookup and registration failures."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest

from aivia.core.errors import ConfigError
from aivia.core.stages import Stage
from aivia.formats import (
    StageContract,
    StageContractRegistry,
    StageRegistration,
    StageRegistry,
    contract_for,
    register_stage,
    registry,
    resolve_stage,
    stage_contract_registry,
    stage_registry,
)


class Topic:
    """A stand-in for the core topic model in a contract."""


class Script:
    """A stand-in for the core script model in a contract."""


class WriteStage(Stage):
    name = "write"
    contract = StageContract("write", inputs=(Topic,), outputs=(Script,))


class RenderStage(Stage):
    name = "render"
    inputs = ("timeline.json",)
    artifacts = ("video.mp4",)


class IncompleteStage(Stage):
    name = "incomplete"


def test_contract_keeps_typed_inputs_and_outputs() -> None:
    contract = StageContract("write", inputs=(Topic,), outputs=(Script,))

    assert contract.name == "write"
    assert contract.inputs == (Topic,)
    assert contract.outputs == (Script,)
    assert contract.input_names == ("Topic",)
    assert contract.output_names == ("Script",)
    assert contract.input_types == contract.inputs
    assert contract.output_types == contract.outputs
    assert contract.requires == contract.inputs
    assert contract.produces == contract.outputs
    assert contract.stage is None


def test_contract_accepts_names_single_values_and_aliases() -> None:
    contract = StageContract(
        "write",
        requires="fact_list",
        produces=["script"],
    )

    assert contract.inputs == ("fact_list",)
    assert contract.outputs == ("script",)
    assert contract.input_contracts == ("fact_list",)
    assert contract.output_contracts == ("script",)

    aliased = StageContract(
        "voice",
        input_types=("script",),
        output_types=("timeline", "audio"),
    )

    assert aliased.input_names == ("script",)
    assert aliased.output_names == ("timeline", "audio")


def test_contract_names_are_stripped_and_blanks_are_rejected() -> None:
    contract = StageContract("  write  ", inputs=" fact_list ", outputs=" script ")

    assert contract.name == "write"
    assert contract.inputs == ("fact_list",)
    assert contract.outputs == ("script",)

    with pytest.raises(ConfigError, match="stage name must not be empty"):
        StageContract("   ")

    with pytest.raises(ConfigError, match="input contract must not be empty"):
        StageContract("write", inputs=(" ",), outputs=("script",))


def test_contract_rejects_malformed_and_duplicate_entries() -> None:
    with pytest.raises(TypeError, match="types or strings"):
        StageContract("write", inputs=cast(Any, [42]), outputs=("script",))

    with pytest.raises(ConfigError, match="duplicate input contract: fact_list"):
        StageContract(
            "write",
            inputs=("fact_list", " fact_list "),
            outputs=("script",),
        )

    with pytest.raises(ConfigError, match="duplicate output contract: script"):
        StageContract(
            "write",
            inputs=("fact_list",),
            outputs=("script", "script"),
        )


def test_contract_can_be_built_from_a_stage_class() -> None:
    contract = StageContract.from_stage(WriteStage, module="writing")

    assert contract.name == "write"
    assert contract.inputs == (Topic,)
    assert contract.outputs == (Script,)
    assert contract.stage is WriteStage
    assert contract.module == "writing"


def test_contract_reads_base_stage_inputs_and_artifacts() -> None:
    contract = StageContract.from_stage(RenderStage)

    assert contract.inputs == ("timeline.json",)
    assert contract.outputs == ("video.mp4",)
    assert contract.stage is RenderStage


def test_contract_from_a_stage_without_declarations_fails() -> None:
    with pytest.raises(ConfigError, match="has no input or output contract"):
        StageContract.from_stage(IncompleteStage)


def test_registry_registers_a_contract_and_looks_it_up() -> None:
    stages = StageRegistry()
    contract = StageContract("write", inputs=("fact_list",), outputs=("script",))

    registration = stages.register(contract)

    assert isinstance(registration, StageRegistration)
    assert registration.name == "write"
    assert registration.contract is contract
    assert registration.stage is None
    assert registration.inputs == ("fact_list",)
    assert registration.outputs == ("script",)
    assert stages.get("write") is contract
    assert stages.resolve("write") is contract
    assert stages.resolve_stage("write") is contract
    assert stages.contract("write") is contract
    assert stages.contract_for("write") is contract
    assert stages.inputs("write") == ("fact_list",)
    assert stages.outputs("write") == ("script",)
    assert stages.names() == ("write",)
    assert stages.keys() == ("write",)
    assert stages.stage_names() == ("write",)
    assert "write" in stages
    assert "missing" not in stages
    assert 1 not in stages
    assert len(stages) == 1
    assert [entry.name for entry in stages] == ["write"]
    assert stages.contracts() == {"write": contract}
    assert stages.stages() == {"write": None}


def test_registry_registers_a_stage_class_and_its_contract() -> None:
    stages = StageRegistry()

    registration = stages.register(WriteStage, module="writing")

    assert registration.name == "write"
    assert registration.stage is WriteStage
    assert registration.module == "writing"
    assert registration.inputs == (Topic,)
    assert registration.outputs == (Script,)
    assert stages.get("write") is WriteStage
    assert stages.contract("write").inputs == (Topic,)
    assert stages.stages() == {"write": WriteStage}


def test_registry_reads_contracts_from_the_base_stage_declarations() -> None:
    stages = StageRegistry()

    stages.register(RenderStage)

    assert stages.contract("render").input_names == ("timeline.json",)
    assert stages.contract("render").output_names == ("video.mp4",)


def test_registry_supports_explicit_contract_values() -> None:
    stages = StageRegistry()

    registration = cast(
        StageRegistration,
        stages.register(
            "write",
            inputs=("fact_list",),
            outputs=("script",),
        ),
    )

    assert registration.contract.input_names == ("fact_list",)
    assert registration.contract.output_names == ("script",)


def test_registry_supports_the_decorator_form() -> None:
    stages = StageRegistry()

    decorator = cast(
        Callable[[type[Stage]], type[Stage]],
        stages.register("custom"),
    )

    @decorator
    class CustomStage(Stage):
        name = "custom"
        artifacts = ("custom.json",)

    assert stages.get("custom") is CustomStage
    assert stages.contract("custom").outputs == ("custom.json",)


def test_registry_rejects_duplicate_stage_names() -> None:
    stages = StageRegistry()
    stages.register("write", inputs=("fact_list",), outputs=("script",))

    with pytest.raises(ConfigError, match="already registered: write") as caught:
        stages.register("write", inputs=("topic",), outputs=("script",))

    assert caught.value.details["stage"] == "write"
    assert caught.value.details["existing"] == "write"
    assert caught.value.details["new"] == "write"


def test_registry_rejects_the_same_stage_twice() -> None:
    stages = StageRegistry()
    stages.register(WriteStage)

    with pytest.raises(ConfigError, match="already registered: write"):
        stages.register(WriteStage)


def test_registry_rejects_a_stage_without_a_contract() -> None:
    stages = StageRegistry()

    with pytest.raises(ConfigError, match="has no input or output contract"):
        stages.register(IncompleteStage)


def test_registry_rejects_an_empty_contract() -> None:
    stages = StageRegistry()

    with pytest.raises(ConfigError, match="has no input or output contract"):
        stages.register(StageContract("write"))


def test_registry_rejects_non_stage_classes() -> None:
    stages = StageRegistry()

    with pytest.raises(TypeError, match="Stage subclass"):
        stages.register("bad", cast(Any, dict))


def test_registry_rejects_malformed_stage_names() -> None:
    stages = StageRegistry()

    with pytest.raises(TypeError, match="stage name must be a string"):
        stages.register(cast(Any, 1), inputs=("a",), outputs=("b",))

    with pytest.raises(ConfigError, match="invalid stage name"):
        stages.register("two words", inputs=("a",), outputs=("b",))


def test_unknown_stage_names_the_registered_stages() -> None:
    stages = StageRegistry()
    stages.register("write", inputs=("fact_list",), outputs=("script",))

    with pytest.raises(ConfigError) as caught:
        stages.resolve("render")

    message = str(caught.value)
    assert "unknown stage: render" in message
    assert "registered: write" in message
    assert caught.value.details["stage"] == "render"
    assert caught.value.details["available"] == ["write"]


def test_registry_unregisters_a_stage() -> None:
    stages = StageRegistry()
    stages.register(WriteStage)

    stages.unregister("write")

    assert stages.names() == ()
    with pytest.raises(ConfigError, match="unknown stage: write"):
        stages.unregister("write")


def test_shared_registry_aliases_and_helpers_agree() -> None:
    assert StageContractRegistry is StageRegistry
    assert stage_contract_registry is stage_registry
    assert registry is stage_registry
    assert register_stage.__self__ is stage_registry  # type: ignore[attr-defined]
    assert resolve_stage.__self__ is stage_registry  # type: ignore[attr-defined]
    assert contract_for.__self__ is stage_registry  # type: ignore[attr-defined]

    decorator = cast(
        Callable[[type[Stage]], type[Stage]],
        register_stage("shared-write"),
    )

    @decorator
    class SharedWriteStage(Stage):
        name = "shared-write"
        contract = StageContract(
            "shared-write",
            inputs=("fact_list",),
            outputs=("script",),
        )

    try:
        assert stage_registry.get("shared-write") is SharedWriteStage
        assert resolve_stage("shared-write") is SharedWriteStage
        assert contract_for("shared-write").outputs == ("script",)
        assert "shared-write" in stage_registry
    finally:
        stage_registry.unregister("shared-write")

    with pytest.raises(ConfigError, match="unknown stage: shared-write"):
        stage_registry.resolve("shared-write")
