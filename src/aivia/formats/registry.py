"""Registry of pipeline stages and their typed input/output contracts.

A format recipe names stages in order. Before a recipe can be checked, the
system needs to know what each stage consumes and produces. This module keeps
that information in a :class:`StageContract`: a stable stage name plus the
typed values the stage reads (``inputs``) and writes (``outputs``). A contract
may also point at the stage class that implements it.

The registry is deliberately strict at registration time. Two stages cannot
claim the same name, a stage cannot be registered without a contract, and a
contract cannot declare the same input or output twice. Failing early keeps a
recipe typo or an incomplete stage from turning into a confusing runtime
failure.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, cast, overload

from aivia.core.errors import ConfigError
from aivia.core.stages import Stage

__all__ = [
    "Contract",
    "ContractType",
    "InputContract",
    "OutputContract",
    "Registry",
    "StageContract",
    "StageContractRegistration",
    "StageContractRegistry",
    "StageDefinition",
    "StageRegistration",
    "StageRegistry",
    "StageSpec",
    "contract_for",
    "register_stage",
    "registry",
    "resolve_stage",
    "stage_contract_registry",
    "stage_registry",
]

ContractType = str | type[Any]
"""An input or output contract: a type or a stable name for one."""

InputContract = ContractType
OutputContract = ContractType

_UNSET = object()
_INPUT_KEYS = (
    "inputs",
    "input",
    "input_contract",
    "input_contracts",
    "input_type",
    "input_types",
    "requires",
    "consumes",
    "accepts",
)
_OUTPUT_KEYS = (
    "outputs",
    "output",
    "output_contract",
    "output_contracts",
    "output_type",
    "output_types",
    "produces",
    "artifacts",
    "results",
)


def _validate_name(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("stage name must be a string")
    name = value.strip()
    if not name:
        raise ConfigError("stage name must not be empty")
    if any(character.isspace() for character in name):
        raise ConfigError(f"invalid stage name: {value!r}")
    return name


def _contract_key(value: ContractType) -> str:
    if isinstance(value, str):
        return value
    return value.__name__


def _pick(values: tuple[tuple[str, Any], ...]) -> Any:
    for _, value in values:
        if value is not _UNSET and value is not None:
            return value
    return _UNSET


def _normalise_contracts(
    value: Any,
    *,
    kind: str,
) -> tuple[ContractType, ...]:
    if value is _UNSET or value is None:
        return ()
    if isinstance(value, (str, type)):
        value = (value,)
    try:
        values = tuple(value)
    except TypeError:
        raise TypeError(
            f"{kind} contracts must be a type, a string or an iterable of them"
        ) from None
    contracts: list[ContractType] = []
    seen: set[str] = set()
    for item in values:
        if isinstance(item, str):
            text = item.strip()
            if not text:
                raise ConfigError(f"{kind} contract must not be empty")
            contract: ContractType = text
        elif isinstance(item, type):
            contract = item
        else:
            raise TypeError(
                f"{kind} contracts must be types or strings, not {type(item).__name__}"
            )
        key = _contract_key(contract)
        if key in seen:
            raise ConfigError(f"duplicate {kind} contract: {key}")
        seen.add(key)
        contracts.append(contract)
    return tuple(contracts)


def _coerce_stage(value: Any) -> type[Any] | None:
    if value is None:
        return None
    if isinstance(value, Stage):
        return type(value)
    if isinstance(value, type) and issubclass(value, Stage):
        return value
    raise TypeError("stage must be a Stage subclass")


def _stage_attr(stage: type[Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if hasattr(stage, key):
            return getattr(stage, key)
    return _UNSET


def _contract_from_stage(
    name: str,
    stage: type[Any],
    *,
    module: str | None,
    source: str | None,
    description: str | None,
) -> StageContract | None:
    declared = getattr(stage, "contract", None)
    if declared is not None:
        if isinstance(declared, StageContract):
            return StageContract(
                name or declared.name,
                declared.inputs,
                declared.outputs,
                stage=stage,
                module=module or declared.module,
                source=source or declared.source,
                description=(
                    declared.description if description is None else description
                ),
            )
        if isinstance(declared, Mapping):
            values = dict(declared)
            values.setdefault("name", name)
            values.setdefault("stage", stage)
            values.setdefault("module", module)
            values.setdefault("source", source)
            if description is not None:
                values.setdefault("description", description)
            return StageContract(**values)
        raise TypeError(f"stage {name!r} contract must be a StageContract or mapping")

    inputs = _stage_attr(stage, _INPUT_KEYS)
    outputs = _stage_attr(stage, _OUTPUT_KEYS)
    if inputs is _UNSET and outputs is _UNSET:
        return None
    contract = StageContract(
        name,
        () if inputs is _UNSET else inputs,
        () if outputs is _UNSET else outputs,
        stage=stage,
        module=module,
        source=source,
        description=description or "",
    )
    if not contract.inputs and not contract.outputs:
        return None
    return contract


@dataclass(frozen=True, slots=True, init=False)
class StageContract:
    """The typed inputs and outputs one stage declares.

    ``inputs`` and ``outputs`` accept a single type or name, or an iterable of
    them. A type is kept as-is so callers can reason about the model a stage
    exchanges; a string is a stable contract name. :attr:`input_names` and
    :attr:`output_names` provide comparable names for recipe validation.
    ``stage`` optionally links the contract to its implementation class.
    """

    name: str
    inputs: tuple[ContractType, ...]
    outputs: tuple[ContractType, ...]
    stage: type[Any] | None
    module: str | None
    source: str | None
    description: str

    def __init__(
        self,
        name: str | type[Any] | Stage | None = None,
        inputs: Any = _UNSET,
        outputs: Any = _UNSET,
        *,
        input: Any = _UNSET,
        input_contract: Any = _UNSET,
        input_contracts: Any = _UNSET,
        input_type: Any = _UNSET,
        input_types: Any = _UNSET,
        requires: Any = _UNSET,
        consumes: Any = _UNSET,
        accepts: Any = _UNSET,
        output: Any = _UNSET,
        output_contract: Any = _UNSET,
        output_contracts: Any = _UNSET,
        output_type: Any = _UNSET,
        output_types: Any = _UNSET,
        produces: Any = _UNSET,
        results: Any = _UNSET,
        stage: Any = None,
        cls: Any = None,
        factory: Any = None,
        module: str | None = None,
        source: str | None = None,
        description: str = "",
    ) -> None:
        if name is not None and not isinstance(name, str):
            if isinstance(name, (Stage, type)):
                if stage is None and cls is None and factory is None:
                    stage = name
                name = getattr(name, "name", "") or type(name).__name__
            else:
                raise TypeError("stage contract name must be a string")

        selected_stage = _pick((("stage", stage), ("cls", cls), ("factory", factory)))
        stage_class = _coerce_stage(
            None if selected_stage is _UNSET else selected_stage
        )

        if name is None:
            name = getattr(stage_class, "name", "") if stage_class is not None else ""
        validated_name = _validate_name(name)

        selected_inputs = _pick(
            (
                ("inputs", inputs),
                ("input", input),
                ("input_contract", input_contract),
                ("input_contracts", input_contracts),
                ("input_type", input_type),
                ("input_types", input_types),
                ("requires", requires),
                ("consumes", consumes),
                ("accepts", accepts),
            )
        )
        selected_outputs = _pick(
            (
                ("outputs", outputs),
                ("output", output),
                ("output_contract", output_contract),
                ("output_contracts", output_contracts),
                ("output_type", output_type),
                ("output_types", output_types),
                ("produces", produces),
                ("results", results),
            )
        )

        object.__setattr__(self, "name", validated_name)
        object.__setattr__(
            self,
            "inputs",
            _normalise_contracts(selected_inputs, kind="input"),
        )
        object.__setattr__(
            self,
            "outputs",
            _normalise_contracts(selected_outputs, kind="output"),
        )
        object.__setattr__(self, "stage", stage_class)
        object.__setattr__(self, "module", module)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "description", description)

    @classmethod
    def from_stage(
        cls,
        stage: type[Any] | Stage,
        *,
        name: str | None = None,
        module: str | None = None,
        source: str | None = None,
        description: str | None = None,
    ) -> StageContract:
        """Build a contract from a stage class or instance.

        A stage may declare a ``contract`` attribute, explicit contract
        attributes (``input_contracts``/``output_contracts`` and their
        aliases), or the base stage ``inputs``/``artifacts`` declarations. A
        stage that declares none of these has no contract and is rejected.
        """
        stage_class = _coerce_stage(stage)
        if stage_class is None:
            raise TypeError("stage must be a Stage subclass")
        resolved_name = name or getattr(stage_class, "name", "")
        if not resolved_name:
            raise ConfigError(
                "stage has no name",
                details={"stage": stage_class.__name__},
            )
        contract = _contract_from_stage(
            _validate_name(resolved_name),
            stage_class,
            module=module,
            source=source,
            description=description,
        )
        if contract is None:
            raise ConfigError(
                f"stage {resolved_name!r} has no input or output contract",
                details={"stage": resolved_name, "class": stage_class.__name__},
            )
        return contract

    @property
    def input_names(self) -> tuple[str, ...]:
        """The declared input contracts as comparable names."""
        return tuple(_contract_key(value) for value in self.inputs)

    @property
    def output_names(self) -> tuple[str, ...]:
        """The declared output contracts as comparable names."""
        return tuple(_contract_key(value) for value in self.outputs)

    @property
    def input(self) -> tuple[ContractType, ...]:
        """Alias for :attr:`inputs`."""
        return self.inputs

    @property
    def output(self) -> tuple[ContractType, ...]:
        """Alias for :attr:`outputs`."""
        return self.outputs

    @property
    def input_types(self) -> tuple[ContractType, ...]:
        """Alias for :attr:`inputs`."""
        return self.inputs

    @property
    def output_types(self) -> tuple[ContractType, ...]:
        """Alias for :attr:`outputs`."""
        return self.outputs

    @property
    def input_contracts(self) -> tuple[ContractType, ...]:
        """Alias for :attr:`inputs`."""
        return self.inputs

    @property
    def output_contracts(self) -> tuple[ContractType, ...]:
        """Alias for :attr:`outputs`."""
        return self.outputs

    @property
    def requires(self) -> tuple[ContractType, ...]:
        """Alias for :attr:`inputs`."""
        return self.inputs

    @property
    def produces(self) -> tuple[ContractType, ...]:
        """Alias for :attr:`outputs`."""
        return self.outputs

    @property
    def cls(self) -> type[Any] | None:
        """Alias for :attr:`stage`."""
        return self.stage


@dataclass(frozen=True, slots=True)
class StageRegistration:
    """One stage name bound to its contract and implementation class."""

    name: str
    contract: StageContract
    stage: type[Any] | None = None
    module: str | None = None
    source: str | None = None

    @property
    def inputs(self) -> tuple[ContractType, ...]:
        """The contract's inputs."""
        return self.contract.inputs

    @property
    def outputs(self) -> tuple[ContractType, ...]:
        """The contract's outputs."""
        return self.contract.outputs

    @property
    def input_names(self) -> tuple[str, ...]:
        """The contract's input names."""
        return self.contract.input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        """The contract's output names."""
        return self.contract.output_names

    @property
    def input_contracts(self) -> tuple[ContractType, ...]:
        """Alias for :attr:`inputs`."""
        return self.inputs

    @property
    def output_contracts(self) -> tuple[ContractType, ...]:
        """Alias for :attr:`outputs`."""
        return self.outputs

    @property
    def cls(self) -> type[Any] | None:
        """Alias for :attr:`stage`."""
        return self.stage

    @property
    def provider(self) -> type[Any] | None:
        """Alias for :attr:`stage`."""
        return self.stage

    @property
    def factory(self) -> type[Any] | None:
        """Alias for :attr:`stage`."""
        return self.stage


class StageRegistry:
    """An ordered, collision-checked collection of stage contracts."""

    def __init__(self) -> None:
        self._stages: dict[str, StageRegistration] = {}

    @overload
    def register(
        self,
        name: str,
        stage: Any = None,
        *,
        contract: StageContract | Mapping[str, Any] | None = None,
        inputs: Any = _UNSET,
        outputs: Any = _UNSET,
        module: str | None = None,
        source: str | None = None,
        description: str | None = None,
    ) -> StageRegistration | Callable[[Any], Any]: ...

    @overload
    def register(
        self,
        name: StageContract | type[Any] | Stage,
        stage: Any = None,
        *,
        contract: StageContract | Mapping[str, Any] | None = None,
        inputs: Any = _UNSET,
        outputs: Any = _UNSET,
        module: str | None = None,
        source: str | None = None,
        description: str | None = None,
    ) -> StageRegistration: ...

    def register(
        self,
        name: str | StageContract | type[Any] | Stage,
        stage: Any = None,
        *,
        contract: StageContract | Mapping[str, Any] | None = None,
        inputs: Any = _UNSET,
        outputs: Any = _UNSET,
        module: str | None = None,
        source: str | None = None,
        description: str | None = None,
    ) -> StageRegistration | Callable[[Any], Any]:
        """Register a stage contract or a stage class with its contract.

        The first argument may be a contract, a stage class or a stage name.
        With only a name, the call returns a decorator so a stage can register
        itself. A contract may be supplied explicitly or read from the stage
        class; a stage that declares neither is rejected.
        """
        contract_value = contract
        stage_value = stage
        explicit_inputs = inputs
        explicit_outputs = outputs

        if isinstance(name, Mapping):
            if contract_value is not None:
                raise TypeError("stage contract was supplied twice")
            contract_value = name
            name_value = str(name.get("name", ""))
            if stage_value is None:
                stage_value = name.get("stage")
        elif isinstance(name, StageContract):
            if contract_value is not None:
                raise TypeError("stage contract was supplied twice")
            contract_value = name
            name_value = name.name
            if stage_value is None:
                stage_value = name.stage
        elif isinstance(name, (Stage, type)):
            if stage_value is not None:
                raise TypeError("stage class was supplied twice")
            stage_value = name
            name_value = getattr(name, "name", "") or type(name).__name__
        elif isinstance(name, str):
            name_value = name
            if (
                stage_value is None
                and contract_value is None
                and explicit_inputs is _UNSET
                and explicit_outputs is _UNSET
            ):

                def decorator(value: Any) -> Any:
                    self.register(
                        name_value,
                        value,
                        module=module,
                        source=source,
                        description=description,
                    )
                    return value

                return decorator
        else:
            raise TypeError("stage name must be a string")

        validated_name = _validate_name(name_value)

        if isinstance(stage_value, (StageContract, Mapping)):
            if contract_value is not None:
                raise TypeError("stage contract was supplied twice")
            contract_value = stage_value
            stage_value = (
                stage_value.stage if isinstance(stage_value, StageContract) else None
            )

        stage_class = _coerce_stage(stage_value)
        built_from_values = False
        if (
            contract_value is None
            and stage_class is None
            and (explicit_inputs is not _UNSET or explicit_outputs is not _UNSET)
        ):
            contract_value = StageContract(
                validated_name,
                explicit_inputs,
                explicit_outputs,
                module=module,
                source=source,
                description=description or "",
            )
            built_from_values = True

        if isinstance(contract_value, Mapping):
            values = dict(contract_value)
            values.setdefault("name", validated_name)
            values.setdefault("stage", stage_class)
            values.setdefault("module", module)
            values.setdefault("source", source)
            if description is not None:
                values.setdefault("description", description)
            contract_value = StageContract(**values)
        elif contract_value is not None and not isinstance(
            contract_value, StageContract
        ):
            raise TypeError("stage contract must be a StageContract or mapping")

        if contract_value is None:
            if stage_class is None:
                raise ConfigError(
                    f"stage {validated_name!r} has no contract",
                    details={"stage": validated_name},
                )
            contract_value = _contract_from_stage(
                validated_name,
                stage_class,
                module=module,
                source=source,
                description=description,
            )
            if contract_value is None:
                raise ConfigError(
                    f"stage {validated_name!r} has no input or output contract",
                    details={
                        "stage": validated_name,
                        "class": stage_class.__name__,
                    },
                )
        elif not built_from_values and (
            explicit_inputs is not _UNSET or explicit_outputs is not _UNSET
        ):
            raise TypeError(
                "supply either a stage contract or inputs/outputs, not both"
            )

        if stage_class is None:
            stage_class = contract_value.stage

        if not contract_value.inputs and not contract_value.outputs:
            raise ConfigError(
                f"stage {validated_name!r} has no input or output contract",
                details={
                    "stage": validated_name,
                    "class": (
                        stage_class.__name__ if stage_class is not None else None
                    ),
                },
            )

        existing = self._stages.get(validated_name)
        if existing is not None:
            raise ConfigError(
                f"stage already registered: {validated_name}",
                details={
                    "stage": validated_name,
                    "existing": _stage_label(existing),
                    "new": _stage_label(
                        StageRegistration(
                            name=validated_name,
                            contract=contract_value,
                            stage=stage_class,
                        )
                    ),
                },
            )

        registration = StageRegistration(
            name=validated_name,
            contract=contract_value,
            stage=stage_class,
            module=module or contract_value.module,
            source=source or contract_value.source,
        )
        self._stages[validated_name] = registration
        return registration

    def register_stage(
        self,
        name: str | StageContract | type[Any] | Stage,
        stage: Any = None,
        *,
        contract: StageContract | Mapping[str, Any] | None = None,
        inputs: Any = _UNSET,
        outputs: Any = _UNSET,
        module: str | None = None,
        source: str | None = None,
        description: str | None = None,
    ) -> StageRegistration | Callable[[Any], Any]:
        """Alias for :meth:`register`."""
        return cast(
            StageRegistration | Callable[[Any], Any],
            self.register(
                cast(Any, name),
                stage,
                contract=contract,
                inputs=inputs,
                outputs=outputs,
                module=module,
                source=source,
                description=description,
            ),
        )

    def unregister(self, name: str) -> None:
        """Remove the stage registered under *name*."""
        validated_name = _validate_name(name)
        if validated_name not in self._stages:
            raise self._unknown(validated_name)
        del self._stages[validated_name]

    def get(self, name: str) -> Any:
        """Return the stage class, or the contract for a contract-only entry."""
        registration = self.registration(name)
        return (
            registration.stage
            if registration.stage is not None
            else registration.contract
        )

    def resolve(self, name: str) -> Any:
        """Alias for :meth:`get`."""
        return self.get(name)

    def resolve_stage(self, name: str) -> Any:
        """Alias for :meth:`get`."""
        return self.get(name)

    def registration(self, name: str) -> StageRegistration:
        """Return the full registration for *name*."""
        validated_name = _validate_name(name)
        registration = self._stages.get(validated_name)
        if registration is None:
            raise self._unknown(validated_name)
        return registration

    def contract(self, name: str) -> StageContract:
        """Return the contract registered under *name*."""
        return self.registration(name).contract

    def contract_for(self, name: str) -> StageContract:
        """Alias for :meth:`contract`."""
        return self.contract(name)

    def inputs(self, name: str) -> tuple[ContractType, ...]:
        """Return the inputs declared by the stage registered under *name*."""
        return self.contract(name).inputs

    def outputs(self, name: str) -> tuple[ContractType, ...]:
        """Return the outputs declared by the stage registered under *name*."""
        return self.contract(name).outputs

    def stage(self, name: str) -> type[Any] | StageContract:
        """Return the implementation class or contract for *name*."""
        return cast(type[Any] | StageContract, self.get(name))

    def names(self) -> tuple[str, ...]:
        """Return the registered stage names in sorted order."""
        return tuple(sorted(self._stages))

    def keys(self) -> tuple[str, ...]:
        """Alias for :meth:`names`."""
        return self.names()

    def stage_names(self) -> tuple[str, ...]:
        """Alias for :meth:`names`."""
        return self.names()

    def registrations(self) -> tuple[StageRegistration, ...]:
        """Return the registrations in sorted name order."""
        return tuple(self._stages[name] for name in self.names())

    def contracts(self) -> dict[str, StageContract]:
        """Return a copy of the name-to-contract mapping in sorted order."""
        return {name: self._stages[name].contract for name in self.names()}

    def stages(self) -> dict[str, type[Any] | None]:
        """Return a copy of the name-to-class mapping in sorted order."""
        return {name: self._stages[name].stage for name in self.names()}

    def _unknown(self, name: str) -> ConfigError:
        known = ", ".join(self.names()) or "none"
        return ConfigError(
            f"unknown stage: {name} (registered: {known})",
            details={"stage": name, "available": list(self.names())},
        )

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return name.strip() in self._stages

    def __getitem__(self, name: str) -> Any:
        return self.get(name)

    def __iter__(self) -> Iterator[StageRegistration]:
        return iter(self.registrations())

    def __len__(self) -> int:
        return len(self._stages)


def _stage_label(registration: StageRegistration) -> str:
    if registration.stage is not None:
        return registration.stage.__name__
    return registration.contract.name


stage_registry = StageRegistry()
"""The shared registry of pipeline stages."""

stage_contract_registry = stage_registry
registry = stage_registry
StageContractRegistry = StageRegistry
Contract = StageContract
StageSpec = StageContract
StageDefinition = StageContract
Registry = StageRegistry
StageContractRegistration = StageRegistration

register_stage = stage_registry.register
"""Register a stage on the shared :data:`stage_registry`."""

resolve_stage = stage_registry.resolve
"""Resolve a stage name through the shared registry."""

contract_for = stage_registry.contract
"""Return the contract for a stage name from the shared registry."""
