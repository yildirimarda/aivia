"""Writing modes, their base class and registry.

A writing mode turns a brief, a fact list or user supplied text into a script
payload. The registry maps a profile name such as ``facts`` to the class that
implements it. A custom mode may be defined by a Python file, a YAML/JSON/TOML
definition or a plain text instruction file, so a user can add a mode without
changing the package.

The registry starts with no registrations. The built in names resolve to a
small base mode that only carries the name and its payload model; later mode
modules replace them with their full implementation. Custom modes are loaded
lazily and cached by path and by name, so resolving the same file twice returns
the same class.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast, overload, runtime_checkable

import yaml

from aivia.core.errors import ConfigError
from aivia.writing.schema import (
    CustomPayload,
    ModePayload,
    WritingMode,
    payload_model_for_mode,
)

__all__ = [
    "BaseWritingMode",
    "BuiltinWritingMode",
    "CustomModeDefinition",
    "ModeBase",
    "ModeRegistry",
    "WritingModeBase",
    "WritingModeLike",
    "WritingModeProtocol",
    "WritingModeRegistration",
    "WritingModeRegistry",
    "builtin_writing_mode",
    "is_known_writing_mode",
    "load_writing_mode",
    "load_writing_mode_file",
    "mode_registry",
    "register_mode",
    "register_writing_mode",
    "resolve_mode",
    "resolve_writing_mode",
    "writing_mode_registry",
]

_BUILTIN_MODULE = __name__
_MODE_FILE_SUFFIXES = {".json", ".toml", ".yaml", ".yml", ".md", ".markdown", ".txt"}
_DATA_SUFFIXES = {".json", ".toml", ".yaml", ".yml"}


@runtime_checkable
class WritingModeProtocol(Protocol):
    """The structural contract every writing mode satisfies."""

    name: str

    def build_payload(self, value: Any = None) -> ModePayload:
        """Return *value* validated as this mode's payload."""


class WritingModeBase:
    """Convenience base for a writing mode.

    Subclasses set :attr:`name` and optionally override :attr:`payload_model`,
    :attr:`description` and :attr:`instructions`. :meth:`build_payload` keeps
    the payload typed for the mode while accepting the same shorthands as the
    script schema.
    """

    name: str = ""
    description: str = ""
    instructions: str = ""
    payload_model: type[ModePayload] = CustomPayload
    aliases: tuple[str, ...] = ()
    default_payload: Mapping[str, Any] = MappingProxyType({})

    def __init__(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
        instructions: str | None = None,
        payload_model: type[ModePayload] | None = None,
        aliases: Sequence[str] | None = None,
        default_payload: Mapping[str, Any] | None = None,
    ) -> None:
        if name is not None:
            self.name = _validate_name(name)
        if description is not None:
            self.description = description
        if instructions is not None:
            self.instructions = instructions
        if payload_model is not None:
            self.payload_model = payload_model
        if aliases is not None:
            self.aliases = tuple(aliases)
        if default_payload is not None:
            self.default_payload = dict(default_payload)

    @property
    def mode(self) -> str:
        """Alias for :attr:`name`."""
        return self.name

    @property
    def payload_type(self) -> type[ModePayload]:
        """Alias for :attr:`payload_model`."""
        return self.payload_model

    @property
    def payload_class(self) -> type[ModePayload]:
        """Alias for :attr:`payload_model`."""
        return self.payload_model

    def build_payload(self, value: Any = None) -> ModePayload:
        """Validate *value* as this mode's payload."""
        model = self.payload_model
        if value is None:
            return model.model_validate(dict(self.default_payload))
        if isinstance(value, model):
            return value
        if isinstance(value, ModePayload):
            return model.model_validate(value.model_dump(mode="python"))
        if isinstance(value, Mapping):
            values = dict(self.default_payload)
            values.update(value)
            return model.model_validate(values)
        return model.model_validate(value)

    def build(self, value: Any = None) -> ModePayload:
        """Alias for :meth:`build_payload`."""
        return self.build_payload(value)

    def payload_for(self, value: Any = None) -> ModePayload:
        """Alias for :meth:`build_payload`."""
        return self.build_payload(value)

    def validate_payload(self, value: Any = None) -> ModePayload:
        """Alias for :meth:`build_payload`."""
        return self.build_payload(value)

    def to_dict(self) -> dict[str, Any]:
        """Return the mode definition as a JSON-compatible mapping."""
        return {
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "aliases": list(self.aliases),
            "default_payload": dict(self.default_payload),
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


BaseWritingMode = WritingModeBase
ModeBase = WritingModeBase

WritingModeLike = type[WritingModeBase] | WritingModeBase


class BuiltinWritingMode(WritingModeBase):
    """The placeholder implementation for a built in mode name."""

    def __init__(self, name: WritingMode | str | None = None) -> None:
        selected = name if name is not None else type(self).name
        mode = selected if isinstance(selected, WritingMode) else WritingMode(selected)
        super().__init__(mode.value, payload_model=payload_model_for_mode(mode))


_BUILTIN_MODES: dict[str, type[WritingModeBase]] = {}
_KNOWN_CUSTOM_MODES: set[str] = set()


def is_known_writing_mode(name: str) -> bool:
    """Return whether *name* is a built in or an already registered mode."""
    if not isinstance(name, str):
        return False
    cleaned = name.strip().lower()
    if not cleaned:
        return False
    try:
        WritingMode(cleaned)
    except ValueError:
        return cleaned in _KNOWN_CUSTOM_MODES
    return True


def builtin_writing_mode(name: WritingMode | str) -> type[WritingModeBase]:
    """Return the placeholder class for a built in mode name."""
    mode = name if isinstance(name, WritingMode) else WritingMode(name)
    cached = _BUILTIN_MODES.get(mode.value)
    if cached is not None:
        return cached
    class_name = "".join(part.title() for part in mode.value.split("_")) + "Mode"

    def __init__(self: BuiltinWritingMode) -> None:
        BuiltinWritingMode.__init__(self, mode)

    mode_class = type(
        class_name,
        (BuiltinWritingMode,),
        {
            "__module__": _BUILTIN_MODULE,
            "name": mode.value,
            "payload_model": payload_model_for_mode(mode),
            "__init__": __init__,
        },
    )
    _BUILTIN_MODES[mode.value] = mode_class
    return mode_class


@dataclass(frozen=True, slots=True)
class WritingModeRegistration:
    """One writing mode registered under a profile name."""

    name: str
    mode: WritingModeLike
    module: str | None = None
    source: str | None = None

    @property
    def provider(self) -> WritingModeLike:
        """Alias for :attr:`mode`."""
        return self.mode

    @property
    def cls(self) -> WritingModeLike:
        """Alias for :attr:`mode`."""
        return self.mode

    @property
    def instructions(self) -> str:
        """The instructions carried by the registered mode, if any."""
        return getattr(self.mode, "instructions", "")

    @property
    def path(self) -> str | None:
        """Alias for :attr:`source`."""
        return self.source


@dataclass(frozen=True, slots=True)
class CustomModeDefinition:
    """The fields a custom mode file may define."""

    name: str
    instructions: str = ""
    description: str = ""
    aliases: tuple[str, ...] = ()
    default_payload: dict[str, Any] | None = None


def _validate_name(name: object) -> str:
    if not isinstance(name, str):
        raise TypeError("writing mode name must be a string")
    stripped = name.strip()
    if not stripped:
        raise ConfigError("writing mode name must not be empty")
    if any(character.isspace() for character in stripped):
        raise ConfigError(f"invalid writing mode name: {name!r}")
    return stripped


def _validate_mode(
    name: str,
    mode: object,
    *,
    source: str = "writing mode",
) -> WritingModeLike:
    if isinstance(mode, WritingModeBase):
        return mode
    if not isinstance(mode, type):
        raise TypeError(f"{source} {name!r} must be a class or mode instance")
    if issubclass(mode, WritingModeBase):
        return mode
    if callable(getattr(mode, "build_payload", None)) or callable(
        getattr(mode, "build", None)
    ):
        return cast(type[WritingModeBase], mode)
    raise TypeError(
        f"{source} {name!r} must subclass WritingModeBase or define build_payload()"
    )


def _mode_name(mode: WritingModeLike) -> str:
    return str(getattr(mode, "name", ""))


def _is_dotted_path(value: str) -> bool:
    return ":" in value or "." in value


def _split_path(value: str) -> tuple[str, str]:
    if ":" in value:
        module_name, _, attribute = value.partition(":")
    else:
        module_name, _, attribute = value.rpartition(".")
    if not module_name or not attribute:
        raise ConfigError(
            f"invalid writing mode path: {value!r} "
            "(expected 'module:Class' or 'module.Class')"
        )
    return module_name, attribute


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"cannot read writing mode file {str(path)!r}: {exc}",
            details={"path": str(path)},
        ) from exc


def _parse_data(text: str, suffix: str) -> Any:
    if suffix == ".json":
        try:
            return json.loads(text)
        except ValueError as exc:
            raise ConfigError(f"invalid writing mode JSON: {exc}") from exc
    if suffix == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError as exc:
            raise ConfigError("TOML writing mode files need Python 3.11+") from exc
        try:
            return tomllib.loads(text)
        except ValueError as exc:
            raise ConfigError(f"invalid writing mode TOML: {exc}") from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid writing mode YAML: {exc}") from exc


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _aliases_value(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _definition_from_mapping(
    data: Mapping[str, Any],
    *,
    default_name: str | None = None,
) -> CustomModeDefinition:
    values = dict(data)
    name = (
        values.pop("name", None)
        or values.pop("id", None)
        or values.pop("mode", None)
        or default_name
    )
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("custom writing mode file must define a name")
    instructions = _text_value(
        values.pop("instructions", None)
        or values.pop("prompt", None)
        or values.pop("guidance", None)
        or values.pop("rules", None)
    )
    description = _text_value(values.pop("description", None))
    aliases = _aliases_value(values.pop("aliases", None) or values.pop("alias", None))
    default_payload = values.pop("payload", None) or values.pop("defaults", None)
    if default_payload is None:
        default_payload = values.pop("default_payload", None)
    if default_payload is not None and not isinstance(default_payload, Mapping):
        raise ConfigError("custom writing mode payload must be a mapping")
    fields = values.pop("fields", None)
    if fields is not None and default_payload is None:
        default_payload = fields
    return CustomModeDefinition(
        name=_validate_name(name),
        instructions=instructions,
        description=description,
        aliases=aliases,
        default_payload=dict(default_payload) if default_payload else None,
    )


def _class_from_definition(definition: CustomModeDefinition) -> type[WritingModeBase]:
    class_name = "".join(
        part.title() for part in definition.name.replace("-", "_").split("_")
    )
    if not class_name:
        class_name = "Custom"
    if not class_name.endswith("Mode"):
        class_name = f"{class_name}Mode"
    return type(
        class_name,
        (WritingModeBase,),
        {
            "__module__": _BUILTIN_MODULE,
            "name": definition.name,
            "description": definition.description,
            "instructions": definition.instructions,
            "aliases": definition.aliases,
            "default_payload": dict(definition.default_payload or {}),
            "payload_model": CustomPayload,
        },
    )


class WritingModeRegistry:
    """An ordered, collision-checked collection of writing mode classes.

    A duplicate name is rejected even when it names the same class, because two
    owners for one profile mode would make selection ambiguous. Resolving a
    built in name returns its placeholder mode, while a name that is neither
    registered nor a loadable file fails with the name and the known modes.
    """

    include_builtins = False

    def __init__(self) -> None:
        self._modes: dict[str, WritingModeRegistration] = {}
        self._loaded: dict[str, WritingModeLike] = {}
        self._loaded_paths: dict[str, WritingModeLike] = {}

    @overload
    def register(
        self,
        name: str,
        mode: WritingModeLike,
        *,
        module: str | None = None,
        source: str | None = None,
    ) -> WritingModeRegistration: ...

    @overload
    def register(
        self,
        name: str,
        mode: None = None,
        *,
        module: str | None = None,
        source: str | None = None,
    ) -> Callable[[WritingModeLike], WritingModeLike]: ...

    @overload
    def register(
        self,
        name: WritingModeLike,
        mode: None = None,
        *,
        module: str | None = None,
        source: str | None = None,
    ) -> WritingModeRegistration: ...

    def register(
        self,
        name: str | WritingModeLike,
        mode: WritingModeLike | None = None,
        *,
        module: str | None = None,
        source: str | None = None,
    ) -> WritingModeRegistration | Callable[[WritingModeLike], WritingModeLike]:
        """Register *mode* under *name* or return a registration decorator."""
        if mode is None and not isinstance(name, str):
            candidate = name
            name = _mode_name(candidate)
            mode = candidate
        validated_name = _validate_name(name)
        if mode is None:

            def decorator(cls: WritingModeLike) -> WritingModeLike:
                self.register(validated_name, cls, module=module, source=source)
                return cls

            return decorator

        validated_mode = _validate_mode(validated_name, mode)
        existing = self._modes.get(validated_name)
        if existing is not None:
            raise ConfigError(
                f"writing mode already registered: {validated_name}",
                details={
                    "mode": validated_name,
                    "existing": type(existing.mode).__name__,
                    "new": type(validated_mode).__name__,
                },
            )
        registration = WritingModeRegistration(
            name=validated_name,
            mode=validated_mode,
            module=module,
            source=source,
        )
        self._modes[validated_name] = registration
        _KNOWN_CUSTOM_MODES.add(validated_name.lower())
        return registration

    @overload
    def register_mode(
        self,
        name: str,
        mode: WritingModeLike,
        *,
        module: str | None = None,
        source: str | None = None,
    ) -> WritingModeRegistration: ...

    @overload
    def register_mode(
        self,
        name: str,
        mode: None = None,
        *,
        module: str | None = None,
        source: str | None = None,
    ) -> Callable[[WritingModeLike], WritingModeLike]: ...

    @overload
    def register_mode(
        self,
        name: WritingModeLike,
        mode: None = None,
        *,
        module: str | None = None,
        source: str | None = None,
    ) -> WritingModeRegistration: ...

    def register_mode(
        self,
        name: str | WritingModeLike,
        mode: WritingModeLike | None = None,
        *,
        module: str | None = None,
        source: str | None = None,
    ) -> WritingModeRegistration | Callable[[WritingModeLike], WritingModeLike]:
        """Alias for :meth:`register`."""
        result = cast(Any, self.register)(name, mode, module=module, source=source)
        return cast(
            WritingModeRegistration | Callable[[WritingModeLike], WritingModeLike],
            result,
        )

    def get(self, name: str) -> WritingModeLike:
        """Return the mode registered under *name*."""
        validated_name = _validate_name(name)
        registration = self._modes.get(validated_name)
        if registration is not None:
            return registration.mode
        loaded = self._loaded.get(validated_name)
        if loaded is not None:
            return loaded
        try:
            return builtin_writing_mode(validated_name)
        except ValueError:
            pass
        raise ConfigError(
            f"unknown writing mode: {validated_name} (registered: {self._known()})",
            details={
                "mode": validated_name,
                "available": list(self.names()),
            },
        )

    def get_mode(self, name: str) -> WritingModeLike:
        """Alias for :meth:`get`."""
        return self.get(name)

    def unregister(self, name: str) -> None:
        """Remove a registered or loaded mode, rejecting an unknown name."""
        validated_name = _validate_name(name)
        if validated_name in self._modes:
            del self._modes[validated_name]
        elif validated_name in self._loaded:
            del self._loaded[validated_name]
        else:
            raise ConfigError(
                f"unknown writing mode: {validated_name} (registered: {self._known()})",
                details={
                    "mode": validated_name,
                    "available": list(self.names()),
                },
            )
        self._loaded.pop(validated_name, None)
        for path, mode in list(self._loaded_paths.items()):
            if _mode_name(mode) == validated_name:
                del self._loaded_paths[path]

    def load(self, path: str | Path) -> WritingModeLike:
        """Load a writing mode from a file path or a dotted module path."""
        if isinstance(path, Path):
            cleaned = str(path)
        elif isinstance(path, str):
            cleaned = path.strip()
        else:
            raise ConfigError("writing mode path must be a string or path")
        if not cleaned:
            raise ConfigError("writing mode path must not be empty")
        cached = self._loaded_paths.get(cleaned)
        if cached is not None:
            return cached

        if ":" in cleaned:
            file_part, _, _ = cleaned.rpartition(":")
            file_candidate = Path(file_part).expanduser()
            if file_candidate.exists():
                return self._load_file(Path(cleaned))

        candidate = Path(cleaned).expanduser()
        if candidate.exists() or candidate.suffix.lower() in _MODE_FILE_SUFFIXES:
            if not candidate.exists():
                raise ConfigError(
                    f"cannot load writing mode {cleaned!r}: file does not exist",
                    details={"path": cleaned},
                )
            return self._load_file(candidate)

        if _is_dotted_path(cleaned):
            return self._load_dotted(cleaned)

        try:
            mode = builtin_writing_mode(cleaned)
        except ValueError:
            raise ConfigError(
                f"cannot load writing mode {cleaned!r}: "
                "not a file, module path or built in mode",
                details={"path": cleaned},
            ) from None
        self._remember(cleaned, mode)
        return mode

    def load_file(self, path: str | Path) -> WritingModeLike:
        """Load a writing mode from *path* and remember it by name."""
        return self.load(str(path))

    def load_path(self, path: str | Path) -> WritingModeLike:
        """Alias for :meth:`load_file`."""
        return self.load_file(path)

    def from_file(self, path: str | Path) -> WritingModeLike:
        """Alias for :meth:`load_file`."""
        return self.load_file(path)

    def register_file(
        self,
        path: str | Path,
        name: str | None = None,
        *,
        source: str | None = None,
    ) -> WritingModeRegistration:
        """Load a mode file and register it under *name* or its own name."""
        mode = self.load_file(path)
        target = _validate_name(name) if name is not None else _mode_name(mode)
        return self.register(target, mode, module=str(path), source=source or str(path))

    def resolve(self, name: str) -> WritingModeLike:
        """Return the mode for a registered *name*, file path or built in."""
        validated_name = _validate_name(name)
        registration = self._modes.get(validated_name)
        if registration is not None:
            return registration.mode
        loaded = self._loaded.get(validated_name)
        if loaded is not None:
            return loaded
        if _is_dotted_path(validated_name) or Path(validated_name).exists():
            return self.load(validated_name)
        try:
            return builtin_writing_mode(validated_name)
        except ValueError:
            pass
        raise ConfigError(
            f"unknown writing mode: {validated_name} (registered: {self._known()})",
            details={
                "mode": validated_name,
                "available": list(self.names()),
            },
        )

    def resolve_mode(self, name: str) -> WritingModeLike:
        """Alias for :meth:`resolve`."""
        return self.resolve(name)

    def names(self) -> tuple[str, ...]:
        """Return the known mode names in sorted order."""
        names = set(self._modes) | set(self._loaded)
        if self.include_builtins:
            names.update(mode.value for mode in WritingMode)
        return tuple(sorted(names))

    def mode_names(self) -> tuple[str, ...]:
        """Alias for :meth:`names`."""
        return self.names()

    def keys(self) -> tuple[str, ...]:
        """Alias for :meth:`names`."""
        return self.names()

    def registrations(self) -> tuple[WritingModeRegistration, ...]:
        """Return the known entries in sorted name order."""
        entries: dict[str, WritingModeRegistration] = dict(self._modes)
        for name, mode in self._loaded.items():
            entries.setdefault(name, WritingModeRegistration(name=name, mode=mode))
        if self.include_builtins:
            for builtin in WritingMode:
                entries.setdefault(
                    builtin.value,
                    WritingModeRegistration(
                        name=builtin.value,
                        mode=builtin_writing_mode(builtin),
                    ),
                )
        return tuple(entries[name] for name in sorted(entries))

    def _known(self) -> str:
        return ", ".join(self.names()) or "none"

    def _remember(self, path: str, mode: WritingModeLike) -> None:
        self._loaded_paths[path] = mode
        name = _mode_name(mode)
        if name:
            self._loaded.setdefault(name, mode)
            _KNOWN_CUSTOM_MODES.add(name.lower())

    def _load_file(self, path: Path) -> WritingModeLike:
        text = str(path)
        target = path
        if ":" in text:
            file_part, _, _ = text.rpartition(":")
            candidate = Path(file_part).expanduser()
            if candidate.exists():
                target = candidate
        resolved = target.resolve()
        cached = self._loaded_paths.get(text) or self._loaded_paths.get(str(resolved))
        if cached is not None:
            return cached
        if target.suffix.lower() == ".py":
            mode = self._load_python_file(path)
        else:
            mode = self._load_data_file(target)
        self._remember(text, mode)
        self._remember(str(resolved), mode)
        return mode

    def _load_data_file(self, path: Path) -> WritingModeLike:
        text = _read_text(path)
        suffix = path.suffix.lower()
        data = _parse_data(text, suffix) if suffix in _DATA_SUFFIXES else text
        if isinstance(data, str):
            definition = CustomModeDefinition(
                name=_validate_name(path.stem),
                instructions=data.strip(),
            )
            return _class_from_definition(definition)
        if not isinstance(data, Mapping):
            raise ConfigError(
                f"writing mode file {str(path)!r} must contain a mapping or text",
                details={"path": str(path)},
            )
        values = dict(data)
        container = values.get("modes")
        if isinstance(container, Mapping):
            loaded: list[WritingModeLike] = []
            for name, entry in container.items():
                if not isinstance(entry, Mapping):
                    raise ConfigError(
                        f"writing mode {name!r} must be a mapping",
                        details={"path": str(path), "mode": name},
                    )
                definition = _definition_from_mapping(entry, default_name=str(name))
                mode = _class_from_definition(definition)
                self._remember(str(path), mode)
                loaded.append(mode)
            if not loaded:
                raise ConfigError(
                    f"writing mode file {str(path)!r} defines no modes",
                    details={"path": str(path)},
                )
            return loaded[0]
        definition = _definition_from_mapping(values, default_name=path.stem)
        return _class_from_definition(definition)

    def _load_python_file(self, path: Path) -> WritingModeLike:
        target_path = path
        attribute: str | None = None
        text = str(path)
        if ":" in text:
            file_part, _, attribute_part = text.rpartition(":")
            file_candidate = Path(file_part).expanduser()
            if file_candidate.exists():
                target_path = file_candidate
                attribute = attribute_part or None
        resolved = target_path.resolve()
        module_name = f"_aivia_writing_mode_{abs(hash(str(resolved)))}"
        spec = importlib.util.spec_from_file_location(module_name, resolved)
        if spec is None or spec.loader is None:
            raise ConfigError(
                f"cannot load writing mode file {str(target_path)!r}",
                details={"path": str(target_path)},
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise ConfigError(
                f"cannot load writing mode file {str(target_path)!r}: {exc}",
                details={"path": str(target_path)},
            ) from exc

        if attribute is not None:
            target: object = module
            for part in attribute.split("."):
                try:
                    target = getattr(target, part)
                except AttributeError:
                    raise ConfigError(
                        f"cannot load writing mode {str(target_path)!r}: "
                        f"no attribute {attribute!r}",
                        details={"path": str(target_path), "attribute": attribute},
                    ) from None
            try:
                return _validate_mode(attribute, target, source="writing mode type")
            except TypeError as exc:
                raise ConfigError(
                    f"cannot load writing mode {str(target_path)!r}: {exc}",
                    details={"path": str(target_path)},
                ) from exc

        candidates: list[WritingModeLike] = []
        for value in vars(module).values():
            if isinstance(value, type):
                if value is WritingModeBase or value.__module__ != module_name:
                    continue
                structural = getattr(value, "name", None) is not None and (
                    callable(getattr(value, "build_payload", None))
                    or callable(getattr(value, "build", None))
                )
                if issubclass(value, WritingModeBase) or structural:
                    candidates.append(value)
            elif isinstance(value, WritingModeBase):
                if type(value).__module__ == module_name:
                    candidates.append(value)
        if not candidates:
            raise ConfigError(
                f"writing mode file {str(target_path)!r} defines no WritingModeBase",
                details={"path": str(target_path)},
            )

        def candidate_name(candidate: WritingModeLike) -> str:
            return str(getattr(candidate, "__name__", type(candidate).__name__))

        selected = candidates[0]
        if len(candidates) > 1:
            preferred = path.stem.replace("-", "_")
            found = next(
                (
                    candidate
                    for candidate in candidates
                    if _mode_name(candidate) in {preferred, path.stem}
                    or candidate_name(candidate).lower() == f"{preferred}mode".lower()
                ),
                None,
            )
            if found is None:
                names = ", ".join(candidate_name(candidate) for candidate in candidates)
                raise ConfigError(
                    f"writing mode file {str(target_path)!r} defines several modes "
                    f"({names}); select one with 'file.py:ClassName'",
                    details={"path": str(target_path)},
                )
            selected = found
        if not _mode_name(selected):
            selected.name = path.stem.replace("-", "_")
        return selected

    def _load_dotted(self, path: str) -> WritingModeLike:
        module_name, attribute = _split_path(path)
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ConfigError(
                f"cannot load writing mode {path!r}: {exc}",
                details={"mode": path, "module": module_name},
            ) from exc

        target: object = module
        for part in attribute.split("."):
            try:
                target = getattr(target, part)
            except AttributeError:
                raise ConfigError(
                    f"cannot load writing mode {path!r}: "
                    f"{module_name!r} has no attribute {attribute!r}",
                    details={
                        "mode": path,
                        "module": module_name,
                        "attribute": attribute,
                    },
                ) from None

        try:
            mode = _validate_mode(path, target, source="writing mode type")
        except TypeError as exc:
            raise ConfigError(
                f"cannot load writing mode {path!r}: {exc}",
                details={"mode": path, "module": module_name},
            ) from exc
        self._remember(path, mode)
        return mode

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        cleaned = name.strip()
        if cleaned in self._modes or cleaned in self._loaded:
            return True
        return self.include_builtins and cleaned in {mode.value for mode in WritingMode}

    def __getitem__(self, name: str) -> WritingModeLike:
        return self.get(name)

    def __iter__(self) -> Iterator[WritingModeRegistration]:
        return iter(self.registrations())

    def __len__(self) -> int:
        return len(self.names())


class _SharedWritingModeRegistry(WritingModeRegistry):
    include_builtins = True


writing_mode_registry = _SharedWritingModeRegistry()
"""The shared registry of built-in and custom writing modes."""

mode_registry = writing_mode_registry
ModeRegistry = WritingModeRegistry

register_writing_mode = writing_mode_registry.register
"""Register a mode on the shared :data:`writing_mode_registry`."""

register_mode = register_writing_mode

resolve_writing_mode = writing_mode_registry.resolve
"""Resolve a mode name, file path or dotted path through the shared registry."""

resolve_mode = resolve_writing_mode

load_writing_mode = writing_mode_registry.load
load_writing_mode_file = writing_mode_registry.load_file
