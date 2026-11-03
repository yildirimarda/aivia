"""Base class and registry for topic modes.

A topic mode is a research strategy: it turns a profile brief, the selected
sources and a run window into topic candidates. The mode registry maps a
profile name such as ``on_this_day`` to the class that implements it, while a
custom mode may be named by a dotted path such as ``my_package.modes:MyMode``
and imported lazily.

Modes do not score, deduplicate or select topics; those policies belong to the
later research stages. A mode only discovers candidates and reports them in the
order it found them. The shared registry starts empty because the built-in
modes are registered by their own modules, keeping the base module free of
strategy-specific dependencies.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol, cast, overload, runtime_checkable

from aivia.core.errors import ConfigError
from aivia.providers.protocols.source import (
    SourceCandidate,
    SourceRequest,
    SourceResponse,
    SourceResult,
)

__all__ = [
    "BaseMode",
    "BaseTopicMode",
    "Mode",
    "ModeBase",
    "ModeRegistry",
    "TopicMode",
    "TopicModeBase",
    "TopicModeCandidate",
    "TopicModeProtocol",
    "TopicModeRegistration",
    "TopicModeRegistry",
    "TopicModeRequest",
    "TopicModeResponse",
    "TopicModeResult",
    "mode_registry",
    "register_mode",
    "register_topic_mode",
    "resolve_mode",
    "resolve_topic_mode",
    "topic_mode_registry",
]

TopicModeRequest = SourceRequest
"""The request a topic mode receives, shared with source providers."""

TopicModeResponse = SourceResponse
"""The response a topic mode returns, shared with source providers."""

TopicModeCandidate = SourceCandidate
"""Alias for :class:`~aivia.core.models.topic.TopicCandidate`."""

TopicModeResult = SourceResult
"""Alias for :class:`~aivia.core.models.topic.TopicCandidate`."""


@runtime_checkable
class TopicMode(Protocol):
    """The structural contract every topic mode satisfies."""

    name: str
    """The mode name used in a profile and the registry."""

    async def discover(self, request: SourceRequest) -> SourceResponse:
        """Return the topic candidates this mode discovers for *request*."""


TopicModeProtocol = TopicMode


class TopicModeBase:
    """Convenience base for a topic mode.

    Subclasses set :attr:`name` and implement :meth:`discover`. :meth:`find`
    and :meth:`collect` delegate to it so callers can use the verb that reads
    naturally for their stage.
    """

    name: str = ""

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            cleaned = name.strip()
            if not cleaned:
                raise ConfigError("topic mode name must not be empty")
            self.name = cleaned

    @property
    def mode(self) -> str:
        """Alias for :attr:`name`."""
        return self.name

    async def discover(self, request: SourceRequest) -> SourceResponse:
        """Discover topic candidates for *request*."""
        raise NotImplementedError

    async def find(self, request: SourceRequest) -> SourceResponse:
        """Alias for :meth:`discover`."""
        return await self.discover(request)

    async def fetch(self, request: SourceRequest) -> SourceResponse:
        """Alias for :meth:`discover`."""
        return await self.discover(request)

    async def collect(self, request: SourceRequest) -> list[SourceCandidate]:
        """Return only the candidates from :meth:`discover`."""
        response = await self.discover(request)
        return list(response.candidates)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


BaseTopicMode = TopicModeBase
BaseMode = TopicModeBase
ModeBase = TopicModeBase
Mode = TopicMode


def _validate_name(name: object) -> str:
    if not isinstance(name, str):
        raise TypeError("topic mode name must be a string")
    stripped = name.strip()
    if not stripped:
        raise ConfigError("topic mode name must not be empty")
    if any(character.isspace() for character in stripped):
        raise ConfigError(f"invalid topic mode name: {name!r}")
    return stripped


def _validate_mode(
    name: str,
    mode: object,
    *,
    source: str = "topic mode",
) -> type[TopicModeBase]:
    if not isinstance(mode, type):
        raise TypeError(f"{source} {name!r} must be a class, not {type(mode).__name__}")
    if issubclass(mode, TopicModeBase):
        return mode
    if callable(getattr(mode, "discover", None)):
        return cast(type[TopicModeBase], mode)
    raise TypeError(
        f"{source} {name!r} must subclass TopicModeBase or define discover()"
    )


def _is_dotted_path(value: str) -> bool:
    return ":" in value or "." in value


def _split_path(value: str) -> tuple[str, str]:
    if ":" in value:
        module_name, _, attribute = value.partition(":")
    else:
        module_name, _, attribute = value.rpartition(".")
    if not module_name or not attribute:
        raise ConfigError(
            f"invalid topic mode path: {value!r} "
            "(expected 'module:Class' or 'module.Class')"
        )
    return module_name, attribute


@dataclass(frozen=True, slots=True)
class TopicModeRegistration:
    """One mode class registered under a profile name."""

    name: str
    mode: type[TopicModeBase]
    module: str | None = None

    @property
    def provider(self) -> type[TopicModeBase]:
        """Alias for :attr:`mode`."""
        return self.mode

    @property
    def cls(self) -> type[TopicModeBase]:
        """Alias for :attr:`mode`."""
        return self.mode


class TopicModeRegistry:
    """An ordered, collision-checked collection of topic mode classes.

    A duplicate name is rejected even when it names the same class, because two
    owners for one profile mode would make selection ambiguous. Resolving a
    name that is neither registered nor a loadable dotted path fails with the
    name and the registered modes, so a profile typo stays visible.
    """

    def __init__(self) -> None:
        self._modes: dict[str, TopicModeRegistration] = {}
        self._loaded: dict[str, type[TopicModeBase]] = {}

    @overload
    def register(
        self,
        name: str,
        mode: type[TopicModeBase],
        *,
        module: str | None = None,
    ) -> TopicModeRegistration: ...

    @overload
    def register(
        self,
        name: str,
        mode: None = None,
        *,
        module: str | None = None,
    ) -> Callable[[type[TopicModeBase]], type[TopicModeBase]]: ...

    def register(
        self,
        name: str,
        mode: type[TopicModeBase] | None = None,
        *,
        module: str | None = None,
    ) -> TopicModeRegistration | Callable[[type[TopicModeBase]], type[TopicModeBase]]:
        """Register *mode* under *name* or return a registration decorator."""
        validated_name = _validate_name(name)
        if mode is None:

            def decorator(cls: type[TopicModeBase]) -> type[TopicModeBase]:
                self.register(validated_name, cls, module=module)
                return cls

            return decorator

        validated_mode = _validate_mode(validated_name, mode)
        existing = self._modes.get(validated_name)
        if existing is not None:
            raise ConfigError(
                f"topic mode already registered: {validated_name}",
                details={
                    "mode": validated_name,
                    "existing": existing.mode.__name__,
                    "new": validated_mode.__name__,
                },
            )
        registration = TopicModeRegistration(
            name=validated_name,
            mode=validated_mode,
            module=module,
        )
        self._modes[validated_name] = registration
        return registration

    @overload
    def register_mode(
        self,
        name: str,
        mode: type[TopicModeBase],
        *,
        module: str | None = None,
    ) -> TopicModeRegistration: ...

    @overload
    def register_mode(
        self,
        name: str,
        mode: None = None,
        *,
        module: str | None = None,
    ) -> Callable[[type[TopicModeBase]], type[TopicModeBase]]: ...

    def register_mode(
        self,
        name: str,
        mode: type[TopicModeBase] | None = None,
        *,
        module: str | None = None,
    ) -> TopicModeRegistration | Callable[[type[TopicModeBase]], type[TopicModeBase]]:
        """Alias for :meth:`register`."""
        return self.register(name, mode, module=module)

    def get(self, name: str) -> type[TopicModeBase]:
        """Return the mode class registered under *name*."""
        validated_name = _validate_name(name)
        registration = self._modes.get(validated_name)
        if registration is None:
            raise ConfigError(
                f"unknown topic mode: {validated_name} (registered: {self._known()})",
                details={
                    "mode": validated_name,
                    "available": list(self.names()),
                },
            )
        return registration.mode

    def get_mode(self, name: str) -> type[TopicModeBase]:
        """Alias for :meth:`get`."""
        return self.get(name)

    def unregister(self, name: str) -> None:
        """Remove a registered mode, rejecting an unknown name."""
        validated_name = _validate_name(name)
        if validated_name not in self._modes:
            raise ConfigError(
                f"unknown topic mode: {validated_name} (registered: {self._known()})",
                details={
                    "mode": validated_name,
                    "available": list(self.names()),
                },
            )
        del self._modes[validated_name]
        self._loaded.pop(validated_name, None)

    def load(self, path: str) -> type[TopicModeBase]:
        """Import and return a mode class from a dotted *path*."""
        validated_path = _validate_name(path)
        cached = self._loaded.get(validated_path)
        if cached is not None:
            return cached

        module_name, attribute = _split_path(validated_path)
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ConfigError(
                f"cannot load topic mode {validated_path!r}: {exc}",
                details={"mode": validated_path, "module": module_name},
            ) from exc

        target: object = module
        for part in attribute.split("."):
            try:
                target = getattr(target, part)
            except AttributeError:
                raise ConfigError(
                    f"cannot load topic mode {validated_path!r}: "
                    f"{module_name!r} has no attribute {attribute!r}",
                    details={
                        "mode": validated_path,
                        "module": module_name,
                        "attribute": attribute,
                    },
                ) from None

        try:
            mode = _validate_mode(validated_path, target, source="topic mode type")
        except TypeError as exc:
            raise ConfigError(
                f"cannot load topic mode {validated_path!r}: {exc}",
                details={"mode": validated_path, "module": module_name},
            ) from exc
        self._loaded[validated_path] = mode
        return mode

    def resolve(self, name: str) -> type[TopicModeBase]:
        """Return the mode class for a registered *name* or a dotted path."""
        validated_name = _validate_name(name)
        registration = self._modes.get(validated_name)
        if registration is not None:
            return registration.mode
        if _is_dotted_path(validated_name):
            return self.load(validated_name)
        raise ConfigError(
            f"unknown topic mode: {validated_name} (registered: {self._known()})",
            details={
                "mode": validated_name,
                "available": list(self.names()),
            },
        )

    def resolve_mode(self, name: str) -> type[TopicModeBase]:
        """Alias for :meth:`resolve`."""
        return self.resolve(name)

    def names(self) -> tuple[str, ...]:
        """Return the registered mode names in sorted order."""
        return tuple(sorted(self._modes))

    def mode_names(self) -> tuple[str, ...]:
        """Alias for :meth:`names`."""
        return self.names()

    def keys(self) -> tuple[str, ...]:
        """Alias for :meth:`names`."""
        return self.names()

    def registrations(self) -> tuple[TopicModeRegistration, ...]:
        """Return the registered entries in sorted name order."""
        return tuple(self._modes[name] for name in self.names())

    def _known(self) -> str:
        return ", ".join(self.names()) or "none"

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return name.strip() in self._modes

    def __getitem__(self, name: str) -> type[TopicModeBase]:
        return self.get(name)

    def __iter__(self) -> Iterator[TopicModeRegistration]:
        return iter(self.registrations())

    def __len__(self) -> int:
        return len(self._modes)


topic_mode_registry = TopicModeRegistry()
"""The shared registry of built-in and custom topic modes."""

mode_registry = topic_mode_registry
ModeRegistry = TopicModeRegistry

register_topic_mode = topic_mode_registry.register
"""Register a mode on the shared :data:`topic_mode_registry`."""

register_mode = register_topic_mode

resolve_topic_mode = topic_mode_registry.resolve
"""Resolve a mode name or dotted path through the shared registry."""

resolve_mode = resolve_topic_mode
