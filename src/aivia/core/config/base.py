"""Base model for validated configuration with environment interpolation.

Configuration values may reference environment variables with ``${NAME}``.
A default is used when the variable is absent with ``${NAME:-default}`` (also
when it is empty) or ``${NAME-default}`` (only when it is unset). Write
``$$`` for a literal dollar sign. Interpolation happens before validation, so
an interpolated value is coerced to the field's declared type.

Unknown keys are rejected by Pydantic and reported as a dotted path, for
example ``pipeline.stages`` or ``providers.0.api_key``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from typing import Any, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    model_validator,
)

from aivia.core.config.errors import (
    describe_validation_issue,
    enrich_config_error,
    source_for_path,
)
from aivia.core.errors import ConfigError

__all__ = [
    "ConfigModel",
    "dotted_path",
    "interpolate_env",
]

_ENV_REFERENCE = re.compile(r"(?<!\$)\$\{(?P<body>[^{}]*)\}")
_ENV_BODY = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:(?P<operator>:-|-)(?P<default>.*))?$",
    re.DOTALL,
)
_ESCAPED_DOLLAR = "\x00"


def dotted_path(location: Sequence[object]) -> str:
    """Render a validation location as a dotted key path."""
    return ".".join(str(part) for part in location)


def _child_path(path: str, key: object) -> str:
    return f"{path}.{key}" if path else str(key)


def _interpolate_string(
    text: str,
    *,
    path: str,
    environ: Mapping[str, str],
) -> str:
    escaped = text.replace("$$", _ESCAPED_DOLLAR)

    def replace(match: re.Match[str]) -> str:
        body = match.group("body")
        parsed = _ENV_BODY.fullmatch(body)
        if parsed is None:
            location = f" for config key {path}" if path else ""
            raise ConfigError(
                f"invalid environment reference ${{{body}}}{location}",
                details={"path": path} if path else None,
            )

        name = parsed.group("name")
        operator = parsed.group("operator")
        default = parsed.group("default") or ""
        present = name in environ
        value = environ.get(name)

        if operator == ":-":
            return value if value else default
        if operator == "-":
            return (value or "") if present else default
        if not present:
            location = f" for config key {path}" if path else ""
            raise ConfigError(
                f"missing environment variable {name}{location}",
                details={"path": path, "variable": name}
                if path
                else {"variable": name},
            )
        return value or ""

    interpolated = _ENV_REFERENCE.sub(replace, escaped)
    return interpolated.replace(_ESCAPED_DOLLAR, "$")


def interpolate_env(
    value: Any,
    *,
    environ: Mapping[str, str] | None = None,
    path: str = "",
) -> Any:
    """Return *value* with ``${NAME}`` references resolved recursively.

    ``environ`` defaults to :data:`os.environ`. Mappings and sequences are
    copied, so the input is never mutated. A reference without a default that
    names an unset variable raises :class:`~aivia.core.errors.ConfigError`
    with the dotted path of the offending value.
    """
    env = os.environ if environ is None else environ
    if isinstance(value, str):
        return _interpolate_string(value, path=path, environ=env)
    if isinstance(value, Mapping):
        return {
            key: interpolate_env(
                item,
                environ=env,
                path=_child_path(path, key),
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            interpolate_env(item, environ=env, path=_child_path(path, index))
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            interpolate_env(item, environ=env, path=_child_path(path, index))
            for index, item in enumerate(value)
        )
    return value


def _validation_error(
    error: ValidationError,
    *,
    context: Mapping[str, Any] | None = None,
) -> ConfigError:
    messages: list[str] = []
    details: list[dict[str, Any]] = []
    for issue in error.errors():
        path = dotted_path(issue.get("loc", ()))
        source = source_for_path(path, context)
        message, detail = describe_validation_issue(
            issue,
            path=path,
            source=source,
        )
        messages.append(message)
        details.append(detail)
    return ConfigError("; ".join(messages), details={"errors": details})


class ConfigModel(BaseModel):
    """Base class for every validated configuration section.

    Use :meth:`from_mapping` to interpolate environment references and turn
    validation failures into :class:`~aivia.core.errors.ConfigError` with a
    dotted key path. Direct Pydantic construction also interpolates using
    :data:`os.environ` and rejects unknown keys.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _resolve_environment(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, Mapping):
            return data
        environ: Mapping[str, str] | None = None
        context = info.context
        if isinstance(context, Mapping):
            candidate = context.get("environ")
            if candidate is not None and not isinstance(candidate, Mapping):
                raise ConfigError("environ must be a mapping")
            environ = cast(Mapping[str, str] | None, candidate)
        return interpolate_env(dict(data), environ=environ)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        environ: Mapping[str, str] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Self:
        """Interpolate, validate and return a configuration section.

        *context* carries cross-section data such as the configured profile
        names. It is passed to Pydantic validators unchanged, with ``environ``
        added under the same key the base model reads.
        """
        if not isinstance(values, Mapping):
            raise ConfigError("config root must be a mapping")
        merged_context: dict[str, Any] = dict(context or {})
        merged_context["environ"] = environ
        try:
            return cls.model_validate(
                dict(values),
                context=merged_context,
            )
        except ValidationError as error:
            raise _validation_error(error, context=merged_context) from error
        except ConfigError as error:
            raise enrich_config_error(error, context=merged_context) from error

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
        *,
        environ: Mapping[str, str] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Self:
        """Alias for :meth:`from_mapping`."""
        return cls.from_mapping(values, environ=environ, context=context)
