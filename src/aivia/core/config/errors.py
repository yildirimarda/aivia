"""Readable configuration errors: source file, dotted path, expectation, hint.

Validation failures are raised as :class:`~aivia.core.errors.ConfigError` with a
message that names the file and key path, contrasts the expected value with the
actual one and suggests how to fix it. Nothing here formats a traceback: callers
catch ``ConfigError`` and print ``str(error)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from aivia.core.errors import ConfigError

__all__ = [
    "describe_validation_issue",
    "enrich_config_error",
    "source_for_path",
]

_MISSING = object()

_EXPECTED_TYPES = {
    "bool": "a boolean",
    "date": "a date",
    "datetime": "a date and time",
    "decimal": "a number",
    "dict": "a mapping",
    "float": "a number",
    "int": "an integer",
    "list": "a list",
    "set": "a set",
    "str": "text",
    "string": "text",
    "tuple": "a tuple",
    "url": "a URL",
    "uuid": "a UUID",
}

_HINTS = {
    "bool": "use true or false",
    "dict": "use a YAML mapping of keys to values",
    "float": "use a decimal number, for example 1.5",
    "int": "use a whole number, for example 30",
    "list": "use a YAML list, for example [a, b]",
    "str": "use text, quoted when it contains YAML punctuation",
    "string": "use text, quoted when it contains YAML punctuation",
}


def _dotted(location: Sequence[object]) -> str:
    return ".".join(str(part) for part in location)


def _short(value: object, *, limit: int = 80) -> str:
    text = repr(value)
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…"


def _expected(issue: Mapping[str, Any]) -> str:
    kind = str(issue.get("type", ""))
    if kind == "missing":
        return "a value"
    if kind == "extra_forbidden":
        return "a known configuration key"
    context = issue.get("ctx")
    if isinstance(context, Mapping):
        expected = context.get("expected")
        if expected is not None:
            return str(expected)
    prefix = kind.split("_", 1)[0]
    if prefix in _EXPECTED_TYPES:
        return _EXPECTED_TYPES[prefix]
    message = str(issue.get("msg", "a valid value"))
    return message.removeprefix("Input should be ").lower()


def _actual(issue: Mapping[str, Any], path: str) -> str:
    kind = str(issue.get("type", ""))
    if kind == "missing":
        return "nothing"
    if kind == "extra_forbidden":
        return _short(path.rsplit(".", 1)[-1] if path else "")
    value = issue.get("input", _MISSING)
    if value is _MISSING:
        return "nothing"
    return _short(value)


def _hint(issue: Mapping[str, Any]) -> str:
    kind = str(issue.get("type", ""))
    if kind == "missing":
        return "add the required key with a value"
    if kind == "extra_forbidden":
        return "remove the key or check its spelling"
    if kind == "literal_error":
        return "use one of the allowed values"
    prefix = kind.split("_", 1)[0]
    if prefix in _HINTS:
        return _HINTS[prefix]
    if kind.startswith(("greater_than", "less_than", "multiple_of")):
        return "choose a value within the documented range"
    if kind.startswith(
        ("string_too_short", "string_too_long", "too_short", "too_long")
    ):
        return "adjust the value to fit the allowed length"
    return "check the value against the configuration schema"


def source_for_path(path: str, context: Mapping[str, Any] | None) -> str | None:
    """Return the file that supplied *path*, falling back to the main file."""
    if not context:
        return None
    origins = context.get("origins")
    origin_files = context.get("origin_files")
    if isinstance(origins, Mapping) and isinstance(origin_files, Mapping):
        candidate = path
        while candidate:
            label = origins.get(candidate)
            if label is not None:
                source = origin_files.get(label)
                if isinstance(source, str) and source:
                    return source
            candidate = candidate.rpartition(".")[0]
    fallback = context.get("file")
    if isinstance(fallback, str) and fallback:
        return fallback
    return None


def _location(source: str | None, path: str) -> str:
    if source and path:
        return f"{source}:{path}"
    if source:
        return source
    return ""


def describe_validation_issue(
    issue: Mapping[str, Any],
    *,
    path: str,
    source: str | None,
) -> tuple[str, dict[str, Any]]:
    """Render one Pydantic issue as ``(message, details)``."""
    kind = str(issue.get("type", ""))
    if kind == "extra_forbidden":
        base = f"unknown config key: {path}"
    elif kind == "missing":
        base = f"missing required config key: {path}"
    else:
        reason = str(issue.get("msg", "invalid value"))
        base = f"invalid value for {path}: {reason}" if path else reason

    expected = _expected(issue)
    actual = _actual(issue, path)
    hint = _hint(issue)
    location = _location(source, path)
    message = f"{location}: {base}" if location else base
    message = f"{message} (expected {expected}, actual {actual}; hint: {hint})"

    details: dict[str, Any] = {
        "path": path,
        "type": kind,
        "message": message,
        "expected": expected,
        "actual": actual,
        "hint": hint,
    }
    if source:
        details["file"] = source
    return message, details


def _error_path(error: ConfigError) -> str:
    details = error.details
    message = error.message
    path = details.get("path")
    if isinstance(path, str) and path:
        return path
    errors = details.get("errors")
    if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)) and errors:
        first = errors[0]
        if isinstance(first, Mapping):
            candidate = first.get("path")
            if isinstance(candidate, str) and candidate:
                return candidate
    unknown = details.get("unknown")
    if (
        isinstance(unknown, Sequence)
        and not isinstance(unknown, (str, bytes))
        and unknown
    ):
        return str(unknown[0])
    preset = details.get("preset")
    if isinstance(preset, str) and preset:
        if "unknown platform account" in message:
            return f"presets.{preset}.accounts"
        if "unknown format" in message:
            return f"presets.{preset}.format"
        if "unknown profile" in message:
            return f"presets.{preset}.profile"
        return f"presets.{preset}"
    schedule = details.get("schedule")
    if isinstance(schedule, str) and schedule:
        return f"schedule.schedules.{schedule}.profiles"
    profile = details.get("profile")
    if isinstance(profile, str) and profile:
        if f"profile {profile} references unknown platform account" in message:
            return f"profiles.{profile}.accounts"
        if f"profile {profile} references unknown format" in message:
            return f"profiles.{profile}.format"
        return f"profiles.{profile}"
    account = details.get("account")
    if isinstance(account, str) and account:
        if "unknown credential" in message:
            return f"publish.accounts.{account}.credential_refs"
        if "unknown profile" in message:
            return f"publish.accounts.{account}.profile"
        return f"publish.accounts.{account}"
    return ""


def _hint_for_message(message: str) -> str:
    lowered = message.lower()
    if "unknown config section" in lowered:
        return "register the module that owns this section or remove the key"
    if "unknown config key" in lowered:
        return "remove the key or check its spelling"
    if "missing required config key" in lowered:
        return "add the required key with a value"
    if "missing environment variable" in lowered:
        return "set the variable or give it a default such as ${NAME:-fallback}"
    if "invalid environment reference" in lowered:
        return "use ${NAME} or ${NAME:-default}"
    if "unknown profile" in lowered:
        return "choose one of the configured profiles"
    if "unknown platform account" in lowered:
        return "add the account to platforms.accounts or fix the name"
    if "unknown format" in lowered:
        return "add the format recipe or fix the name"
    if "config file not found" in lowered:
        return "create the file or pass an existing --config path"
    if "invalid yaml" in lowered:
        return "fix the YAML syntax near the reported line"
    if "not a mapping" in lowered:
        return "use a YAML mapping of keys to values"
    return "check the value against the configuration schema"


def enrich_config_error(
    error: ConfigError,
    *,
    context: Mapping[str, Any] | None,
) -> ConfigError:
    """Add a source file, dotted path and hint to *error* when missing.

    Validation errors already carry the full form; this fills in the gaps for
    errors raised directly by a model validator, such as a missing environment
    variable or a cross-section reference. Calling it twice is harmless.
    """
    path = _error_path(error)
    source = source_for_path(path, context)
    message = error.message
    details = dict(error.details)

    if "hint:" not in message:
        hint = _hint_for_message(message)
        message = f"{message} (hint: {hint})"
        details.setdefault("hint", hint)

    already_sourced = details.get("file")
    if source and already_sourced != source:
        location = _location(source, path)
        if location and not message.startswith(f"{location}:"):
            message = f"{location}: {message}"
        details["file"] = source
    if path:
        details.setdefault("path", path)
    return ConfigError(message, details=details)
