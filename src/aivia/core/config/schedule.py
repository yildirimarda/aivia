"""Scheduling and publishing configuration with cross-reference checks.

A schedule names a cron expression and the content profile it runs. Cron
expressions accept the usual five fields (minute, hour, day of month, month and
day of week) or six fields with a leading seconds field. Names for months and
weekdays, lists, ranges and steps are supported. A schedule may inherit the
section timezone and may be disabled without deleting it.

Publishing configuration maps accounts to the secret references they need.
Credential references are names resolved by the secrets loader, never values,
so the configuration is safe to copy. Profiles and credentials can be checked
when the rest of the configuration is available: pass them as keyword
arguments, as validation context, or call the explicit ``validate_*`` helpers.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, ValidationInfo, field_validator, model_validator

from aivia.core.config.base import ConfigModel
from aivia.core.errors import ConfigError

__all__ = [
    "CronScheduleConfig",
    "PublishAccount",
    "PublishAccountConfig",
    "PublishAccountsConfig",
    "PublishConfig",
    "PublishSettingsConfig",
    "PublishTargetConfig",
    "ScheduleConfig",
    "ScheduleConfigEntry",
    "ScheduleEntry",
    "ScheduleEntryConfig",
    "ScheduleItemConfig",
    "ScheduleJobConfig",
    "ScheduledRunConfig",
    "SchedulesConfig",
    "credential_names",
    "is_valid_cron",
    "profile_names",
    "validate_cron",
]

_CRON_MACROS = {
    "@yearly": "0 0 1 1 *",
    "@annually": "0 0 1 1 *",
    "@monthly": "0 0 1 * *",
    "@weekly": "0 0 * * 0",
    "@daily": "0 0 * * *",
    "@midnight": "0 0 * * *",
    "@hourly": "0 * * * *",
}

_MONTH_NAMES = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_WEEKDAY_NAMES = {
    "sun": 0,
    "mon": 1,
    "tue": 2,
    "wed": 3,
    "thu": 4,
    "fri": 5,
    "sat": 6,
}

_REFERENCE_WHITESPACE = re.compile(r"\s")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


def _normalise_names(values: Sequence[str], label: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for entry in values:
        name = _text(entry, label)
        if name in seen:
            raise ValueError(f"duplicate {label}: {name}")
        seen.add(name)
        names.append(name)
    return names


def _name_values(value: Any, label: str) -> Any:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return list(value)
    return value


def _cron_number(token: str, names: Mapping[str, int] | None) -> int:
    key = token.strip().lower()
    if names is not None and key in names:
        return names[key]
    try:
        return int(key, 10)
    except ValueError:
        raise ValueError(f"invalid cron value: {token}") from None


def _validate_cron_field(
    field: str,
    *,
    low: int,
    high: int,
    label: str,
    names: Mapping[str, int] | None = None,
) -> None:
    if not field or field != field.strip():
        raise ValueError(f"invalid cron {label} field: {field!r}")
    for part in field.split(","):
        if not part:
            raise ValueError(f"invalid cron {label} field: {field!r}")

        base, separator, step_text = part.partition("/")
        if separator:
            try:
                step = int(step_text, 10)
            except ValueError:
                raise ValueError(f"invalid cron step: {step_text}") from None
            if step <= 0:
                raise ValueError(f"cron step must be positive: {step_text}")
        else:
            step = 1

        if base == "*":
            start, end = low, high
        elif "-" in base:
            start_text, end_text = base.split("-", 1)
            start = _cron_number(start_text, names)
            end = _cron_number(end_text, names)
            if start > end:
                raise ValueError(f"invalid cron range: {base}")
        else:
            start = _cron_number(base, names)
            end = high if separator else start

        if not low <= start <= high or not low <= end <= high:
            raise ValueError(f"cron {label} field out of range {low}-{high}: {part}")
        if separator and step > 0 and start > high:
            raise ValueError(f"cron {label} field out of range: {part}")


def validate_cron(expression: str) -> str:
    """Validate *expression* and return its normalised text.

    Five fields are minute, hour, day of month, month and day of week. Six
    fields add a leading seconds field. ``@daily`` and the other common macros
    are accepted. An invalid expression raises :class:`ValueError` naming the
    offending field.
    """
    text = _text(expression, "cron expression")
    text = _CRON_MACROS.get(text.lower(), text)
    fields = text.split()
    if len(fields) not in (5, 6):
        raise ValueError("cron expression must have 5 or 6 fields: " + " ".join(fields))

    if len(fields) == 6:
        second, minute, hour, day, month, weekday = fields
        _validate_cron_field(second, low=0, high=59, label="second")
    else:
        minute, hour, day, month, weekday = fields

    _validate_cron_field(minute, low=0, high=59, label="minute")
    _validate_cron_field(hour, low=0, high=23, label="hour")
    _validate_cron_field(day, low=1, high=31, label="day of month")
    _validate_cron_field(
        month,
        low=1,
        high=12,
        label="month",
        names=_MONTH_NAMES,
    )
    _validate_cron_field(
        weekday,
        low=0,
        high=7,
        label="day of week",
        names=_WEEKDAY_NAMES,
    )
    return " ".join(fields)


def is_valid_cron(expression: str) -> bool:
    """Return whether *expression* is a valid five or six field cron."""
    try:
        validate_cron(expression)
    except (TypeError, ValueError):
        return False
    return True


def _cron_expression_from_mapping(values: Mapping[str, Any]) -> str:
    fields = {
        "second": values.get("second", values.get("seconds")),
        "minute": values.get("minute", values.get("minutes")),
        "hour": values.get("hour", values.get("hours")),
        "day": values.get(
            "day",
            values.get("day_of_month", values.get("dom")),
        ),
        "month": values.get("month", values.get("months")),
        "weekday": values.get(
            "weekday",
            values.get("day_of_week", values.get("dow")),
        ),
    }
    order = (
        ["second", "minute", "hour", "day", "month", "weekday"]
        if fields["second"] is not None
        else ["minute", "hour", "day", "month", "weekday"]
    )
    return " ".join(
        str(fields[name]) if fields[name] is not None else "*" for name in order
    )


def _validate_timezone(value: str, label: str = "timezone") -> str:
    text = value.strip()
    if not text:
        return ""
    try:
        ZoneInfo(text)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"unknown {label}: {text}") from None
    return text


def _profile_names(value: Any) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if "profiles" in value and isinstance(value["profiles"], Mapping):
            return {str(name) for name in value["profiles"]}
        return {str(name) for name in value}
    profiles = getattr(value, "profiles", None)
    if isinstance(profiles, Mapping):
        return {str(name) for name in profiles}
    names = getattr(value, "names", None)
    if names is not None:
        return {str(name) for name in names}
    if isinstance(value, str):
        return {value}
    if isinstance(value, Iterable):
        return {str(name) for name in value}
    raise ValueError("profiles must be a mapping or an iterable of names")


def _credential_names(value: Any) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        names = {str(name) for name in value}
        names.update(str(item) for item in value.values() if isinstance(item, str))
        return names
    if isinstance(value, str):
        return {value}
    if isinstance(value, Iterable):
        return {str(name) for name in value}
    raise ValueError("credentials must be a mapping or an iterable of names")


def _language_names(value: Any) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        entries = value.get("languages")
        if isinstance(entries, Mapping):
            return {str(name) for name in entries}
        return {str(name) for name in value if str(name) != "default"}
    entries = getattr(value, "languages", None)
    if isinstance(entries, Mapping):
        return {str(name) for name in entries}
    names = getattr(value, "names", None)
    if names is not None:
        return {str(name) for name in names}
    if isinstance(value, str):
        return {value}
    if isinstance(value, Iterable):
        return {str(name) for name in value}
    raise ValueError("languages must be a mapping or an iterable of names")


def _section(source: Any, key: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _context(info: ValidationInfo, key: str) -> Any:
    context = info.context
    if isinstance(context, Mapping):
        return context.get(key)
    return None


profile_names = _profile_names
credential_names = _credential_names


class ScheduleEntryConfig(ConfigModel):
    """One cron entry: when to run and which content profile to use."""

    name: str = ""
    profile: str = ""
    profiles: list[str] = Field(default_factory=list)
    cron: str = ""
    timezone: str = ""
    enabled: bool = True
    languages: list[str] = Field(default_factory=list)
    jitter_seconds: int = Field(default=0, ge=0)
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"cron": data}
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias in ("id", "key", "schedule", "job"):
            if alias in values:
                values.setdefault("name", values.pop(alias))
        for alias in ("profile_name", "profile_id", "content_profile"):
            if alias in values:
                values.setdefault("profile", values.pop(alias))
        if isinstance(values.get("profiles"), Mapping):
            values["profiles"] = list(values["profiles"])
        elif isinstance(values.get("profiles"), str):
            values["profiles"] = [values["profiles"]]
        for alias in ("expression", "cron_expression", "schedule_cron", "when"):
            if alias in values:
                values.setdefault("cron", values.pop(alias))
        for alias in ("tz", "time_zone"):
            if alias in values:
                values.setdefault("timezone", values.pop(alias))
        for alias in ("lang", "language"):
            if alias in values:
                values.setdefault("languages", values.pop(alias))
        if "languages" in values:
            values["languages"] = _name_values(values["languages"], "language")
        if isinstance(values.get("cron"), Mapping):
            values["cron"] = _cron_expression_from_mapping(values["cron"])

        cron_fields = {
            "second": values.pop("second", values.pop("seconds", None)),
            "minute": values.pop("minute", values.pop("minutes", None)),
            "hour": values.pop("hour", values.pop("hours", None)),
            "day": values.pop(
                "day",
                values.pop("day_of_month", values.pop("dom", None)),
            ),
            "month": values.pop("month", values.pop("months", None)),
            "weekday": values.pop(
                "weekday",
                values.pop("day_of_week", values.pop("dow", None)),
            ),
        }
        present = [name for name, value in cron_fields.items() if value is not None]
        if present and "cron" not in values:
            order = (
                ["second", "minute", "hour", "day", "month", "weekday"]
                if cron_fields["second"] is not None
                else ["minute", "hour", "day", "month", "weekday"]
            )
            values["cron"] = " ".join(
                str(cron_fields[name]) if cron_fields[name] is not None else "*"
                for name in order
            )
        return values

    @field_validator("name", "profile", "timezone")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("cron")
    @classmethod
    def _validate_cron(cls, value: str) -> str:
        text = value.strip()
        return validate_cron(text) if text else ""

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        return _validate_timezone(value, "schedule timezone")

    @field_validator("languages")
    @classmethod
    def _validate_languages(cls, value: list[str]) -> list[str]:
        return _normalise_names(value, "schedule language")

    @field_validator("profiles")
    @classmethod
    def _validate_profiles(cls, value: list[str]) -> list[str]:
        return _normalise_names(value, "schedule profile")

    @field_validator("options", mode="before")
    @classmethod
    def _accept_options(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("schedule options must be a mapping")
        return dict(value)

    @model_validator(mode="after")
    def _profile_without_cron(self) -> ScheduleEntryConfig:
        if "cron" in self.model_fields_set and not self.cron:
            raise ValueError("cron expression must be non-empty")
        if self.enabled and (self.profile or self.profiles) and not self.cron:
            raise ValueError(
                f"schedule {self.name or self.profile or self.profiles!r} "
                "requires a cron expression"
            )
        return self


class ScheduleConfig(ConfigModel):
    """Named cron schedules, their timezone and the default entry.

    Entries may be written directly under the section
    (``schedule: {daily: {...}}``) or nested under ``schedules``. A single
    top-level ``cron`` and ``profile`` pair is treated as the ``default`` entry.
    """

    enabled: bool = True
    default: str = ""
    timezone: str = "UTC"
    profile: str = ""
    profiles: list[str] = Field(default_factory=list)
    cron: str = ""
    schedules: dict[str, ScheduleEntryConfig] = Field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        environ: Mapping[str, str] | None = None,
        context: Mapping[str, Any] | None = None,
        profiles: Any = None,
    ) -> Self:
        """Validate a schedule section and optionally check profile names."""
        merged = dict(context or {})
        if profiles is not None:
            merged["profiles"] = profiles
        return super().from_mapping(values, environ=environ, context=merged)

    @model_validator(mode="before")
    @classmethod
    def _accept_shapes(cls, data: Any) -> Any:
        if data is None:
            return {}
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes, Mapping)):
            entries: dict[str, Any] = {}
            for index, entry in enumerate(data):
                if not isinstance(entry, Mapping):
                    raise ValueError("schedule entries must be mappings")
                values = dict(entry)
                name = values.get("name", values.get("id", values.get("key")))
                if name is None:
                    name = values.get("profile", f"schedule-{index + 1}")
                entries[str(name)] = values
            return {"schedules": entries}
        if not isinstance(data, Mapping):
            return data

        values = dict(data)
        for alias in ("entries", "items", "jobs", "crons"):
            if alias in values:
                values.setdefault("schedules", values.pop(alias))
        for alias in ("tz", "time_zone"):
            if alias in values:
                values.setdefault("timezone", values.pop(alias))
        if isinstance(values.get("cron"), Mapping):
            values["cron"] = _cron_expression_from_mapping(values["cron"])

        if "schedules" in values:
            schedules = values["schedules"]
            if isinstance(schedules, Sequence) and not isinstance(
                schedules, (str, bytes, Mapping)
            ):
                listed: dict[str, Any] = {}
                for index, entry in enumerate(schedules):
                    if not isinstance(entry, Mapping):
                        raise ValueError("schedule entries must be mappings")
                    item = dict(entry)
                    name = item.get("name", item.get("id", item.get("key")))
                    if name is None:
                        name = item.get("profile", f"schedule-{index + 1}")
                    listed[str(name)] = item
                values["schedules"] = listed
            return values

        global_keys = {
            "enabled",
            "default",
            "timezone",
            "profile",
            "profiles",
            "cron",
        }
        entries = {
            key: value for key, value in values.items() if key not in global_keys
        }
        if entries:
            for key in list(entries):
                values.pop(key)
            values["schedules"] = entries
        elif values.get("cron") or values.get("profile") or values.get("profiles"):
            values["schedules"] = {
                "default": {
                    "profile": values.get("profile", ""),
                    "profiles": values.get("profiles", []),
                    "cron": values.get("cron", ""),
                    "timezone": values.get("timezone", ""),
                }
            }
        return values

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        return _validate_timezone(value, "schedule timezone")

    @field_validator("profile")
    @classmethod
    def _strip_profile(cls, value: str) -> str:
        return value.strip()

    @field_validator("profiles")
    @classmethod
    def _validate_profiles(cls, value: list[str]) -> list[str]:
        return _normalise_names(value, "schedule profile")

    @field_validator("cron")
    @classmethod
    def _validate_cron(cls, value: str) -> str:
        text = value.strip()
        return validate_cron(text) if text else ""

    @field_validator("schedules")
    @classmethod
    def _name_schedules(
        cls,
        value: dict[str, ScheduleEntryConfig],
    ) -> dict[str, ScheduleEntryConfig]:
        schedules: dict[str, ScheduleEntryConfig] = {}
        for name, entry in value.items():
            key = _text(name, "schedule name")
            if key in schedules:
                raise ValueError(f"duplicate schedule: {key}")
            object.__setattr__(entry, "name", key)
            if not entry.timezone:
                object.__setattr__(entry, "timezone", "")
            schedules[key] = entry
        return schedules

    @model_validator(mode="after")
    def _fill_defaults_and_check(self) -> ScheduleConfig:
        if not self.schedules and self.cron:
            object.__setattr__(
                self,
                "schedules",
                {
                    "default": ScheduleEntryConfig(
                        name="default",
                        profile=self.profile,
                        cron=self.cron,
                        timezone=self.timezone,
                    )
                },
            )
        if self.default and self.schedules and self.default not in self.schedules:
            raise ValueError(f"default schedule is not configured: {self.default}")
        for entry in self.schedules.values():
            if not entry.timezone and self.timezone:
                object.__setattr__(entry, "timezone", self.timezone)
            if not entry.profile and not entry.profiles:
                if self.profile:
                    object.__setattr__(entry, "profile", self.profile)
                elif self.profiles:
                    object.__setattr__(entry, "profiles", list(self.profiles))
        return self

    @model_validator(mode="after")
    def _validate_context(self, info: ValidationInfo) -> ScheduleConfig:
        self.validate_profiles(_context(info, "profiles"))
        return self

    def validate_profiles(self, profiles: Any) -> Self:
        """Check every entry profile against *profiles*.

        *profiles* may be a :class:`ProfilesConfig`, a mapping of names or any
        iterable of profile names. ``None`` skips the check.
        """
        available = _profile_names(profiles)
        if available is None:
            return self
        for name, entry in self.schedules.items():
            referenced = list(entry.profiles)
            if entry.profile:
                referenced.insert(0, entry.profile)
            for profile in referenced:
                if profile not in available:
                    names = ", ".join(sorted(available)) or "none"
                    raise ConfigError(
                        f"schedule {name} references unknown profile: "
                        f"{profile} (available: {names})",
                        details={
                            "schedule": name,
                            "profile": profile,
                            "available": sorted(available),
                        },
                    )
        return self

    def check_profiles(self, profiles: Any) -> Self:
        """Alias for :meth:`validate_profiles`."""
        return self.validate_profiles(profiles)

    def validate_languages(self, languages: Any) -> Self:
        """Check every entry language against *languages*.

        *languages* may be a :class:`LanguagesConfig`, a mapping of names or any
        iterable of language names. ``None`` skips the check.
        """
        available = _language_names(languages)
        if available is None:
            return self
        for name, entry in self.schedules.items():
            for language in entry.languages:
                if language not in available:
                    names = ", ".join(sorted(available)) or "none"
                    raise ConfigError(
                        f"schedule {name} references unknown language: "
                        f"{language} (available: {names})",
                        details={
                            "schedule": name,
                            "language": language,
                            "available": sorted(available),
                        },
                    )
        return self

    def check_languages(self, languages: Any) -> Self:
        """Alias for :meth:`validate_languages`."""
        return self.validate_languages(languages)

    def validate_references(self, config: Any) -> Self:
        """Validate profiles and languages from a parsed root or mapping."""
        self.validate_profiles(_section(config, "profiles"))
        self.validate_languages(_section(config, "languages"))
        return self

    @property
    def entries(self) -> dict[str, ScheduleEntryConfig]:
        """Alias for :attr:`schedules`."""
        return self.schedules

    @property
    def names(self) -> tuple[str, ...]:
        """The configured schedule names in mapping order."""
        return tuple(self.schedules)


ScheduleConfigEntry = ScheduleEntryConfig
ScheduleEntry = ScheduleEntryConfig
ScheduleItemConfig = ScheduleEntryConfig
ScheduleJobConfig = ScheduleEntryConfig
ScheduledRunConfig = ScheduleEntryConfig
CronScheduleConfig = ScheduleEntryConfig


class PublishAccountConfig(ConfigModel):
    """One publishing account and the secret references it needs."""

    platform: str = ""
    language: str = ""
    profile: str = ""
    enabled: bool = True
    credential_refs: list[str] = Field(default_factory=list)
    visibility: str = ""
    dry_run: bool | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"credential_refs": [data]}
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias in ("account", "name", "id"):
            if alias in values:
                values.setdefault("profile", values.pop(alias))
        for alias in ("platform_name", "service"):
            if alias in values:
                values.setdefault("platform", values.pop(alias))
        for alias in ("lang", "locale"):
            if alias in values:
                values.setdefault("language", values.pop(alias))
        for alias in ("profile_name", "profile_id", "content_profile"):
            if alias in values:
                values.setdefault("profile", values.pop(alias))
        for alias in ("credentials", "credential", "secrets", "token_refs"):
            if alias in values:
                values.setdefault("credential_refs", values.pop(alias))
        if "credential_ref" in values:
            ref = values.pop("credential_ref")
            values.setdefault(
                "credential_refs",
                [ref] if isinstance(ref, str) else ref,
            )
        if "credential_refs" in values:
            values["credential_refs"] = _name_values(
                values["credential_refs"], "credential reference"
            )
        if "options" not in values:
            values["options"] = {}
        return values

    @field_validator("platform", "language", "profile", "visibility")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("credential_refs")
    @classmethod
    def _validate_refs(cls, value: list[str]) -> list[str]:
        refs = _normalise_names(value, "credential reference")
        for ref in refs:
            if _REFERENCE_WHITESPACE.search(ref):
                raise ValueError(
                    f"credential reference must not contain whitespace: {ref!r}"
                )
        return refs

    @field_validator("options", mode="before")
    @classmethod
    def _accept_options(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("publish account options must be a mapping")
        return dict(value)

    @model_validator(mode="after")
    def _account_shape(self) -> PublishAccountConfig:
        if bool(self.platform) != bool(self.language) and (
            self.platform or self.language
        ):
            raise ValueError("publish account requires both platform and language")
        return self

    @property
    def credential_ref(self) -> str:
        """The first credential reference, if any."""
        return self.credential_refs[0] if self.credential_refs else ""

    @property
    def credentials(self) -> list[str]:
        """Alias for :attr:`credential_refs`."""
        return self.credential_refs


PublishAccount = PublishAccountConfig
PublishTargetConfig = PublishAccountConfig


class PublishConfig(ConfigModel):
    """Publishing accounts, dry-run policy and reference validation."""

    enabled: bool = True
    dry_run: bool = True
    default_profile: str = ""
    accounts: dict[str, PublishAccountConfig] = Field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        environ: Mapping[str, str] | None = None,
        context: Mapping[str, Any] | None = None,
        profiles: Any = None,
        credentials: Any = None,
        platforms: Any = None,
    ) -> Self:
        """Validate a publish section and optionally check its references."""
        merged = dict(context or {})
        if profiles is not None:
            merged["profiles"] = profiles
        if credentials is not None:
            merged["credentials"] = credentials
        if platforms is not None:
            merged["platforms"] = platforms
        return super().from_mapping(values, environ=environ, context=merged)

    @model_validator(mode="before")
    @classmethod
    def _accept_shapes(cls, data: Any) -> Any:
        if data is None:
            return {}
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias in ("targets", "channels", "destinations"):
            if alias in values:
                values.setdefault("accounts", values.pop(alias))
        for alias in ("profile", "profile_name"):
            if alias in values:
                values.setdefault("default_profile", values.pop(alias))
        if "accounts" in values:
            accounts = values["accounts"]
            if isinstance(accounts, Sequence) and not isinstance(
                accounts, (str, bytes, Mapping)
            ):
                listed: dict[str, Any] = {}
                for index, account in enumerate(accounts):
                    if isinstance(account, str):
                        listed[account] = {}
                    elif isinstance(account, Mapping):
                        item = dict(account)
                        name = item.pop(
                            "name",
                            item.pop(
                                "id", item.get("platform", f"account-{index + 1}")
                            ),
                        )
                        listed[str(name)] = item
                    else:
                        raise ValueError("publish accounts must be names or mappings")
                values["accounts"] = listed
            elif isinstance(accounts, Mapping):
                values["accounts"] = {
                    str(name): ({} if entry is None else entry)
                    for name, entry in accounts.items()
                }
            return values
        global_keys = {"enabled", "dry_run", "default_profile"}
        entries = {
            key: value for key, value in values.items() if key not in global_keys
        }
        if entries:
            for key in list(entries):
                values.pop(key)
            values["accounts"] = entries
        return values

    @field_validator("default_profile")
    @classmethod
    def _strip_profile(cls, value: str) -> str:
        return value.strip()

    @field_validator("accounts")
    @classmethod
    def _name_accounts(
        cls,
        value: dict[str, PublishAccountConfig],
    ) -> dict[str, PublishAccountConfig]:
        accounts: dict[str, PublishAccountConfig] = {}
        for name, account in value.items():
            key = _text(name, "publish account name")
            if key in accounts:
                raise ValueError(f"duplicate publish account: {key}")
            platform, separator, language = key.partition("/")
            if separator:
                if account.platform and account.platform != platform:
                    raise ValueError(
                        f"publish account {key} platform does not match its key"
                    )
                if account.language and account.language != language:
                    raise ValueError(
                        f"publish account {key} language does not match its key"
                    )
                object.__setattr__(account, "platform", platform)
                object.__setattr__(account, "language", language)
            accounts[key] = account
        return accounts

    @model_validator(mode="after")
    def _validate_context(self, info: ValidationInfo) -> PublishConfig:
        self.validate_profiles(_context(info, "profiles"))
        self.validate_credentials(_context(info, "credentials"))
        platforms = _context(info, "platforms")
        if platforms is not None:
            self.validate_platforms(platforms)
        return self

    def validate_profiles(self, profiles: Any) -> Self:
        """Check account profiles against *profiles* when provided."""
        available = _profile_names(profiles)
        if available is None:
            return self
        for name, account in self.accounts.items():
            if account.profile and account.profile not in available:
                names = ", ".join(sorted(available)) or "none"
                raise ConfigError(
                    f"publish account {name} references unknown profile: "
                    f"{account.profile} (available: {names})",
                    details={
                        "account": name,
                        "profile": account.profile,
                        "available": sorted(available),
                    },
                )
        if self.default_profile and self.default_profile not in available:
            names = ", ".join(sorted(available)) or "none"
            raise ConfigError(
                f"publish default profile is not configured: "
                f"{self.default_profile} (available: {names})",
                details={
                    "profile": self.default_profile,
                    "available": sorted(available),
                },
            )
        return self

    def validate_credentials(self, credentials: Any) -> Self:
        """Check every account credential reference against *credentials*.

        *credentials* may be a mapping, an iterable of names or ``None`` to
        skip the existence check. Empty and duplicate references are always
        rejected by the field validator.
        """
        available = _credential_names(credentials)
        if available is None:
            return self
        for name, account in self.accounts.items():
            missing = [ref for ref in account.credential_refs if ref not in available]
            if missing:
                names = ", ".join(sorted(available)) or "none"
                raise ConfigError(
                    f"publish account {name} references unknown credential: "
                    f"{', '.join(missing)} (available: {names})",
                    details={
                        "account": name,
                        "missing": missing,
                        "available": sorted(available),
                    },
                )
        return self

    def check_profiles(self, profiles: Any) -> Self:
        """Alias for :meth:`validate_profiles`."""
        return self.validate_profiles(profiles)

    def check_credentials(self, credentials: Any) -> Self:
        """Alias for :meth:`validate_credentials`."""
        return self.validate_credentials(credentials)

    def validate_platforms(self, platforms: Any) -> Self:
        """Check that publish accounts exist in the platform account list."""
        accounts = _section(platforms, "accounts")
        if not isinstance(accounts, Mapping):
            return self
        available = {str(name) for name in accounts}
        for name in self.accounts:
            if name not in available:
                names = ", ".join(sorted(available)) or "none"
                raise ConfigError(
                    f"publish account is not configured in platforms: {name} "
                    f"(available: {names})",
                    details={"account": name, "available": sorted(available)},
                )
        return self

    def validate_references(self, config: Any) -> Self:
        """Validate profiles, platform accounts and credentials from a root."""
        self.validate_profiles(_section(config, "profiles"))
        platforms = _section(config, "platforms")
        if platforms is not None:
            self.validate_platforms(platforms)
            accounts = _section(platforms, "accounts")
            if isinstance(accounts, Mapping):
                refs: set[str] = set()
                for account in accounts.values():
                    refs.update(getattr(account, "credential_refs", []) or [])
                self.validate_credentials(refs)
        return self

    @property
    def targets(self) -> dict[str, PublishAccountConfig]:
        """Alias for :attr:`accounts`."""
        return self.accounts

    @property
    def names(self) -> tuple[str, ...]:
        """The configured account names in mapping order."""
        return tuple(self.accounts)


PublishAccountsConfig = PublishConfig
PublishSettings = PublishConfig
PublishSettingsConfig = PublishConfig
PublishTargetsConfig = PublishConfig
SchedulesConfig = ScheduleConfig
