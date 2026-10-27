"""Occasion, order and delivery configuration.

Occasions describe the variables a personalized order may carry. Orders define
the personalization levels that select which stages run, how often a spoken
name is verified by transcription and the simple pricing used by receipts.
Delivery defines the per-provider fields and the fallback chain.

Every field has a default, so an empty mapping is a valid configuration. Names
are the canonical identifiers used by the orders module: an occasion is keyed
by its name and a personalization level is keyed by its level name. Provider
credentials are referenced by name and resolved through the secrets loader,
never written into the configuration file.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel

__all__ = [
    "DeliveryConfig",
    "DeliveryProviderConfig",
    "DeliveryProviderType",
    "EmailDeliveryConfig",
    "FolderDeliveryConfig",
    "LinkDeliveryConfig",
    "LocalDeliveryConfig",
    "MailDeliveryConfig",
    "NameVerificationConfig",
    "NameVerifyConfig",
    "OccasionConfig",
    "OccasionVariableConfig",
    "OccasionsConfig",
    "OrderPricingConfig",
    "OrderVariableConfig",
    "OrdersConfig",
    "PersonalizationLevel",
    "PersonalizationLevelConfig",
    "PricingConfig",
    "VariableConfig",
    "VariableKind",
    "VariableType",
    "WebhookDeliveryConfig",
]


class VariableType(StrEnum):
    """The value type a personalized order variable accepts."""

    STRING = "string"
    TEXT = "text"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    DATE = "date"
    CHOICE = "choice"
    FILE = "file"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


VariableKind = VariableType
_VARIABLE_TYPE_ALIASES = {
    "str": VariableType.STRING,
    "text": VariableType.TEXT,
    "string": VariableType.STRING,
    "number": VariableType.NUMBER,
    "float": VariableType.NUMBER,
    "int": VariableType.INTEGER,
    "integer": VariableType.INTEGER,
    "bool": VariableType.BOOLEAN,
    "boolean": VariableType.BOOLEAN,
    "date": VariableType.DATE,
    "enum": VariableType.CHOICE,
    "choice": VariableType.CHOICE,
    "select": VariableType.CHOICE,
    "upload": VariableType.FILE,
    "file": VariableType.FILE,
    "image": VariableType.IMAGE,
    "audio": VariableType.AUDIO,
    "video": VariableType.VIDEO,
}


def _text(value: str, label: str) -> str:
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


def _normalise_variables(value: Any) -> Any:
    """Accept a mapping, a list of names or a list of variable mappings."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        variables: dict[str, Any] = {}
        for name, entry in value.items():
            key = _text(str(name), "variable name")
            if isinstance(entry, OccasionVariableConfig):
                variables[key] = entry
                continue
            if isinstance(entry, Mapping):
                variable = dict(entry)
                variable.setdefault("name", key)
            elif isinstance(entry, bool):
                variable = {"name": key, "required": entry}
            elif isinstance(entry, str):
                variable = {"name": key, "type": entry}
            else:
                variable = {"name": key, "default": entry}
            variables[key] = variable
        return variables
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        variables = {}
        for entry in value:
            if isinstance(entry, str):
                key = _text(entry, "variable name")
                variable = {"name": key}
            elif isinstance(entry, Mapping):
                variable = dict(entry)
                raw_name = variable.get(
                    "name",
                    variable.get("key", variable.get("id")),
                )
                if not isinstance(raw_name, str):
                    raise ValueError("variable entries must have a name")
                key = _text(raw_name, "variable name")
                variable.setdefault("name", key)
            else:
                raise ValueError("variables must be names or mappings")
            if key in variables:
                raise ValueError(f"duplicate variable: {key}")
            variables[key] = variable
        return variables
    raise ValueError("variables must be a mapping or a list")


class OccasionVariableConfig(ConfigModel):
    """One variable a personalized occasion asks the customer to provide.

    A bare name is accepted as shorthand. ``type`` selects the validation the
    order input applies, ``required`` marks mandatory fields and ``default`` is
    used when an optional variable is omitted. Choice variables list their
    allowed ``options``.
    """

    name: str = ""
    label: str = ""
    type: VariableType = VariableType.STRING
    required: bool = False
    default: Any = None
    description: str = ""
    options: list[str] = Field(default_factory=list)
    min_length: int | None = Field(default=None, ge=0)
    max_length: int | None = Field(default=None, gt=0)
    pattern: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"name": data}
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias in ("key", "id", "variable"):
            if alias in values:
                values.setdefault("name", values.pop(alias))
        for alias in ("kind", "value_type", "format"):
            if alias in values:
                values.setdefault("type", values.pop(alias))
        for alias in ("values", "choices", "allowed"):
            if alias in values:
                values.setdefault("options", values.pop(alias))
        if "max" in values:
            values.setdefault("max_length", values.pop("max"))
        if "min" in values:
            values.setdefault("min_length", values.pop("min"))
        if "help" in values:
            values.setdefault("description", values.pop("help"))
        return values

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return _text(value, "variable name") if value else ""

    @field_validator("type", mode="before")
    @classmethod
    def _normalise_type(cls, value: Any) -> Any:
        if isinstance(value, VariableType):
            return value
        if isinstance(value, str):
            key = value.strip().lower()
            if key in _VARIABLE_TYPE_ALIASES:
                return _VARIABLE_TYPE_ALIASES[key]
            if key in {item.value for item in VariableType}:
                return VariableType(key)
        return value

    @field_validator("options")
    @classmethod
    def _validate_options(cls, value: list[str]) -> list[str]:
        return _normalise_names(value, "variable option")

    @model_validator(mode="after")
    def _ordered_lengths(self) -> OccasionVariableConfig:
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.min_length > self.max_length
        ):
            raise ValueError("variable min_length must not exceed max_length")
        return self

    @property
    def key(self) -> str:
        """Alias for :attr:`name`."""
        return self.name

    @property
    def kind(self) -> VariableType:
        """Alias for :attr:`type`."""
        return self.type

    @property
    def choices(self) -> list[str]:
        """Alias for :attr:`options`."""
        return self.options


VariableConfig = OccasionVariableConfig
OrderVariableConfig = OccasionVariableConfig


class OccasionConfig(ConfigModel):
    """One config-defined personalization type, such as a birthday.

    ``variables`` are the fields an order of this occasion may fill.
    ``personalization_levels`` restricts which order levels are allowed, while
    ``files`` lists optional prompt or template files shipped with a custom
    occasion.
    """

    name: str = ""
    label: str = ""
    description: str = ""
    variables: dict[str, OccasionVariableConfig] = Field(default_factory=dict)
    personalization_levels: list[str] = Field(default_factory=list)
    formats: list[str] = Field(default_factory=list)
    format: str = ""
    template: str = ""
    prompt: str = ""
    files: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    default_level: str = ""
    enabled: bool = True

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias in ("id", "key"):
            if alias in values:
                values.setdefault("name", values.pop(alias))
        for alias in ("title", "display_name"):
            if alias in values:
                values.setdefault("label", values.pop(alias))
        for alias in ("levels", "allowed_levels"):
            if alias in values:
                values.setdefault("personalization_levels", values.pop(alias))
        for alias in ("recipes", "outputs"):
            if alias in values:
                values.setdefault("formats", values.pop(alias))
        for alias in ("recipe", "format_name"):
            if alias in values:
                values.setdefault("format", values.pop(alias))
        for alias in ("templates", "prompt_files", "attachments"):
            if alias in values:
                values.setdefault("files", values.pop(alias))
        if "variables" in values:
            values["variables"] = _normalise_variables(values["variables"])
        levels = values.get("personalization_levels")
        if isinstance(levels, Mapping):
            values["personalization_levels"] = list(levels)
        return values

    @field_validator("personalization_levels")
    @classmethod
    def _validate_levels(cls, value: list[str]) -> list[str]:
        return _normalise_names(value, "personalization level")

    @field_validator("formats", "files", "languages")
    @classmethod
    def _validate_names(cls, value: list[str]) -> list[str]:
        return _normalise_names(value, "name")

    @field_validator("variables")
    @classmethod
    def _name_variables(
        cls,
        value: dict[str, OccasionVariableConfig],
    ) -> dict[str, OccasionVariableConfig]:
        for key, variable in value.items():
            if not key.strip():
                raise ValueError("variable name must not be empty")
            object.__setattr__(variable, "name", key)
        return value

    @property
    def id(self) -> str:
        """Alias for :attr:`name`."""
        return self.name

    @property
    def levels(self) -> list[str]:
        """Alias for :attr:`personalization_levels`."""
        return self.personalization_levels

    @property
    def variable_names(self) -> list[str]:
        """The names of every variable in declaration order."""
        return list(self.variables)


Occasion = OccasionConfig


class OccasionsConfig(ConfigModel):
    """The configured occasions and the one selected by default.

    Occasions may be written directly under the section
    (``occasions: {birthday: {...}}``) or nested under ``occasions``.
    """

    default: str = ""
    enabled: bool = True
    occasions: dict[str, OccasionConfig] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_direct_mapping(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias in ("items", "definitions", "types"):
            if alias in values:
                values.setdefault("occasions", values.pop(alias))
        if "occasions" not in values:
            default = values.pop("default", "")
            enabled = values.pop("enabled", True)
            return {
                "default": default,
                "enabled": enabled,
                "occasions": values,
            }
        return values

    @field_validator("occasions")
    @classmethod
    def _name_occasions(
        cls,
        value: dict[str, OccasionConfig],
    ) -> dict[str, OccasionConfig]:
        for key, occasion in value.items():
            if not key.strip():
                raise ValueError("occasion name must not be empty")
            object.__setattr__(occasion, "name", key)
        return value

    @model_validator(mode="after")
    def _default_exists(self) -> OccasionsConfig:
        if self.occasions and self.default and self.default not in self.occasions:
            raise ValueError(f"default occasion is not configured: {self.default}")
        return self


class PersonalizationLevelConfig(ConfigModel):
    """One personalization level: the stages it runs and what it costs.

    ``stages`` is the ordered subset of pipeline stages the level runs.
    ``price`` is the line-item amount before the flat tax estimate. The feature
    flags say which optional order variables the level uses.
    """

    name: str = ""
    description: str = ""
    stages: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)
    price: float = Field(default=0.0, ge=0)
    cost_multiplier: float = Field(default=1.0, ge=0)
    personalize_name: bool = True
    include_age: bool = False
    include_interests: bool = False
    include_message: bool = True
    background: str = ""
    format: str = ""
    enabled: bool = True

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias in ("id", "key", "level"):
            if alias in values:
                values.setdefault("name", values.pop(alias))
        for alias in ("steps", "stage_names"):
            if alias in values:
                values.setdefault("stages", values.pop(alias))
        for alias in ("media", "artifacts", "output_types"):
            if alias in values:
                values.setdefault("outputs", values.pop(alias))
        for alias in ("amount", "cost", "unit_price", "base_price"):
            if alias in values:
                values.setdefault("price", values.pop(alias))
        if "features" in values:
            features = values.pop("features")
            if isinstance(features, Mapping):
                for key, item in features.items():
                    values.setdefault(key, item)
        if "price" in values and isinstance(values["price"], Mapping):
            price = values["price"]
            for key in ("amount", "value", "price", "usd"):
                if key in price:
                    values["price"] = price[key]
                    break
        return values

    @field_validator("stages", "outputs", "variables")
    @classmethod
    def _validate_unique_names(cls, value: list[str]) -> list[str]:
        return _normalise_names(value, "name")

    @field_validator("price", "cost_multiplier")
    @classmethod
    def _finite_amount(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("amount must be finite")
        return value

    @property
    def steps(self) -> list[str]:
        """Alias for :attr:`stages`."""
        return self.stages

    @property
    def amount(self) -> float:
        """Alias for :attr:`price`."""
        return self.price


PersonalizationLevel = PersonalizationLevelConfig


class NameVerifyConfig(ConfigModel):
    """How often a generated name is transcribed and checked before accepting.

    Verification regenerates the spoken name up to ``retries`` times when the
    transcription does not match within ``threshold``.
    """

    enabled: bool = True
    retries: int = Field(default=3, ge=0)
    threshold: float = Field(default=0.8, ge=0, le=1)
    regenerate: bool = True
    language: str = ""

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthand(cls, data: Any) -> Any:
        if isinstance(data, bool):
            return {"enabled": data}
        if isinstance(data, int):
            return {"retries": data}
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias in ("max_retries", "max_attempts", "verify_retries", "attempts"):
            if alias in values:
                values.setdefault("retries", values.pop(alias))
        if "min_score" in values:
            values.setdefault("threshold", values.pop("min_score"))
        if "on_mismatch" in values:
            values.setdefault("regenerate", values.pop("on_mismatch"))
        return values

    @property
    def max_retries(self) -> int:
        """Alias for :attr:`retries`."""
        return self.retries

    @property
    def max_attempts(self) -> int:
        """Alias for :attr:`retries`."""
        return self.retries

    @property
    def verify_retries(self) -> int:
        """Alias for :attr:`retries`."""
        return self.retries


NameVerificationConfig = NameVerifyConfig


class OrderPricingConfig(ConfigModel):
    """Currency, line-item amount and one flat tax estimate for receipts.

    This is deliberately not a tax engine: ``tax_rate`` is a single flat rate
    and receipts state that they are not tax invoices (decision D-011).
    """

    currency: str = "USD"
    amount: float = Field(default=0.0, ge=0)
    tax_rate: float = Field(default=0.0, ge=0, le=1)
    tax_included: bool = False
    tax_label: str = "Estimated tax"
    per_output: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias in ("currency_code", "code"):
            if alias in values:
                values.setdefault("currency", values.pop(alias))
        for alias in ("price", "base_price", "unit_price", "subtotal"):
            if alias in values:
                values.setdefault("amount", values.pop(alias))
        for alias in ("flat_tax_rate", "tax_percent", "vat_rate", "tax"):
            if alias in values:
                values.setdefault("tax_rate", values.pop(alias))
        for alias in ("taxes_included", "include_tax"):
            if alias in values:
                values.setdefault("tax_included", values.pop(alias))
        return values

    @field_validator("currency", mode="before")
    @classmethod
    def _normalise_currency(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("currency")
    @classmethod
    def _validate_currency(cls, value: str) -> str:
        if len(value) != 3 or not value.isalpha():
            raise ValueError("currency must be a three-letter code")
        return value

    @field_validator("amount", "tax_rate")
    @classmethod
    def _finite_amount(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("amount must be finite")
        return value

    @field_validator("per_output")
    @classmethod
    def _validate_per_output(cls, value: dict[str, float]) -> dict[str, float]:
        for name, amount in value.items():
            if not name.strip():
                raise ValueError("output name must not be empty")
            if not math.isfinite(amount) or amount < 0:
                raise ValueError(
                    "per output amount must be a finite non-negative number"
                )
        return value

    @property
    def price(self) -> float:
        """Alias for :attr:`amount`."""
        return self.amount

    @property
    def flat_tax_rate(self) -> float:
        """Alias for :attr:`tax_rate`."""
        return self.tax_rate

    @property
    def tax_percent(self) -> float:
        """The flat tax rate expressed as a percentage."""
        return self.tax_rate * 100

    @property
    def tax_amount(self) -> float:
        """The flat tax estimate for the line-item amount."""
        return self.amount * self.tax_rate

    @property
    def currency_code(self) -> str:
        """Alias for :attr:`currency`."""
        return self.currency


PricingConfig = OrderPricingConfig


def _normalise_levels(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        levels: dict[str, Any] = {}
        for name, entry in value.items():
            key = _text(str(name), "personalization level")
            if isinstance(entry, PersonalizationLevelConfig):
                levels[key] = entry
                continue
            if isinstance(entry, Mapping):
                level = dict(entry)
                level.setdefault("name", key)
            elif isinstance(entry, bool):
                level = {"name": key, "enabled": entry}
            elif isinstance(entry, (int, float)):
                level = {"name": key, "price": entry}
            elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
                level = {"name": key, "stages": list(entry)}
            else:
                level = {"name": key}
            levels[key] = level
        return levels
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        levels = {}
        for entry in value:
            if isinstance(entry, str):
                key = _text(entry, "personalization level")
                level = {"name": key}
            elif isinstance(entry, Mapping):
                level = dict(entry)
                raw_name = level.get("name", level.get("id", level.get("key")))
                if not isinstance(raw_name, str):
                    raise ValueError("personalization levels must have a name")
                key = _text(raw_name, "personalization level")
                level.setdefault("name", key)
            else:
                raise ValueError("personalization levels must be names or mappings")
            if key in levels:
                raise ValueError(f"duplicate personalization level: {key}")
            levels[key] = level
        return levels
    raise ValueError("personalization levels must be a mapping or a list")


class OrdersConfig(ConfigModel):
    """Personalized order defaults: levels, name verification and pricing."""

    enabled: bool = True
    variables: dict[str, OccasionVariableConfig] = Field(default_factory=dict)
    personalization_levels: dict[str, PersonalizationLevelConfig] = Field(
        default_factory=dict
    )
    default_level: str = ""
    name_verify_retries: int = Field(default=3, ge=0)
    name_verify: NameVerifyConfig = Field(default_factory=NameVerifyConfig)
    pricing: OrderPricingConfig = Field(default_factory=OrderPricingConfig)
    timezone: str = "UTC"
    max_batch_size: int = Field(default=100, ge=1)
    max_name_length: int = Field(default=80, gt=0)
    max_message_length: int = Field(default=1000, gt=0)
    max_variables: int = Field(default=32, ge=0)
    allow_custom_occasion: bool = True
    require_age: bool = False
    require_message: bool = False

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias in ("levels", "personalization"):
            if alias in values:
                values.setdefault("personalization_levels", values.pop(alias))
        for alias in ("level", "default_personalization"):
            if alias in values:
                values.setdefault("default_level", values.pop(alias))
        for alias in ("price", "prices"):
            if alias in values:
                price = values.pop(alias)
                values.setdefault(
                    "pricing",
                    {"amount": price} if isinstance(price, (int, float)) else price,
                )
        for alias in ("name_verification", "verify_name", "name_check"):
            if alias in values:
                values.setdefault("name_verify", values.pop(alias))
        for alias in (
            "verify_retries",
            "name_retries",
            "name_verification_retries",
            "name_verify_attempts",
        ):
            if alias in values:
                values.setdefault("name_verify_retries", values.pop(alias))

        if "variables" in values:
            values["variables"] = _normalise_variables(values["variables"])
        if "personalization_levels" in values:
            values["personalization_levels"] = _normalise_levels(
                values["personalization_levels"]
            )

        verify = values.get("name_verify")
        if verify is None:
            verify = {}
        elif isinstance(verify, (bool, int)):
            verify = (
                {"enabled": bool(verify)}
                if isinstance(verify, bool)
                else {"retries": verify}
            )
        if isinstance(verify, Mapping):
            nested = dict(verify)
            if "name_verify_retries" in values and "retries" not in nested:
                nested["retries"] = values["name_verify_retries"]
            elif "retries" in nested:
                values["name_verify_retries"] = nested["retries"]
            values["name_verify"] = nested
        return values

    @field_validator("variables")
    @classmethod
    def _name_variables(
        cls,
        value: dict[str, OccasionVariableConfig],
    ) -> dict[str, OccasionVariableConfig]:
        for key, variable in value.items():
            if not key.strip():
                raise ValueError("variable name must not be empty")
            object.__setattr__(variable, "name", key)
        return value

    @field_validator("personalization_levels")
    @classmethod
    def _name_levels(
        cls,
        value: dict[str, PersonalizationLevelConfig],
    ) -> dict[str, PersonalizationLevelConfig]:
        for key, level in value.items():
            if not key.strip():
                raise ValueError("personalization level name must not be empty")
            object.__setattr__(level, "name", key)
        return value

    @model_validator(mode="after")
    def _default_level_exists(self) -> OrdersConfig:
        if (
            self.personalization_levels
            and self.default_level
            and self.default_level not in self.personalization_levels
        ):
            raise ValueError(
                f"default personalization level is not configured: {self.default_level}"
            )
        if self.name_verify.retries != self.name_verify_retries:
            self.name_verify.retries = self.name_verify_retries
        return self

    @property
    def levels(self) -> dict[str, PersonalizationLevelConfig]:
        """Alias for :attr:`personalization_levels`."""
        return self.personalization_levels

    @property
    def default_personalization(self) -> str:
        """Alias for :attr:`default_level`."""
        return self.default_level

    @property
    def price(self) -> OrderPricingConfig:
        """Alias for :attr:`pricing`."""
        return self.pricing


OrderConfig = OrdersConfig


class DeliveryProviderType(StrEnum):
    """The delivery providers the orders module knows how to configure."""

    FOLDER = "folder"
    LINK = "link"
    EMAIL = "email"
    WEBHOOK = "webhook"

    LOCAL = "folder"
    FILE = "folder"
    STORAGE_LINK = "link"
    MAIL = "email"
    HTTP = "webhook"


_DELIVERY_TYPE_ALIASES = {
    "folder": DeliveryProviderType.FOLDER,
    "local": DeliveryProviderType.FOLDER,
    "file": DeliveryProviderType.FOLDER,
    "filesystem": DeliveryProviderType.FOLDER,
    "link": DeliveryProviderType.LINK,
    "storage_link": DeliveryProviderType.LINK,
    "signed_link": DeliveryProviderType.LINK,
    "expiring_link": DeliveryProviderType.LINK,
    "local_folder": DeliveryProviderType.FOLDER,
    "expiring_storage_link": DeliveryProviderType.LINK,
    "email": DeliveryProviderType.EMAIL,
    "mail": DeliveryProviderType.EMAIL,
    "smtp": DeliveryProviderType.EMAIL,
    "webhook": DeliveryProviderType.WEBHOOK,
    "http": DeliveryProviderType.WEBHOOK,
    "signed_webhook": DeliveryProviderType.WEBHOOK,
}

_PROVIDER_KEY_ALIASES = {
    "local": "folder",
    "local_folder": "folder",
    "file": "folder",
    "filesystem": "folder",
    "folder": "folder",
    "storage_link": "link",
    "signed_link": "link",
    "expiring_link": "link",
    "expiring_storage_link": "link",
    "link": "link",
    "mail": "email",
    "smtp": "email",
    "email": "email",
    "webhook": "webhook",
    "signed_webhook": "webhook",
    "http": "webhook",
}


def _canonical_provider(name: str) -> str:
    key = name.strip().lower()
    return _PROVIDER_KEY_ALIASES.get(key, key)


class DeliveryProviderConfig(ConfigModel):
    """Fields shared by every delivery provider.

    Provider-specific fields are present but only the ones relevant to the
    selected ``type`` are used. Credentials are referenced by name.
    """

    type: str = ""
    enabled: bool = True
    timeout_seconds: int = Field(default=30, gt=0)
    max_attempts: int = Field(default=3, ge=1)
    retry_seconds: int = Field(default=60, ge=0)

    path: str = ""
    expires_seconds: int = Field(default=604_800, gt=0)

    from_address: str = ""
    reply_to: str = ""
    subject: str = ""
    smtp_host: str = ""
    smtp_port: int = Field(default=587, gt=0, le=65_535)
    username_ref: str = ""
    password_ref: str = ""

    url: str = ""
    secret_ref: str = ""
    allowlist: list[str] = Field(default_factory=list)
    signature_header: str = "X-Aivia-Signature"

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias in ("kind", "transport", "provider"):
            if alias in values:
                values.setdefault("type", values.pop(alias))
        for alias in ("directory", "folder", "dir", "local_dir", "output_dir"):
            if alias in values:
                values.setdefault("path", values.pop(alias))
        for alias in ("ttl", "ttl_seconds", "expiry_seconds", "link_ttl"):
            if alias in values:
                values.setdefault("expires_seconds", values.pop(alias))
        for alias in ("from", "from_email", "sender", "sender_address"):
            if alias in values:
                values.setdefault("from_address", values.pop(alias))
        for alias in ("allowed_hosts", "hosts", "domains"):
            if alias in values:
                values.setdefault("allowlist", values.pop(alias))
        for alias in ("webhook_url", "callback_url"):
            if alias in values:
                values.setdefault("url", values.pop(alias))
        for alias in ("secret", "secret_name", "signing_secret_ref"):
            if alias in values:
                values.setdefault("secret_ref", values.pop(alias))
        for alias in ("retries", "attempts"):
            if alias in values:
                values.setdefault("max_attempts", values.pop(alias))
        if "timeout" in values:
            values.setdefault("timeout_seconds", values.pop("timeout"))
        return values

    @field_validator("type", mode="before")
    @classmethod
    def _normalise_type(cls, value: Any) -> Any:
        if isinstance(value, DeliveryProviderType):
            return value.value
        if isinstance(value, str):
            key = value.strip().lower()
            if key in _DELIVERY_TYPE_ALIASES:
                return _DELIVERY_TYPE_ALIASES[key].value
            if key in {item.value for item in DeliveryProviderType}:
                return DeliveryProviderType(key).value
        return value

    @field_validator("allowlist")
    @classmethod
    def _normalise_allowlist(cls, value: list[str]) -> list[str]:
        hosts: list[str] = []
        seen: set[str] = set()
        for entry in value:
            host = _text(entry, "allowlist host").lower()
            if host in seen:
                raise ValueError(f"duplicate allowlist host: {host}")
            seen.add(host)
            hosts.append(host)
        return hosts

    @model_validator(mode="after")
    def _validate_provider(self) -> DeliveryProviderConfig:
        if (
            self.type == DeliveryProviderType.WEBHOOK.value
            and self.url
            and not self.url.startswith(("http://", "https://"))
        ):
            raise ValueError("webhook url must be http or https")
        return self

    @property
    def kind(self) -> str:
        """Alias for :attr:`type`."""
        return self.type

    @property
    def folder(self) -> str:
        """Alias for :attr:`path`."""
        return self.path

    @property
    def ttl_seconds(self) -> int:
        """Alias for :attr:`expires_seconds`."""
        return self.expires_seconds

    @property
    def sender(self) -> str:
        """Alias for :attr:`from_address`."""
        return self.from_address

    @property
    def allowed_hosts(self) -> list[str]:
        """Alias for :attr:`allowlist`."""
        return self.allowlist


class LocalDeliveryConfig(DeliveryProviderConfig):
    """Deliver by copying final outputs into a local folder."""

    type: str = DeliveryProviderType.FOLDER.value
    path: str = "deliveries"


class LinkDeliveryConfig(DeliveryProviderConfig):
    """Deliver an expiring link to the storage backend."""

    type: str = DeliveryProviderType.LINK.value


class EmailDeliveryConfig(DeliveryProviderConfig):
    """Deliver by email; credentials are referenced by name."""

    type: str = DeliveryProviderType.EMAIL.value
    subject: str = "Your personalized card"


class WebhookDeliveryConfig(DeliveryProviderConfig):
    """Deliver a signed webhook to an allowlisted host."""

    type: str = DeliveryProviderType.WEBHOOK.value


FolderDeliveryConfig = LocalDeliveryConfig
MailDeliveryConfig = EmailDeliveryConfig

_PROVIDER_MODELS: dict[str, type[DeliveryProviderConfig]] = {
    DeliveryProviderType.FOLDER.value: LocalDeliveryConfig,
    DeliveryProviderType.LINK.value: LinkDeliveryConfig,
    DeliveryProviderType.EMAIL.value: EmailDeliveryConfig,
    DeliveryProviderType.WEBHOOK.value: WebhookDeliveryConfig,
}


def _provider_from_entry(name: str, entry: Any) -> Any:
    canonical = _canonical_provider(name)
    if isinstance(entry, str):
        if canonical == "folder":
            return {"type": canonical, "path": entry}
        if canonical == "email":
            return {"type": canonical, "from_address": entry}
        if canonical == "webhook":
            return {"type": canonical, "url": entry}
        if canonical == "link":
            return {"type": canonical, "expires_seconds": entry}
        return entry
    if isinstance(entry, Mapping):
        provider = dict(entry)
        provider.setdefault("type", canonical)
        return provider
    return entry


class DeliveryConfig(ConfigModel):
    """Delivery providers, the fallback chain and shared retry settings.

    Providers may be nested under ``providers`` or written directly as
    ``folder``, ``link``, ``email`` and ``webhook`` keys. The ``chain`` lists
    provider names in fallback order and every entry must name a provider.
    """

    enabled: bool = True
    providers: dict[str, DeliveryProviderConfig] = Field(default_factory=dict)
    chain: list[str] = Field(default_factory=list)
    default: str = "folder"
    retry_seconds: int = Field(default=60, ge=0)
    max_attempts: int = Field(default=3, ge=1)
    notify_on_failure: bool = True

    @model_validator(mode="before")
    @classmethod
    def _accept_direct_providers(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias in ("targets", "channels", "backends", "transports"):
            if alias in values:
                values.setdefault("providers", values.pop(alias))
        for alias in ("fallback", "order", "fallback_chain"):
            if alias in values:
                values.setdefault("chain", values.pop(alias))
        for alias in ("provider", "primary", "default_provider"):
            if alias in values:
                values.setdefault("default", values.pop(alias))

        providers = values.get("providers", {})
        providers = {} if not isinstance(providers, Mapping) else dict(providers)

        for key in (
            "folder",
            "local",
            "file",
            "filesystem",
            "local_folder",
            "link",
            "storage_link",
            "signed_link",
            "expiring_link",
            "expiring_storage_link",
            "email",
            "mail",
            "smtp",
            "webhook",
            "signed_webhook",
            "http",
        ):
            if key in values:
                provider = values.pop(key)
                providers.setdefault(key, provider)

        normalised: dict[str, Any] = {}
        for name, entry in providers.items():
            key = _text(str(name), "delivery provider name")
            normalised[key] = _provider_from_entry(key, entry)
        values["providers"] = normalised
        return values

    @field_validator("providers")
    @classmethod
    def _specialise_providers(
        cls,
        value: dict[str, DeliveryProviderConfig],
    ) -> dict[str, DeliveryProviderConfig]:
        specialised: dict[str, DeliveryProviderConfig] = {}
        for name, provider in value.items():
            model = _PROVIDER_MODELS.get(provider.type)
            if model is None:
                specialised[name] = provider
            else:
                specialised[name] = model.model_validate(
                    provider.model_dump(exclude_unset=True)
                )
        return specialised

    @field_validator("chain")
    @classmethod
    def _validate_chain(cls, value: list[str]) -> list[str]:
        return _normalise_names(value, "delivery chain entry")

    @model_validator(mode="after")
    def _references_exist(self) -> DeliveryConfig:
        if not self.providers:
            return self
        available = set(self.providers)
        aliases = {name: _canonical_provider(name) for name in self.providers}
        canonical_available = set(aliases.values())

        resolved: list[str] = []
        for entry in self.chain:
            if entry in available:
                resolved.append(entry)
            elif _canonical_provider(entry) in canonical_available:
                resolved.append(
                    next(
                        name
                        for name, canonical in aliases.items()
                        if canonical == _canonical_provider(entry)
                    )
                )
            else:
                raise ValueError(
                    f"delivery chain references an unknown provider: {entry}"
                )
        object.__setattr__(self, "chain", resolved)

        if self.default:
            if self.default in available:
                return self
            canonical = _canonical_provider(self.default)
            if canonical in canonical_available:
                self.default = next(
                    name for name, value in aliases.items() if value == canonical
                )
                return self
            raise ValueError(
                f"default delivery provider is not configured: {self.default}"
            )
        return self

    def provider(self, name: str) -> DeliveryProviderConfig | None:
        """Return the configured provider *name* or its alias."""
        if name in self.providers:
            return self.providers[name]
        canonical = _canonical_provider(name)
        for key, provider in self.providers.items():
            if _canonical_provider(key) == canonical:
                return provider
        return None

    @property
    def targets(self) -> dict[str, DeliveryProviderConfig]:
        """Alias for :attr:`providers`."""
        return self.providers

    @property
    def fallback(self) -> list[str]:
        """Alias for :attr:`chain`."""
        return self.chain

    @property
    def local(self) -> DeliveryProviderConfig | None:
        """The local folder provider, if configured."""
        return self.provider("folder")

    @property
    def folder(self) -> DeliveryProviderConfig | None:
        """The local folder provider, if configured."""
        return self.provider("folder")

    @property
    def local_folder(self) -> DeliveryProviderConfig | None:
        """The local folder provider, if configured."""
        return self.provider("folder")

    @property
    def link(self) -> DeliveryProviderConfig | None:
        """The expiring link provider, if configured."""
        return self.provider("link")

    @property
    def storage_link(self) -> DeliveryProviderConfig | None:
        """The expiring link provider, if configured."""
        return self.provider("link")

    @property
    def email(self) -> DeliveryProviderConfig | None:
        """The email provider, if configured."""
        return self.provider("email")

    @property
    def mail(self) -> DeliveryProviderConfig | None:
        """The email provider, if configured."""
        return self.provider("email")

    @property
    def webhook(self) -> DeliveryProviderConfig | None:
        """The webhook provider, if configured."""
        return self.provider("webhook")


Delivery = DeliveryConfig
