"""Declarative HTTP provider engine.

A provider that speaks a plain JSON-over-HTTP API does not need a Python class
of its own. A YAML spec can describe the endpoint, the request body, the
response fields, the polling loop and the price. This module turns such a spec
into a normal :class:`~aivia.providers.base.Provider`, so the router, budget
tracker and media service treat it exactly like a hand-written adapter.

Two API shapes are supported:

* **sync** -- one request returns the finished media. ``request`` and
  ``response`` describe it.
* **async** -- a submit request returns a remote job id and a poll request is
  repeated until the job is terminal. ``async.submit``, ``async.poll`` and an
  optional ``async.cancel`` describe the queue.

Request bodies, URLs, headers and query values are rendered from Jinja2
templates in a :class:`~jinja2.sandbox.SandboxedEnvironment` with strict
undefined variables. A template may not include, extend or import another file,
and the sandbox blocks Python attribute escapes such as ``__class__``. User text
is therefore data even when it reaches a template.

Response extraction uses a small path language: ``data.images[0].url`` or a JSON
pointer such as ``/data/images/0/url``. Canonical fields (``url``, ``base64``,
``width``, ``duration_seconds`` and so on) may be remapped per spec, which keeps
vendor field names out of the engine.

Prices are ordinary :class:`~aivia.core.models.provider.ProviderPrice` values,
so the provider metadata exposes them to the router and the pricing resolver
without a second price model.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import httpx
import yaml
from jinja2 import StrictUndefined, TemplateError, Undefined, nodes
from jinja2.sandbox import SandboxedEnvironment
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from sqlalchemy import Engine

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import (
    ConfigError,
    ContentPolicyError,
    InvalidRequestError,
    ValidationError,
)
from aivia.core.http import HttpClient
from aivia.core.models.asset import AssetLicense
from aivia.core.models.provider import ProviderMeta, ProviderPrice, RateLimit
from aivia.core.secrets import SecretsLoader
from aivia.providers.base import Provider
from aivia.providers.media.async_job import (
    AsyncJob,
    AsyncJobState,
    AsyncJobStatus,
    AsyncMediaJobHelper,
)
from aivia.providers.protocols.audio import AudioFormat
from aivia.providers.protocols.base import validate_http_url
from aivia.providers.protocols.image import ImageRequest, ImageResponse, ImageResult
from aivia.providers.protocols.media import ImageFormat, VideoFormat
from aivia.providers.protocols.music import MusicRequest, MusicResponse, MusicTrack
from aivia.providers.protocols.tts import TTSRequest, TTSResponse
from aivia.providers.protocols.video import VideoRequest, VideoResponse, VideoResult

__all__ = [
    "DEFAULT_RESULT_FIELDS",
    "DeclarativeHttpProvider",
    "HTTPProvider",
    "HttpAsyncSpec",
    "HttpProviderSpec",
    "HttpRequestSpec",
    "HttpResponseSpec",
    "HttpSpec",
    "HttpSpecEngine",
    "HttpSpecError",
    "HttpSpecProvider",
    "HttpSpecResult",
    "HttpSpecTemplateError",
    "ProviderSpec",
    "build_http_provider",
    "extract_path",
    "http_provider_from_spec",
    "load_http_spec",
    "load_http_spec_file",
    "load_http_specs",
    "parse_http_spec",
]

_SINGLE_EXPRESSION = re.compile(r"^\s*\{\{(?P<expression>.*?)\}\}\s*$", re.DOTALL)
_SECRET_REFERENCE = re.compile(
    r"secrets(?:\.(?P<dot>[A-Za-z_][A-Za-z0-9_]*)"
    r"|\[\s*['\"](?P<bracket>[A-Za-z_][A-Za-z0-9_]*)['\"]\s*\])"
)
_COMPOSITION_TAGS: dict[type[nodes.Node], str] = {
    nodes.Include: "include",
    nodes.Extends: "extends",
    nodes.Import: "import",
    nodes.FromImport: "from",
}

_METHODS = frozenset(
    {"GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE", "TRACE"}
)
_ENCODINGS = frozenset({"json", "text", "binary"})

DEFAULT_RESULT_FIELDS: dict[str, tuple[str, ...]] = {
    "url": ("url", "result_url", "download_url", "image_url", "video_url", "audio_url"),
    "base64": (
        "b64_json",
        "b64",
        "base64",
        "image_base64",
        "video_base64",
        "audio_base64",
    ),
    "data": ("data", "bytes"),
    "format": ("format", "mime_type", "content_type"),
    "width": ("width",),
    "height": ("height",),
    "duration_seconds": ("duration_seconds", "duration", "seconds"),
    "fps": ("fps", "frame_rate"),
    "seed": ("seed",),
    "revised_prompt": ("revised_prompt",),
    "thumbnail_url": ("thumbnail_url", "thumbnail"),
    "has_audio": ("has_audio", "audio"),
    "title": ("title", "name"),
    "lyrics": ("lyrics",),
    "sample_rate": ("sample_rate", "sample_rate_hz"),
    "channels": ("channels",),
    "model": ("model", "model_id"),
}
"""Canonical result fields and the vendor names tried when no path is given."""

_FIELD_ALIASES: dict[str, str] = {
    "b64": "base64",
    "b64_json": "base64",
    "base64_data": "base64",
    "audio": "base64",
    "audio_base64": "base64",
    "duration": "duration_seconds",
    "seconds": "duration_seconds",
    "image_url": "url",
    "video_url": "url",
    "download_url": "url",
    "result_url": "url",
    "frame_rate": "fps",
    "thumbnail": "thumbnail_url",
    "mime_type": "format",
    "model_id": "model",
}

_RESULT_LIST_KEYS = (
    "data",
    "results",
    "items",
    "images",
    "videos",
    "tracks",
    "outputs",
    "output",
    "files",
)
_ERROR_LIST_KEYS = ("error", "errors", "detail", "message")
_STATE_ALIASES = {
    "queued": AsyncJobState.PENDING,
    "queue": AsyncJobState.PENDING,
    "pending": AsyncJobState.PENDING,
    "submitted": AsyncJobState.PENDING,
    "waiting": AsyncJobState.PENDING,
    "running": AsyncJobState.RUNNING,
    "processing": AsyncJobState.RUNNING,
    "in_progress": AsyncJobState.RUNNING,
    "in-progress": AsyncJobState.RUNNING,
    "complete": AsyncJobState.SUCCEEDED,
    "completed": AsyncJobState.SUCCEEDED,
    "succeeded": AsyncJobState.SUCCEEDED,
    "success": AsyncJobState.SUCCEEDED,
    "done": AsyncJobState.SUCCEEDED,
    "failed": AsyncJobState.FAILED,
    "error": AsyncJobState.FAILED,
    "cancelled": AsyncJobState.CANCELLED,
    "canceled": AsyncJobState.CANCELLED,
    "timed_out": AsyncJobState.TIMED_OUT,
    "timeout": AsyncJobState.TIMED_OUT,
}


class HttpSpecError(ConfigError):
    """A declarative provider spec is missing or invalid."""

    code = "http_spec"


class HttpSpecTemplateError(ValidationError):
    """A request template could not be rendered safely."""

    code = "http_spec_template"


def _text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    text = value.strip()
    if not text and not allow_empty:
        raise ValueError(f"{label} must be non-empty")
    return text


def _optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _string_mapping(value: Any, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, item in _mapping(value, label).items():
        name = _text(key, f"{label} key")
        if not isinstance(item, str):
            raise ValueError(f"{label} value for {name!r} must be a string")
        result[name] = item
    return result


def _coerce_method(value: Any) -> str:
    method = _text(value, "request method").upper()
    if method not in _METHODS:
        raise ValueError(f"unsupported HTTP method: {method}")
    return method


def _coerce_encoding(value: Any, *, label: str = "encoding") -> str:
    encoding = _text(value, label).lower()
    if encoding not in _ENCODINGS:
        raise ValueError(f"{label} must be one of: {', '.join(sorted(_ENCODINGS))}")
    return encoding


def _coerce_string_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        entries: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        entries = value
    else:
        raise ValueError(f"{label} must be a string or a list of strings")
    result: list[str] = []
    for entry in entries:
        text = _text(entry, f"{label} entry")
        if text not in result:
            result.append(text)
    return result


def _coerce_pricing(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, ProviderPrice):
        return {"default": value}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {"default": {"amount_usd": value}}
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, price in value.items():
            name = _text(key, "pricing key")
            if isinstance(price, (int, float)) and not isinstance(price, bool):
                result[name] = {"amount_usd": price}
            else:
                result[name] = price
        return result
    return value


def _coerce_states(value: Any) -> dict[str, list[str]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("async states must be a mapping")
    result: dict[str, list[str]] = {}
    for key, entries in value.items():
        canonical = _text(key, "async state").lower()
        if canonical not in _STATE_ALIASES:
            raise ValueError(f"unknown async state: {canonical}")
        result[canonical] = [
            entry.lower() for entry in _coerce_string_list(entries, "async state value")
        ]
    return result


class HttpRequestSpec(BaseModel):
    """One HTTP operation: method, URL, headers, query and body template."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    method: str = "POST"
    url: str | None = None
    path: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    query: dict[str, Any] = Field(default_factory=dict)
    body: Any = None
    body_template: str | None = Field(
        default=None,
        validation_alias=AliasChoices("body_template", "template"),
    )
    encoding: str = "json"
    content_type: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0.0)

    @field_validator("method", mode="before")
    @classmethod
    def _validate_method(cls, value: Any) -> str:
        return _coerce_method(value)

    @field_validator("url", "path", "body_template", "content_type", mode="before")
    @classmethod
    def _strip_optional(cls, value: Any) -> str | None:
        return _optional_text(value, "request field")

    @field_validator("headers", mode="before")
    @classmethod
    def _validate_headers(cls, value: Any) -> dict[str, str]:
        return _string_mapping(value, "request headers")

    @field_validator("query", mode="before")
    @classmethod
    def _validate_query(cls, value: Any) -> dict[str, Any]:
        return _mapping(value, "request query")

    @field_validator("encoding", mode="before")
    @classmethod
    def _validate_encoding(cls, value: Any) -> str:
        return _coerce_encoding(value)

    @model_validator(mode="after")
    def _validate_target_and_body(self) -> Self:
        if self.url is not None and self.path is not None:
            raise ValueError("request must set either url or path, not both")
        if self.url is not None:
            validate_http_url(self.url, field="request url")
        if self.body is not None and self.body_template is not None:
            raise ValueError("request must set either body or body_template, not both")
        return self


class HttpResponseSpec(BaseModel):
    """How to read results and errors out of a JSON, text or binary response."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    results: str = ""
    fields: dict[str, str] = Field(default_factory=dict)
    encoding: str = "json"
    request_id: str = ""
    error_code: str = ""
    error_message: str = ""

    @field_validator(
        "results",
        "request_id",
        "error_code",
        "error_message",
        mode="before",
    )
    @classmethod
    def _strip_optional(cls, value: Any) -> str:
        return _text(value or "", "response path", allow_empty=True)

    @field_validator("fields", mode="before")
    @classmethod
    def _validate_fields(cls, value: Any) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, path in _mapping(value, "response fields").items():
            name = _text(key, "response field").lower()
            canonical = _FIELD_ALIASES.get(name, name)
            if canonical not in DEFAULT_RESULT_FIELDS:
                raise ValueError(f"unknown response field: {name}")
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"response field {name!r} needs a non-empty path")
            result[canonical] = path.strip()
        return result

    @field_validator("encoding", mode="before")
    @classmethod
    def _validate_encoding(cls, value: Any) -> str:
        return _coerce_encoding(value, label="response encoding")


class HttpAsyncSpec(BaseModel):
    """Submit, poll and optional cancel operations for a queued provider."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    submit: HttpRequestSpec | None = None
    poll: HttpRequestSpec | None = None
    cancel: HttpRequestSpec | None = None
    remote_id: str = Field(
        default="",
        validation_alias=AliasChoices("remote_id", "job_id", "id"),
    )
    status: str = "status"
    progress: str = ""
    results: str = ""
    error_code: str = ""
    error_message: str = ""
    states: dict[str, list[str]] = Field(default_factory=dict)
    poll_interval_seconds: float = Field(default=1.0, gt=0.0)
    timeout_seconds: float = Field(default=600.0, gt=0.0)

    @field_validator(
        "remote_id",
        "status",
        "progress",
        "results",
        "error_code",
        "error_message",
        mode="before",
    )
    @classmethod
    def _strip_paths(cls, value: Any) -> str:
        return _text(value or "", "async path", allow_empty=True)

    @field_validator("states", mode="before")
    @classmethod
    def _validate_states(cls, value: Any) -> dict[str, list[str]]:
        return _coerce_states(value)

    @model_validator(mode="after")
    def _require_submit_and_poll(self) -> Self:
        if self.submit is None:
            raise ValueError("async spec requires a submit operation")
        if self.poll is None:
            raise ValueError("async spec requires a poll operation")
        return self


class HttpProviderSpec(BaseModel):
    """A complete declarative provider definition.

    ``capabilities`` names the protocols the provider serves. A sync spec uses
    ``request`` and ``response``; an async spec uses ``async`` and the same
    response mapping for the final result. Prices are keyed by capability and
    may also use ``default``.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    name: str = ""
    capabilities: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("capabilities", "capability"),
    )
    description: str = ""
    base_url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    query: dict[str, Any] = Field(default_factory=dict)
    request: HttpRequestSpec | None = None
    response: HttpResponseSpec = Field(default_factory=HttpResponseSpec)
    async_: HttpAsyncSpec | None = Field(
        default=None,
        validation_alias=AliasChoices("async", "async_spec", "queue"),
    )
    pricing: dict[str, ProviderPrice] = Field(default_factory=dict)
    secrets: dict[str, str] = Field(default_factory=dict)
    default_model: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    rate_limit: RateLimit = Field(default_factory=RateLimit)
    supported_options: dict[str, list[str]] = Field(default_factory=dict)
    quality_rank: int | None = Field(default=None, ge=0)
    license: AssetLicense = Field(default_factory=AssetLicense)
    commercial_use_notes: str = ""

    @field_validator("name", "description", "commercial_use_notes", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> str:
        return _text(value or "", "spec field", allow_empty=True)

    @field_validator("capabilities", mode="before")
    @classmethod
    def _validate_capabilities(cls, value: Any) -> list[str]:
        capabilities = _coerce_string_list(value, "capability")
        if not capabilities:
            raise ValueError("spec needs at least one capability")
        return capabilities

    @field_validator("base_url", mode="before")
    @classmethod
    def _validate_base_url(cls, value: Any) -> str:
        text = _optional_text(value, "base_url")
        if text is None:
            return ""
        return validate_http_url(text, field="base_url").rstrip("/")

    @field_validator("headers", mode="before")
    @classmethod
    def _validate_headers(cls, value: Any) -> dict[str, str]:
        return _string_mapping(value, "spec headers")

    @field_validator("query", mode="before")
    @classmethod
    def _validate_query(cls, value: Any) -> dict[str, Any]:
        return _mapping(value, "spec query")

    @field_validator("pricing", mode="before")
    @classmethod
    def _validate_pricing(cls, value: Any) -> Any:
        return _coerce_pricing(value)

    @field_validator("secrets", mode="before")
    @classmethod
    def _validate_secrets(cls, value: Any) -> dict[str, str]:
        return _string_mapping(value, "spec secrets")

    @field_validator("default_model", mode="before")
    @classmethod
    def _validate_default_model(cls, value: Any) -> str | None:
        return _optional_text(value, "default_model")

    @field_validator("supported_options", mode="before")
    @classmethod
    def _validate_supported_options(cls, value: Any) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for key, entries in _mapping(value, "supported_options").items():
            name = _text(key, "supported option capability")
            result[name] = _coerce_string_list(entries, f"options for {name}")
        return result

    @model_validator(mode="before")
    @classmethod
    def _accept_async_shorthand(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        has_async = any(key in values for key in ("async", "async_spec", "queue"))
        if not has_async:
            operations = {
                key: values.pop(key)
                for key in ("submit", "poll", "cancel")
                if key in values
            }
            if operations:
                values["async"] = operations
        if "result" in values and "response" not in values:
            values["response"] = values.pop("result")
        return values

    @model_validator(mode="after")
    def _require_an_operation(self) -> Self:
        if not self.capabilities:
            raise ValueError("spec needs at least one capability")
        if self.async_ is None and self.request is None:
            raise ValueError("spec needs either a request or an async queue")
        return self

    @property
    def is_async(self) -> bool:
        """Whether the spec describes a submit/poll queue."""
        return self.async_ is not None

    @property
    def mode(self) -> str:
        """``"async"`` or ``"sync"``."""
        return "async" if self.is_async else "sync"

    @property
    def async_spec(self) -> HttpAsyncSpec | None:
        """Alias for the async queue definition."""
        return self.async_

    @property
    def operations(self) -> tuple[HttpRequestSpec, ...]:
        """Every HTTP operation the spec can send, in a stable order."""
        if self.async_ is not None:
            operations = [self.async_.submit, self.async_.poll, self.async_.cancel]
        else:
            operations = [self.request]
        return tuple(operation for operation in operations if operation is not None)

    @property
    def target_hosts(self) -> tuple[str, ...]:
        """Every host the spec can contact, lower-case and deduplicated."""
        hosts: set[str] = set()
        if self.base_url:
            host = httpx.URL(self.base_url).host
            if host:
                hosts.add(host.lower())
        for operation in self.operations:
            if operation.url is None:
                continue
            host = httpx.URL(operation.url).host
            if host:
                hosts.add(host.lower())
        return tuple(sorted(hosts))

    @property
    def secret_names(self) -> tuple[str, ...]:
        """Names referenced through ``secrets.<name>`` plus declared names."""
        names = set(self.secrets)
        for value in _template_strings(self):
            for match in _SECRET_REFERENCE.finditer(value):
                name = match.group("dot") or match.group("bracket")
                if name:
                    names.add(name)
        return tuple(sorted(names))

    @property
    def secret_references(self) -> tuple[str, ...]:
        """Secret-loader references the spec declares, without values."""
        return tuple(sorted(set(self.secrets.values())))

    def validate_hosts(self, allowlist: Iterable[str] | None = None) -> None:
        """Reject a target host outside *allowlist*.

        An empty or missing allowlist permits every host. Entries match a host
        exactly, as a parent domain (``example.com`` allows ``api.example.com``)
        or with a leading ``*.`` wildcard.
        """
        allowed = _normalize_hosts(allowlist)
        if not allowed:
            return
        rejected = [
            host for host in self.target_hosts if not _host_allowed(host, allowed)
        ]
        if rejected:
            raise HttpSpecError(
                "provider spec targets a host outside the allowlist: "
                + ", ".join(rejected),
                details={"hosts": rejected, "allowlist": list(allowed)},
            )

    def to_metadata(self, name: str | None = None) -> ProviderMeta:
        """Build the router-facing metadata for this spec."""
        return ProviderMeta(
            name=name or self.name,
            capabilities=list(self.capabilities),
            pricing=dict(self.pricing),
            supported_options={
                key: list(value) for key, value in self.supported_options.items()
            },
            rate_limit=self.rate_limit,
            quality_rank=self.quality_rank,
            license=self.license,
            commercial_use_notes=self.commercial_use_notes,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> HttpProviderSpec:
        """Validate a mapping, converting pydantic errors to :class:`HttpSpecError`."""
        try:
            return cls.model_validate(value)
        except Exception as exc:
            raise HttpSpecError(
                f"invalid HTTP provider spec: {exc}",
                details={"spec": _safe_mapping(value)},
            ) from exc

    @classmethod
    def from_yaml(cls, text: str) -> HttpProviderSpec:
        """Validate a YAML document."""
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise HttpSpecError(f"invalid YAML provider spec: {exc}") from exc
        if not isinstance(data, Mapping):
            raise HttpSpecError("provider spec must be a YAML mapping")
        return cls.from_mapping(data)

    @classmethod
    def from_file(cls, path: str | Path) -> HttpProviderSpec:
        """Read and validate a YAML file."""
        target = Path(path)
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise HttpSpecError(
                f"cannot read provider spec {target}: {exc}",
                details={"path": str(target)},
            ) from exc
        return cls.from_yaml(text)


HttpSpec = HttpProviderSpec
"""Short alias for :class:`HttpProviderSpec`."""

ProviderSpec = HttpProviderSpec
"""Alias matching the CLI's ``providers add-spec`` wording."""

HTTPProviderSpec = HttpProviderSpec
"""Alias for callers that spell out HTTP."""


def _safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a spec mapping without anything that looks like a secret."""
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(key, str) and "secret" in key.lower():
            result[key] = "[REDACTED]"
        elif isinstance(item, Mapping):
            result[str(key)] = _safe_mapping(item)
        else:
            result[str(key)] = item
    return result


def _template_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, BaseModel):
        yield from _template_strings(value.model_dump(mode="python"))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                yield key
            yield from _template_strings(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _template_strings(item)


def _normalize_hosts(value: Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    hosts: list[str] = []
    for entry in value:
        text = _text(entry, "allowlist host").lower()
        if text.startswith("*."):
            text = text[2:]
        if text.startswith("."):
            text = text[1:]
        if text and text not in hosts:
            hosts.append(text)
    return tuple(hosts)


def _host_allowed(host: str, allowed: Iterable[str]) -> bool:
    candidate = host.lower().rstrip(".")
    for entry in allowed:
        if candidate == entry or candidate.endswith(f".{entry}"):
            return True
    return False


def extract_path(value: Any, path: str | None) -> Any:
    """Extract *path* from *value*.

    Supports dotted keys, list indexes (``images[0]``) and JSON pointers
    (``/images/0/url``). An empty path, ``.`` or ``$`` returns the root. A
    missing key or an out-of-range index returns ``None``.
    """
    if path is None:
        return None
    text = path.strip()
    if text in {"", ".", "$"}:
        return value
    if text.startswith("/"):
        current = value
        for part in text.lstrip("/").split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            current = _path_step(current, key)
            if current is None:
                return None
        return current

    current = value
    for token in _path_tokens(text):
        current = _path_step(current, token)
        if current is None:
            return None
    return current


def _path_tokens(path: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    length = len(path)
    while index < length:
        character = path[index]
        if character == ".":
            index += 1
            continue
        if character == "[":
            end = path.find("]", index + 1)
            if end == -1:
                tokens.append(path[index:])
                break
            token = path[index + 1 : end].strip()
            if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
                token = token[1:-1]
            tokens.append(token)
            index = end + 1
            continue
        end = index
        while end < length and path[end] not in ".[":
            end += 1
        tokens.append(path[index:end])
        index = end
    return [token for token in tokens if token]


def _path_step(value: Any, token: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(token)
    if isinstance(value, (list, tuple)):
        if not token.lstrip("-").isdigit():
            return None
        try:
            return value[int(token)]
        except (IndexError, ValueError):
            return None
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _decode_base64(value: str, *, label: str) -> bytes:
    text = value.strip()
    if text.startswith("data:") and "," in text:
        text = text.split(",", 1)[1]
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValidationError(
            f"{label} is not valid base64",
            details={"field": label},
        ) from exc


def _image_format(value: Any, fallback: ImageFormat) -> ImageFormat:
    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("image/"):
            text = text.split("/", 1)[1]
        if text == "jpg":
            text = "jpeg"
        try:
            return ImageFormat(text)
        except ValueError:
            pass
    return fallback


def _video_format(value: Any, fallback: VideoFormat) -> VideoFormat:
    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("video/"):
            text = text.split("/", 1)[1]
        try:
            return VideoFormat(text)
        except ValueError:
            pass
    return fallback


def _audio_format(value: Any, fallback: AudioFormat) -> AudioFormat:
    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("audio/"):
            text = text.split("/", 1)[1]
        if text == "mpeg":
            text = "mp3"
        try:
            return AudioFormat(text)
        except ValueError:
            pass
    return fallback


def _extract_optional(value: Any, path: str | None) -> Any:
    """Extract *path* unless it is empty, where the root is not a value."""
    if not path:
        return None
    return extract_path(value, path)


def _state_for(value: Any, states: Mapping[str, list[str]]) -> AsyncJobState:
    if isinstance(value, AsyncJobState):
        return value
    text = str(value or "").strip().lower()
    for canonical, entries in states.items():
        if text in entries:
            return _STATE_ALIASES[canonical]
    return _STATE_ALIASES.get(text, AsyncJobState.RUNNING)


@dataclass(frozen=True, slots=True)
class _BuiltRequest:
    method: str
    url: str
    headers: dict[str, str]
    params: dict[str, Any]
    content: Any = None
    json_body: Any = None
    timeout: float | None = None

    @property
    def has_body(self) -> bool:
        return self.content is not None or self.json_body is not None


class HttpSpecResult(BaseModel):
    """The raw result of one declarative HTTP operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    body: Any = None
    status_code: int = 200
    headers: dict[str, str] = Field(default_factory=dict)
    url: str = ""


class HttpSpecEngine:
    """Render and execute the operations of one :class:`HttpProviderSpec`.

    The engine owns the sandbox, path extraction and response mapping. A
    provider adapter wraps it with the capability protocol, while tests and the
    ``providers add-spec`` command can inspect the engine without a provider.
    """

    def __init__(
        self,
        spec: HttpProviderSpec,
        *,
        secrets: Mapping[str, str] | None = None,
        http: HttpClient | None = None,
        http_config: HttpClientConfig | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.spec = spec
        self.secrets = dict(secrets or {})
        self._environment = self._create_environment()
        self._http = http
        self._http_config = http_config or HttpClientConfig()
        self._transport = transport
        self._owns_http = http is None
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _create_environment() -> SandboxedEnvironment:
        environment = SandboxedEnvironment(
            autoescape=False,
            keep_trailing_newline=True,
            undefined=StrictUndefined,
        )
        environment.globals.clear()
        return environment

    @property
    def http(self) -> HttpClient:
        """The HTTP client used by the engine."""
        if self._http is None:
            self._http = HttpClient(self._http_config, transport=self._transport)
        return self._http

    def _context(
        self,
        request: Any,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if isinstance(request, BaseModel):
            request_data = request.model_dump(mode="python")
        elif isinstance(request, Mapping):
            request_data = dict(request)
        elif request is None:
            request_data = {}
        else:
            request_data = {"value": request}
        context: dict[str, Any] = {
            "request": request_data,
            "secrets": dict(self.secrets),
            "spec": self.spec,
        }
        if extra:
            context.update(extra)
        return context

    def render(self, template: str, context: Mapping[str, Any]) -> str:
        """Render *template* in the sandbox and return text."""
        parsed = self._parse(template)
        for node in parsed.find_all(tuple(_COMPOSITION_TAGS)):
            tag = _COMPOSITION_TAGS[type(node)]
            raise HttpSpecTemplateError(
                f"provider request templates may not use {tag}",
                details={"tag": tag},
            )
        try:
            compiled = self._environment.from_string(template)
            return compiled.render(**dict(context))
        except TemplateError as exc:
            raise HttpSpecTemplateError(
                f"cannot render provider request template: {exc}",
            ) from exc

    def _parse(self, template: str) -> nodes.Template:
        try:
            return self._environment.parse(template)
        except TemplateError as exc:
            raise HttpSpecTemplateError(
                f"cannot parse provider request template: {exc}",
            ) from exc

    def render_value(self, value: Any, context: Mapping[str, Any]) -> Any:
        """Render every string inside *value*, preserving native types."""
        if isinstance(value, str):
            return self._render_scalar(value, context)
        if isinstance(value, Mapping):
            return {
                key: self.render_value(item, context) for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.render_value(item, context) for item in value]
        if isinstance(value, tuple):
            return tuple(self.render_value(item, context) for item in value)
        return value

    def _render_scalar(self, value: str, context: Mapping[str, Any]) -> Any:
        match = _SINGLE_EXPRESSION.match(value)
        if match is None:
            return self.render(value, context)
        expression = match.group("expression").strip()
        if not expression or "{{" in expression or "}}" in expression:
            return self.render(value, context)
        self._parse(value)
        try:
            compiled = self._environment.compile_expression(expression)
            result = compiled(**dict(context))
        except TemplateError as exc:
            raise HttpSpecTemplateError(
                f"cannot render provider request template: {exc}",
            ) from exc
        if isinstance(result, Undefined):
            raise HttpSpecTemplateError(
                "provider request template used an undefined value",
            )
        return result

    def build_request(
        self,
        operation: HttpRequestSpec,
        context: Mapping[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> _BuiltRequest:
        """Render one operation into a concrete HTTP request."""
        url = self._operation_url(operation, context)
        headers = dict(self.spec.headers)
        headers.update(operation.headers)
        rendered_headers = {
            key: str(self.render(value, context)) for key, value in headers.items()
        }
        params = dict(self.spec.query)
        params.update(operation.query)
        rendered_params = {
            key: self.render_value(value, context) for key, value in params.items()
        }
        rendered_params = {
            key: value for key, value in rendered_params.items() if value is not None
        }

        body = self._render_body(operation, context)
        encoding = operation.encoding
        json_body: Any = None
        content: Any = None
        if body is not None:
            if encoding == "json":
                if isinstance(body, str):
                    try:
                        body = json.loads(body)
                    except json.JSONDecodeError as exc:
                        raise ValidationError(
                            "rendered provider request body is not valid JSON",
                            details={"error": str(exc)},
                        ) from exc
                json_body = body
            elif encoding == "text":
                content = body if isinstance(body, str) else json.dumps(body)
            else:
                content = body
        content_type = operation.content_type
        if content_type is not None:
            rendered_headers["content-type"] = str(self.render(content_type, context))
        elif body is not None and encoding == "json":
            rendered_headers.setdefault("content-type", "application/json")

        selected_timeout = operation.timeout_seconds or timeout_seconds
        if selected_timeout is None:
            selected_timeout = self.timeout_seconds
        return _BuiltRequest(
            method=operation.method,
            url=url,
            headers=rendered_headers,
            params=rendered_params,
            content=content,
            json_body=json_body,
            timeout=selected_timeout,
        )

    def _operation_url(
        self,
        operation: HttpRequestSpec,
        context: Mapping[str, Any],
    ) -> str:
        if operation.url is not None:
            return validate_http_url(
                self.render(operation.url, context),
                field="request url",
            )
        path = self.render(operation.path or "", context).strip()
        if not self.spec.base_url:
            raise HttpSpecError(
                "provider spec needs a base_url or an absolute request url",
            )
        if not path:
            return self.spec.base_url
        return f"{self.spec.base_url}/{path.lstrip('/')}"

    def _render_body(
        self,
        operation: HttpRequestSpec,
        context: Mapping[str, Any],
    ) -> Any:
        if operation.body_template is not None:
            return self.render(operation.body_template, context)
        if operation.body is None:
            return None
        return self.render_value(operation.body, context)

    async def execute(
        self,
        operation: HttpRequestSpec,
        request: Any,
        *,
        extra: Mapping[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> HttpSpecResult:
        """Send one operation and return the raw response."""
        context = self._context(request, extra)
        built = self.build_request(
            operation,
            context,
            timeout_seconds=timeout_seconds,
        )
        response = await self.http.request(
            built.method,
            built.url,
            headers=built.headers,
            params=built.params,
            content=built.content,
            json=built.json_body,
            timeout=built.timeout,
            raise_for_status=True,
        )
        try:
            body = self._response_body(response, self.spec.response.encoding)
            return HttpSpecResult(
                body=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                url=str(response.url),
            )
        finally:
            await response.aclose()

    @staticmethod
    def _response_body(response: httpx.Response, encoding: str) -> Any:
        if encoding == "binary":
            return response.content
        if encoding == "text":
            return response.text
        if not response.content:
            return None
        try:
            return response.json()
        except (json.JSONDecodeError, ValueError):
            return response.text

    def entries(
        self,
        body: Any,
        *,
        results_path: str | None = None,
        response: HttpResponseSpec | None = None,
    ) -> list[dict[str, Any]]:
        """Return canonical result entries extracted from *body*."""
        mapping = response or self.spec.response
        path = mapping.results if results_path is None else results_path
        raw_entries = self._result_values(body, path)
        return [self._canonical_entry(entry, mapping) for entry in raw_entries]

    @staticmethod
    def _result_values(body: Any, path: str) -> list[Any]:
        if path:
            value = extract_path(body, path)
            return _as_entries(value)
        if isinstance(body, list):
            return list(body)
        if isinstance(body, Mapping):
            for key in _RESULT_LIST_KEYS:
                if key in body:
                    value = body[key]
                    if isinstance(value, Mapping) and key == "output":
                        return [value]
                    entries = _as_entries(value)
                    if entries:
                        return entries
        return _as_entries(body)

    def _canonical_entry(
        self,
        entry: Any,
        response: HttpResponseSpec,
    ) -> dict[str, Any]:
        if isinstance(entry, str):
            return {"url": entry}
        if not isinstance(entry, Mapping):
            return {"data": entry}
        values: dict[str, Any] = {}
        for canonical, aliases in DEFAULT_RESULT_FIELDS.items():
            override = response.fields.get(canonical)
            if override is not None:
                values[canonical] = extract_path(entry, override)
                continue
            for alias in aliases:
                if alias in entry:
                    values[canonical] = entry[alias]
                    break
        return values

    def result_urls(
        self,
        body: Any,
        *,
        results_path: str | None = None,
        response: HttpResponseSpec | None = None,
    ) -> list[str]:
        """Return absolute result URLs from *body*."""
        urls: list[str] = []
        for entry in self.entries(
            body,
            results_path=results_path,
            response=response,
        ):
            value = entry.get("url")
            if not isinstance(value, str) or not value.strip():
                continue
            url = self._absolute_url(value)
            if url not in urls:
                urls.append(url)
        return urls

    def _absolute_url(self, value: str) -> str:
        text = value.strip()
        if text.startswith(("http://", "https://")):
            return validate_http_url(text, field="result url")
        if not self.spec.base_url:
            raise ValidationError(
                f"result url {text!r} is relative and the spec has no base_url",
            )
        if text.startswith("/"):
            base = httpx.URL(self.spec.base_url)
            return validate_http_url(
                str(base.copy_with(path=text, query=None, fragment=None)),
                field="result url",
            )
        return validate_http_url(
            f"{self.spec.base_url}/{text.lstrip('/')}",
            field="result url",
        )

    async def aclose(self) -> None:
        """Close the HTTP client when the engine created it."""
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None


def _as_entries(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping):
        return [value]
    return [value]


class _SpecAsyncBackend:
    """The submit, poll and cancel callbacks the async job helper drives."""

    def __init__(self, provider: HttpSpecProvider) -> None:
        self._provider = provider
        self._requests: dict[str, Any] = {}

    async def submit(self, request: Any) -> Any:
        provider = self._provider
        spec = provider.spec.async_spec
        if spec is None or spec.submit is None:
            raise InvalidRequestError("provider spec has no async submit operation")
        result = await provider.engine.execute(spec.submit, request)
        remote_id = _extract_optional(result.body, spec.remote_id)
        if remote_id is None and isinstance(result.body, Mapping):
            for key in ("job_id", "remote_id", "request_id", "id", "uuid"):
                if key in result.body:
                    remote_id = result.body[key]
                    break
        if not isinstance(remote_id, (str, int)) or not str(remote_id).strip():
            raise InvalidRequestError(
                "provider submit response did not contain a remote job id",
                details={"provider": provider.name},
            )
        remote_text = str(remote_id).strip()
        self._requests[remote_text] = request
        return {"job_id": remote_text, "response": result.body}

    async def poll(self, remote_id: str) -> AsyncJobStatus:
        provider = self._provider
        spec = provider.spec.async_spec
        if spec is None or spec.poll is None:
            raise InvalidRequestError("provider spec has no async poll operation")
        request = self._requests.get(remote_id)
        extra = {
            "remote_id": remote_id,
            "job": {"remote_id": remote_id},
        }
        result = await provider.engine.execute(spec.poll, request, extra=extra)
        body = result.body
        raw_state = _extract_optional(body, spec.status) or ""
        if not raw_state and isinstance(body, Mapping):
            raw_state = body.get("status") or body.get("state") or ""
        state = _state_for(raw_state, spec.states)
        results_path = spec.results or provider.spec.response.results
        urls = provider.engine.result_urls(body, results_path=results_path)
        progress = _as_float(_extract_optional(body, spec.progress))
        if progress is None and isinstance(body, Mapping):
            progress = _as_float(body.get("progress"))
            if progress is None:
                progress = _as_float(body.get("percent"))
        error_code = _extract_optional(body, spec.error_code) or ""
        error_message = _extract_optional(body, spec.error_message) or ""
        if not error_message and isinstance(body, Mapping):
            error = body.get("error")
            if isinstance(error, Mapping):
                error_code = error_code or error.get("code", "")
                error_message = error.get("message", "")
            elif isinstance(error, str):
                error_message = error
            if not error_message:
                for key in _ERROR_LIST_KEYS:
                    value = body.get(key)
                    if isinstance(value, str) and value.strip():
                        error_message = value
                        break
        return AsyncJobStatus(
            state=state,
            progress=progress,
            result_urls=tuple(urls),
            error_code=str(error_code),
            error_message=str(error_message),
            metadata={"result": body},
        )

    async def cancel(self, remote_id: str) -> None:
        provider = self._provider
        spec = provider.spec.async_spec
        if spec is None or spec.cancel is None:
            return
        request = self._requests.get(remote_id)
        extra = {
            "remote_id": remote_id,
            "job": {"remote_id": remote_id},
        }
        await provider.engine.execute(spec.cancel, request, extra=extra)


class HttpSpecProvider(Provider):
    """A provider implemented entirely by a declarative spec."""

    default_metadata: ClassVar[ProviderMeta | Mapping[str, Any] | None] = None

    def __init__(
        self,
        spec: HttpProviderSpec | Mapping[str, Any] | str | Path,
        *,
        secrets: Mapping[str, str] | SecretsLoader | None = None,
        secret_values: Mapping[str, str] | None = None,
        name: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        http_config: HttpClientConfig | None = None,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float | None = None,
        engine: Engine | None = None,
        download_results: bool = False,
        max_download_bytes: int | None = None,
        clock: Any = None,
        sleep: Any = None,
        health_timeout_seconds: float | None = None,
        close_timeout_seconds: float | None = None,
    ) -> None:
        resolved_spec = _coerce_spec(spec)
        resolved_secrets = _resolve_secrets(resolved_spec, secrets, secret_values)
        super().__init__(
            resolved_spec.to_metadata(name),
            name=name,
            health_timeout_seconds=health_timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            http_config=http_config,
        )
        if transport is not None and not isinstance(
            transport, httpx.AsyncBaseTransport
        ):
            raise TypeError("transport must be an httpx.AsyncBaseTransport")
        if transport is not None:
            self._http = self.create_http_client(transport=transport)
        self._spec = resolved_spec
        self._engine = HttpSpecEngine(
            resolved_spec,
            secrets=resolved_secrets,
            http=self.http,
            http_config=http_config,
            timeout_seconds=timeout_seconds,
        )
        self._download_results = bool(download_results)
        self._max_download_bytes = max_download_bytes
        self._helper: AsyncMediaJobHelper | None = None
        if resolved_spec.is_async:
            async_spec = resolved_spec.async_spec
            assert async_spec is not None
            self._helper = AsyncMediaJobHelper(
                _SpecAsyncBackend(self),
                engine=engine,
                provider=self.name,
                kind=f"http_spec:{self.name}",
                http=self.http,
                timeout_seconds=(
                    timeout_seconds
                    if timeout_seconds is not None
                    else async_spec.timeout_seconds
                ),
                poll_interval_seconds=(
                    poll_interval_seconds
                    if poll_interval_seconds is not None
                    else async_spec.poll_interval_seconds
                ),
                max_download_bytes=max_download_bytes,
                clock=clock,
                sleep=sleep,
            )

    @property
    def spec(self) -> HttpProviderSpec:
        """The declarative spec backing this provider."""
        return self._spec

    @property
    def engine(self) -> HttpSpecEngine:
        """The renderer and HTTP executor for the spec."""
        return self._engine

    @property
    def pricing(self) -> dict[str, ProviderPrice]:
        """Prices keyed by capability, as declared by the spec."""
        return dict(self._spec.pricing)

    @property
    def base_url(self) -> str:
        """The configured API root."""
        return self._spec.base_url

    @property
    def default_model(self) -> str | None:
        """The model id used when a request does not name one."""
        return self._spec.default_model

    @property
    def target_hosts(self) -> tuple[str, ...]:
        """Hosts this provider may contact."""
        return self._spec.target_hosts

    @property
    def secret_names(self) -> tuple[str, ...]:
        """Secret names referenced by the spec."""
        return self._spec.secret_names

    def price_for(self, capability: str | None = None) -> ProviderPrice | None:
        """Return the declared price for *capability* or the default price."""
        selected = capability or self.metadata.capability
        return self.metadata.pricing_for(selected)

    def estimate_cost(self, capability: str, units: float = 1.0) -> float:
        """Return the declared cost for *units* of *capability*."""
        price = self.metadata.pricing_for(capability)
        if price is None:
            return 0.0
        return price.cost_for(units)

    async def execute(self, request: Any) -> HttpSpecResult:
        """Execute the sync operation for *request* without mapping media."""
        if self._spec.request is None:
            raise InvalidRequestError(
                "provider spec has no sync request operation",
                details={"provider": self.name},
            )
        timeout = getattr(request, "timeout_seconds", None)
        return await self._engine.execute(
            self._spec.request,
            request,
            timeout_seconds=timeout,
        )

    async def generate(
        self,
        request: ImageRequest | VideoRequest | MusicRequest,
    ) -> ImageResponse | VideoResponse | MusicResponse:
        """Generate media according to the request type."""
        if isinstance(request, ImageRequest):
            body, request_id, results_path = await self._run(request)
            return self._image_response(body, request, request_id, results_path)
        if isinstance(request, VideoRequest):
            body, request_id, results_path = await self._run(request)
            return self._video_response(body, request, request_id, results_path)
        if isinstance(request, MusicRequest):
            body, request_id, results_path = await self._run(request)
            return self._music_response(body, request, request_id, results_path)
        raise TypeError("request must be an ImageRequest, VideoRequest or MusicRequest")

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Synthesize speech from a declarative TTS spec."""
        body, request_id, results_path = await self._run(request)
        return self._tts_response(body, request, request_id, results_path)

    async def _run(
        self,
        request: Any,
    ) -> tuple[Any, str | None, str | None]:
        if self._spec.is_async:
            body, request_id, results_path = await self._run_async(request)
        else:
            result = await self.execute(request)
            body = result.body
            request_id = self._response_request_id(body)
            results_path = None
        if self._download_results and not isinstance(body, (bytes, bytearray)):
            body, results_path = await self._materialize(body, results_path)
        return body, request_id, results_path

    async def _materialize(
        self,
        body: Any,
        results_path: str | None,
    ) -> tuple[Any, str | None]:
        entries = self._entries_for(body, results_path=results_path)
        if not entries:
            return body, results_path
        materialized: list[dict[str, Any]] = []
        for entry in entries:
            if entry.get("data") is not None or entry.get("base64") is not None:
                materialized.append(entry)
                continue
            url = self._result_url(entry)
            if url is None:
                materialized.append(entry)
                continue
            downloaded = dict(entry)
            downloaded["data"] = await self._download(url)
            downloaded["url"] = None
            materialized.append(downloaded)
        return {"data": materialized}, "data"

    async def _run_async(
        self,
        request: Any,
    ) -> tuple[Any, str | None, str | None]:
        helper = self._helper
        if helper is None:
            raise InvalidRequestError("provider spec has no async helper")
        idempotency_key = ""
        metadata = getattr(request, "metadata", None)
        if isinstance(metadata, Mapping):
            value = metadata.get("idempotency_key")
            if isinstance(value, str):
                idempotency_key = value.strip()
        job = await helper.submit(request, idempotency_key=idempotency_key)
        job = await helper.poll(
            job,
            timeout_seconds=getattr(request, "timeout_seconds", None),
        )
        if job.state is AsyncJobState.FAILED:
            self._raise_job_failure(job)
        if not job.succeeded:
            raise InvalidRequestError(
                f"provider job ended in state {job.state.value}",
                details={"provider": self.name, "job_id": job.id},
            )
        body = job.metadata.get("result")
        if body is None:
            body = {"urls": list(job.result_urls)}
        async_spec = self._spec.async_spec
        results_path = None
        if async_spec is not None and async_spec.results:
            results_path = async_spec.results
        return body, job.remote_id or None, results_path

    def _raise_job_failure(self, job: AsyncJob) -> None:
        message = job.error_message or "provider job failed"
        if _looks_like_content_policy(message, job.error_code):
            raise ContentPolicyError(
                message,
                details={"provider": self.name, "job_id": job.id},
            )
        raise InvalidRequestError(
            message,
            details={
                "provider": self.name,
                "job_id": job.id,
                "code": job.error_code,
            },
        )

    def _response_request_id(self, body: Any) -> str | None:
        path = self._spec.response.request_id
        if not path:
            return None
        value = extract_path(body, path)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _entries_for(
        self,
        body: Any,
        *,
        results_path: str | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(body, Mapping) and "urls" in body:
            values = body.get("urls")
            if isinstance(values, list):
                return [{"url": value} for value in values]
        return self._engine.entries(body, results_path=results_path)

    def _result_url(self, entry: Mapping[str, Any]) -> str | None:
        value = _entry_url(entry)
        if value is None:
            return None
        return self._engine._absolute_url(value)

    def _image_response(
        self,
        body: Any,
        request: ImageRequest,
        request_id: str | None,
        results_path: str | None = None,
    ) -> ImageResponse:
        results: list[ImageResult] = []
        for entry in self._entries_for(body, results_path=results_path):
            data = _entry_bytes(entry, "image")
            url = self._result_url(entry)
            if data is not None:
                url = None
            if data is None and url is None:
                continue
            width = _as_int(entry.get("width"))
            height = _as_int(entry.get("height"))
            results.append(
                ImageResult(
                    data=data,
                    url=url,
                    format=_image_format(entry.get("format"), request.format),
                    width=width,
                    height=height,
                    seed=_as_int(entry.get("seed")),
                    revised_prompt=_optional_entry_text(entry.get("revised_prompt")),
                )
            )
        if not results:
            raise ValidationError(
                "provider response did not contain an image result",
                details={"provider": self.name},
            )
        return ImageResponse(
            images=results,
            model=request.model or self._spec.default_model or "",
            provider=self.name,
            request_id=request_id,
        )

    def _video_response(
        self,
        body: Any,
        request: VideoRequest,
        request_id: str | None,
        results_path: str | None = None,
    ) -> VideoResponse:
        results: list[VideoResult] = []
        for entry in self._entries_for(body, results_path=results_path):
            data = _entry_bytes(entry, "video")
            url = self._result_url(entry)
            if data is not None:
                url = None
            if data is None and url is None:
                continue
            width = _as_int(entry.get("width"))
            height = _as_int(entry.get("height"))
            results.append(
                VideoResult(
                    data=data,
                    url=url,
                    format=_video_format(entry.get("format"), request.format),
                    width=width,
                    height=height,
                    duration_seconds=_as_float(entry.get("duration_seconds")),
                    fps=_as_int(entry.get("fps")),
                    has_audio=_as_bool(entry.get("has_audio")),
                    seed=_as_int(entry.get("seed")),
                    thumbnail_url=self._result_url({"url": entry.get("thumbnail_url")}),
                )
            )
        if not results:
            raise ValidationError(
                "provider response did not contain a video result",
                details={"provider": self.name},
            )
        return VideoResponse(
            videos=results,
            model=request.model or self._spec.default_model or "",
            provider=self.name,
            request_id=request_id,
        )

    def _music_response(
        self,
        body: Any,
        request: MusicRequest,
        request_id: str | None,
        results_path: str | None = None,
    ) -> MusicResponse:
        results: list[MusicTrack] = []
        for entry in self._entries_for(body, results_path=results_path):
            data = _entry_bytes(entry, "music")
            url = self._result_url(entry)
            if data is not None:
                url = None
            if data is None and url is None:
                continue
            results.append(
                MusicTrack(
                    data=data,
                    url=url,
                    format=_audio_format(entry.get("format"), request.format),
                    duration_seconds=_as_float(entry.get("duration_seconds")),
                    sample_rate=_as_int(entry.get("sample_rate")),
                    channels=_as_int(entry.get("channels")),
                    seed=_as_int(entry.get("seed")),
                    title=_optional_entry_text(entry.get("title")),
                    lyrics=_optional_entry_text(entry.get("lyrics")),
                )
            )
        if not results:
            raise ValidationError(
                "provider response did not contain a music track",
                details={"provider": self.name},
            )
        return MusicResponse(
            tracks=results,
            model=request.model or self._spec.default_model or "",
            provider=self.name,
            request_id=request_id,
        )

    def _tts_response(
        self,
        body: Any,
        request: TTSRequest,
        request_id: str | None,
        results_path: str | None = None,
    ) -> TTSResponse:
        if isinstance(body, (bytes, bytearray)):
            audio = bytes(body)
            format_value = request.format
            duration = None
            sample_rate = None
        else:
            entries = self._entries_for(body, results_path=results_path)
            entry = entries[0] if entries else {}
            data = _entry_bytes(entry, "speech")
            if data is None:
                raise ValidationError(
                    "provider response did not contain speech audio",
                    details={"provider": self.name},
                )
            audio = data
            format_value = _audio_format(entry.get("format"), request.format)
            duration = _as_float(entry.get("duration_seconds"))
            sample_rate = _as_int(entry.get("sample_rate"))
        if not audio:
            raise ValidationError(
                "provider response did not contain speech audio",
                details={"provider": self.name},
            )
        return TTSResponse(
            audio=audio,
            format=format_value,
            sample_rate=sample_rate,
            duration_seconds=duration,
            voice=request.voice,
            model=request.model or self._spec.default_model or "",
            provider=self.name,
            request_id=request_id,
        )

    async def _download(self, url: str) -> bytes:
        response = await self.http.get(url, raise_for_status=True)
        try:
            data = response.content
        finally:
            await response.aclose()
        limit = self._max_download_bytes
        if limit is not None and len(data) > limit:
            raise ValidationError(
                f"download is {len(data)} bytes, above the {limit} byte cap",
                details={"url": url, "size_bytes": len(data), "limit": limit},
            )
        return data

    async def close(self) -> None:
        """Close the helper's HTTP client and then this provider's client."""
        if self._helper is not None:
            await self._helper.aclose()
        await self._engine.aclose()
        await super().close()


DeclarativeHttpProvider = HttpSpecProvider
"""Alias for the provider built from a spec."""

HTTPProvider = HttpSpecProvider
"""Alias for callers that spell out HTTP."""


def _looks_like_content_policy(*values: str) -> bool:
    text = " ".join(value for value in values if value).lower()
    return any(
        marker in text
        for marker in (
            "content policy",
            "safety",
            "nsfw",
            "blocked",
            "moderation",
            "prohibited",
        )
    )


def _optional_entry_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _entry_url(entry: Mapping[str, Any]) -> str | None:
    value = entry.get("url")
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _entry_bytes(entry: Mapping[str, Any], label: str) -> bytes | None:
    data = entry.get("data")
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str) and data:
        return _decode_base64(data, label=label)
    value = entry.get("base64")
    if isinstance(value, str) and value:
        return _decode_base64(value, label=label)
    return None


def _coerce_spec(
    value: HttpProviderSpec | Mapping[str, Any] | str | Path,
) -> HttpProviderSpec:
    if isinstance(value, HttpProviderSpec):
        return value
    if isinstance(value, Mapping):
        return HttpProviderSpec.from_mapping(value)
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.exists():
            return HttpProviderSpec.from_file(path)
        if isinstance(value, str):
            return HttpProviderSpec.from_yaml(value)
        raise HttpSpecError(f"provider spec does not exist: {path}")
    raise TypeError("spec must be an HttpProviderSpec, mapping, YAML string or path")


def _resolve_secrets(
    spec: HttpProviderSpec,
    secrets: Mapping[str, str] | SecretsLoader | None,
    values: Mapping[str, str] | None,
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    if isinstance(secrets, SecretsLoader):
        for name, reference in spec.secrets.items():
            resolved[name] = secrets.get(reference)
    elif isinstance(secrets, Mapping):
        resolved.update({str(key): str(value) for key, value in secrets.items()})
        for name, reference in spec.secrets.items():
            if name not in resolved and reference in secrets:
                resolved[name] = str(secrets[reference])
    elif secrets is not None:
        raise TypeError("secrets must be a mapping, a SecretsLoader or None")
    if values is not None:
        resolved.update({str(key): str(value) for key, value in values.items()})
    for name in spec.secret_names:
        if name not in resolved and isinstance(secrets, Mapping) and name in secrets:
            resolved[name] = str(secrets[name])
    return resolved


def load_http_spec(
    source: HttpProviderSpec | Mapping[str, Any] | str | Path,
) -> HttpProviderSpec:
    """Load a spec from an object, mapping, YAML text or file path."""
    return _coerce_spec(source)


parse_http_spec = load_http_spec
"""Alias for :func:`load_http_spec`."""


def load_http_spec_file(path: str | Path) -> HttpProviderSpec:
    """Load one YAML spec file."""
    return HttpProviderSpec.from_file(path)


def load_http_specs(directory: str | Path) -> list[HttpProviderSpec]:
    """Load every ``*.yaml`` and ``*.yml`` spec in *directory*, sorted by name."""
    root = Path(directory)
    if not root.is_dir():
        raise HttpSpecError(f"provider spec directory does not exist: {root}")
    specs: list[HttpProviderSpec] = []
    for path in sorted((*root.glob("*.yaml"), *root.glob("*.yml"))):
        specs.append(HttpProviderSpec.from_file(path))
    return specs


def build_http_provider(
    spec: HttpProviderSpec | Mapping[str, Any] | str | Path,
    **options: Any,
) -> HttpSpecProvider:
    """Build a provider from a spec with the given constructor options."""
    return HttpSpecProvider(spec, **options)


http_provider_from_spec = build_http_provider
"""Alias for :func:`build_http_provider`."""
