"""S3 compatible storage with SigV4 signing and presigned expiring URLs.

The provider speaks the S3 REST API directly over the shared HTTP client, so it
does not depend on a vendor SDK. Objects live under ``storage.bucket`` with an
optional key prefix. Credentials are referenced by name in configuration and
resolved through :class:`~aivia.core.secrets.SecretsLoader`; they are never
written into the config file.

A put without an explicit key derives a content addressed key from the SHA-256
digest, mirroring the local provider. Identical bytes under the same key are a
no-op, different bytes require ``overwrite``. :meth:`url` returns a presigned
GET URL that expires after ``expires_seconds`` (or ``storage.presign_seconds``
when omitted).
"""

from __future__ import annotations

import hashlib
import hmac
import mimetypes
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from aivia.core.config.http import HttpClientConfig
from aivia.core.config.ops import StorageConfig
from aivia.core.errors import ConfigError, InvalidRequestError
from aivia.core.http import error_for_response
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.core.secrets import SecretsLoader
from aivia.providers.base import HealthStatus, Provider, ProviderHealth
from aivia.providers.protocols.storage import (
    CAPABILITY_STORAGE,
    StorageObject,
    StoragePutRequest,
)
from aivia.providers.registry import register_provider

__all__ = [
    "S3Storage",
    "S3StorageProvider",
]

_DEFAULT_METADATA = ProviderMeta(
    name="s3",
    capabilities=[CAPABILITY_STORAGE],
    pricing={CAPABILITY_STORAGE: ProviderPrice(amount_usd=0.0)},
)

_KEY_SEPARATOR = "/"
_CONTENT_PREFIX = "objects"
_ALGORITHM = "AWS4-HMAC-SHA256"
_SERVICE = "s3"
_UNSIGNED_PAYLOAD = "UNSIGNED-PAYLOAD"
_MAX_PRESIGN_SECONDS = 7 * 24 * 60 * 60


def _normalise_key(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("storage key must be a string")
    text = value.strip().replace("\\", _KEY_SEPARATOR)
    if not text:
        raise InvalidRequestError("storage key must be non-empty")
    if "\x00" in text:
        raise InvalidRequestError("storage key must not contain NUL")
    if text.startswith(_KEY_SEPARATOR):
        raise InvalidRequestError(f"storage key must be relative: {value!r}")

    parts: list[str] = []
    for part in text.split(_KEY_SEPARATOR):
        if part in {"", "."}:
            continue
        if part == "..":
            raise InvalidRequestError(f"storage key must not traverse: {value!r}")
        if ":" in part:
            raise InvalidRequestError(f"storage key must not name a drive: {value!r}")
        parts.append(part)
    if not parts:
        raise InvalidRequestError("storage key must be non-empty")
    return _KEY_SEPARATOR.join(parts)


def _content_key(digest: str) -> str:
    return f"{_CONTENT_PREFIX}/{digest[:2]}/{digest}"


def _guess_content_type(key: str) -> str | None:
    guessed, _ = mimetypes.guess_type(key)
    return guessed


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def _signing_key(secret_key: str, date_stamp: str, region: str) -> bytes:
    date_key = _hmac(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    region_key = _hmac(date_key, region)
    service_key = _hmac(region_key, _SERVICE)
    return _hmac(service_key, "aws4_request")


def _encode(value: str) -> str:
    return quote(value, safe="-_.~")


def _canonical_query(query: Mapping[str, str]) -> str:
    pairs = sorted((_encode(key), _encode(value)) for key, value in query.items())
    return "&".join(f"{key}={value}" for key, value in pairs)


def _canonical_headers(headers: Mapping[str, str]) -> tuple[str, str]:
    values = {
        name.lower(): " ".join(str(value).split()) for name, value in headers.items()
    }
    names = sorted(values)
    rendered = "".join(f"{name}:{values[name]}\n" for name in names)
    return rendered, ";".join(names)


def _signature(
    secret_key: str,
    date_stamp: str,
    region: str,
    string_to_sign: str,
) -> str:
    key = _signing_key(secret_key, date_stamp, region)
    return hmac.new(key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()


class S3StorageProvider(Provider):
    """A storage provider backed by an S3 compatible endpoint.

    ``config`` is a :class:`~aivia.core.config.StorageConfig` with
    ``backend='s3'``. The same values may be supplied as keywords, which is
    convenient for tests and one-off instances. Credential references are
    resolved through ``secrets`` (a :class:`SecretsLoader` by default), while
    ``access_key`` and ``secret_key`` accept already resolved values.

    ``transport`` lets the offline suite inject an ``httpx`` mock transport.
    ``clock`` returns the current time and exists so expiry tests are
    deterministic.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        config: StorageConfig | None = None,
        *,
        bucket: str | None = None,
        endpoint: str | None = None,
        region: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        access_key_ref: str | None = None,
        secret_key_ref: str | None = None,
        session_token: str | None = None,
        secrets: SecretsLoader | None = None,
        prefix: str | None = None,
        presign_seconds: int | None = None,
        use_path_style: bool | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], datetime] | None = None,
        name: str | None = None,
        health_timeout_seconds: float | None = None,
        close_timeout_seconds: float | None = None,
        http_config: HttpClientConfig | None = None,
    ) -> None:
        super().__init__(
            name=name,
            health_timeout_seconds=health_timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            http_config=http_config,
        )
        if config is not None and not isinstance(config, StorageConfig):
            raise TypeError("config must be a StorageConfig")
        if config is not None and config.backend != "s3":
            raise ConfigError(
                "s3 storage provider requires storage.backend='s3'",
                details={"backend": config.backend},
            )
        if secrets is not None and not isinstance(secrets, SecretsLoader):
            raise TypeError("secrets must be a SecretsLoader")
        if transport is not None and not isinstance(
            transport, httpx.AsyncBaseTransport
        ):
            raise TypeError("transport must be an httpx.AsyncBaseTransport")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")

        self._config = config
        self._bucket = _required(
            bucket if bucket is not None else getattr(config, "bucket", None),
            "bucket",
        )
        self._endpoint = _required(
            endpoint if endpoint is not None else getattr(config, "endpoint", None),
            "endpoint",
        )
        self._region = _required(
            region if region is not None else getattr(config, "region", None),
            "region",
        )
        parsed = urlsplit(self._endpoint)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ConfigError(
                "storage.endpoint must be an http or https URL with a host",
                details={"endpoint": self._endpoint},
            )
        self._scheme = parsed.scheme.lower()
        self._netloc = parsed.netloc
        self._base_path = "/".join(part for part in parsed.path.split("/") if part)

        loader = secrets or SecretsLoader()
        selected_access_ref = (
            access_key_ref
            if access_key_ref is not None
            else getattr(config, "access_key_ref", None)
        )
        selected_secret_ref = (
            secret_key_ref
            if secret_key_ref is not None
            else getattr(config, "secret_key_ref", None)
        )
        self._access_key = _credential(
            access_key, selected_access_ref, loader, "access key"
        )
        self._secret_key = _credential(
            secret_key, selected_secret_ref, loader, "secret key"
        )
        self._session_token = session_token

        selected_prefix = (
            prefix if prefix is not None else getattr(config, "prefix", "")
        )
        self._prefix = _normalise_prefix(selected_prefix or "")

        selected_presign = (
            presign_seconds
            if presign_seconds is not None
            else getattr(config, "presign_seconds", 3600)
        )
        self._presign_seconds = _validate_presign(selected_presign)

        selected_path_style = (
            use_path_style
            if use_path_style is not None
            else getattr(config, "use_path_style", True)
        )
        if not isinstance(selected_path_style, bool):
            raise TypeError("use_path_style must be a boolean")
        self._use_path_style = selected_path_style

        self._clock = clock or (lambda: datetime.now(UTC))
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def bucket(self) -> str:
        """The configured bucket name."""
        return self._bucket

    @property
    def endpoint(self) -> str:
        """The configured endpoint URL."""
        return self._endpoint

    @property
    def region(self) -> str:
        """The configured signing region."""
        return self._region

    @property
    def prefix(self) -> str:
        """The key prefix applied to every object, without slashes."""
        return self._prefix

    @property
    def presign_seconds(self) -> int:
        """The default lifetime of a presigned URL."""
        return self._presign_seconds

    def _now(self) -> datetime:
        return _utc(self._clock())

    def _host(self) -> str:
        if self._use_path_style:
            return self._netloc
        return f"{self._bucket}.{self._netloc}"

    def _object_key(self, key: str) -> str:
        logical = _normalise_key(key)
        if not self._prefix:
            return logical
        return f"{self._prefix}/{logical}"

    def _object_path(self, key: str) -> str:
        object_key = self._object_key(key)
        segments: list[str] = []
        if self._base_path:
            segments.extend(self._base_path.split("/"))
        if self._use_path_style:
            segments.append(self._bucket)
        segments.extend(object_key.split("/"))
        return "/" + "/".join(_encode(segment) for segment in segments)

    def _bucket_path(self) -> str:
        segments: list[str] = []
        if self._base_path:
            segments.extend(self._base_path.split("/"))
        if self._use_path_style:
            segments.append(self._bucket)
        return "/" + "/".join(_encode(segment) for segment in segments)

    def _url_for(self, path: str, query: Mapping[str, str] | None = None) -> str:
        url = f"{self._scheme}://{self._host()}{path}"
        if query:
            url = f"{url}?{_canonical_query(query)}"
        return url

    def _authorized_headers(
        self,
        method: str,
        path: str,
        query: Mapping[str, str],
        payload_hash: str,
        extra: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        now = self._now()
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")
        scope = f"{date_stamp}/{self._region}/{_SERVICE}/aws4_request"

        headers = dict(extra or {})
        headers["host"] = self._host()
        headers["x-amz-content-sha256"] = payload_hash
        headers["x-amz-date"] = amz_date
        if self._session_token:
            headers["x-amz-security-token"] = self._session_token

        canonical_headers, signed_headers = _canonical_headers(headers)
        canonical_request = "\n".join(
            [
                method,
                path,
                _canonical_query(query),
                canonical_headers,
                signed_headers,
                payload_hash,
            ]
        )
        string_to_sign = "\n".join(
            [
                _ALGORITHM,
                amz_date,
                scope,
                _sha256_hex(canonical_request.encode("utf-8")),
            ]
        )
        signature = _signature(
            self._secret_key, date_stamp, self._region, string_to_sign
        )
        headers["authorization"] = (
            f"{_ALGORITHM} Credential={self._access_key}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        return headers

    def _presigned(self, key: str, expires_seconds: int) -> tuple[str, datetime]:
        now = self._now()
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")
        scope = f"{date_stamp}/{self._region}/{_SERVICE}/aws4_request"
        path = self._object_path(key)

        query = {
            "X-Amz-Algorithm": _ALGORITHM,
            "X-Amz-Credential": f"{self._access_key}/{scope}",
            "X-Amz-Date": amz_date,
            "X-Amz-Expires": str(expires_seconds),
            "X-Amz-SignedHeaders": "host",
        }
        if self._session_token:
            query["X-Amz-Security-Token"] = self._session_token

        canonical_headers = f"host:{self._host()}\n"
        canonical_request = "\n".join(
            [
                "GET",
                path,
                _canonical_query(query),
                canonical_headers,
                "host",
                _UNSIGNED_PAYLOAD,
            ]
        )
        string_to_sign = "\n".join(
            [
                _ALGORITHM,
                amz_date,
                scope,
                _sha256_hex(canonical_request.encode("utf-8")),
            ]
        )
        signature = _signature(
            self._secret_key, date_stamp, self._region, string_to_sign
        )
        url = (
            f"{self._scheme}://{self._host()}{path}?"
            f"{_canonical_query(query)}&X-Amz-Signature={signature}"
        )
        return url, now + timedelta(seconds=expires_seconds)

    def url(self, key: str, *, expires_seconds: int | None = None) -> str:
        """Return a presigned GET URL for *key* that expires after some time."""
        if expires_seconds is None:
            seconds = self._presign_seconds
        else:
            seconds = _validate_presign(expires_seconds)
        url, _ = self._presigned(key, seconds)
        return url

    async def put(self, request: StoragePutRequest) -> StorageObject:
        """Store *request*, deduping identical bytes under the same key."""
        if not isinstance(request, StoragePutRequest):
            raise TypeError("request must be a StoragePutRequest")
        digest = _sha256_hex(request.data)
        key = (
            _normalise_key(request.key)
            if request.key is not None
            else _content_key(digest)
        )

        existing = await self.stat(key)
        if existing is not None:
            if existing.sha256 == digest:
                return existing
            if existing.sha256 is None:
                try:
                    if await self.get(key) == request.data:
                        return existing
                except KeyError:
                    pass
            if not request.overwrite:
                raise InvalidRequestError(
                    f"storage key already exists: {key}",
                    details={"key": key},
                )

        content_type = request.content_type or _guess_content_type(key)
        extra: dict[str, str] = {}
        if content_type:
            extra["content-type"] = content_type
        for name, value in request.metadata.items():
            header = f"x-amz-meta-{str(name).strip().lower()}"
            extra[header] = str(value)
        extra["x-amz-meta-sha256"] = digest

        path = self._object_path(key)
        headers = self._authorized_headers("PUT", path, {}, digest, extra)
        response = await self.http.request(
            "PUT",
            self._url_for(path),
            headers=headers,
            content=request.data,
        )
        try:
            if response.status_code >= 400:
                error = error_for_response(response)
                if error is not None:
                    raise error
        finally:
            await response.aclose()

        url, expires_at = self._presigned(key, self._presign_seconds)
        return StorageObject(
            key=key,
            size_bytes=len(request.data),
            content_type=content_type,
            sha256=digest,
            url=url,
            expires_at=expires_at,
            created_at=self._now(),
            metadata=dict(request.metadata),
        )

    async def get(self, key: str) -> bytes:
        """Read *key*, raising :class:`KeyError` when it does not exist."""
        logical = _normalise_key(key)
        path = self._object_path(logical)
        headers = self._authorized_headers("GET", path, {}, _sha256_hex(b""))
        response = await self.http.request("GET", self._url_for(path), headers=headers)
        try:
            if response.status_code == 404:
                raise KeyError(logical)
            if response.status_code >= 400:
                error = error_for_response(response)
                if error is not None:
                    raise error
            return response.content
        finally:
            await response.aclose()

    async def stat(self, key: str) -> StorageObject | None:
        """Return the record for *key*, or ``None`` when it is missing."""
        logical = _normalise_key(key)
        path = self._object_path(logical)
        headers = self._authorized_headers("HEAD", path, {}, _sha256_hex(b""))
        response = await self.http.request("HEAD", self._url_for(path), headers=headers)
        try:
            if response.status_code == 404:
                return None
            if response.status_code >= 400:
                error = error_for_response(response)
                if error is not None:
                    raise error

            response_headers = response.headers
            size = _content_length(response_headers)
            content_type = response_headers.get("content-type")
            sha256 = response_headers.get("x-amz-meta-sha256") or _etag_sha256(
                response_headers.get("etag")
            )
            metadata = _user_metadata(response_headers)
            created_at = _last_modified(response_headers.get("last-modified"))
            url, expires_at = self._presigned(logical, self._presign_seconds)
            return StorageObject(
                key=logical,
                size_bytes=size,
                content_type=content_type,
                sha256=sha256,
                url=url,
                expires_at=expires_at,
                created_at=created_at,
                metadata=metadata,
            )
        finally:
            await response.aclose()

    async def delete(self, key: str) -> bool:
        """Delete *key* and report whether an object was removed."""
        if await self.stat(key) is None:
            return False
        logical = _normalise_key(key)
        path = self._object_path(logical)
        headers = self._authorized_headers("DELETE", path, {}, _sha256_hex(b""))
        response = await self.http.request(
            "DELETE", self._url_for(path), headers=headers
        )
        try:
            if response.status_code == 404:
                return False
            if response.status_code >= 400:
                error = error_for_response(response)
                if error is not None:
                    raise error
            return True
        finally:
            await response.aclose()

    async def check_health(self) -> ProviderHealth:
        """Probe the bucket with a signed HEAD request."""
        path = self._bucket_path()
        headers = self._authorized_headers("HEAD", path, {}, _sha256_hex(b""))
        response = await self.http.request("HEAD", self._url_for(path), headers=headers)
        try:
            if 200 <= response.status_code < 300:
                return ProviderHealth(
                    status=HealthStatus.HEALTHY,
                    message="ok",
                    details={"bucket": self._bucket},
                )
            return ProviderHealth(
                status=HealthStatus.UNHEALTHY,
                message=f"bucket probe returned HTTP {response.status_code}",
                details={"status": response.status_code, "bucket": self._bucket},
            )
        finally:
            await response.aclose()


S3Storage = S3StorageProvider
"""Alias for :class:`S3StorageProvider`."""


def _required(value: str | None, field: str) -> str:
    if value is None or not str(value).strip():
        raise ConfigError(
            f"s3 storage requires {field}",
            details={"field": field},
        )
    return str(value).strip()


def _credential(
    direct: str | None,
    reference: str | None,
    loader: SecretsLoader,
    label: str,
) -> str:
    if direct is not None:
        if not direct.strip():
            raise ConfigError(f"s3 storage {label} must not be blank")
        return direct
    if reference is None or not reference.strip():
        raise ConfigError(f"s3 storage requires a {label} reference")
    return loader.get(reference.strip())


def _normalise_prefix(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("prefix must be a string")
    text = value.strip().strip(_KEY_SEPARATOR)
    if not text:
        return ""
    return _normalise_key(text)


def _validate_presign(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("expires_seconds must be an integer")
    if value <= 0:
        raise ValueError("expires_seconds must be positive")
    if value > _MAX_PRESIGN_SECONDS:
        raise ValueError(
            f"expires_seconds must not exceed {_MAX_PRESIGN_SECONDS} seconds"
        )
    return value


def _content_length(headers: Mapping[str, str]) -> int:
    raw = headers.get("content-length")
    if raw is None:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _etag_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    digest = value.strip().strip('"').lower()
    if len(digest) == 64 and all(char in "0123456789abcdef" for char in digest):
        return digest
    return None


def _user_metadata(headers: Mapping[str, str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, value in headers.items():
        lowered = name.lower()
        if not lowered.startswith("x-amz-meta-"):
            continue
        key = lowered[len("x-amz-meta-") :]
        if key == "sha256":
            continue
        values[key] = value
    return values


def _last_modified(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return _utc(parsed)


register_provider("s3", S3StorageProvider, module=__name__)
