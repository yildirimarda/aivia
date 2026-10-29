"""S3 storage: SigV4 requests and presigned expiring URLs against a local mock."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Any
from urllib.parse import parse_qsl, quote, unquote, urlsplit

import httpx
import pytest

from aivia.core.config import StorageConfig
from aivia.core.errors import ConfigError, InvalidRequestError
from aivia.core.secrets import SecretsLoader
from aivia.providers import (
    ProviderRegistry,
    S3Storage,
    S3StorageProvider,
    provider_registry,
)
from aivia.providers.protocols import (
    CAPABILITY_STORAGE,
    StorageObject,
    StorageProvider,
    StoragePutRequest,
)

ACCESS_KEY = "AKIDEXAMPLE"
SECRET_KEY = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
REGION = "eu-central-1"
BUCKET = "media"
ENDPOINT = "https://s3.example.test"


def _encode(value: str) -> str:
    return quote(value, safe="-_.~")


def _sign(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def _signature(secret: str, date_stamp: str, region: str, message: str) -> str:
    key = _sign(("AWS4" + secret).encode("utf-8"), date_stamp)
    key = _sign(key, region)
    key = _sign(key, "s3")
    key = _sign(key, "aws4_request")
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).hexdigest()


def _canonical_query(pairs: list[tuple[str, str]]) -> str:
    encoded = sorted((_encode(key), _encode(value)) for key, value in pairs)
    return "&".join(f"{key}={value}" for key, value in encoded)


def _canonical_uri(path: str) -> str:
    segments = [unquote(segment) for segment in path.split("/")]
    return "/".join(_encode(segment) for segment in segments)


class MockS3:
    """A tiny S3 endpoint that validates SigV4 and stores objects in memory."""

    def __init__(self, *, now: Callable[[], datetime] | None = None) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.requests: list[httpx.Request] = []
        self.now = now or (lambda: datetime.now(UTC))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._authorized(request):
            return httpx.Response(403, text="signature mismatch")

        segments = [unquote(part) for part in request.url.path.strip("/").split("/")]
        virtual_hosted = request.url.host == f"{BUCKET}.s3.example.test"
        if virtual_hosted:
            key = "/".join(segments)
        else:
            if not segments or segments[0] != BUCKET:
                return httpx.Response(404)
            key = "/".join(segments[1:])
        if not key:
            return httpx.Response(200)

        if request.method == "PUT":
            self.objects[key] = {
                "data": request.content,
                "content_type": request.headers.get("content-type"),
                "metadata": {
                    name.lower(): value
                    for name, value in request.headers.items()
                    if name.lower().startswith("x-amz-meta-")
                },
                "modified": self.now(),
            }
            return httpx.Response(200)

        if request.method == "GET":
            stored = self.objects.get(key)
            if stored is None:
                return httpx.Response(404)
            headers = {
                "content-type": stored["content_type"] or "application/octet-stream"
            }
            headers.update(stored["metadata"])
            return httpx.Response(200, content=stored["data"], headers=headers)

        if request.method == "HEAD":
            stored = self.objects.get(key)
            if stored is None:
                return httpx.Response(404)
            digest = hashlib.sha256(stored["data"]).hexdigest()
            headers = {
                "content-length": str(len(stored["data"])),
                "content-type": stored["content_type"] or "application/octet-stream",
                "etag": f'"{digest}"',
                "last-modified": format_datetime(stored["modified"], usegmt=True),
            }
            headers.update(stored["metadata"])
            return httpx.Response(200, headers=headers)

        if request.method == "DELETE":
            self.objects.pop(key, None)
            return httpx.Response(204)

        return httpx.Response(405)

    def _authorized(self, request: httpx.Request) -> bool:
        params = dict(
            parse_qsl(request.url.query.decode("ascii"), keep_blank_values=True)
        )
        if "X-Amz-Signature" in params:
            return self._verify_presigned(request, params)
        return self._verify_header(request)

    def _verify_presigned(self, request: httpx.Request, params: dict[str, str]) -> bool:
        if request.method != "GET":
            return False
        if params.get("X-Amz-Algorithm") != "AWS4-HMAC-SHA256":
            return False
        credential = params.get("X-Amz-Credential", "")
        access_key, _, scope = credential.partition("/")
        date_stamp, region, service, terminator = scope.split("/")
        if (
            access_key != ACCESS_KEY
            or region != REGION
            or service != "s3"
            or terminator != "aws4_request"
        ):
            return False
        try:
            signed_at = datetime.strptime(
                params["X-Amz-Date"], "%Y%m%dT%H%M%SZ"
            ).replace(tzinfo=UTC)
            expires = int(params["X-Amz-Expires"])
        except (KeyError, ValueError):
            return False
        if self.now() > signed_at + timedelta(seconds=expires):
            return False
        if params.get("X-Amz-SignedHeaders") != "host":
            return False

        pairs = [
            (key, value) for key, value in params.items() if key != "X-Amz-Signature"
        ]
        host = str(request.headers.get("host") or request.url.netloc)
        canonical_request = "\n".join(
            [
                "GET",
                _canonical_uri(request.url.path),
                _canonical_query(pairs),
                f"host:{host}\n",
                "host",
                "UNSIGNED-PAYLOAD",
            ]
        )
        string_to_sign = "\n".join(
            [
                "AWS4-HMAC-SHA256",
                params["X-Amz-Date"],
                scope,
                hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
            ]
        )
        expected = _signature(SECRET_KEY, date_stamp, region, string_to_sign)
        return hmac.compare_digest(expected, params["X-Amz-Signature"])

    def _verify_header(self, request: httpx.Request) -> bool:
        authorization = request.headers.get("authorization", "")
        if not authorization.startswith("AWS4-HMAC-SHA256 "):
            return False
        fields = dict(
            item.strip().split("=", 1)
            for item in authorization[len("AWS4-HMAC-SHA256 ") :].split(",")
        )
        credential = fields.get("Credential", "")
        access_key, _, scope = credential.partition("/")
        date_stamp, region, service, terminator = scope.split("/")
        if (
            access_key != ACCESS_KEY
            or region != REGION
            or service != "s3"
            or terminator != "aws4_request"
        ):
            return False
        signed_headers = fields.get("SignedHeaders", "")
        names = signed_headers.split(";")
        canonical_headers = "".join(
            f"{name}:{' '.join(request.headers.get(name, '').split())}\n"
            for name in names
        )
        payload_hash = request.headers.get("x-amz-content-sha256", "")
        canonical_request = "\n".join(
            [
                request.method,
                _canonical_uri(request.url.path),
                _canonical_query(list(request.url.params.multi_items())),
                canonical_headers,
                signed_headers,
                payload_hash,
            ]
        )
        amz_date = request.headers.get("x-amz-date", "")
        string_to_sign = "\n".join(
            [
                "AWS4-HMAC-SHA256",
                amz_date,
                scope,
                hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
            ]
        )
        expected = _signature(SECRET_KEY, date_stamp, region, string_to_sign)
        return hmac.compare_digest(expected, fields.get("Signature", ""))


def _provider(mock: MockS3, **kwargs: Any) -> S3StorageProvider:
    values: dict[str, Any] = {
        "bucket": BUCKET,
        "endpoint": ENDPOINT,
        "region": REGION,
        "access_key": ACCESS_KEY,
        "secret_key": SECRET_KEY,
        "transport": httpx.MockTransport(mock),
    }
    values.update(kwargs)
    return S3StorageProvider(**values)


def put(provider: S3StorageProvider, data: bytes, **kwargs: Any) -> StorageObject:
    return asyncio.run(provider.put(StoragePutRequest(data=data, **kwargs)))


def test_s3_storage_declares_the_storage_capability() -> None:
    mock = MockS3()
    provider = _provider(mock)

    assert isinstance(provider, StorageProvider)
    assert provider.supports(CAPABILITY_STORAGE)
    assert provider.name == "s3"
    assert provider.bucket == BUCKET
    assert provider.endpoint == ENDPOINT
    assert provider.region == REGION
    assert provider.presign_seconds == 3600
    assert S3Storage is S3StorageProvider
    assert provider_registry.resolve("s3") is S3StorageProvider


def test_config_resolves_credential_references_through_the_secrets_loader() -> None:
    mock = MockS3()
    config = StorageConfig.from_mapping(
        {
            "backend": "s3",
            "bucket": BUCKET,
            "endpoint": ENDPOINT,
            "region": REGION,
            "access_key_ref": "AIVIA_S3_ACCESS_KEY",
            "secret_key_ref": "AIVIA_S3_SECRET_KEY",
            "prefix": "runs",
        }
    )
    secrets = SecretsLoader(
        environ={
            "AIVIA_S3_ACCESS_KEY": ACCESS_KEY,
            "AIVIA_S3_SECRET_KEY": SECRET_KEY,
        }
    )

    provider = S3StorageProvider(
        config,
        secrets=secrets,
        transport=httpx.MockTransport(mock),
    )
    stored = put(provider, b"hello", key="outputs/hello.txt")

    assert provider.prefix == "runs"
    assert stored.key == "outputs/hello.txt"
    assert "runs/outputs/hello.txt" in mock.objects
    assert asyncio.run(provider.get(stored.key)) == b"hello"


def test_put_get_stat_and_delete_round_trip_through_the_mock() -> None:
    mock = MockS3()
    provider = _provider(mock)
    data = b"hello world"
    digest = hashlib.sha256(data).hexdigest()

    stored = put(provider, data, content_type="text/plain", metadata={"run": "run-1"})

    assert stored.key == f"objects/{digest[:2]}/{digest}"
    assert stored.sha256 == digest
    assert stored.size_bytes == len(data)
    assert stored.content_type == "text/plain"
    assert stored.metadata == {"run": "run-1"}
    assert stored.created_at is not None
    assert stored.expires_at is not None
    assert stored.url is not None
    assert "X-Amz-Signature=" in stored.url
    assert asyncio.run(provider.get(stored.key)) == data

    restated = asyncio.run(provider.stat(stored.key))
    assert restated is not None
    assert restated.key == stored.key
    assert restated.sha256 == digest
    assert restated.size_bytes == len(data)
    assert restated.content_type == "text/plain"
    assert restated.metadata == {"run": "run-1"}

    assert asyncio.run(provider.delete(stored.key)) is True
    assert asyncio.run(provider.delete(stored.key)) is False
    assert asyncio.run(provider.stat(stored.key)) is None
    with pytest.raises(KeyError):
        asyncio.run(provider.get(stored.key))


def test_put_dedupes_identical_bytes_and_refuses_to_clobber() -> None:
    mock = MockS3()
    provider = _provider(mock)
    first = put(provider, b"same bytes", key="fixed.bin")
    writes = len(mock.requests)

    second = put(provider, b"same bytes", key="fixed.bin")

    assert second.key == first.key
    assert second.sha256 == first.sha256
    assert len(mock.requests) == writes + 1

    with pytest.raises(InvalidRequestError, match="already exists"):
        put(provider, b"different", key="fixed.bin")

    replaced = put(provider, b"different", key="fixed.bin", overwrite=True)
    assert replaced.sha256 == hashlib.sha256(b"different").hexdigest()
    assert asyncio.run(provider.get("fixed.bin")) == b"different"


def test_a_presigned_url_verifies_and_expires_at_the_mock() -> None:
    current = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    clock = [current]
    mock = MockS3(now=lambda: clock[0])
    provider = _provider(mock, clock=lambda: clock[0])
    stored = put(provider, b"presigned payload", key="links/file.txt")

    url = provider.url(stored.key, expires_seconds=60)
    parts = urlsplit(url)
    params = dict(parse_qsl(parts.query))
    assert parts.scheme == "https"
    assert parts.netloc == "s3.example.test"
    assert parts.path == f"/{BUCKET}/links/file.txt"
    assert params["X-Amz-Algorithm"] == "AWS4-HMAC-SHA256"
    assert params["X-Amz-Expires"] == "60"
    assert params["X-Amz-SignedHeaders"] == "host"
    assert params["X-Amz-Credential"].startswith(ACCESS_KEY + "/")
    assert params["X-Amz-Date"] == "20260101T120000Z"
    assert "X-Amz-Signature" in params

    async def fetch(target: str) -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock)) as client:
            return await client.get(target)

    response = asyncio.run(fetch(url))
    assert response.status_code == 200
    assert response.content == b"presigned payload"

    tampered = url[:-1] + ("0" if url[-1] != "0" else "1")
    assert asyncio.run(fetch(tampered)).status_code == 403

    clock[0] = current + timedelta(seconds=61)
    assert asyncio.run(fetch(url)).status_code == 403


def test_presign_lifetime_defaults_to_the_storage_config() -> None:
    mock = MockS3()
    config = StorageConfig.from_mapping(
        {
            "backend": "s3",
            "bucket": BUCKET,
            "endpoint": ENDPOINT,
            "region": REGION,
            "access_key_ref": "AIVIA_S3_ACCESS_KEY",
            "secret_key_ref": "AIVIA_S3_SECRET_KEY",
            "presign_seconds": 120,
        }
    )
    provider = S3StorageProvider(
        config,
        secrets=SecretsLoader(
            environ={
                "AIVIA_S3_ACCESS_KEY": ACCESS_KEY,
                "AIVIA_S3_SECRET_KEY": SECRET_KEY,
            }
        ),
        transport=httpx.MockTransport(mock),
    )

    assert provider.presign_seconds == 120
    params = dict(parse_qsl(urlsplit(provider.url("a.txt")).query))
    assert params["X-Amz-Expires"] == "120"
    with pytest.raises(ValueError):
        provider.url("a.txt", expires_seconds=0)
    with pytest.raises(ValueError):
        provider.url("a.txt", expires_seconds=7 * 24 * 60 * 60 + 1)


def test_path_style_prefix_and_virtual_host_addressing() -> None:
    mock = MockS3()
    provider = _provider(mock, prefix="tenant/run-1")
    put(provider, b"x", key="a/b.txt")

    assert "tenant/run-1/a/b.txt" in mock.objects
    assert urlsplit(provider.url("a/b.txt")).path == (f"/{BUCKET}/tenant/run-1/a/b.txt")

    virtual = _provider(mock, use_path_style=False)
    stored = put(virtual, b"y", key="a/b.txt")
    assert virtual.url(stored.key).startswith(
        f"https://{BUCKET}.s3.example.test/a/b.txt?"
    )


def test_unsafe_keys_cannot_escape_the_bucket() -> None:
    provider = _provider(MockS3())

    for key in ("../escape.txt", "/etc/passwd", "C:\\windows\\system32", "a/../../b"):
        with pytest.raises(InvalidRequestError):
            put(provider, b"x", key=key)

    with pytest.raises(InvalidRequestError):
        provider.url("../../etc/passwd")


def test_the_s3_provider_rejects_a_local_configuration_and_missing_values() -> None:
    local = StorageConfig.from_mapping({"backend": "local"})
    with pytest.raises(ConfigError, match="backend"):
        S3StorageProvider(local)

    with pytest.raises(ConfigError, match="bucket"):
        S3StorageProvider(
            endpoint=ENDPOINT,
            region=REGION,
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
        )

    with pytest.raises(ConfigError, match="access key reference"):
        S3StorageProvider(
            bucket=BUCKET,
            endpoint=ENDPOINT,
            region=REGION,
            secret_key=SECRET_KEY,
        )

    with pytest.raises(ConfigError, match="endpoint"):
        S3StorageProvider(
            bucket=BUCKET,
            endpoint="ftp://example.test",
            region=REGION,
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
        )


def test_health_probes_the_bucket() -> None:
    mock = MockS3()
    provider = _provider(mock)

    report = asyncio.run(provider.health())

    assert report.healthy
    assert report.details == {"bucket": BUCKET}


def test_the_provider_is_registered_once() -> None:
    registry = ProviderRegistry()
    registry.register("s3", S3StorageProvider)

    assert registry.resolve("s3") is S3StorageProvider
