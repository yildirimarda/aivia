"""Delivery providers: folder, expiring storage link, email and signed webhook.

Covers the four transports, the webhook host allowlist and the chain fallback
that retries transient failures, falls back between providers and raises when
every provider is exhausted.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from aivia.core.config.http import HttpClientConfig
from aivia.core.config.orders import DeliveryConfig
from aivia.core.errors import (
    ConfigError,
    InvalidRequestError,
    ProviderExhaustedError,
    RateLimitError,
)
from aivia.core.http import HttpClient
from aivia.core.models.base import utcnow
from aivia.core.models.provider import ProviderMeta
from aivia.core.secrets import SecretsLoader
from aivia.providers import (
    DeliveryChain,
    EmailDeliveryProvider,
    ExpiringStorageLinkDeliveryProvider,
    LocalFolderDeliveryProvider,
    LocalStorageProvider,
    SignedWebhookDeliveryProvider,
    build_delivery_provider,
    build_delivery_providers,
)
from aivia.providers.base import HealthStatus, ProviderHealth
from aivia.providers.protocols.delivery import (
    CAPABILITY_DELIVERY,
    DeliveryArtifact,
    DeliveryMethod,
    DeliveryProvider,
    DeliveryReceipt,
    DeliveryRequest,
)
from aivia.providers.protocols.storage import StorageObject

__all__ = ["_FakeDelivery", "_SmtpServer"]


def _artifact(data: bytes = b"card", *, filename: str = "card.png") -> DeliveryArtifact:
    return DeliveryArtifact(data=data, media_type="image/png", filename=filename)


def _request(
    method: DeliveryMethod = DeliveryMethod.FOLDER,
    target: str = "/tmp/deliveries",
    **overrides: Any,
) -> DeliveryRequest:
    values: dict[str, Any] = {
        "method": method,
        "target": target,
        "order_id": "order-1",
        "run_id": "run-1",
        "subject": "Your card",
        "message": "Enjoy",
        "artifacts": [_artifact()],
    }
    values.update(overrides)
    return DeliveryRequest(**values)


class _SmtpServer:
    """A minimal asyncio SMTP server sufficient for smtplib to deliver mail."""

    def __init__(self) -> None:
        self.messages: list[bytes] = []
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> int:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self._server.sockets[0].getsockname()[1]  # type: ignore[no-any-return]

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        writer.write(b"220 smtp.example.test ESMTP ready\r\n")
        await writer.drain()
        while True:
            line = await reader.readline()
            if not line:
                break
            command = line.decode("utf-8", errors="replace").strip().upper()
            token = command.split()[0] if command else ""
            if token == "QUIT":
                writer.write(b"221 bye\r\n")
                await writer.drain()
                break
            if token in {"EHLO", "HELO"}:
                writer.write(b"250-smtp.example.test\r\n250 8BITMIME\r\n")
            elif token in {"MAIL", "RCPT"}:
                writer.write(b"250 OK\r\n")
            elif command == "DATA":
                writer.write(b"354 end data with <CR><LF>.<CR><LF>\r\n")
                await writer.drain()
                data = b""
                while True:
                    chunk = await reader.readline()
                    if chunk in {b".\r\n", b".\n", b""}:
                        break
                    data += chunk
                self.messages.append(data)
                writer.write(b"250 OK queued\r\n")
            else:
                writer.write(b"502 unknown command\r\n")
            await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()


class _FakeStorage:
    """An in-memory storage double recording every put for the link provider."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.put_calls: list[Any] = []

    async def put(self, request: Any) -> StorageObject:
        self.put_calls.append(request)
        digest = hashlib.sha256(request.data).hexdigest()
        key = request.key or f"objects/{digest[:2]}/{digest}"
        self.objects[key] = request.data
        return StorageObject(
            key=key,
            size_bytes=len(request.data),
            content_type=request.content_type,
            sha256=digest,
            url=f"https://storage.example.test/{key}",
            created_at=utcnow(),
        )

    async def get(self, key: str) -> bytes:
        if key not in self.objects:
            raise KeyError(key)
        return self.objects[key]

    def url(self, key: str, *, expires_seconds: int | None = None) -> str:
        return f"https://storage.example.test/{key}"


class _FakeDelivery:
    """A structural delivery provider double recording calls and outcomes."""

    def __init__(
        self,
        name: str,
        *,
        outcome: str = "ok",
        error: BaseException | None = None,
    ) -> None:
        self.name = name
        self._outcome = outcome
        self._error = error or RuntimeError("boom")
        self.calls = 0
        self.last_request: DeliveryRequest | None = None

    @property
    def metadata(self) -> ProviderMeta:
        return ProviderMeta(name=self.name, capabilities=[CAPABILITY_DELIVERY])

    def supports(self, capability: str) -> bool:
        return capability == CAPABILITY_DELIVERY

    async def health(self) -> ProviderHealth:
        return ProviderHealth(status=HealthStatus.HEALTHY, message="ok")

    async def close(self) -> None:
        return None

    async def deliver(self, request: DeliveryRequest) -> DeliveryReceipt:
        self.calls += 1
        self.last_request = request
        if self._outcome == "fail":
            raise self._error
        return DeliveryReceipt(
            method=request.method,
            target=request.target,
            provider=self.name,
            delivered_at=utcnow(),
        )


def _noop_sleep(_seconds: float) -> None:
    return None


def _chain(
    config_values: dict[str, Any], providers: dict[str, _FakeDelivery]
) -> DeliveryChain:
    values = dict(config_values)
    values.setdefault("providers", {})
    for name in providers:
        values["providers"].setdefault(name, {})
    values.setdefault("default", next(iter(providers)))
    config = DeliveryConfig.from_mapping(values)
    return DeliveryChain(config, providers, sleep=_noop_sleep)


# ---------------------------------------------------------------------------
# Local folder provider
# ---------------------------------------------------------------------------


def test_folder_provider_copies_artifacts_into_the_target_directory(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "deliveries"
    provider = LocalFolderDeliveryProvider(path=str(folder))
    receipt = asyncio.run(provider.deliver(_request()))

    assert receipt.method is DeliveryMethod.FOLDER
    assert receipt.target == str(folder.resolve())
    assert receipt.provider == provider.name
    assert receipt.delivered_at is not None
    assert (folder / "card.png").read_bytes() == b"card"


def test_folder_provider_uses_artifact_filename_and_writes_bytes(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "out"
    provider = LocalFolderDeliveryProvider(path=str(folder))
    request = _request(
        artifacts=[
            DeliveryArtifact(data=b"one", media_type="text/plain", filename="one.txt"),
            DeliveryArtifact(data=b"two", media_type="text/plain", filename="two.txt"),
        ]
    )
    receipt = asyncio.run(provider.deliver(request))

    assert (folder / "one.txt").read_bytes() == b"one"
    assert (folder / "two.txt").read_bytes() == b"two"
    assert receipt.metadata["artifacts"] == 2


def test_folder_provider_rejects_the_wrong_method() -> None:
    provider = LocalFolderDeliveryProvider(path="/tmp/deliveries")
    request = _request(method=DeliveryMethod.LINK)
    with pytest.raises(ValueError, match="folder provider cannot deliver"):
        asyncio.run(provider.deliver(request))


# ---------------------------------------------------------------------------
# Expiring storage link provider
# ---------------------------------------------------------------------------


def test_link_provider_stores_artifacts_and_returns_an_expiring_link(
    tmp_path: Path,
) -> None:
    storage = _FakeStorage()
    provider = ExpiringStorageLinkDeliveryProvider(
        storage=storage, expires_seconds=3600
    )
    request = _request(method=DeliveryMethod.LINK)
    receipt = asyncio.run(provider.deliver(request))

    assert receipt.method is DeliveryMethod.LINK
    assert receipt.url is not None and receipt.url.startswith("https://")
    assert receipt.expires_at is not None
    assert len(storage.put_calls) == 1
    assert storage.put_calls[0].data == b"card"
    key = storage.put_calls[0].key
    assert receipt.url == f"https://storage.example.test/{key}"
    assert asyncio.run(storage.get(key)) == b"card"
    assert receipt.metadata["expires_seconds"] == 3600


def test_link_provider_requires_positive_expiry() -> None:
    with pytest.raises(ValueError, match="expires_seconds must be positive"):
        ExpiringStorageLinkDeliveryProvider(storage=_FakeStorage(), expires_seconds=0)


def test_link_provider_rejects_the_wrong_method(tmp_path: Path) -> None:
    provider = ExpiringStorageLinkDeliveryProvider(
        storage=_FakeStorage(), expires_seconds=3600
    )
    request = _request(method=DeliveryMethod.FOLDER)
    with pytest.raises(ValueError, match="link provider cannot deliver"):
        asyncio.run(provider.deliver(request))


# ---------------------------------------------------------------------------
# Email provider
# ---------------------------------------------------------------------------


async def _run_email_test(
    provider: EmailDeliveryProvider, request: DeliveryRequest
) -> tuple[DeliveryReceipt, _SmtpServer]:
    server = _SmtpServer()
    port = await server.start()
    try:
        provider._smtp_port = port
        receipt = await provider.deliver(request)
    finally:
        await server.stop()
    return receipt, server


def test_email_sends_via_local_smtp() -> None:
    provider = EmailDeliveryProvider(
        smtp_host="127.0.0.1",
        smtp_port=0,
        from_address="aivia@example.test",
        subject="Your card",
    )
    request = _request(
        method=DeliveryMethod.EMAIL,
        target="recipient@example.test",
        artifacts=[
            _artifact(),
            DeliveryArtifact(
                data=b"audio", media_type="audio/wav", filename="voice.wav"
            ),
        ],
    )

    receipt, server = asyncio.run(_run_email_test(provider, request))

    assert receipt.method is DeliveryMethod.EMAIL
    assert receipt.provider == provider.name
    assert len(server.messages) == 1
    text = server.messages[0].decode("utf-8", errors="replace")
    assert "Subject: Your card" in text
    assert "recipient@example.test" in text
    assert "From: aivia@example.test" in text
    assert 'filename="card.png"' in text
    assert 'filename="voice.wav"' in text


def test_email_requires_a_from_address_and_recipient() -> None:
    provider = EmailDeliveryProvider(smtp_host="127.0.0.1", from_address="")
    with pytest.raises(ValueError, match="from_address"):
        asyncio.run(provider.deliver(_request(method=DeliveryMethod.EMAIL)))

    provider = EmailDeliveryProvider(
        smtp_host="127.0.0.1", from_address="a@example.test"
    )
    request = DeliveryRequest.model_construct(
        method=DeliveryMethod.EMAIL,
        target="",
        order_id="order-1",
        subject="x",
        message="m",
    )
    with pytest.raises(ValueError, match="recipient"):
        asyncio.run(provider.deliver(request))

    with pytest.raises(ValueError, match="email provider cannot deliver"):
        asyncio.run(provider.deliver(_request(method=DeliveryMethod.FOLDER)))


# ---------------------------------------------------------------------------
# Signed webhook provider
# ---------------------------------------------------------------------------


def _captured_requests() -> tuple[list[httpx.Request], httpx.MockTransport]:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, text="delivered")

    return captured, httpx.MockTransport(handler)


def _signature(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def test_webhook_posts_a_signed_payload_to_an_allowlisted_host() -> None:
    captured, transport = _captured_requests()
    client = HttpClient(transport=transport, config=HttpClientConfig())
    provider = SignedWebhookDeliveryProvider(
        url="https://hooks.example.test/deliver",
        secret="s3cr3t",
        allowlist=["hooks.example.test"],
        http_client=client,
    )
    request = _request(
        method=DeliveryMethod.WEBHOOK,
        target="https://hooks.example.test/deliver",
    )
    receipt = asyncio.run(provider.deliver(request))

    assert receipt.method is DeliveryMethod.WEBHOOK
    assert receipt.target == "https://hooks.example.test/deliver"
    assert receipt.url == "https://hooks.example.test/deliver"
    assert receipt.provider == provider.name
    assert len(captured) == 1

    call = captured[0]
    assert call.method == "POST"
    assert call.url == httpx.URL("https://hooks.example.test/deliver")
    assert call.headers["Content-Type"] == "application/json"
    assert call.headers["X-Aivia-Signature"] == _signature("s3cr3t", call.content)
    payload = json.loads(call.content)
    assert payload["order_id"] == "order-1"
    assert payload["subject"] == "Your card"
    assert len(payload["artifacts"]) == 1
    assert receipt.metadata["signature_header"] == "X-Aivia-Signature"


def test_webhook_refuses_a_host_that_is_not_allowlisted() -> None:
    captured, transport = _captured_requests()
    client = HttpClient(transport=transport, config=HttpClientConfig())
    provider = SignedWebhookDeliveryProvider(
        url="https://evil.example.test/deliver",
        secret="s3cr3t",
        allowlist=["hooks.example.test"],
        http_client=client,
    )
    request = _request(method=DeliveryMethod.WEBHOOK)
    with pytest.raises(InvalidRequestError, match="not allowlisted"):
        asyncio.run(provider.deliver(request))
    assert captured == []  # no request was made


def test_webhook_accepts_any_host_when_the_allowlist_is_empty() -> None:
    captured, transport = _captured_requests()
    client = HttpClient(transport=transport, config=HttpClientConfig())
    provider = SignedWebhookDeliveryProvider(
        url="https://anywhere.example.test/deliver",
        secret="",
        allowlist=[],
        http_client=client,
    )
    receipt = asyncio.run(provider.deliver(_request(method=DeliveryMethod.WEBHOOK)))
    assert receipt.method is DeliveryMethod.WEBHOOK
    assert len(captured) == 1


def test_webhook_rejects_the_wrong_method() -> None:
    provider = SignedWebhookDeliveryProvider(
        url="https://hooks.example.test/deliver", secret="s"
    )
    with pytest.raises(ValueError, match="webhook provider cannot deliver"):
        asyncio.run(provider.deliver(_request(method=DeliveryMethod.FOLDER)))


# ---------------------------------------------------------------------------
# Factory: build providers from the delivery config section
# ---------------------------------------------------------------------------


def test_factory_builds_each_provider_type_from_config(tmp_path: Path) -> None:
    config = DeliveryConfig.from_mapping(
        {
            "chain": ["folder", "link", "email", "webhook"],
            "providers": {
                "folder": {"path": str(tmp_path / "d")},
                "link": {"expires_seconds": 900},
                "email": {
                    "smtp_host": "smtp.example.test",
                    "from_address": "aivia@example.test",
                    "username_ref": "AIVIA_SMTP_USER",
                    "password_ref": "AIVIA_SMTP_PASS",
                },
                "webhook": {
                    "url": "https://hooks.example.test/deliver",
                    "secret_ref": "AIVIA_WEBHOOK_SECRET",
                    "allowlist": ["Hooks.Example.Test"],
                },
            },
        }
    )
    storage = LocalStorageProvider(root=tmp_path / "storage")
    secrets = SecretsLoader(
        environ={
            "AIVIA_SMTP_USER": "user",
            "AIVIA_SMTP_PASS": "pass",
            "AIVIA_WEBHOOK_SECRET": "wh-secret",
        }
    )
    providers = build_delivery_providers(config, storage=storage, secrets=secrets)

    assert set(providers) == {"folder", "link", "email", "webhook"}
    assert isinstance(providers["folder"], LocalFolderDeliveryProvider)
    assert providers["folder"].path == tmp_path / "d"
    assert isinstance(providers["link"], ExpiringStorageLinkDeliveryProvider)
    assert providers["link"].expires_seconds == 900
    assert isinstance(providers["email"], EmailDeliveryProvider)
    assert providers["email"].from_address == "aivia@example.test"
    assert providers["email"]._username == "user"
    assert providers["email"]._password == "pass"
    assert isinstance(providers["webhook"], SignedWebhookDeliveryProvider)
    assert providers["webhook"].secret == "wh-secret"
    assert providers["webhook"].allowlist == ["hooks.example.test"]


def test_factory_skips_disabled_providers(tmp_path: Path) -> None:
    config = DeliveryConfig.from_mapping(
        {
            "providers": {
                "folder": {"path": str(tmp_path / "d"), "enabled": True},
                "link": {"enabled": False},
            }
        }
    )
    providers = build_delivery_providers(config, storage=LocalStorageProvider())
    assert set(providers) == {"folder"}


def test_factory_link_provider_requires_storage() -> None:
    config = DeliveryConfig.from_mapping({"providers": {"link": {}}, "default": "link"})
    with pytest.raises(ConfigError, match="requires a storage backend"):
        build_delivery_providers(config)


def test_build_delivery_provider_accepts_a_type_string() -> None:
    provider = build_delivery_provider(
        {"type": "folder", "path": "/tmp/out"}, name="out"
    )
    assert isinstance(provider, LocalFolderDeliveryProvider)
    assert provider.name == "out"


# ---------------------------------------------------------------------------
# Delivery chain: retries, fallback and exhaustion
# ---------------------------------------------------------------------------


def test_chain_delivers_with_the_first_healthy_provider() -> None:
    providers = {
        "a": _FakeDelivery("a"),
        "b": _FakeDelivery("b"),
    }
    chain = _chain({"chain": ["a", "b"]}, providers)
    receipt = asyncio.run(chain.deliver(_request()))

    assert receipt.provider == "a"
    assert providers["a"].calls == 1
    assert providers["b"].calls == 0


def test_chain_retries_a_retryable_provider_then_falls_back() -> None:
    providers = {
        "a": _FakeDelivery("a", outcome="fail", error=RateLimitError("slow down")),
        "b": _FakeDelivery("b"),
    }
    chain = _chain(
        {"chain": ["a", "b"], "max_attempts": 3, "retry_seconds": 0}, providers
    )
    receipt = asyncio.run(chain.deliver(_request()))

    assert receipt.provider == "b"
    assert providers["a"].calls == 3
    assert providers["b"].calls == 1


def test_chain_falls_back_immediately_on_a_permanent_error() -> None:
    providers = {
        "a": _FakeDelivery("a", outcome="fail", error=InvalidRequestError("bad")),
        "b": _FakeDelivery("b"),
    }
    chain = _chain({"chain": ["a", "b"], "max_attempts": 3}, providers)
    receipt = asyncio.run(chain.deliver(_request()))

    assert receipt.provider == "b"
    assert providers["a"].calls == 1  # permanent errors are not retried


def test_chain_raises_provider_exhausted_when_every_provider_fails() -> None:
    providers = {
        "a": _FakeDelivery("a", outcome="fail", error=RateLimitError("down")),
    }
    chain = _chain({"chain": ["a"], "max_attempts": 2, "retry_seconds": 0}, providers)
    with pytest.raises(ProviderExhaustedError) as caught:
        asyncio.run(chain.deliver(_request()))

    assert providers["a"].calls == 2
    assert "chain" in caught.value.details


def test_chain_uses_the_default_provider_when_the_chain_is_empty() -> None:
    providers = {"folder": _FakeDelivery("folder")}
    chain = _chain({"default": "folder", "chain": []}, providers)
    receipt = asyncio.run(chain.deliver(_request()))

    assert receipt.provider == "folder"
    assert providers["folder"].calls == 1


def test_chain_skips_unknown_chain_entries() -> None:
    runtime = {"folder": _FakeDelivery("folder")}
    config = DeliveryConfig.from_mapping(
        {"chain": ["missing", "folder"], "providers": {"missing": {}, "folder": {}}}
    )
    chain = DeliveryChain(config, runtime, sleep=_noop_sleep)
    receipt = asyncio.run(chain.deliver(_request()))

    assert receipt.provider == "folder"


def test_chain_rejects_a_non_delivery_request() -> None:
    chain = _chain({"chain": ["a"]}, {"a": _FakeDelivery("a")})
    with pytest.raises(TypeError, match="DeliveryRequest"):
        asyncio.run(chain.deliver("not a request"))  # type: ignore[arg-type]


def test_fakes_satisfy_the_delivery_provider_protocol() -> None:
    fake = _FakeDelivery("fake")
    assert isinstance(fake, DeliveryProvider)
    assert fake.supports(CAPABILITY_DELIVERY)
    assert not fake.supports("storage")


def test_delivery_receipt_normalizes_timestamps() -> None:
    naive = utcnow()
    receipt = DeliveryReceipt(
        method=DeliveryMethod.FOLDER,
        reference="  delivery-1  ",
        target="  /tmp/deliveries  ",
        provider="  folder  ",
        delivered_at=naive,
    )
    assert receipt.reference == "delivery-1"
    assert receipt.target == "/tmp/deliveries"
    assert receipt.provider == "folder"
    assert receipt.delivered is True
    assert receipt.delivered_at == naive


def test_delivery_artifact_requires_exactly_one_source() -> None:
    with pytest.raises(ValueError):
        DeliveryArtifact()
    with pytest.raises(ValueError):
        DeliveryArtifact(data=b"x", url="https://example.test/card.png")
    with pytest.raises(ValueError):
        DeliveryArtifact(url="ftp://example.test/card.png")
