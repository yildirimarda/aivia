"""Remote references are probed through the configured download policy."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any

import httpx
import pytest

from aivia.core.config.reference import ReferenceDownloadConfig
from aivia.core.errors import ValidationError
from aivia.reference import (
    ReferenceKind,
    load_reference,
    load_reference_inputs,
)


def png(width: int = 32, height: int = 32) -> bytes:
    header = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR"
    body = struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    return header + body + b"\x00\x00\x00\x00"


def text(content: str = "A calm opening with short cuts.\n") -> bytes:
    return content.encode("utf-8")


def transport_for(payload: bytes, *, status: int = 200) -> httpx.AsyncBaseTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            content=payload,
            headers={"content-length": str(len(payload))},
        )

    return httpx.MockTransport(handler)


def policy(**overrides: Any) -> ReferenceDownloadConfig:
    data = {"mode": "allow"}
    data.update(overrides)
    return ReferenceDownloadConfig.model_validate(data)


def test_remote_reference_denied_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match="not allowed"):
        load_reference(
            "https://example.test/look.png",
            download_policy=policy(mode="deny"),
        )


def test_remote_reference_ask_requires_confirmation() -> None:
    with pytest.raises(ValidationError, match="confirmation"):
        load_reference(
            "https://example.test/look.png",
            download_policy=policy(mode="ask"),
        )


def test_remote_reference_ask_downloads_when_confirmed(tmp_path: Path) -> None:
    payload = png()
    confirmed: list[str] = []

    def confirm(url: str) -> bool:
        confirmed.append(url)
        return True

    spec = load_reference(
        "https://example.test/look.png",
        download_policy=policy(mode="ask"),
        confirm_download=confirm,
        transport=transport_for(payload),
        temp_dir=tmp_path,
    )

    assert confirmed == ["https://example.test/look.png"]
    assert spec.kind is ReferenceKind.IMAGE
    assert spec.path
    assert Path(spec.path).is_file()
    assert spec.url == ""
    assert spec.size_bytes == len(payload)
    assert spec.content_hash == hashlib.sha256(payload).hexdigest()
    assert spec.mime_type == "image/png"
    assert spec.metadata["probe"] == "download"
    assert spec.metadata["source_url"] == "https://example.test/look.png"


def test_remote_reference_allow_downloads_and_probes(tmp_path: Path) -> None:
    payload = png()

    spec = load_reference(
        "https://cdn.example.test/look.png",
        download_policy=policy(),
        transport=transport_for(payload),
        temp_dir=tmp_path,
    )

    assert spec.kind is ReferenceKind.IMAGE
    assert Path(spec.path).is_file()
    assert spec.size_bytes == len(payload)
    assert spec.content_hash == hashlib.sha256(payload).hexdigest()
    assert spec.mime_type == "image/png"
    assert spec.metadata["probe"] == "download"


def test_remote_reference_download_text(tmp_path: Path) -> None:
    payload = text()

    spec = load_reference(
        "https://example.test/notes.txt",
        download_policy=policy(),
        transport=transport_for(payload),
        temp_dir=tmp_path,
    )

    assert spec.kind is ReferenceKind.TEXT
    assert spec.mime_type == "text/plain"
    assert spec.size_bytes == len(payload)
    assert spec.content_hash == hashlib.sha256(payload).hexdigest()


def test_remote_reference_download_rejects_oversized_body(tmp_path: Path) -> None:
    payload = png()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    with pytest.raises(ValidationError, match="byte cap"):
        load_reference(
            "https://example.test/look.png",
            download_policy=policy(max_bytes=1),
            transport=httpx.MockTransport(handler),
            temp_dir=tmp_path,
        )


def test_remote_reference_download_rejects_disallowed_host(tmp_path: Path) -> None:
    payload = png()

    with pytest.raises(ValidationError, match="allowlist"):
        load_reference(
            "https://evil.example.test/look.png",
            download_policy=policy(allowed_hosts=["cdn.example.test"]),
            transport=transport_for(payload),
            temp_dir=tmp_path,
        )


def test_remote_reference_download_rejects_unknown_content(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="not recognizable"):
        load_reference(
            "https://example.test/look",
            download_policy=policy(),
            transport=transport_for(b"\x00\x01\x02\x03"),
            temp_dir=tmp_path,
        )


def test_remote_reference_download_propagates_transport_errors(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    with pytest.raises(ValidationError, match="could not download"):
        load_reference(
            "https://example.test/look.png",
            download_policy=policy(),
            transport=httpx.MockTransport(handler),
            temp_dir=tmp_path,
        )


def test_remote_reference_without_policy_classifies_by_extension() -> None:
    spec = load_reference("https://example.test/look.png")
    assert spec.kind is ReferenceKind.IMAGE
    assert spec.url == "https://example.test/look.png"
    assert spec.path == ""
    assert spec.metadata["probe"] == "extension"


def test_load_reference_inputs_downloads_each_remote(tmp_path: Path) -> None:
    payload = png()
    specs = load_reference_inputs(
        ["https://example.test/a.png", "https://example.test/b.png"],
        download_policy=policy(),
        transport=transport_for(payload),
        temp_dir=tmp_path,
    )
    assert len(specs) == 2
    assert all(spec.kind is ReferenceKind.IMAGE for spec in specs)
    assert {spec.metadata["source_url"] for spec in specs} == {
        "https://example.test/a.png",
        "https://example.test/b.png",
    }


def test_local_reference_still_probed_without_download_policy(tmp_path: Path) -> None:
    image = tmp_path / "look.png"
    image.write_bytes(png())

    spec = load_reference(image)
    assert spec.kind is ReferenceKind.IMAGE
    assert spec.metadata["probe"] == "content"
    assert spec.size_bytes == len(png())


def test_download_policy_bool_false_denies(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="not allowed"):
        load_reference(
            "https://example.test/look.png",
            download_policy=ReferenceDownloadConfig.model_validate(False),
            temp_dir=tmp_path,
        )
