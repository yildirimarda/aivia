"""Collect user-supplied reference inputs from the CLI or the API.

A reference may arrive as a bare path, a URL, a mapping or an already validated
:class:`~aivia.reference.schema.ReferenceSpec`. Every local file is classified
twice: once by its extension and once by probing the bytes. The two answers must
agree, so a video renamed to ``.png`` is rejected before any paid analysis runs.
Images are checked from their header, video and audio through ffprobe, and text
by decoding it as UTF-8. Remote references are classified by extension because
their content is only available after a download.

The module returns plain :class:`~aivia.reference.schema.ReferenceSpec` values.
It never calls a provider, writes a file or imports the media services.
"""

from __future__ import annotations

import hashlib
import mimetypes
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from pydantic import ValidationError as PydanticValidationError

from aivia.core.errors import ValidationError
from aivia.core.media_probe import (
    FFprobeRunner,
    MediaCategory,
    MediaLimits,
    sniff_media,
    validate_media,
)
from aivia.reference.schema import ReferenceKind, ReferenceSpec

__all__ = [
    "collect_references",
    "load_reference",
    "load_reference_inputs",
    "probe_reference",
]

_CATEGORY_KINDS = {
    MediaCategory.IMAGE: ReferenceKind.IMAGE,
    MediaCategory.VIDEO: ReferenceKind.VIDEO,
    MediaCategory.AUDIO: ReferenceKind.AUDIO,
}
"""Map a sniffed media family onto a reference kind."""

_REFERENCE_KEYS = frozenset({"kind", "type", "reference_kind"})


def load_reference_inputs(
    values: Sequence[Any] | Any | None,
    *,
    probe: bool = True,
    limits: MediaLimits | None = None,
    ffprobe_runner: FFprobeRunner | None = None,
    ffprobe_executable: str = "ffprobe",
    max_bytes: int | None = None,
) -> list[ReferenceSpec]:
    """Return one validated reference per entry in *values*.

    A single value is accepted as well as a sequence, which makes the same
    helper serve an ``argparse`` append option and a JSON array. Order is
    preserved. ``probe=False`` validates only the extension, which is useful
    when the content is not on this machine yet; a remote URL is never probed
    here.
    """
    entries = _entries(values)
    return [
        load_reference(
            entry,
            probe=probe,
            limits=limits,
            ffprobe_runner=ffprobe_runner,
            ffprobe_executable=ffprobe_executable,
            max_bytes=max_bytes,
        )
        for entry in entries
    ]


collect_references = load_reference_inputs
"""Alias for :func:`load_reference_inputs`."""


def load_reference(
    value: Any,
    *,
    probe: bool = True,
    limits: MediaLimits | None = None,
    ffprobe_runner: FFprobeRunner | None = None,
    ffprobe_executable: str = "ffprobe",
    max_bytes: int | None = None,
) -> ReferenceSpec:
    """Validate one reference value and return its normalized spec.

    *value* may be a path, an HTTP(S) URL, a mapping or a
    :class:`~aivia.reference.schema.ReferenceSpec`. A local path is probed
    unless *probe* is false. The inferred kind must match both the extension
    and, when probed, the content.
    """
    spec, explicit_kind = _coerce(value)
    if spec.path:
        return _local_reference(
            spec,
            explicit_kind=explicit_kind,
            probe=probe,
            limits=limits,
            ffprobe_runner=ffprobe_runner,
            ffprobe_executable=ffprobe_executable,
            max_bytes=max_bytes,
        )
    if spec.url:
        return _remote_reference(spec, explicit_kind=explicit_kind)
    raise ValidationError(
        "a reference needs a path or a URL",
        details={"reference": _safe_repr(value)},
    )


def probe_reference(
    path: str | Path,
    *,
    limits: MediaLimits | None = None,
    ffprobe_runner: FFprobeRunner | None = None,
    ffprobe_executable: str = "ffprobe",
    max_bytes: int | None = None,
) -> ReferenceSpec:
    """Probe one local file and return its reference spec."""
    return _local_reference(
        ReferenceSpec(path=str(path)),
        explicit_kind=None,
        probe=True,
        limits=limits,
        ffprobe_runner=ffprobe_runner,
        ffprobe_executable=ffprobe_executable,
        max_bytes=max_bytes,
    )


def _entries(values: Sequence[Any] | Any | None) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, (str, bytes, bytearray, Path, Mapping, ReferenceSpec)):
        return [values]
    if isinstance(values, Sequence):
        return list(values)
    return [values]


def _coerce(value: Any) -> tuple[ReferenceSpec, ReferenceKind | None]:
    if isinstance(value, ReferenceSpec):
        return value.model_copy(deep=True), None

    explicit_kind: ReferenceKind | None = None
    if isinstance(value, (str, Path)):
        text = str(value).strip()
        if not text:
            raise ValidationError("a reference path or URL must not be empty")
        spec = _spec_for_text(text)
    elif isinstance(value, Mapping):
        data = dict(value)
        raw_kind = next(
            (data[key] for key in _REFERENCE_KEYS if key in data),
            None,
        )
        if raw_kind is not None:
            try:
                explicit_kind = ReferenceKind(raw_kind)
            except ValueError as exc:
                allowed = ", ".join(item.value for item in ReferenceKind)
                raise ValidationError(
                    f"unknown reference kind {raw_kind!r} (allowed: {allowed})",
                    details={"kind": str(raw_kind)},
                ) from exc
        try:
            spec = ReferenceSpec.model_validate(data)
        except PydanticValidationError as exc:
            raise ValidationError(
                f"invalid reference: {exc}",
                details={"reference": _safe_repr(data)},
            ) from exc
    else:
        raise ValidationError(
            "a reference must be a path, URL, mapping or reference spec, "
            f"got {type(value).__name__}",
            details={"reference": _safe_repr(value)},
        )
    return spec, explicit_kind


def _spec_for_text(text: str) -> ReferenceSpec:
    parsed = urlsplit(text)
    if parsed.scheme.lower() in {"http", "https"}:
        return ReferenceSpec(url=text)
    if parsed.scheme.lower() == "file":
        return ReferenceSpec(path=unquote(parsed.path))
    return ReferenceSpec(path=text)


def _local_reference(
    spec: ReferenceSpec,
    *,
    explicit_kind: ReferenceKind | None,
    probe: bool,
    limits: MediaLimits | None,
    ffprobe_runner: FFprobeRunner | None,
    ffprobe_executable: str,
    max_bytes: int | None,
) -> ReferenceSpec:
    path = Path(spec.path).expanduser()
    if not path.is_file():
        raise ValidationError(
            f"reference file not found: {path}",
            details={"path": str(path)},
        )

    extension_kind = ReferenceKind.from_path(str(path))
    _check_kind("extension", extension_kind, explicit_kind, path)
    if not probe:
        kind = explicit_kind or extension_kind
        if kind is None:
            raise ValidationError(
                f"cannot tell the reference type from its extension: {path}",
                details={"path": str(path), "extension": path.suffix},
            )
        return _with_file_metadata(spec, path, kind=kind, probed=False)

    active_limits = limits or MediaLimits()
    payload = _read_payload(path, max_bytes or active_limits.max_bytes)
    detected = _probe_payload(
        payload,
        path=path,
        limits=active_limits,
        ffprobe_runner=ffprobe_runner,
        ffprobe_executable=ffprobe_executable,
    )
    if detected is None:
        if extension_kind is ReferenceKind.TEXT:
            detected = ReferenceKind.TEXT
        else:
            raise ValidationError(
                f"reference content is not recognizable as "
                f"{extension_kind.value if extension_kind else 'media'}: {path}",
                details={"path": str(path), "extension": path.suffix},
            )

    _check_kind("extension", extension_kind, detected, path)
    _check_kind("declared kind", explicit_kind, detected, path)

    mime_type = _mime_type(path, detected, payload)
    return _with_file_metadata(
        spec,
        path,
        kind=detected,
        probed=True,
        payload=payload,
        mime_type=mime_type,
    )


def _remote_reference(
    spec: ReferenceSpec,
    *,
    explicit_kind: ReferenceKind | None,
) -> ReferenceSpec:
    url = spec.url.strip()
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValidationError(
            f"unsupported reference URL: {url}",
            details={"url": url},
        )
    extension_kind = ReferenceKind.from_path(url)
    _check_kind("extension", extension_kind, explicit_kind, url)
    kind = explicit_kind or extension_kind
    if kind is None:
        raise ValidationError(
            f"cannot tell the reference type from its URL extension: {url}",
            details={"url": url},
        )
    mime_type = mimetypes.guess_type(parsed.path)[0] or ""
    metadata = dict(spec.metadata)
    metadata.setdefault("probe", "extension")
    return spec.model_copy(
        update={
            "kind": kind,
            "url": url,
            "mime_type": mime_type,
            "metadata": metadata,
        }
    )


def _probe_payload(
    payload: bytes,
    *,
    path: Path,
    limits: MediaLimits,
    ffprobe_runner: FFprobeRunner | None,
    ffprobe_executable: str,
) -> ReferenceKind | None:
    signature = sniff_media(payload)
    if signature is not None:
        try:
            validate_media(
                payload,
                path=path,
                limits=limits,
                ffprobe_executable=ffprobe_executable,
                ffprobe_runner=ffprobe_runner,
            )
        except ValidationError as exc:
            raise ValidationError(
                f"reference {path} is not valid {signature.category.value}: "
                f"{exc.message}",
                details={"path": str(path), **exc.details},
            ) from exc
        return _CATEGORY_KINDS[signature.category]

    if _is_text(payload):
        return ReferenceKind.TEXT
    return None


def _read_payload(path: Path, max_bytes: int) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValidationError(
            f"cannot read reference file {path}: {exc}",
            details={"path": str(path)},
        ) from exc
    if size > max_bytes:
        raise ValidationError(
            f"reference {path} is {size} bytes, above the {max_bytes} byte cap",
            details={"path": str(path), "size_bytes": size, "limit": max_bytes},
        )
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValidationError(
            f"cannot read reference file {path}: {exc}",
            details={"path": str(path)},
        ) from exc


def _is_text(payload: bytes) -> bool:
    if not payload or b"\x00" in payload:
        return False
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return bool(text.strip())


def _check_kind(
    source: str,
    expected: ReferenceKind | None,
    actual: ReferenceKind | None,
    target: str | Path,
) -> None:
    if expected is None or actual is None or expected is actual:
        return
    raise ValidationError(
        f"reference {source} says {expected.value} but {target} is {actual.value}",
        details={
            "path": str(target),
            "expected_kind": expected.value,
            "detected_kind": actual.value,
        },
    )


def _with_file_metadata(
    spec: ReferenceSpec,
    path: Path,
    *,
    kind: ReferenceKind,
    probed: bool,
    payload: bytes | None = None,
    mime_type: str = "",
) -> ReferenceSpec:
    data = payload if payload is not None else _read_payload(path, _MAX_METADATA_BYTES)
    metadata = dict(spec.metadata)
    metadata.setdefault("probe", "content" if probed else "extension")
    return spec.model_copy(
        update={
            "kind": kind,
            "path": str(path),
            "mime_type": mime_type or _mime_type(path, kind, data),
            "size_bytes": len(data),
            "content_hash": hashlib.sha256(data).hexdigest(),
            "metadata": metadata,
        }
    )


def _mime_type(path: Path, kind: ReferenceKind, payload: bytes) -> str:
    signature = sniff_media(payload)
    if signature is not None:
        return signature.mime_type
    guessed = mimetypes.guess_type(path.name)[0]
    if guessed:
        return guessed
    if kind is ReferenceKind.TEXT:
        return "text/plain"
    return ""


def _safe_repr(value: Any) -> str:
    try:
        text = repr(value)
    except Exception:  # pragma: no cover - defensive for hostile objects
        return type(value).__name__
    return text if len(text) <= 200 else text[:197] + "..."


_MAX_METADATA_BYTES = MediaLimits().max_bytes
"""Cap used only when a caller asks for metadata without a probed payload."""
