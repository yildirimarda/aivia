"""Persist visual results as asset rows with their full provenance.

An asset row is only useful if it records where the file came from and what it
may be used for. This module turns a generated asset, a stock result or a
visual QA result into one :class:`~aivia.core.models.asset.Asset` and writes it
through the asset repository, so provider, model, prompt hash, license, cost
links, quality score, alt text and the pinned flag all survive together.

Conversion is deterministic: an existing asset keeps its values unless an
explicit override is supplied, and a stock result maps its provider, license,
attribution and measured dimensions onto the same row shape.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from aivia.core.db.repos import AssetRepository
from aivia.core.models.asset import (
    Asset,
    AssetAnalysis,
    AssetKind,
    AssetLicense,
    compute_prompt_hash,
)
from aivia.providers.protocols.stock import StockAsset, StockMediaKind

__all__ = [
    "asset_row",
    "asset_row_from_stock",
    "asset_rows",
    "persist_asset_rows",
    "record_asset_rows",
    "write_asset_rows",
]

_LICENSE_KEYS = (
    "license",
    "licence",
    "spdx",
    "license_url",
    "licence_url",
    "terms_url",
    "attribution",
    "credit",
    "attribution_text",
    "attribution_required",
    "requires_attribution",
    "commercial_use",
    "allow_commercial",
    "share_alike",
)
_ALT_TEXT_KEYS = ("alt_text", "alt", "accessibility_text")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _license_from(value: Any) -> AssetLicense | None:
    if isinstance(value, AssetLicense):
        return value
    metadata = _mapping(value)
    if metadata is None:
        return None
    raw = metadata.get("license", metadata.get("licence"))
    if isinstance(raw, Mapping):
        return AssetLicense.model_validate(dict(raw))
    values = {key: metadata[key] for key in _LICENSE_KEYS if key in metadata}
    if not values:
        return None
    return AssetLicense.model_validate(values)


def _alt_text_from(value: Any) -> str:
    metadata = _mapping(value)
    if metadata is None:
        return ""
    for key in _ALT_TEXT_KEYS:
        text = _text(metadata.get(key))
        if text:
            return text
    return ""


def _score_of(value: Any) -> float | None:
    score = getattr(value, "score", None)
    if score is None:
        score = getattr(value, "quality_score", None)
    score = getattr(score, "overall", score)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    return float(score)


def _prompt_of(value: Any, asset: Asset | None, stock: StockAsset | None) -> str:
    request = getattr(value, "request", None)
    if request is not None:
        prompt = _text(getattr(request, "prompt", ""))
        if prompt:
            return prompt
        query = _text(getattr(request, "query", ""))
        if query:
            return query
    prompt = _text(getattr(value, "prompt", ""))
    if prompt:
        return prompt
    if asset is not None:
        return asset.prompt
    if stock is not None:
        return (
            _text(stock.metadata.get("description"))
            or _text(stock.metadata.get("alt"))
            or _text(stock.attribution)
            or stock.id
        )
    return ""


def _result_parts(value: Any) -> tuple[Asset | None, StockAsset | None]:
    if isinstance(value, Asset):
        return value, None
    if isinstance(value, StockAsset):
        return None, value
    asset = getattr(value, "asset", None)
    if isinstance(asset, Asset):
        return asset, None
    if isinstance(asset, StockAsset):
        return None, asset
    stock = getattr(value, "stock", None)
    if isinstance(stock, StockAsset):
        return None, stock
    if isinstance(value, Mapping):
        nested = value.get("asset")
        if isinstance(nested, Asset):
            return nested, None
        if isinstance(nested, Mapping):
            return Asset.model_validate(dict(nested)), None
        nested = value.get("stock")
        if isinstance(nested, StockAsset):
            return None, nested
        if isinstance(nested, Mapping):
            return None, StockAsset.model_validate(dict(nested))
    return None, None


def asset_row_from_stock(
    stock: StockAsset,
    *,
    prompt: str = "",
    run_id: str = "",
    quality_score: float | None = None,
    alt_text: str = "",
    pinned: bool = True,
    cost_links: Iterable[Any] | None = None,
) -> Asset:
    """Return the asset row for one licensed stock result."""
    kind = AssetKind.VIDEO if stock.kind is StockMediaKind.VIDEO else AssetKind.IMAGE
    license_ = AssetLicense(
        name=_text(stock.license),
        url=_text(stock.license_url),
        attribution=_text(stock.attribution),
        attribution_required=bool(_text(stock.attribution)),
    )
    resolved_prompt = prompt or _prompt_of(stock, None, stock)
    resolved_alt = (
        alt_text
        or _alt_text_from(stock.metadata)
        or _text(stock.attribution)
        or resolved_prompt
    )
    return Asset(
        kind=kind,
        label=stock.id,
        provider=_text(stock.provider),
        model=_text(stock.metadata.get("model")),
        prompt=resolved_prompt,
        path=stock.url,
        content_hash=_text(stock.metadata.get("content_hash")),
        license=license_,
        cost_links=list(cost_links or ()),
        analysis=AssetAnalysis(
            width=stock.width,
            height=stock.height,
            duration_seconds=(
                stock.duration_seconds if stock.duration_seconds else None
            ),
            format=(stock.format.value if stock.format is not None else ""),
        ),
        alt_text=resolved_alt,
        pinned=pinned,
        quality_score=quality_score,
        run_id=run_id,
    )


def asset_row(
    value: Any,
    *,
    prompt: str = "",
    run_id: str = "",
    quality_score: float | None = None,
    alt_text: str = "",
    license: AssetLicense | Mapping[str, Any] | None = None,
    cost_links: Iterable[Any] | None = None,
    pinned: bool | None = None,
) -> Asset | None:
    """Return the asset row for one asset, stock result or visual result.

    Explicit keyword overrides win over values already carried by *value*.
    ``None`` is returned when *value* holds neither an asset nor a stock result.
    """
    asset, stock = _result_parts(value)
    if asset is None and stock is None:
        return None
    if asset is None and stock is not None:
        row = asset_row_from_stock(
            stock,
            prompt=prompt or _prompt_of(value, None, stock),
            run_id=run_id or _text(getattr(value, "run_id", "")),
            quality_score=(
                quality_score if quality_score is not None else _score_of(value)
            ),
            alt_text=alt_text,
            pinned=True if pinned is None else bool(pinned),
            cost_links=cost_links,
        )
        return row

    assert asset is not None
    resolved_prompt = prompt or _prompt_of(value, asset, None)
    updates: dict[str, Any] = {}
    if prompt and prompt != asset.prompt:
        updates["prompt"] = prompt
        updates["prompt_hash"] = compute_prompt_hash(prompt)
    elif not asset.prompt_hash and (resolved_prompt or asset.prompt):
        updates["prompt_hash"] = compute_prompt_hash(resolved_prompt or asset.prompt)
    if run_id and run_id != asset.run_id:
        updates["run_id"] = run_id
    if quality_score is not None:
        if asset.quality_score != quality_score:
            updates["quality_score"] = quality_score
    else:
        resolved_score = _score_of(value)
        if resolved_score is not None and asset.quality_score is None:
            updates["quality_score"] = resolved_score
    if alt_text:
        if alt_text != asset.alt_text:
            updates["alt_text"] = alt_text
    elif not asset.alt_text:
        resolved_alt = _alt_text_from(getattr(value, "request", None))
        if resolved_alt:
            updates["alt_text"] = resolved_alt
    resolved_license = _license_from(license)
    if resolved_license is not None:
        updates["license"] = resolved_license
    if cost_links is not None:
        updates["cost_links"] = list(cost_links)
    if pinned is not None and asset.pinned != bool(pinned):
        updates["pinned"] = bool(pinned)
    return asset.model_copy(update=updates) if updates else asset


def _values(value: Any) -> list[Any]:
    if isinstance(value, (Asset, StockAsset, Mapping)):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def asset_rows(values: Any, **kwargs: Any) -> list[Asset]:
    """Convert *values* into asset rows, skipping entries without an asset."""
    rows: list[Asset] = []
    seen: set[str] = set()
    for value in _values(values):
        row = asset_row(value, **kwargs)
        if row is None or row.id in seen:
            continue
        seen.add(row.id)
        rows.append(row)
    return rows


def persist_asset_rows(
    repository: AssetRepository | Any,
    values: Any,
    **kwargs: Any,
) -> list[Asset]:
    """Write every asset row in *values* and return the stored records.

    *repository* may be an :class:`AssetRepository` or an SQLAlchemy engine.
    An existing row with the same id is updated, so a later quality score, alt
    text or cost link is persisted without creating a duplicate.
    """
    target = (
        repository
        if isinstance(repository, AssetRepository)
        else AssetRepository(repository)
    )
    stored: list[Asset] = []
    for row in asset_rows(values, **kwargs):
        if target.get(row.id) is None:
            stored.append(target.create(row))
            continue
        changes = row.model_dump(
            mode="python",
            exclude={"created_at", "updated_at", "schema_version"},
        )
        updated = target.update(row.id, **changes)
        stored.append(updated if updated is not None else row)
    return stored


record_asset_rows = persist_asset_rows
write_asset_rows = persist_asset_rows
