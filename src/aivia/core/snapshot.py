"""Masked configuration snapshots recorded for replay.

A run snapshot is the reproducibility record a later replay reads: the resolved
configuration with credentials and prompt text masked, the seeds that make
random choices repeatable, the provider and model ids that were selected and
the renderer toolchain fingerprint. It is written into the run directory as
canonical JSON so the same inputs always produce the same bytes.

The snapshot only describes configuration and provenance. It never resolves a
provider, calls a model or reads media; those belong to the runtime. Values are
masked with :func:`aivia.core.config.masking.mask_snapshot` before they reach
disk, so a secret registered by the secrets loader cannot leak through a nested
provider option or a prompt.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from aivia.core.config.masking import mask_snapshot
from aivia.core.config.resolve import ResolvedConfig
from aivia.core.config.sections import PrivacyConfig
from aivia.core.models.base import canonical_json
from aivia.core.models.run import Run, ToolchainFingerprint

__all__ = [
    "CONFIG_SNAPSHOT_FILENAME",
    "CONFIG_SNAPSHOT_NAME",
    "RUN_SNAPSHOT_FILENAME",
    "SNAPSHOT_FILE",
    "SNAPSHOT_FILENAME",
    "SNAPSHOT_NAME",
    "SNAPSHOT_VERSION",
    "ConfigSnapshot",
    "Snapshot",
    "build_config_snapshot",
    "build_run_snapshot",
    "build_snapshot",
    "capture_snapshot",
    "create_config_snapshot",
    "create_snapshot",
    "load_config_snapshot",
    "load_run_snapshot",
    "load_snapshot",
    "read_config_snapshot",
    "read_run_snapshot",
    "read_snapshot",
    "save_config_snapshot",
    "save_run_snapshot",
    "save_snapshot",
    "snapshot_path",
    "snapshot_run",
    "write_config_snapshot",
    "write_run_snapshot",
    "write_snapshot",
]

SNAPSHOT_FILENAME = "snapshot.json"
"""The file in a run directory that holds the replay snapshot."""

SNAPSHOT_NAME = SNAPSHOT_FILENAME
CONFIG_SNAPSHOT_FILENAME = SNAPSHOT_FILENAME
CONFIG_SNAPSHOT_NAME = SNAPSHOT_FILENAME
RUN_SNAPSHOT_FILENAME = SNAPSHOT_FILENAME
SNAPSHOT_FILE = SNAPSHOT_FILENAME

SNAPSHOT_VERSION = 1
"""The snapshot schema version written by this release."""

_SNAPSHOT_ALIASES = ("config.snapshot.json", "config.json")


def _normalise_key(key: object) -> str:
    return str(key).strip().lower().replace("-", "_")


def _as_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _dedupe(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _as_text(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _config_values(config: Any) -> tuple[dict[str, Any], str, str]:
    if config is None:
        return {}, "", ""
    if isinstance(config, ResolvedConfig):
        return (
            copy.deepcopy(config.values),
            _as_text(config.profile),
            _as_text(config.environment),
        )
    if isinstance(config, BaseModel):
        values = config.model_dump(mode="python")
        return (
            copy.deepcopy(values),
            _as_text(getattr(config, "profile", "")),
            _as_text(getattr(config, "environment", "")),
        )
    if isinstance(config, Mapping):
        return copy.deepcopy(dict(config)), "", ""
    raise TypeError("config must be a mapping, a model or None")


def _privacy_from(config: Mapping[str, Any]) -> PrivacyConfig | None:
    raw = config.get("privacy")
    if raw is None:
        return None
    try:
        if isinstance(raw, PrivacyConfig):
            return raw
        if isinstance(raw, Mapping):
            return PrivacyConfig.model_validate(dict(raw))
    except (TypeError, ValueError):
        return None
    return None


def _walk_seeds(
    value: Any,
    *,
    prefix: str,
    found: dict[str, Any],
    seen: set[int],
) -> None:
    if isinstance(value, Mapping):
        if id(value) in seen:
            return
        seen.add(id(value))
        for key, item in value.items():
            name = _normalise_key(key)
            path = f"{prefix}.{key}" if prefix else str(key)
            if name == "seeds" and isinstance(item, Mapping):
                for seed_key, seed_value in item.items():
                    if seed_value is not None:
                        found[f"{path}.{seed_key}"] = seed_value
            elif name == "seed" and item is not None:
                found[path] = item
            _walk_seeds(item, prefix=path, found=found, seen=seen)
    elif isinstance(value, list):
        if id(value) in seen:
            return
        seen.add(id(value))
        for index, item in enumerate(value):
            _walk_seeds(
                item,
                prefix=f"{prefix}.{index}" if prefix else str(index),
                found=found,
                seen=seen,
            )


def _seeds_from(config: Mapping[str, Any]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    _walk_seeds(config, prefix="", found=found, seen=set())
    return found


def _provider_names(config: Mapping[str, Any]) -> list[str]:
    names: list[str] = []

    providers = config.get("providers")
    if isinstance(providers, Mapping):
        nested = providers.get("providers")
        entries = nested if isinstance(nested, Mapping) else providers
        names.extend(
            key
            for key in entries
            if _normalise_key(key) not in {"enabled", "providers"}
        )

    def walk(value: Any, seen: set[int]) -> None:
        if isinstance(value, Mapping):
            if id(value) in seen:
                return
            seen.add(id(value))
            for key, item in value.items():
                name = _normalise_key(key)
                if name in {"chain", "provider_chain"}:
                    if isinstance(item, str):
                        names.append(item)
                    elif isinstance(item, Sequence) and not isinstance(
                        item, (str, bytes)
                    ):
                        names.extend(entry for entry in item if isinstance(entry, str))
                walk(item, seen)
        elif isinstance(value, list):
            if id(value) in seen:
                return
            seen.add(id(value))
            for item in value:
                walk(item, seen)

    walk(config, set())
    return _dedupe(names)


def _model_roles(config: Mapping[str, Any]) -> dict[str, str]:
    llm = config.get("llm")
    if not isinstance(llm, Mapping):
        return {}
    roles = llm.get("roles")
    if not isinstance(roles, Mapping):
        return {}
    return {
        str(role): _as_text(model) for role, model in roles.items() if _as_text(model)
    }


def _model_ids(config: Mapping[str, Any]) -> list[str]:
    llm = config.get("llm")
    if not isinstance(llm, Mapping):
        return []
    models: list[object] = []
    roles = llm.get("roles")
    if isinstance(roles, Mapping):
        models.extend(roles.values())
    pricing = llm.get("pricing")
    if isinstance(pricing, Mapping):
        models.extend(pricing.keys())
    return _dedupe(models)


def _normalise_providers(
    value: Any,
    config: Mapping[str, Any],
) -> Any:
    if value is None:
        return _provider_names(config)
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return _dedupe(value)


def _normalise_models(
    value: Any,
    config: Mapping[str, Any],
) -> Any:
    if value is None:
        return _model_ids(config)
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return _dedupe(value)


def _normalise_seeds(
    value: Any,
    config: Mapping[str, Any],
    seed: int | None,
) -> Any:
    if value is None:
        seeds: Any = _seeds_from(config)
    elif isinstance(value, Mapping):
        seeds = {str(key): item for key, item in value.items()}
    elif isinstance(value, str) or not isinstance(value, Iterable):
        seeds = value
    else:
        seeds = list(value)
    if seed is not None:
        if isinstance(seeds, Mapping):
            seeds = {**seeds, "seed": seed}
        else:
            seeds = {"seed": seed}
    return seeds


def _toolchain_payload(
    value: ToolchainFingerprint | Mapping[str, Any] | str | None,
    run: Run | None,
) -> ToolchainFingerprint:
    if value is None and run is not None:
        value = run.toolchain_fingerprint
    return ToolchainFingerprint.model_validate(value or {})


class _AttributeDict(dict[str, Any]):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _wrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _AttributeDict({key: _wrap(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_wrap(item) for item in value)
    return value


class ConfigSnapshot(dict[str, Any]):
    """A snapshot payload that also exposes its fields as attributes."""

    def __init__(
        self,
        values: Mapping[str, Any] | None = None,
        **options: Any,
    ) -> None:
        super().__init__(_wrap(dict(values or {})), **options)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @property
    def config(self) -> dict[str, Any]:
        """The masked configuration values."""
        return cast(dict[str, Any], self["config"])

    @property
    def seeds(self) -> Any:
        """The recorded seeds."""
        return self["seeds"]

    @property
    def providers(self) -> Any:
        """The recorded provider ids."""
        return self["providers"]

    @property
    def models(self) -> Any:
        """The recorded model ids."""
        return self["models"]

    @property
    def toolchain(self) -> _AttributeDict:
        """Alias for :attr:`toolchain_fingerprint`."""
        return cast(_AttributeDict, self["toolchain_fingerprint"])

    @property
    def fingerprint(self) -> str:
        """The toolchain digest."""
        return cast(str, self["fingerprint"])

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable copy."""
        return copy.deepcopy(dict(self))

    def to_json(self, *, indent: int | None = None) -> str:
        """Return the snapshot as canonical JSON."""
        return canonical_json(self.to_dict(), indent=indent)


def build_config_snapshot(
    config: ResolvedConfig | BaseModel | Mapping[str, Any] | None = None,
    *,
    resolved: ResolvedConfig | BaseModel | Mapping[str, Any] | None = None,
    run: Run | None = None,
    seeds: Any = None,
    seed: int | None = None,
    providers: Any = None,
    models: Any = None,
    model_roles: Mapping[str, str] | None = None,
    toolchain_fingerprint: ToolchainFingerprint | Mapping[str, Any] | str | None = None,
    toolchain: ToolchainFingerprint | Mapping[str, Any] | str | None = None,
    privacy: PrivacyConfig | None = None,
    run_id: str | None = None,
    profile: str | None = None,
    environment: str | None = None,
    generated_at: datetime | None = None,
) -> ConfigSnapshot:
    """Build a masked replay snapshot from *config* and its provenance.

    ``seeds`` may be a mapping, a single value or an iterable. When omitted the
    seed fields found in the configuration are recorded with their dotted path.
    ``providers`` and ``models`` default to the ids named by the configuration.
    The toolchain fingerprint comes from *toolchain_fingerprint*, *toolchain* or
    the run record. Secret values and prompt text are masked according to the
    configuration's ``privacy`` section.
    """
    source = config if config is not None else resolved
    values, source_profile, source_environment = _config_values(source)
    policy = privacy if privacy is not None else _privacy_from(values)
    fingerprint = _toolchain_payload(
        toolchain_fingerprint if toolchain_fingerprint is not None else toolchain,
        run,
    )
    roles = _model_roles(values) if model_roles is None else dict(model_roles)

    payload: dict[str, Any] = {
        "schema_version": SNAPSHOT_VERSION,
        "run_id": _as_text(run_id) or (run.id if run is not None else ""),
        "profile": _as_text(profile)
        or source_profile
        or (_as_text(run.profile) if run is not None else ""),
        "environment": _as_text(environment) or source_environment,
        "config": values,
        "seeds": _normalise_seeds(seeds, values, seed),
        "providers": _normalise_providers(providers, values),
        "models": _normalise_models(models, values),
        "model_roles": roles,
        "toolchain_fingerprint": {
            "renderer_version": fingerprint.renderer_version,
            "ffmpeg_version": fingerprint.ffmpeg_version,
            "font_set": fingerprint.font_set,
            "identity": fingerprint.identity,
            "digest": fingerprint.digest,
            "is_known": fingerprint.is_known,
        },
        "fingerprint": fingerprint.digest,
    }
    if generated_at is not None:
        payload["generated_at"] = generated_at.isoformat()
    masked = mask_snapshot(payload, privacy=policy)
    return ConfigSnapshot(masked)


def build_snapshot(
    config: ResolvedConfig | BaseModel | Mapping[str, Any] | None = None,
    **options: Any,
) -> ConfigSnapshot:
    """Alias for :func:`build_config_snapshot`."""
    return build_config_snapshot(config, **options)


def build_run_snapshot(
    config: ResolvedConfig | BaseModel | Mapping[str, Any] | None = None,
    **options: Any,
) -> ConfigSnapshot:
    """Alias for :func:`build_config_snapshot`."""
    return build_config_snapshot(config, **options)


def snapshot_run(
    config: ResolvedConfig | BaseModel | Mapping[str, Any] | None = None,
    **options: Any,
) -> ConfigSnapshot:
    """Alias for :func:`build_config_snapshot`."""
    return build_config_snapshot(config, **options)


def create_config_snapshot(
    config: ResolvedConfig | BaseModel | Mapping[str, Any] | None = None,
    **options: Any,
) -> ConfigSnapshot:
    """Alias for :func:`build_config_snapshot`."""
    return build_config_snapshot(config, **options)


def create_snapshot(
    config: ResolvedConfig | BaseModel | Mapping[str, Any] | None = None,
    **options: Any,
) -> ConfigSnapshot:
    """Alias for :func:`build_config_snapshot`."""
    return build_config_snapshot(config, **options)


def capture_snapshot(
    config: ResolvedConfig | BaseModel | Mapping[str, Any] | None = None,
    **options: Any,
) -> ConfigSnapshot:
    """Alias for :func:`build_config_snapshot`."""
    return build_config_snapshot(config, **options)


def snapshot_path(run_dir: str | Path) -> Path:
    """Return the snapshot path inside *run_dir* without creating it."""
    return Path(run_dir).expanduser() / SNAPSHOT_FILENAME


def _directory_path(run_dir: Any) -> Path:
    path = getattr(run_dir, "path", run_dir)
    return Path(path).expanduser()


def _write_atomic(path: Path, text: str) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def write_config_snapshot(
    run_dir: str | Path,
    snapshot: Mapping[str, Any] | None = None,
    *,
    config: ResolvedConfig | BaseModel | Mapping[str, Any] | None = None,
    resolved: ResolvedConfig | BaseModel | Mapping[str, Any] | None = None,
    run: Run | None = None,
    seeds: Any = None,
    seed: int | None = None,
    providers: Any = None,
    models: Any = None,
    model_roles: Mapping[str, str] | None = None,
    toolchain_fingerprint: ToolchainFingerprint | Mapping[str, Any] | str | None = None,
    toolchain: ToolchainFingerprint | Mapping[str, Any] | str | None = None,
    privacy: PrivacyConfig | None = None,
    run_id: str | None = None,
    profile: str | None = None,
    environment: str | None = None,
    generated_at: datetime | None = None,
    filename: str | None = None,
) -> Path:
    """Write a snapshot into *run_dir* and return its path.

    When *snapshot* is omitted it is built from the configuration and the
    keyword provenance. An existing snapshot file is replaced atomically so a
    crash never leaves a half-written record.
    """
    directory = _directory_path(run_dir)
    if snapshot is None:
        payload = build_config_snapshot(
            config if config is not None else resolved,
            run=run,
            seeds=seeds,
            seed=seed,
            providers=providers,
            models=models,
            model_roles=model_roles,
            toolchain_fingerprint=toolchain_fingerprint,
            toolchain=toolchain,
            privacy=privacy,
            run_id=run_id,
            profile=profile,
            environment=environment,
            generated_at=generated_at,
        )
    else:
        payload = ConfigSnapshot(copy.deepcopy(dict(snapshot)))
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (filename or SNAPSHOT_FILENAME)
    _write_atomic(target, canonical_json(payload, indent=2) + "\n")
    if run is not None and not run.config_snapshot:
        run.config_snapshot = dict(payload)
    return target


def write_snapshot(
    run_dir: str | Path,
    snapshot: Mapping[str, Any] | None = None,
    **options: Any,
) -> Path:
    """Alias for :func:`write_config_snapshot`."""
    return write_config_snapshot(run_dir, snapshot, **options)


def write_run_snapshot(
    run_dir: str | Path,
    snapshot: Mapping[str, Any] | None = None,
    **options: Any,
) -> Path:
    """Alias for :func:`write_config_snapshot`."""
    return write_config_snapshot(run_dir, snapshot, **options)


def save_config_snapshot(
    run_dir: str | Path,
    snapshot: Mapping[str, Any] | None = None,
    **options: Any,
) -> Path:
    """Alias for :func:`write_config_snapshot`."""
    return write_config_snapshot(run_dir, snapshot, **options)


def save_snapshot(
    run_dir: str | Path,
    snapshot: Mapping[str, Any] | None = None,
    **options: Any,
) -> Path:
    """Alias for :func:`write_config_snapshot`."""
    return write_config_snapshot(run_dir, snapshot, **options)


def save_run_snapshot(
    run_dir: str | Path,
    snapshot: Mapping[str, Any] | None = None,
    **options: Any,
) -> Path:
    """Alias for :func:`write_config_snapshot`."""
    return write_config_snapshot(run_dir, snapshot, **options)


def _snapshot_file(run_dir: str | Path, filename: str | None = None) -> Path:
    path = Path(run_dir).expanduser()
    if path.is_file():
        return path
    if filename is not None:
        return path / filename
    primary = path / SNAPSHOT_FILENAME
    if primary.is_file():
        return primary
    for alias in _SNAPSHOT_ALIASES:
        candidate = path / alias
        if candidate.is_file():
            return candidate
    return primary


def read_config_snapshot(
    run_dir: str | Path,
    *,
    filename: str | None = None,
) -> ConfigSnapshot:
    """Read and validate a snapshot from a run directory or file path."""
    path = _snapshot_file(run_dir, filename)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"invalid snapshot: {path}")
    return ConfigSnapshot(dict(data))


def load_config_snapshot(
    run_dir: str | Path,
    *,
    filename: str | None = None,
) -> ConfigSnapshot:
    """Alias for :func:`read_config_snapshot`."""
    return read_config_snapshot(run_dir, filename=filename)


def read_snapshot(
    run_dir: str | Path,
    *,
    filename: str | None = None,
) -> ConfigSnapshot:
    """Alias for :func:`read_config_snapshot`."""
    return read_config_snapshot(run_dir, filename=filename)


def read_run_snapshot(
    run_dir: str | Path,
    *,
    filename: str | None = None,
) -> ConfigSnapshot:
    """Alias for :func:`read_config_snapshot`."""
    return read_config_snapshot(run_dir, filename=filename)


def load_snapshot(
    run_dir: str | Path,
    *,
    filename: str | None = None,
) -> ConfigSnapshot:
    """Alias for :func:`read_config_snapshot`."""
    return read_config_snapshot(run_dir, filename=filename)


def load_run_snapshot(
    run_dir: str | Path,
    *,
    filename: str | None = None,
) -> ConfigSnapshot:
    """Alias for :func:`read_config_snapshot`."""
    return read_config_snapshot(run_dir, filename=filename)


Snapshot = ConfigSnapshot
