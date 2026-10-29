"""Masked configuration snapshots recorded for replay."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from aivia.core.config.masking import mask_snapshot
from aivia.core.config.resolve import ResolvedConfig
from aivia.core.config.sections import PrivacyConfig
from aivia.core.models.run import Run, ToolchainFingerprint
from aivia.core.redact import REDACTED, clear_secrets, register_secret
from aivia.core.run_dirs import RunDirectory
from aivia.core.snapshot import (
    SNAPSHOT_FILENAME,
    ConfigSnapshot,
    build_config_snapshot,
    read_config_snapshot,
    snapshot_path,
    write_config_snapshot,
)


@pytest.fixture(autouse=True)
def _empty_secret_registry() -> Iterator[None]:
    clear_secrets()
    yield
    clear_secrets()


def _fingerprint() -> ToolchainFingerprint:
    return ToolchainFingerprint(
        renderer_version="renderer-1",
        ffmpeg_version="ffmpeg-7",
        font_set="fonts-abc",
    )


def test_snapshot_records_masked_config_seeds_providers_models_and_toolchain() -> None:
    register_secret("opaque-snapshot-secret")
    config = {
        "llm": {
            "chain": ["acme", "other"],
            "roles": {"default": "model-1", "writing": "model-2"},
            "pricing": {"model-1": {"input_per_million": 1.0}},
            "api_key": "plain-api-key",
        },
        "providers": {
            "acme": {"type": "acme", "options": {"api_key": "provider-secret"}},
            "other": {"type": "other"},
        },
        "planner": {"defaults": {"seed": 42}},
        "note": "opaque-snapshot-secret",
        "prompt": "draw a fox",
    }
    run = Run(toolchain_fingerprint=_fingerprint(), profile="short")

    snapshot = build_config_snapshot(config, run=run)

    assert isinstance(snapshot, ConfigSnapshot)
    assert snapshot["run_id"] == run.id
    assert snapshot["profile"] == "short"
    assert snapshot["config"]["note"] == REDACTED
    assert snapshot["config"]["llm"]["api_key"] == REDACTED
    assert snapshot["config"]["providers"]["acme"]["options"]["api_key"] == REDACTED
    assert snapshot["config"]["prompt"] == "draw a fox"
    assert snapshot["seeds"] == {"planner.defaults.seed": 42}
    assert snapshot["providers"] == ["acme", "other"]
    assert snapshot["models"] == ["model-1", "model-2"]
    assert snapshot["model_roles"] == {"default": "model-1", "writing": "model-2"}
    assert snapshot["toolchain_fingerprint"]["renderer_version"] == "renderer-1"
    assert snapshot["toolchain_fingerprint"]["ffmpeg_version"] == "ffmpeg-7"
    assert snapshot["toolchain_fingerprint"]["font_set"] == "fonts-abc"
    assert snapshot["fingerprint"] == _fingerprint().digest

    serialised = json.dumps(snapshot)
    for secret in ("plain-api-key", "provider-secret", "opaque-snapshot-secret"):
        assert secret not in serialised


def test_explicit_replay_inputs_override_values_extracted_from_config() -> None:
    config = {
        "llm": {"chain": ["acme"], "roles": {"default": "model-1"}},
        "providers": {"acme": {"type": "acme"}},
    }

    snapshot = build_config_snapshot(
        config,
        seeds={"visuals": 7},
        providers=["chosen"],
        models=["chosen-model"],
        toolchain_fingerprint=_fingerprint(),
    )

    assert snapshot.seeds == {"visuals": 7}
    assert snapshot.providers == ["chosen"]
    assert snapshot.models == ["chosen-model"]
    assert snapshot.toolchain.renderer_version == "renderer-1"


def test_resolved_config_keeps_profile_and_environment_provenance() -> None:
    resolved = ResolvedConfig(
        values={"planner": {"defaults": {"seed": 3}}},
        origins={"planner.defaults.seed": "user"},
        profile="portrait",
        environment="staging",
    )

    snapshot = build_config_snapshot(resolved)

    assert snapshot["config"] == resolved.values
    assert snapshot["profile"] == "portrait"
    assert snapshot["environment"] == "staging"
    assert snapshot["seeds"] == {"planner.defaults.seed": 3}


def test_prompt_text_follows_the_privacy_settings() -> None:
    config = {"privacy": {"redact_prompts": True}, "prompt": "draw a fox"}

    snapshot = build_config_snapshot(config)

    assert snapshot["config"]["prompt"] == REDACTED

    omitted = build_config_snapshot(
        {"privacy": {"store_prompt_text": False}, "prompt": "draw a fox"},
        privacy=PrivacyConfig(store_prompt_text=False),
    )
    assert omitted["config"]["prompt"] is None


def test_snapshot_round_trips_through_the_run_directory(tmp_path: Path) -> None:
    directory = RunDirectory(tmp_path).create()
    run = Run(toolchain_fingerprint=_fingerprint())
    snapshot = build_config_snapshot(
        {"planner": {"defaults": {"seed": 11}}},
        run=run,
        seeds={"visuals": 11},
        providers=["acme"],
        models=["model-1"],
    )

    written = directory.write_snapshot(snapshot)
    assert written == snapshot_path(tmp_path)
    assert written.name == SNAPSHOT_FILENAME
    assert written.is_file()

    loaded = directory.read_snapshot()
    assert loaded == snapshot
    assert loaded.seeds == {"visuals": 11}
    assert loaded.providers == ["acme"]
    assert loaded.toolchain["digest"] == _fingerprint().digest


def test_write_builds_the_snapshot_and_returns_the_path(tmp_path: Path) -> None:
    path = write_config_snapshot(
        tmp_path,
        config={"llm": {"roles": {"default": "model-1"}}},
        toolchain_fingerprint=_fingerprint(),
        seeds={"image": 5},
    )

    assert path == tmp_path / SNAPSHOT_FILENAME
    loaded = read_config_snapshot(path)
    assert loaded.models == ["model-1"]
    assert loaded.seeds == {"image": 5}
    assert loaded.fingerprint == _fingerprint().digest


def test_read_accepts_an_explicit_snapshot_file(tmp_path: Path) -> None:
    path = write_config_snapshot(tmp_path, build_config_snapshot({}))

    assert read_config_snapshot(path)["schema_version"] == 1


def test_mask_snapshot_contract_is_used_for_the_whole_payload() -> None:
    register_secret("payload-secret")
    snapshot = build_config_snapshot({"note": "payload-secret"})

    assert snapshot == mask_snapshot(snapshot)


def test_orchestrator_saves_the_snapshot_and_records_it_on_the_run(
    tmp_path: Path,
) -> None:
    import asyncio

    from aivia.core.models.run import StageResult
    from aivia.core.orchestrator import Orchestrator
    from aivia.core.stages import Stage, StageContext

    class Write(Stage):
        name = "write"

        async def run(self, context: StageContext) -> StageResult:
            return StageResult(stage=self.name)

    run = Run(toolchain_fingerprint=_fingerprint())
    result = asyncio.run(
        Orchestrator(
            stages=[Write()],
            run=run,
            run_dir=tmp_path,
            config={
                "llm": {"roles": {"default": "model-1"}},
                "planner": {"defaults": {"seed": 9}},
            },
        ).execute()
    )

    loaded = read_config_snapshot(tmp_path)
    assert loaded["run_id"] == run.id
    assert loaded.models == ["model-1"]
    assert loaded.seeds == {"planner.defaults.seed": 9}
    assert loaded.fingerprint == _fingerprint().digest
    assert result.run.config_snapshot == dict(loaded)
