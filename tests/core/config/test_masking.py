"""Masking contract for resolved config, run snapshots and sidecars."""

# secret-scan: allow-file -- this module intentionally contains fake tokens.

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from aivia.core.config.masking import (
    mask_artifact,
    mask_prompt,
    mask_resolved_config,
    mask_secrets,
    mask_sidecar,
    mask_snapshot,
)
from aivia.core.config.resolve import ResolvedConfig
from aivia.core.config.sections import PrivacyConfig
from aivia.core.config_show import dump_config
from aivia.core.redact import REDACTED, clear_secrets, register_secret


@pytest.fixture(autouse=True)
def _empty_secret_registry() -> Iterator[None]:
    clear_secrets()
    yield
    clear_secrets()


def test_resolved_config_masking_keeps_runtime_values_intact() -> None:
    register_secret("opaque-runtime-secret")
    resolved = ResolvedConfig(
        values={
            "api": {"key": "plain-api-key", "base_url": "https://example.test"},
            "llm": {"api_key": "plain-llm-key", "model": "example-model"},
            "note": "opaque-runtime-secret",
            "video": {"fps": 30},
        },
        origins={"api.key": "user", "video.fps": "cli"},
        profile="short",
        environment="dev",
        sources=(Path("/tmp/config.yaml"),),
        origin_files={"user": "/tmp/config.yaml"},
    )

    masked = mask_resolved_config(resolved)

    assert masked is not resolved
    assert masked.values == {
        "api": {"key": REDACTED, "base_url": "https://example.test"},
        "llm": {"api_key": REDACTED, "model": "example-model"},
        "note": REDACTED,
        "video": {"fps": 30},
    }
    assert masked.origins == resolved.origins
    assert masked.profile == "short"
    assert masked.environment == "dev"
    assert masked.sources == resolved.sources
    assert masked.origin_files == resolved.origin_files
    assert resolved.values["api"]["key"] == "plain-api-key"
    assert resolved.values["note"] == "opaque-runtime-secret"


def test_config_snapshot_never_contains_secret_values() -> None:
    register_secret("opaque-snapshot-secret")
    token_like = "sk-" + "x" * 24
    snapshot = {
        "config": {
            "api": {"key": "raw-api-key"},
            "note": "opaque-snapshot-secret",
            "model": "example-model",
        },
        "prompt": "draw a fox with opaque-snapshot-secret",
        "items": [{"password": "raw-password"}, token_like],
    }

    masked = mask_snapshot(snapshot)

    serialised = json.dumps(masked)
    for secret in ("raw-api-key", "opaque-snapshot-secret", "raw-password", token_like):
        assert secret not in serialised
    assert masked["config"]["api"]["key"] == REDACTED
    assert masked["config"]["note"] == REDACTED
    assert masked["config"]["model"] == "example-model"
    assert masked["prompt"] == f"draw a fox with {REDACTED}"
    assert masked["items"] == [{"password": REDACTED}, REDACTED]


def test_sidecar_prompt_obeys_the_redact_and_store_settings() -> None:
    prompt = "draw a red fox"

    kept = mask_sidecar({"prompt": prompt, "prompt_hash": "abc123"})
    assert kept["prompt"] == prompt
    assert kept["prompt_hash"] == "abc123"

    redacted = mask_sidecar(
        {"prompt": prompt},
        privacy=PrivacyConfig(redact_prompts=True),
    )
    assert redacted["prompt"] == REDACTED

    omitted = mask_sidecar(
        {"prompt": prompt},
        privacy=PrivacyConfig(store_prompt_text=False),
    )
    assert omitted["prompt"] is None

    both = mask_sidecar(
        {"prompt": prompt},
        privacy=PrivacyConfig(redact_prompts=True, store_prompt_text=False),
    )
    assert both["prompt"] is None


def test_prompt_key_variants_and_nested_prompts_are_redacted() -> None:
    sidecar = {
        "prompt_text": "one",
        "nested": {
            "system_prompt": "two",
            "user_prompt": "three",
            "prompt_hash": "keep-me",
        },
        "prompts": ["four", "five"],
    }

    masked = mask_sidecar(sidecar, privacy=PrivacyConfig(redact_prompts=True))

    assert masked["prompt_text"] == REDACTED
    assert masked["nested"]["system_prompt"] == REDACTED
    assert masked["nested"]["user_prompt"] == REDACTED
    assert masked["nested"]["prompt_hash"] == "keep-me"
    assert masked["prompts"] == REDACTED


def test_mask_prompt_masks_secrets_without_hiding_ordinary_text() -> None:
    register_secret("prompt-secret")
    prompt = f"draw prompt-secret and sk-{'y' * 24}"

    masked = mask_prompt(prompt)

    assert "prompt-secret" not in masked
    assert "sk-" + "y" * 24 not in masked
    assert masked == f"draw {REDACTED} and {REDACTED}"


def test_mask_artifact_and_mask_sidecar_share_the_same_contract() -> None:
    register_secret("artifact-secret")
    artifact: dict[str, Any] = {
        "token": "artifact-secret",
        "note": "artifact-secret",
        "prompt": "artifact-secret",
    }

    assert mask_artifact(artifact) == mask_sidecar(artifact)
    assert mask_artifact(artifact) == {
        "token": REDACTED,
        "note": REDACTED,
        "prompt": REDACTED,
    }


def test_token_like_mapping_keys_are_masked() -> None:
    token_key = "sk-" + "k" * 24

    masked = mask_secrets({token_key: "value"})

    assert token_key not in masked
    assert masked == {REDACTED: "value"}


def test_dump_config_masks_registered_secrets_in_arbitrary_keys() -> None:
    register_secret("arbitrary-config-secret")
    resolved = ResolvedConfig(
        values={"note": "arbitrary-config-secret", "model": "example-model"},
        origins={},
    )

    rendered = dump_config(resolved)
    parsed = yaml.safe_load(rendered)

    assert "arbitrary-config-secret" not in rendered
    assert parsed == {"note": REDACTED, "model": "example-model"}
