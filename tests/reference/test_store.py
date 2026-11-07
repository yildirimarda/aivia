"""Style specs are stored as editable YAML and reloaded by name."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from aivia.core.errors import ConfigError
from aivia.reference import (
    STYLE_SPEC_DIR,
    StyleSpec,
    StyleSpecError,
    StyleSpecInvalidError,
    StyleSpecNotFoundError,
    StyleSpecStore,
    load_style_spec,
    resolve_style_spec_path,
    save_style_spec,
    style_spec_names,
    style_spec_path,
)


def _spec(name: str = "cinematic") -> StyleSpec:
    return StyleSpec.model_validate(
        {
            "name": name,
            "description": "A calm opening with one payoff.",
            "prompt_suffix": "Natural light and shallow depth of field.",
            "structure": {
                "hook": "Open on a question",
                "beats": ["hook", "explain", "payoff"],
            },
            "pacing": {"cuts_per_minute": 24},
            "palette": {"colors": ["#112233", "#ABCDEF"], "temperature": "warm"},
            "typography": {"font": "Inter", "weight": "bold"},
            "music": {"genre": "ambient", "tempo": 92, "moods": ["calm"]},
            "aspect": "9:16",
            "duration": 12.5,
            "caption_style": "subtitles",
            "reference_ids": ["ref-1"],
        }
    )


def _write(store: StyleSpecStore, name: str, text: str) -> Path:
    path = store.user_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_a_saved_spec_is_editable_yaml_and_reloads_by_name(tmp_path: Path) -> None:
    config = tmp_path / "config"
    store = StyleSpecStore(config_dir=config)
    spec = _spec()

    path = store.save(spec, "cinematic")

    assert path == config / STYLE_SPEC_DIR / "cinematic.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, Mapping)
    assert payload["schema_version"] == spec.schema_version
    assert payload["name"] == "cinematic"
    assert payload["palette"]["colors"] == ["#112233", "#abcdef"]
    assert payload["duration_seconds"] == 12.5

    assert store.load("cinematic") == spec
    assert store.load("cinematic.yaml") == spec
    assert load_style_spec("cinematic", config_dir=config) == spec
    assert style_spec_path("cinematic", config_dir=config) == path
    assert resolve_style_spec_path("cinematic", config_dir=config) == path


def test_an_external_edit_is_picked_up_on_reload(tmp_path: Path) -> None:
    store = StyleSpecStore(config_dir=tmp_path / "config")
    path = store.save(_spec(), "cinematic")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["prompt_suffix"] = "Edited by hand."
    payload["palette"]["colors"] = ["#000000"]
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    reloaded = store.reload("cinematic")

    assert reloaded.prompt_suffix == "Edited by hand."
    assert reloaded.palette.colors == ["#000000"]
    assert store.load("cinematic") == reloaded


def test_save_uses_the_spec_name_and_keeps_it_when_saved_under_an_alias(
    tmp_path: Path,
) -> None:
    store = StyleSpecStore(config_dir=tmp_path / "config")
    spec = _spec("Cinematic short")

    default_path = store.save(spec)
    alias_path = store.save(spec, "alias")

    assert default_path.name == "Cinematic short.yaml"
    assert alias_path.name == "alias.yaml"
    assert store.load("Cinematic short").name == "Cinematic short"
    assert store.load("alias").name == "Cinematic short"


def test_names_lists_yaml_specs_and_ignores_other_files(tmp_path: Path) -> None:
    store = StyleSpecStore(config_dir=tmp_path / "config")
    spec = _spec()

    store.save(spec, "alpha")
    store.save(spec, "beta.yml")
    _write(store, "notes.txt", "not a spec\n")
    _write(store, ".hidden.yaml", "name: hidden\n")

    assert store.names() == ("alpha", "beta")
    assert style_spec_names(config_dir=store.config_dir) == ("alpha", "beta")
    assert "alpha" in store
    assert "missing" not in store
    assert list(store) == ["alpha", "beta"]
    assert len(store) == 2
    assert store.load("beta") == spec


def test_a_missing_spec_names_the_directory_and_available_names(
    tmp_path: Path,
) -> None:
    store = StyleSpecStore(config_dir=tmp_path / "config")
    store.save(_spec(), "cinematic")

    with pytest.raises(StyleSpecNotFoundError, match="not found") as caught:
        store.load("missing")

    assert isinstance(caught.value, ConfigError)
    assert caught.value.code == "style_spec_not_found"
    assert "missing" in caught.value.message
    assert str(store.user_dir) in caught.value.message
    assert caught.value.details["available"] == ["cinematic"]

    with pytest.raises(StyleSpecNotFoundError, match="not found") as direct:
        store.load(tmp_path / "elsewhere" / "missing.yaml")

    assert str(tmp_path / "elsewhere" / "missing.yaml") in direct.value.message


def test_malformed_yaml_and_a_non_mapping_root_fail_clearly(tmp_path: Path) -> None:
    store = StyleSpecStore(config_dir=tmp_path / "config")
    broken = _write(store, "broken.yaml", "palette: [\n")
    sequence = _write(store, "sequence.yaml", "- one\n- two\n")

    with pytest.raises(
        StyleSpecInvalidError, match="invalid style spec YAML"
    ) as caught:
        store.load("broken")

    assert str(broken) in caught.value.message
    assert caught.value.code == "style_spec_invalid"

    with pytest.raises(StyleSpecInvalidError, match="root must be a mapping") as caught:
        store.load("sequence")

    assert str(sequence) in caught.value.message


def test_a_schema_violation_names_the_file_and_field(tmp_path: Path) -> None:
    store = StyleSpecStore(config_dir=tmp_path / "config")
    path = _write(store, "bad.yaml", "name: bad\nduration_seconds: -1\n")

    with pytest.raises(StyleSpecInvalidError) as caught:
        store.load("bad")

    assert str(path) in caught.value.message
    assert "duration_seconds" in caught.value.message
    assert caught.value.code == "style_spec_invalid"


def test_an_empty_file_is_rejected(tmp_path: Path) -> None:
    store = StyleSpecStore(config_dir=tmp_path / "config")
    _write(store, "empty.yaml", "\n")

    with pytest.raises(StyleSpecInvalidError, match="empty style spec"):
        store.load("empty")


def test_names_cannot_escape_the_user_directory(tmp_path: Path) -> None:
    store = StyleSpecStore(config_dir=tmp_path / "config")
    spec = _spec()

    for name in ("../escape", "nested/name", tmp_path / "absolute", ""):
        with pytest.raises(StyleSpecError):
            store.save(spec, name)

    with pytest.raises(StyleSpecError, match="unsupported style spec extension"):
        store.save(spec, "notes.txt")


def test_saving_twice_is_deterministic(tmp_path: Path) -> None:
    store = StyleSpecStore(config_dir=tmp_path / "config")
    spec = _spec()

    path = store.save(spec, "cinematic")
    first = path.read_text(encoding="utf-8")
    store.save(spec, "cinematic")

    assert path.read_text(encoding="utf-8") == first


def test_the_store_defaults_to_the_xdg_config_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    spec: Any = _spec("environment")

    path = save_style_spec(spec, "environment")

    assert path == tmp_path / "aivia" / STYLE_SPEC_DIR / "environment.yaml"
    assert load_style_spec("environment") == spec
