"""``aivia config show`` contract: merge, profiles, origins and masking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from aivia.cli import main
from aivia.core import paths
from aivia.core.redact import REDACTED


def packaged_defaults() -> dict[str, Any]:
    text = (paths.package_defaults_dir() / "config.example.yaml").read_text(
        encoding="utf-8"
    )
    return cast(dict[str, Any], yaml.safe_load(text))


def write_config(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def show(
    config: Path,
    capsys: pytest.CaptureFixture[str],
    *extra: str,
) -> tuple[int, dict[str, Any], str, str]:
    code = main(["config", "show", "--config", str(config), *extra])
    captured = capsys.readouterr()
    data = (
        cast(dict[str, Any], yaml.safe_load(captured.out))
        if captured.out.strip()
        else {}
    )
    return code, data, captured.out, captured.err


def test_show_merges_defaults_user_config_and_profile_overrides(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path / "config.yaml",
        """
schema_version: 1
pipeline:
  timezone: UTC
video:
  fps: 24
profiles:
  short:
    overrides:
      pipeline:
        timezone: Europe/Istanbul
      render:
        require_bundled_fonts: false
""".lstrip(),
    )

    code, data, output, _ = show(
        config, capsys, "--profile", "short", "--origins", "--resolved"
    )

    defaults = packaged_defaults()
    assert code == 0
    assert data["schema_version"] == 1
    assert data["pipeline"]["timezone"] == "Europe/Istanbul"
    assert data["pipeline"]["stages"] == defaults["pipeline"]["stages"]
    assert data["video"]["fps"] == 24
    assert data["render"]["require_bundled_fonts"] is False
    assert "# profile: short" in output
    assert "# schema_version: user" in output
    assert "# pipeline.stages: defaults" in output
    assert "# pipeline.timezone: profile:short" in output
    assert "# video.fps: user" in output
    assert "# render.require_bundled_fonts: profile:short" in output


def test_show_deep_merges_dicts_and_replaces_lists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path / "config.yaml",
        """
pipeline:
  stages: [write, render]
""".lstrip(),
    )

    code, data, _, _ = show(config, capsys)

    defaults = packaged_defaults()
    assert code == 0
    assert data["pipeline"]["stages"] == ["write", "render"]
    assert data["pipeline"]["timezone"] == defaults["pipeline"]["timezone"]
    assert data["render"] == defaults["render"]


def test_show_works_without_a_user_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "does-not-exist" / "config.yaml"

    code, data, _, err = show(missing, capsys)

    assert code == 0
    assert err == ""
    assert data == packaged_defaults()


def test_show_rejects_an_unknown_profile_with_its_peers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path / "config.yaml",
        """
profiles:
  alpha:
    overrides: {}
  beta:
    overrides: {}
""".lstrip(),
    )

    code, data, output, err = show(config, capsys, "--profile", "missing")

    assert code == 1
    assert data == {}
    assert output == ""
    assert "unknown profile: missing" in err
    assert "available: alpha, beta" in err


def test_show_masks_secret_keys_and_token_like_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    token_like = "sk-" + "x" * 24
    config = write_config(
        tmp_path / "config.yaml",
        f"""
api:
  key: fake-key
  token: fake-token
  password: fakepass
  base_url: https://example.test
llm:
  api_key: fake-api
  model: example-model
note: {token_like}
budget:
  token_budget: 1000
render:
  sort_keys: false
""".lstrip(),
    )

    code, data, output, _ = show(config, capsys)

    assert code == 0
    for secret in ("fake-key", "fake-token", "fakepass", "fake-api", token_like):
        assert secret not in output
    assert data["api"]["key"] == REDACTED
    assert data["api"]["token"] == REDACTED
    assert data["api"]["password"] == REDACTED
    assert data["llm"]["api_key"] == REDACTED
    assert data["note"] == REDACTED
    assert data["api"]["base_url"] == "https://example.test"
    assert data["llm"]["model"] == "example-model"
    assert data["budget"]["token_budget"] == 1000
    assert data["render"]["sort_keys"] is False


def test_show_reports_invalid_yaml_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path / "config.yaml", "pipeline: [unclosed\n")

    code, data, output, err = show(config, capsys)

    assert code == 1
    assert data == {}
    assert output == ""
    assert "invalid YAML" in err
    assert "Traceback" not in err


def test_show_rejects_a_non_mapping_config_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path / "config.yaml", "- not\n- a\n- mapping\n")

    code, data, output, err = show(config, capsys)

    assert code == 1
    assert data == {}
    assert output == ""
    assert "root must be a mapping" in err


def test_show_applies_the_environment_file_and_set_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        tmp_path / "config.yaml",
        """
pipeline:
  timezone: UTC
video:
  fps: 30
""".lstrip(),
    )
    write_config(
        tmp_path / "config.dev.yaml",
        """
pipeline:
  timezone: Europe/Paris
""".lstrip(),
    )

    code, data, output, _ = show(
        config,
        capsys,
        "--env",
        "dev",
        "--set",
        "video.fps=60",
        "--origins",
    )

    assert code == 0
    assert data["pipeline"]["timezone"] == "Europe/Paris"
    assert data["video"]["fps"] == 60
    assert "# pipeline.timezone: env:dev" in output
    assert "# video.fps: cli" in output


def test_show_rejects_a_malformed_set_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path / "config.yaml", "schema_version: 1\n")

    code, data, output, err = show(config, capsys, "--set", "missing-equals")

    assert code == 1
    assert data == {}
    assert output == ""
    assert "expected key.path=value" in err
